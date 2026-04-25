"""
src/api/app.py
FastAPI backend for the traffic enforcement system.

Endpoints:
  POST /auth/token            — Login, returns JWT
  POST /upload_video          — Upload MP4/AVI → batch pipeline (background)
  POST /start_live            — Start live camera pipeline
  POST /stop_live             — Stop live pipeline
  GET  /violations            — List violations (filterable)
  GET  /vehicle/{id}          — Vehicle detail + violation history
  GET  /score/{vehicle_id}    — Credit score + category
  PATCH /review/{id}          — Approve / Reject violation

Security: JWT Bearer token (python-jose), RBAC, CORS enabled.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, HTTPException,
    Query, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.db import (
    get_session, init_db, log_audit, update_violation_status,
    check_permission,
)
from src.database.models import CreditScore, User, Vehicle, Violation
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════
# App Setup
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Smart Traffic Enforcement System API",
    version="1.0.0",
    description="Vision-based vehicle behavior analysis and enforcement.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to dashboard URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────
_CFG          = load_config()
_SEC_CFG      = _CFG.get("security", {})
JWT_SECRET    = os.environ.get("JWT_SECRET", _SEC_CFG.get("jwt_secret", "changeme"))
JWT_ALGORITHM = _SEC_CFG.get("jwt_algorithm", "HS256")
JWT_EXPIRE    = int(_SEC_CFG.get("token_expire_minutes", 60))

_pwd_ctx      = CryptContext(schemes=["bcrypt"], deprecated="auto")
_oauth2       = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ── Live pipeline state ───────────────────────────────────────────────
_live_threads: Dict[str, threading.Thread] = {}
_live_stop:    Dict[str, threading.Event]  = {}


# ══════════════════════════════════════════════════════════════════════
# Pydantic Schemas
# ══════════════════════════════════════════════════════════════════════

class Token(BaseModel):
    access_token: str
    token_type:   str

class TokenData(BaseModel):
    username: Optional[str] = None
    role:     Optional[str] = None

class ViolationSchema(BaseModel):
    id:             int
    vehicle_id:     int
    type:           str
    timestamp:      str
    speed_kmh:      Optional[float]
    fine_inr:       Optional[float]
    plate_text:     Optional[str]
    ocr_confidence: Optional[float]
    evidence_image: Optional[str]
    evidence_clip:  Optional[str]
    status:         str
    mv_act_section: Optional[str]
    camera_id:      Optional[str]
    class Config:
        from_attributes = True

class VehicleSchema(BaseModel):
    id:         int
    class_name: str
    camera_id:  Optional[str]
    first_seen: str
    last_seen:  str
    class Config:
        from_attributes = True

class ScoreSchema(BaseModel):
    vehicle_id:     int
    score:          int
    category:       str
    total_fines_inr: float

class ReviewRequest(BaseModel):
    status:  str        # "approved" | "rejected"
    comment: Optional[str] = None

class LiveStartRequest(BaseModel):
    camera_id:   str = "cam_00"
    rtsp_url:    Optional[str] = None
    webcam_index: Optional[int] = None

class HighwayRestrictionConfig(BaseModel):
    enabled: bool


# ══════════════════════════════════════════════════════════════════════
# Auth Helpers
# ══════════════════════════════════════════════════════════════════════

def _verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)

def _create_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + expires_delta
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def _get_current_user(token: str = Depends(_oauth2)) -> TokenData:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        role:     str = payload.get("role", "readonly")
        if username is None:
            raise exc
        return TokenData(username=username, role=role)
    except JWTError:
        raise exc

def _require_role(action: str):
    """Dependency factory: verify JWT and RBAC permission."""
    async def _dep(current: TokenData = Depends(_get_current_user)):
        if not check_permission(current.role, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current.role}' cannot perform '{action}'.",
            )
        return current
    return _dep


# ══════════════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    db_url = os.environ.get("DATABASE_URL", _CFG["system"]["db_url"])
    init_db(db_url)
    logger.info("FastAPI startup complete.")


# ══════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════

# ── Auth ─────────────────────────────────────────────────────────────

@app.post("/auth/token", response_model=Token, tags=["Auth"])
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Login with username/password → returns JWT Bearer token."""
    with get_session() as db:
        user = db.query(User).filter_by(username=form.username).first()
        if not user or not _verify_password(form.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password.",
            )
        token = _create_token(
            {"sub": user.username, "role": user.role},
            timedelta(minutes=JWT_EXPIRE),
        )
        username = user.username

    log_audit("LOGIN", actor=username)
    return {"access_token": token, "token_type": "bearer"}


# ── Video Upload ──────────────────────────────────────────────────────

@app.post("/upload_video", tags=["Pipeline"])
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current: TokenData = Depends(_require_role("upload")),
):
    """Upload an MP4/AVI file and run the batch pipeline in background."""
    allowed = {".mp4", ".avi", ".mov"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use {allowed}.")

    # Save uploaded file to a temp path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    finally:
        tmp.close()

    background_tasks.add_task(_run_batch_bg, tmp_path, current.username)
    log_audit("UPLOAD_VIDEO", actor=current.username, new_value={"file": file.filename})
    return {"status": "processing", "file": file.filename, "tmp_path": tmp_path}


def _run_batch_bg(video_path: str, actor: str) -> None:
    """Background task: run batch pipeline then remove temp file."""
    try:
        from src.pipeline.batch_pipeline import BatchPipeline
        cfg = load_config()
        pipeline = BatchPipeline(cfg)
        pipeline.run(video_path)
    except Exception as e:
        logger.error(f"Batch pipeline error: {e}")
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass


# ── Live Pipeline ────────────────────────────────────────────────────

@app.post("/start_live", tags=["Pipeline"])
async def start_live(
    req:     LiveStartRequest,
    current: TokenData = Depends(_require_role("start_live")),
):
    """Start a live camera pipeline for the given camera_id."""
    cam_id = req.camera_id
    if cam_id in _live_threads and _live_threads[cam_id].is_alive():
        return {"status": "already_running", "camera_id": cam_id}

    stop_event = threading.Event()
    _live_stop[cam_id] = stop_event

    thread = threading.Thread(
        target=_run_live_bg,
        args=(req, stop_event),
        daemon=True,
        name=f"live-{cam_id}",
    )
    thread.start()
    _live_threads[cam_id] = thread
    log_audit("START_LIVE", actor=current.username, new_value={"camera_id": cam_id})
    return {"status": "started", "camera_id": cam_id}


def _run_live_bg(req: LiveStartRequest, stop_event: threading.Event) -> None:
    try:
        from src.pipeline.realtime_pipeline import RealtimePipeline
        cfg = load_config()
        pipeline = RealtimePipeline(cfg, stop_event=stop_event)
        pipeline.run(
            camera_id=req.camera_id,
            rtsp_url=req.rtsp_url,
            webcam_index=req.webcam_index,
        )
    except Exception as e:
        logger.error(f"Live pipeline error ({req.camera_id}): {e}")


@app.post("/stop_live", tags=["Pipeline"])
async def stop_live(
    camera_id: str = Query(...),
    current:   TokenData = Depends(_require_role("stop_live")),
):
    """Stop a running live pipeline."""
    if camera_id not in _live_stop:
        raise HTTPException(404, f"No live pipeline for camera '{camera_id}'.")
    _live_stop[camera_id].set()
    log_audit("STOP_LIVE", actor=current.username, new_value={"camera_id": camera_id})
    return {"status": "stopping", "camera_id": camera_id}


# ── Violations ────────────────────────────────────────────────────────

@app.get("/violations", response_model=List[ViolationSchema], tags=["Data"])
async def list_violations(
    status_filter: Optional[str] = Query(None, alias="status"),
    vtype:         Optional[str] = Query(None, alias="type"),
    plate:         Optional[str] = Query(None),
    date_from:     Optional[str] = Query(None),
    date_to:       Optional[str] = Query(None),
    limit:         int = Query(100, le=500),
    offset:        int = Query(0),
    current:       TokenData = Depends(_require_role("view")),
):
    """List violations with optional filters."""
    with get_session() as db:
        q = db.query(Violation)
        if status_filter:
            q = q.filter(Violation.status == status_filter)
        if vtype:
            q = q.filter(Violation.type == vtype)
        if plate:
            q = q.filter(Violation.plate_text.ilike(f"%{plate}%"))
        if date_from:
            q = q.filter(Violation.timestamp >= datetime.fromisoformat(date_from))
        if date_to:
            dt_to = datetime.fromisoformat(date_to)
            if len(date_to) == 10:
                dt_to = dt_to.replace(hour=23, minute=59, second=59, microsecond=999999)
            q = q.filter(Violation.timestamp <= dt_to)
        rows = q.order_by(Violation.timestamp.desc()).offset(offset).limit(limit).all()

        return [
            ViolationSchema(
                id=r.id, vehicle_id=r.vehicle_id, type=r.type,
                timestamp=str(r.timestamp), speed_kmh=r.speed_kmh,
                fine_inr=r.fine_inr, plate_text=r.plate_text,
                ocr_confidence=r.ocr_confidence, evidence_image=r.evidence_image,
                evidence_clip=r.evidence_clip, status=r.status,
                mv_act_section=r.mv_act_section, camera_id=r.camera_id,
            )
            for r in rows
        ]


# ── Vehicle Detail ────────────────────────────────────────────────────

@app.get("/vehicle/{vehicle_id}", tags=["Data"])
async def get_vehicle(
    vehicle_id: int,
    current:    TokenData = Depends(_require_role("view")),
):
    """Get vehicle details and its full violation history."""
    with get_session() as db:
        v = db.query(Vehicle).filter_by(id=vehicle_id).first()
        if not v:
            raise HTTPException(404, "Vehicle not found.")
        violations = [
            ViolationSchema(
                id=r.id, vehicle_id=r.vehicle_id, type=r.type,
                timestamp=str(r.timestamp), speed_kmh=r.speed_kmh,
                fine_inr=r.fine_inr, plate_text=r.plate_text,
                ocr_confidence=r.ocr_confidence, evidence_image=r.evidence_image,
                evidence_clip=r.evidence_clip, status=r.status,
                mv_act_section=r.mv_act_section, camera_id=r.camera_id,
            )
            for r in v.violations
        ]
        return {
            "vehicle": VehicleSchema(
                id=v.id, class_name=v.class_name, camera_id=v.camera_id,
                first_seen=str(v.first_seen), last_seen=str(v.last_seen),
            ),
            "violations": violations,
        }


# ── Credit Score ──────────────────────────────────────────────────────

@app.get("/score/{vehicle_id}", response_model=ScoreSchema, tags=["Data"])
async def get_score(
    vehicle_id: int,
    current:    TokenData = Depends(_require_role("view")),
):
    """Get the credit score and risk category for a vehicle."""
    with get_session() as db:
        row = db.query(CreditScore).filter_by(vehicle_id=vehicle_id).first()
        if not row:
            raise HTTPException(404, "Score not found for this vehicle.")
        return ScoreSchema(
            vehicle_id=row.vehicle_id,
            score=row.score,
            category=row.category,
            total_fines_inr=row.total_fines_inr or 0.0,
        )


# ── Human Review ──────────────────────────────────────────────────────

@app.patch("/review/{violation_id}", tags=["Review"])
async def review_violation(
    violation_id: int,
    req:          ReviewRequest,
    background_tasks: BackgroundTasks,
    current:      TokenData = Depends(_require_role("review")),
):
    """
    Approve or reject a violation.
    On approval → triggers notification (in background).
    """
    if req.status not in ("approved", "rejected"):
        raise HTTPException(400, "status must be 'approved' or 'rejected'.")

    ok = update_violation_status(violation_id, req.status, actor=current.username)
    if not ok:
        raise HTTPException(404, f"Violation #{violation_id} not found.")

    if req.status == "approved":
        background_tasks.add_task(_send_notification_bg, violation_id, current.username)

    return {"violation_id": violation_id, "new_status": req.status}


def _send_notification_bg(violation_id: int, actor: str) -> None:
    """Fire notification after approval (runs in background thread)."""
    try:
        from src.notifications.notifier import Notifier
        cfg  = load_config()
        notifier = Notifier(cfg.get("notifications", {}))
        with get_session() as db:
            viol = db.query(Violation).filter_by(id=violation_id).first()
            if viol:
                notifier.send_email(viol)
                notifier.send_sms(viol)
    except Exception as e:
        logger.error(f"Notification error for violation #{violation_id}: {e}")
# ── Configuration ─────────────────────────────────────────────────────

@app.patch("/config/highway_restriction", tags=["Config"])
async def update_highway_restriction(
    req: HighwayRestrictionConfig,
    current: TokenData = Depends(_require_role("admin")),
):
    """Dynamically toggle the highway restriction detection."""
    import re
    from src.utils.helpers import load_config, _CONFIG_LOCK
    
    cfg = load_config()
    with _CONFIG_LOCK:
        if "behavior" not in cfg:
            cfg["behavior"] = {}
        if "highway_restriction" not in cfg["behavior"]:
            cfg["behavior"]["highway_restriction"] = {}
        cfg["behavior"]["highway_restriction"]["enabled"] = req.enabled
        
        # Persist to settings.yaml
        try:
            with open("config/settings.yaml", "r") as f:
                content = f.read()
            new_val = "true" if req.enabled else "false"
            new_content = re.sub(
                r'(highway_restriction:\s*\n\s*enabled:\s*)(true|false)', 
                rf'\g<1>{new_val}', 
                content, 
                flags=re.IGNORECASE
            )
            with open("config/settings.yaml", "w") as f:
                f.write(new_content)
        except Exception as e:
            logger.error(f"Failed to write config file: {e}")
            
    log_audit("CONFIG_UPDATE", actor=current.username, new_value={"highway_restriction_enabled": req.enabled})
    return {"status": "success", "highway_restriction_enabled": req.enabled}


# ══════════════════════════════════════════════════════════════════════
# Run directly
# ══════════════════════════════════════════════════════════════════════

def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)
