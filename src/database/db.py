"""
src/database/db.py
Database session management, AES-256 encryption, RBAC helpers,
audit logging, and blockchain hash chaining.

Supports:
  - SQLite (default, no setup needed)
  - PostgreSQL (set DATABASE_URL env var)
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import (
    AuditLog, Base, BlockchainLog, CreditScore,
    User, Vehicle, Violation,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Engine Setup
# ══════════════════════════════════════════════════════════════════════

def _build_engine(db_url: str):
    """Create SQLAlchemy engine. Adds check_same_thread=False for SQLite."""
    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    
    engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        echo=False,
    )

    if db_url.startswith("sqlite"):
        from sqlalchemy import event
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


_ENGINE  = None
_SESSION = None


def init_db(db_url: Optional[str] = None) -> None:
    """
    Initialise the database engine, create all tables, and seed the
    default admin user (if not already present).

    Args:
        db_url: SQLAlchemy connection string.
                Defaults to DATABASE_URL env var → 'sqlite:///traffic.db'.
    """
    global _ENGINE, _SESSION

    url = db_url or os.environ.get("DATABASE_URL", "sqlite:///traffic.db")
    _ENGINE  = _build_engine(url)
    _SESSION = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)

    Base.metadata.create_all(_ENGINE)
    logger.info(f"Database initialised: {url}")

    _check_security_config()
    _seed_admin()


_DEFAULT_SECRETS = {
    "CHANGE_ME_IN_PRODUCTION_32CHARS!!",
    "CHANGEME_32bytes_key!!",
    "CHANGE_ME_JWT_SECRET_KEY",
    "",
}


def _check_security_config() -> None:
    """
    Warn loudly if any placeholder secrets are still active.
    Reads from environment variables so values can be overridden without
    editing the config file (recommended for production).
    """
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = {}

    aes_key  = os.environ.get("AES_SECRET_KEY",  cfg.get("system",   {}).get("secret_key", ""))
    jwt_key  = os.environ.get("JWT_SECRET_KEY",  cfg.get("security", {}).get("jwt_secret",  ""))
    adm_pass = os.environ.get("ADMIN_PASS",      cfg.get("security", {}).get("default_admin_password", ""))

    if aes_key in _DEFAULT_SECRETS:
        logger.critical(
            "SECURITY: system.secret_key is still the default placeholder! "
            "Plate encryption is WEAK. Set AES_SECRET_KEY environment variable "
            "or change secret_key in config/settings.yaml before production use."
        )
    if jwt_key in _DEFAULT_SECRETS:
        logger.critical(
            "SECURITY: security.jwt_secret is the default placeholder! "
            "API tokens can be forged. Set JWT_SECRET_KEY env var."
        )
    if adm_pass in {"admin123", "CHANGE_ME", "admin", ""}:
        logger.warning(
            "SECURITY: Default admin password 'admin123' is active. "
            "Change via ADMIN_PASS environment variable or settings.yaml."
        )


def _seed_admin() -> None:
    """Create the default admin user if not present."""
    try:
        from passlib.context import CryptContext
        ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

        admin_user = os.environ.get("ADMIN_USER", "admin")
        admin_pass = os.environ.get("ADMIN_PASS", "admin123")
        # bcrypt limit: 72 bytes
        admin_pass_bytes = admin_pass.encode("utf-8")[:72].decode("utf-8", errors="ignore")

        with get_session() as db:
            existing = db.query(User).filter_by(username=admin_user).first()
            if not existing:
                user = User(
                    username=admin_user,
                    hashed_password=ctx.hash(admin_pass_bytes),
                    role="admin",
                )
                db.add(user)
                db.commit()
                logger.info(f"Default admin user '{admin_user}' created.")
    except Exception as e:
        logger.warning(f"_seed_admin skipped: {e}")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy session as a context manager.

    Usage:
        with get_session() as db:
            db.query(Vehicle).all()
    """
    if _SESSION is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    session: Session = _SESSION()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════
# AES-256 Encryption (plate numbers, GPS coords)
# ══════════════════════════════════════════════════════════════════════

def _get_aes_key(secret: str) -> bytes:
    """Derive a 32-byte AES key from the secret string via SHA-256."""
    return hashlib.sha256(secret.encode()).digest()


def encrypt_field(value: str, secret: str) -> str:
    """
    AES-256-CBC encrypt a string field.

    Returns hex-encoded ciphertext (IV prepended).
    Falls back to plain base64 if PyCryptodome not installed.

    Args:
        value:  Plaintext string.
        secret: Secret key (from config or env var).

    Returns:
        Encrypted hex string.
    """
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad
        from Crypto.Random import get_random_bytes
        import base64

        key = _get_aes_key(secret)
        iv  = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ct     = cipher.encrypt(pad(value.encode(), AES.block_size))
        return base64.b64encode(iv + ct).decode()
    except ImportError:
        logger.warning("PyCryptodome not installed — storing plain text.")
        return value
    except Exception as e:
        logger.error(f"encrypt_field error: {e}")
        return value


def decrypt_field(ciphertext: str, secret: str) -> str:
    """
    Decrypt a value previously encrypted with encrypt_field().

    Args:
        ciphertext: Hex-encoded ciphertext.
        secret:     Must match the key used during encryption.

    Returns:
        Decrypted plaintext string, or ciphertext on failure.
    """
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad
        import base64

        key  = _get_aes_key(secret)
        raw  = base64.b64decode(ciphertext)
        iv   = raw[:16]
        ct   = raw[16:]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), AES.block_size).decode()
    except Exception:
        return ciphertext  # return as-is if not encrypted


# ══════════════════════════════════════════════════════════════════════
# Audit Logging
# ══════════════════════════════════════════════════════════════════════

def log_audit(
    action:       str,
    actor:        str  = "system",
    target_table: str  = "",
    target_id:    Optional[int] = None,
    old_value:    Optional[dict] = None,
    new_value:    Optional[dict] = None,
    ip_address:   Optional[str] = None,
) -> None:
    """
    Write one audit row. Called for every significant DB mutation.

    Args:
        action:       Free-text action name (e.g. "CREATE_VIOLATION", "APPROVE").
        actor:        Username or "system".
        target_table: Table name affected.
        target_id:    Primary key of affected row.
        old_value:    Dict snapshot of old state.
        new_value:    Dict snapshot of new state.
        ip_address:   Requester IP (for API calls).
    """
    try:
        with get_session() as db:
            entry = AuditLog(
                action=action,
                actor=actor,
                target_table=target_table,
                target_id=target_id,
                old_value=json.dumps(old_value) if old_value else None,
                new_value=json.dumps(new_value) if new_value else None,
                ip_address=ip_address,
            )
            db.add(entry)
    except Exception as e:
        logger.error(f"Audit log write failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# Blockchain Hash Chaining
# ══════════════════════════════════════════════════════════════════════

_GENESIS_HASH = "0" * 64   # Genesis block previous hash


def compute_block_hash(violation_id: int, violation_data: dict, prev_hash: str) -> str:
    """
    Compute SHA-256(violation_data_json + prev_hash).

    Args:
        violation_id:   Primary key of the Violation row.
        violation_data: Dictionary of violation fields to hash.
        prev_hash:      Hash of the previous block (or genesis hash).

    Returns:
        64-character hex digest.
    """
    payload = json.dumps({
        "violation_id": violation_id,
        "data": violation_data,
        "prev_hash": prev_hash,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def append_blockchain_entry(violation_id: int, violation_data: dict) -> str:
    """
    Append a new tamper-evident entry to BlockchainLog.

    Reads the last tx_hash as the prev_hash, computes SHA-256,
    and inserts a new row.

    Args:
        violation_id:   ID of the just-saved Violation row.
        violation_data: Dict of key violation fields.

    Returns:
        The new tx_hash.
    """
    with get_session() as db:
        last_entry = (
            db.query(BlockchainLog)
            .order_by(BlockchainLog.id.desc())
            .first()
        )
        prev_hash = last_entry.tx_hash if last_entry else _GENESIS_HASH
        tx_hash   = compute_block_hash(violation_id, violation_data, prev_hash)
        entry     = BlockchainLog(
            violation_id=violation_id,
            tx_hash=tx_hash,
            prev_hash=prev_hash,
        )
        db.add(entry)
    logger.debug(f"Blockchain entry: violation_id={violation_id} hash={tx_hash[:16]}…")
    return tx_hash


# ══════════════════════════════════════════════════════════════════════
# RBAC Permission Check
# ══════════════════════════════════════════════════════════════════════

_ROLE_PERMISSIONS = {
    "admin":    {"all"},
    "officer":  {"upload", "start_live", "stop_live", "view"},
    "reviewer": {"view", "review"},
    "readonly": {"view"},
}


def check_permission(role: str, action: str) -> bool:
    """
    Return True if the given role is allowed to perform the action.

    Args:
        role:   User role string.
        action: Action name (e.g. "view", "review", "upload").
    """
    perms = _ROLE_PERMISSIONS.get(role, set())
    return "all" in perms or action in perms


# ══════════════════════════════════════════════════════════════════════
# High-Level DB Write Helpers (used by pipeline)
# ══════════════════════════════════════════════════════════════════════

def save_violation(
    plate_text:     str,
    vehicle_class:  str,
    violation_type: str,
    speed_kmh:      float,
    fine_inr:       float,
    evidence_image: str,
    evidence_clip:  str,
    ocr_confidence: float,
    mv_act:         str,
    camera_id:      str,
    gps_lat:        float,
    gps_lon:        float,
    metadata_dict:  dict,
    secret_key:     str,
    ocr_status:     str = "pending",
    dedup_window_sec: int = 120,
) -> Optional[int]:
    """
    Persist a complete violation record including vehicle, score, and blockchain entry.

    Deduplication: if the same plate already has a violation of the same type
    saved within `dedup_window_sec` seconds, the new record is silently skipped
    and None is returned.  This prevents a vehicle getting multiple fines for
    the same continuous event (e.g. sustained over-speed across several frames).

    Args:
        All fields map 1:1 to Violation and related table columns.
        secret_key:       AES encryption key for plate numbers.
        dedup_window_sec: Cooldown in seconds (default 120 = 2 minutes).

    Returns:
        violation.id on success, None on duplicate/failure.
    """
    try:
        with get_session() as db:
            # ── Upsert Vehicle ────────────────────────────────────────
            enc_plate = encrypt_field(plate_text, secret_key)
            vehicle   = db.query(Vehicle).filter_by(
                plate_number=enc_plate
            ).first()
            if not vehicle:
                vehicle = Vehicle(
                    plate_number=enc_plate,
                    class_name=vehicle_class,
                    camera_id=camera_id,
                    gps_lat=gps_lat,
                    gps_lon=gps_lon,
                )
                db.add(vehicle)
                db.flush()  # get vehicle.id
            else:
                vehicle.last_seen  = datetime.utcnow()
                vehicle.class_name = vehicle_class

            # ── Duplicate / Cooldown Guard ────────────────────────────
            # Skip if the same plate got the same violation type recently.
            # Handles: track-ID resets, frame-skip re-detection, batch re-runs.
            if plate_text and dedup_window_sec > 0:
                cutoff = datetime.utcnow() - timedelta(seconds=dedup_window_sec)
                recent = (
                    db.query(Violation)
                    .filter(
                        Violation.plate_text == plate_text,
                        Violation.type       == violation_type,
                        Violation.timestamp  >= cutoff,
                    )
                    .first()
                )
                if recent:
                    logger.warning(
                        f"[DEDUP] Skipped {violation_type} for plate '{plate_text}' — "
                        f"duplicate of violation #{recent.id} "
                        f"({int((datetime.utcnow() - recent.timestamp).total_seconds())}s ago). "
                        f"Cooldown window: {dedup_window_sec}s."
                    )
                    return None

            # ── Insert Violation ──────────────────────────────────────
            viol = Violation(
                vehicle_id=vehicle.id,
                type=violation_type,
                speed_kmh=speed_kmh,
                fine_inr=fine_inr,
                evidence_image=evidence_image,
                evidence_clip=evidence_clip,
                ocr_confidence=ocr_confidence,
                plate_text=plate_text,
                mv_act_section=mv_act,
                camera_id=camera_id,
                gps_lat=gps_lat,
                gps_lon=gps_lon,
                metadata_json=json.dumps(metadata_dict, default=str),
                status=ocr_status,
            )
            db.add(viol)
            db.flush()

            # ── Upsert Credit Score ───────────────────────────────────
            score_row = db.query(CreditScore).filter_by(vehicle_id=vehicle.id).first()
            if not score_row:
                score_row = CreditScore(vehicle_id=vehicle.id)
                db.add(score_row)
            score_row.total_fines_inr = (score_row.total_fines_inr or 0.0) + fine_inr
            score_row.updated_at      = datetime.utcnow()

            violation_id = viol.id

        # ── Blockchain entry (outside inner session to get committed ID) ──
        block_data = {
            "type":      violation_type,
            "plate":     plate_text,
            "speed":     speed_kmh,
            "fine":      fine_inr,
            "camera":    camera_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        append_blockchain_entry(violation_id, block_data)

        # ── Audit log ────────────────────────────────────────────────
        log_audit(
            action="CREATE_VIOLATION",
            actor="system",
            target_table="violations",
            target_id=violation_id,
            new_value=block_data,
        )

        logger.info(
            f"Saved violation #{violation_id}: {violation_type} | "
            f"plate={plate_text} | fine=INR{fine_inr:.0f}"
        )
        return violation_id

    except Exception as e:
        logger.error(f"save_violation failed: {e}")
        return None


def update_violation_status(
    violation_id: int,
    new_status:   str,
    actor:        str = "reviewer",
) -> bool:
    """
    Update the human-review status of a violation (approve / reject).

    Args:
        violation_id: Row ID in violations table.
        new_status:   "approved" | "rejected".
        actor:        Username performing the review.

    Returns:
        True on success.
    """
    try:
        with get_session() as db:
            viol = db.query(Violation).filter_by(id=violation_id).first()
            if not viol:
                logger.warning(f"update_violation_status: ID {violation_id} not found.")
                return False
            old_status  = viol.status
            viol.status = new_status
        log_audit(
            action=f"REVIEW_{new_status.upper()}",
            actor=actor,
            target_table="violations",
            target_id=violation_id,
            old_value={"status": old_status},
            new_value={"status": new_status},
        )
        logger.info(f"Violation #{violation_id} → {new_status} by {actor}")
        return True
    except Exception as e:
        logger.error(f"update_violation_status failed: {e}")
        return False


def get_score_from_db(vehicle_id: int) -> Optional[CreditScore]:
    """Fetch the CreditScore row for a given vehicle DB id."""
    with get_session() as db:
        return db.query(CreditScore).filter_by(vehicle_id=vehicle_id).first()


def sync_score_to_db(vehicle_id: int, score: int, category: str) -> None:
    """Write updated credit score back to the database."""
    try:
        with get_session() as db:
            row = db.query(CreditScore).filter_by(vehicle_id=vehicle_id).first()
            if row:
                row.score    = score
                row.category = category
    except Exception as e:
        logger.error(f"sync_score_to_db failed: {e}")
