"""
main.py
Single entry point for the Traffic Enforcement System.

Sub-commands:
  live   — start real-time camera pipeline
  batch  — process an uploaded video file
  api    — start the FastAPI backend server
  dash   — launch the Streamlit dashboard
  all    — start API + dashboard (background threads) + live pipeline

Usage examples:
  python main.py live --camera_id cam_00 --webcam 0
  python main.py live --camera_id cam_01 --rtsp rtsp://192.168.1.100/stream
  python main.py batch --video path/to/clip.mp4
  python main.py api   --host 0.0.0.0 --port 8000
  python main.py dash
  python main.py all   --camera_id cam_00 --webcam 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path

from src.utils.helpers import load_config
from src.utils.logger import setup_logger, get_logger

# ── Logger must be set up before any module logs anything ────────────
setup_logger(log_dir="logs", log_level="INFO")
logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Sub-command runners
# ══════════════════════════════════════════════════════════════════════

def run_live(args: argparse.Namespace, stop_event: threading.Event = None) -> None:
    """Launch the real-time pipeline."""
    from src.pipeline.realtime_pipeline import RealtimePipeline
    cfg = load_config()
    pipeline = RealtimePipeline(cfg, stop_event=stop_event)
    pipeline.run(
        camera_id=args.camera_id,
        rtsp_url=getattr(args, "rtsp", None),
        webcam_index=getattr(args, "webcam", None),
    )


def run_batch(args: argparse.Namespace) -> None:
    """Launch the batch file pipeline."""
    from src.pipeline.batch_pipeline import BatchPipeline
    cfg = load_config()
    pipeline = BatchPipeline(cfg)
    video_path = args.video
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    summary = pipeline.run(video_path, camera_id=getattr(args, "camera_id", "batch"))
    print("\n── Batch processing complete ──")
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")


def run_clear(args: argparse.Namespace) -> None:
    """Clear all violations from the database."""
    from src.database.db import init_db, get_session
    from src.database.models import Violation, BlockchainLog, AuditLog, CreditScore, Vehicle
    
    cfg = load_config()
    db_url = cfg.get("system", {}).get("db_url", "sqlite:///traffic.db")
    init_db(db_url)
    
    try:
        with get_session() as db:
            count = db.query(Violation).count()
            if count == 0:
                print("Database is already empty. No violations to clear.")
                return
                
            confirm = input(f"Are you sure you want to delete {count} violations and related logs? (y/N): ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return
                
            db.query(BlockchainLog).delete()
            db.query(AuditLog).delete()
            db.query(Violation).delete()
            # Optionally clear vehicles and scores to fully reset
            if getattr(args, "full", False):
                db.query(CreditScore).delete()
                db.query(Vehicle).delete()
                print("Cleared all Vehicles and Credit Scores as well.")
                
            print(f"Successfully cleared {count} violations and associated logs.")
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        sys.exit(1)


def run_api(args: argparse.Namespace) -> None:
    """Start FastAPI backend via uvicorn."""
    import uvicorn
    from src.api.app import app
    uvicorn.run(
        app,
        host=getattr(args, "host", "0.0.0.0"),
        port=getattr(args, "port", 8000),
    )


def run_dashboard() -> None:
    """Launch Streamlit dashboard as a subprocess."""
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/dashboard.py",
        "--server.headless", "true",
        "--server.port", "8501",
    ]
    logger.info(f"Launching dashboard: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_all(args: argparse.Namespace) -> None:
    """
    Start API server + dashboard + live pipeline concurrently.
    API and dashboard run in daemon threads; pipeline runs in main thread.
    """
    stop_event = threading.Event()

    # ── API thread ────────────────────────────────────────────────────
    api_thread = threading.Thread(
        target=run_api, args=(args,), daemon=True, name="api"
    )
    api_thread.start()
    logger.info("API server started in background thread.")

    # ── Dashboard thread ──────────────────────────────────────────────
    dash_thread = threading.Thread(
        target=run_dashboard, daemon=True, name="dashboard"
    )
    dash_thread.start()
    logger.info("Streamlit dashboard started in background thread.")

    # ── Live pipeline (foreground) ────────────────────────────────────
    try:
        run_live(args, stop_event=stop_event)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down.")
        stop_event.set()


# ══════════════════════════════════════════════════════════════════════
# Argument Parser
# ══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Vision-Based Vehicle Behavior Analysis & "
            "Smart Traffic Enforcement System"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── live ──────────────────────────────────────────────────────────
    p_live = sub.add_parser("live", help="Run real-time camera pipeline.")
    p_live.add_argument(
        "--camera_id", default="cam_00",
        help="Camera identifier (must match config or be any unique string).",
    )
    p_live.add_argument(
        "--rtsp", default=None,
        help="RTSP stream URL (e.g. rtsp://192.168.1.100/stream1).",
    )
    p_live.add_argument(
        "--webcam", type=int, default=None,
        help="Webcam index (e.g. 0 for default camera). Overrides --rtsp.",
    )

    # ── batch ─────────────────────────────────────────────────────────
    p_batch = sub.add_parser("batch", help="Process an uploaded video file.")
    p_batch.add_argument("--video", required=True, help="Path to MP4/AVI file.")
    p_batch.add_argument("--camera_id", default="batch", help="Label for DB records.")

    # ── api ───────────────────────────────────────────────────────────
    p_api = sub.add_parser("api", help="Start the FastAPI backend server.")
    p_api.add_argument("--host", default="0.0.0.0")
    p_api.add_argument("--port", type=int, default=8000)

    # ── dash ──────────────────────────────────────────────────────────
    sub.add_parser("dash", help="Launch the Streamlit dashboard.")
    
    # ── clear ─────────────────────────────────────────────────────────
    p_clear = sub.add_parser("clear", help="Empty the database violations.")
    p_clear.add_argument("--full", action="store_true", help="Also delete vehicles and credit scores.")

    # ── all ───────────────────────────────────────────────────────────
    p_all = sub.add_parser(
        "all",
        help="Start API + dashboard + live pipeline together.",
    )
    p_all.add_argument("--camera_id", default="cam_00")
    p_all.add_argument("--rtsp", default=None)
    p_all.add_argument("--webcam", type=int, default=None)
    p_all.add_argument("--host", default="0.0.0.0")
    p_all.add_argument("--port", type=int, default=8000)

    return parser


# ══════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    logger.info(f"Starting command: '{args.command}'")

    if args.command == "live":
        run_live(args)
    elif args.command == "batch":
        run_batch(args)
    elif args.command == "api":
        run_api(args)
    elif args.command == "dash":
        run_dashboard()
    elif args.command == "clear":
        run_clear(args)
    elif args.command == "all":
        run_all(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
