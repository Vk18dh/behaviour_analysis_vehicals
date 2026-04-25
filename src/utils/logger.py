"""
src/utils/logger.py
Centralized Loguru-based logging for the traffic enforcement system.
Two sinks: system.log (all events) + violations.log (violation events only).
"""

import sys
from pathlib import Path
from loguru import logger


# Custom violation level (between WARNING and ERROR)
VIOLATION_LEVEL = "VIOLATION"
VIOLATION_NO = 25  # between DEBUG(10) and WARNING(30)


def setup_logger(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """
    Configure Loguru with file and console sinks.
    Call once at application startup.

    Args:
        log_dir: Directory where log files will be written.
        log_level: Minimum log level for console output.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Add custom VIOLATION level if not already added
    try:
        logger.level(VIOLATION_LEVEL, no=VIOLATION_NO, color="<yellow>", icon="🚨")
    except TypeError:
        pass  # already registered

    # ── Console Sink ────────────────────────────────────────────────────────
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<9}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        enqueue=True,  # thread-safe
    )

    # ── System Log (all events, rotation 10 MB) ──────────────────────────
    logger.add(
        log_path / "system.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<9} | {name}:{line} | {message}",
        enqueue=True,
    )

    # ── Violations Log (only VIOLATION level) ───────────────────────────
    logger.add(
        log_path / "violations.log",
        level=VIOLATION_LEVEL,
        rotation="50 MB",
        retention="90 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | VIOLATION | {message}",
        filter=lambda record: record["level"].name == VIOLATION_LEVEL,
        enqueue=True,
    )

    logger.info(f"Logger initialised — level={log_level}, log_dir={log_path.resolve()}")


def get_logger(name: str):
    """Return a named logger (Loguru uses module binding via opt)."""
    return logger.bind(name=name)


def log_violation(track_id: int, violation_type: str, plate: str,
                  speed: float, fine: float, camera_id: str) -> None:
    """
    Structured violation log entry written to violations.log.

    Args:
        track_id: ByteTrack track ID.
        violation_type: e.g. "ZIGZAG", "OVERSPEED".
        plate: OCR plate text.
        speed: Vehicle speed in km/h.
        fine: Fine amount in INR.
        camera_id: Source camera identifier.
    """
    logger.log(
        VIOLATION_LEVEL,
        f"track_id={track_id} | type={violation_type} | plate={plate!r} "
        f"| speed={speed:.1f}km/h | fine=INR{fine:.0f} | cam={camera_id}",
    )
