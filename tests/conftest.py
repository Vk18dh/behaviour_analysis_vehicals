"""
tests/conftest.py
Shared pytest fixtures for the traffic enforcement test suite.
"""
import os
import pytest

# ── Force SQLite in-memory DB for all tests ──────────────────────────
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_traffic.db")

from src.database.db import init_db

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Initialise an in-memory SQLite DB once per test session."""
    init_db("sqlite:///./test_traffic.db")
    yield
    # Cleanup
    from src.database.db import _ENGINE
    if _ENGINE is not None:
        _ENGINE.dispose()
    if os.path.exists("test_traffic.db"):
        try:
            os.remove("test_traffic.db")
        except Exception as e:
            pass
