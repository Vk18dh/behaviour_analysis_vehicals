"""
src/database/models.py
SQLAlchemy ORM models for the traffic enforcement system.

Tables:
  - Vehicle       : registered vehicles (plate = AES-256 encrypted)
  - Violation     : individual violations with evidence paths
  - CreditScore   : per-vehicle running credit score
  - AuditLog      : tamper-evident record of all DB write operations
  - BlockchainLog : SHA-256 chained hash per violation (tamper-evident)
  - User          : API users with RBAC roles
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ══════════════════════════════════════════════════════════════════════
# Vehicle
# ══════════════════════════════════════════════════════════════════════

class Vehicle(Base):
    __tablename__ = "vehicles"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    # Plate number stored AES-256 encrypted (see db.py encrypt_field)
    plate_number  = Column(String(512), nullable=False, index=True)
    class_name    = Column(String(64),  nullable=False)  # car / truck / motorcycle …
    camera_id     = Column(String(64),  nullable=True)
    gps_lat       = Column(Float, nullable=True)
    gps_lon       = Column(Float, nullable=True)
    first_seen    = Column(DateTime, default=datetime.utcnow)
    last_seen     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    violations    = relationship("Violation",   back_populates="vehicle",
                                 cascade="all, delete-orphan")
    score         = relationship("CreditScore", back_populates="vehicle",
                                 uselist=False, cascade="all, delete-orphan")


# ══════════════════════════════════════════════════════════════════════
# Violation
# ══════════════════════════════════════════════════════════════════════

class Violation(Base):
    __tablename__ = "violations"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id         = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)
    type               = Column(String(64),  nullable=False)   # e.g. "ZIGZAG"
    timestamp          = Column(DateTime,    default=datetime.utcnow, index=True)
    speed_kmh          = Column(Float,       nullable=True)
    fine_inr           = Column(Float,       nullable=True)
    evidence_image     = Column(String(512), nullable=True)    # absolute file path
    evidence_clip      = Column(String(512), nullable=True)
    ocr_confidence     = Column(Float,       nullable=True)
    plate_text         = Column(String(64),  nullable=True)    # plain (not encrypted)
    mv_act_section     = Column(String(256), nullable=True)
    camera_id          = Column(String(64),  nullable=True)
    gps_lat            = Column(Float,       nullable=True)
    gps_lon            = Column(Float,       nullable=True)
    metadata_json      = Column(Text,        nullable=True)    # JSON blob of extra info
    # Human-review status
    status             = Column(
        String(32), default="pending", index=True
    )  # pending | approved | rejected | low_confidence

    vehicle            = relationship("Vehicle",      back_populates="violations")
    blockchain_entry   = relationship("BlockchainLog", back_populates="violation",
                                      uselist=False, cascade="all, delete-orphan")


# ══════════════════════════════════════════════════════════════════════
# Credit Score
# ══════════════════════════════════════════════════════════════════════

class CreditScore(Base):
    __tablename__ = "credit_scores"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id      = Column(Integer, ForeignKey("vehicles.id"),
                             nullable=False, unique=True, index=True)
    score           = Column(Integer, default=100)
    category        = Column(String(32), default="Safe")  # Safe | Moderate | Risky
    total_fines_inr = Column(Float,   default=0.0)
    updated_at      = Column(DateTime, default=datetime.utcnow,
                             onupdate=datetime.utcnow)

    vehicle = relationship("Vehicle", back_populates="score")


# ══════════════════════════════════════════════════════════════════════
# Audit Log
# ══════════════════════════════════════════════════════════════════════

class AuditLog(Base):
    """
    Every INSERT / UPDATE / logical action is logged here.
    Append-only — rows are never modified.
    """
    __tablename__ = "audit_logs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    timestamp    = Column(DateTime, default=datetime.utcnow, index=True)
    action       = Column(String(64),  nullable=False)  # CREATE_VIOLATION, APPROVE, REJECT …
    actor        = Column(String(128), nullable=True)   # username / system
    target_table = Column(String(64),  nullable=True)
    target_id    = Column(Integer,     nullable=True)
    old_value    = Column(Text,        nullable=True)   # JSON snapshot
    new_value    = Column(Text,        nullable=True)
    ip_address   = Column(String(64),  nullable=True)


# ══════════════════════════════════════════════════════════════════════
# Blockchain Log (tamper-evident SHA-256 chain)
# ══════════════════════════════════════════════════════════════════════

class BlockchainLog(Base):
    """
    Simple linked-hash chain providing tamper-evidence for violations.
    Each row stores: SHA-256(violation_data + prev_hash) → tx_hash.
    To verify: recompute hashes from genesis and compare.
    For production, mirror to IPFS or a permissioned blockchain.
    """
    __tablename__ = "blockchain_logs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    violation_id = Column(Integer, ForeignKey("violations.id"),
                          nullable=False, unique=True)
    tx_hash      = Column(String(256), nullable=False)   # SHA-256 of this record
    prev_hash    = Column(String(256), nullable=False)   # prev row's tx_hash
    timestamp    = Column(DateTime, default=datetime.utcnow)

    violation    = relationship("Violation", back_populates="blockchain_entry")


# ══════════════════════════════════════════════════════════════════════
# User (RBAC)
# ══════════════════════════════════════════════════════════════════════

class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(128), nullable=False, unique=True, index=True)
    hashed_password = Column(String(256), nullable=False)
    role          = Column(String(32),  nullable=False, default="readonly")
                   # admin | officer | reviewer | readonly
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    last_login    = Column(DateTime, nullable=True)
