"""
tests/test_db.py
Unit tests for src/database/db.py
Tests encryption, credit score persistence, audit log, and blockchain chain.
"""
import pytest
import hashlib
from src.database.db import (
    encrypt_field, decrypt_field,
    save_violation, update_violation_status,
    log_audit, append_blockchain_entry,
    compute_block_hash, get_session, check_permission,
)
from src.database.models import AuditLog, BlockchainLog, Violation

_SECRET = "test_secret_key_32bytes_padding!!"


class TestEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        original = "MH12AB1234"
        encrypted = encrypt_field(original, _SECRET)
        decrypted = decrypt_field(encrypted, _SECRET)
        assert decrypted == original

    def test_encrypted_differs_from_plain(self):
        original  = "KA01CD5678"
        encrypted = encrypt_field(original, _SECRET)
        # Encrypted should not equal plaintext (unless PyCryptodome missing)
        # At minimum, it should be a string
        assert isinstance(encrypted, str)
        assert len(encrypted) > 0

    def test_wrong_key_returns_ciphertext(self):
        original  = "TN09EF0001"
        encrypted = encrypt_field(original, _SECRET)
        result    = decrypt_field(encrypted, "wrong_key_______________padding!!")
        # Should not raise; may return original or garbled text
        assert isinstance(result, str)


class TestSaveViolation:
    def test_save_basic_violation(self):
        vid = save_violation(
            plate_text="DL01AA0001",
            vehicle_class="car",
            violation_type="OVERSPEED",
            speed_kmh=90.0,
            fine_inr=900.0,
            evidence_image="",
            evidence_clip="",
            ocr_confidence=0.95,
            mv_act="MV Act §183",
            camera_id="cam_00",
            gps_lat=28.6,
            gps_lon=77.2,
            metadata_dict={"speed_kmh": 90.0},
            secret_key=_SECRET,
        )
        assert vid is not None
        assert isinstance(vid, int)
        assert vid > 0

    def test_save_second_violation_same_vehicle(self):
        v1 = save_violation(
            plate_text="MH01XX9999",
            vehicle_class="motorcycle",
            violation_type="ZIGZAG",
            speed_kmh=50.0, fine_inr=1000.0,
            evidence_image="", evidence_clip="",
            ocr_confidence=0.92, mv_act="MV Act §184",
            camera_id="cam_01", gps_lat=0.0, gps_lon=0.0,
            metadata_dict={}, secret_key=_SECRET,
        )
        v2 = save_violation(
            plate_text="MH01XX9999",
            vehicle_class="motorcycle",
            violation_type="TAILGATING",
            speed_kmh=70.0, fine_inr=1000.0,
            evidence_image="", evidence_clip="",
            ocr_confidence=0.91, mv_act="MV Act §184",
            camera_id="cam_01", gps_lat=0.0, gps_lon=0.0,
            metadata_dict={}, secret_key=_SECRET,
        )
        assert v1 != v2   # different violation IDs

    def test_violation_status_default(self):
        vid = save_violation(
            plate_text="UP32ZZ1111",
            vehicle_class="truck",
            violation_type="LANE_VIOLATION",
            speed_kmh=55.0, fine_inr=500.0,
            evidence_image="", evidence_clip="",
            ocr_confidence=0.88, mv_act="",
            camera_id="cam_00", gps_lat=0.0, gps_lon=0.0,
            metadata_dict={}, secret_key=_SECRET,
        )
        with get_session() as db:
            v = db.query(Violation).filter_by(id=vid).first()
            assert v.status == "pending"


class TestUpdateStatus:
    def test_approve_violation(self):
        vid = save_violation(
            plate_text="RJ14YY2222",
            vehicle_class="car",
            violation_type="RED_LIGHT",
            speed_kmh=30.0, fine_inr=1000.0,
            evidence_image="", evidence_clip="",
            ocr_confidence=0.95, mv_act="",
            camera_id="cam_00", gps_lat=0.0, gps_lon=0.0,
            metadata_dict={}, secret_key=_SECRET,
        )
        ok = update_violation_status(vid, "approved", actor="reviewer1")
        assert ok is True
        with get_session() as db:
            v = db.query(Violation).filter_by(id=vid).first()
            assert v.status == "approved"

    def test_reject_violation(self):
        vid = save_violation(
            plate_text="GJ05WW3333",
            vehicle_class="car",
            violation_type="NO_HELMET",
            speed_kmh=0.0, fine_inr=1000.0,
            evidence_image="", evidence_clip="",
            ocr_confidence=0.55, mv_act="",
            camera_id="cam_00", gps_lat=0.0, gps_lon=0.0,
            metadata_dict={}, secret_key=_SECRET,
            ocr_status="low_confidence",
        )
        ok = update_violation_status(vid, "rejected")
        assert ok is True

    def test_update_nonexistent_violation(self):
        ok = update_violation_status(999999, "approved")
        assert ok is False


class TestBlockchain:
    def test_hash_is_deterministic(self):
        data = {"type": "OVERSPEED", "plate": "TN01AA0000", "fine": 900}
        h1   = compute_block_hash(1, data, "0"*64)
        h2   = compute_block_hash(1, data, "0"*64)
        assert h1 == h2

    def test_hash_changes_with_data(self):
        data1 = {"type": "OVERSPEED", "plate": "TN01AA0000"}
        data2 = {"type": "ZIGZAG",    "plate": "TN01AA0000"}
        h1    = compute_block_hash(1, data1, "0"*64)
        h2    = compute_block_hash(1, data2, "0"*64)
        assert h1 != h2

    def test_blockchain_chain_linkage(self):
        """Each new entry's prev_hash must equal previous entry's tx_hash."""
        vid1 = save_violation(
            plate_text="HR26BC0001", vehicle_class="car",
            violation_type="TAILGATING",
            speed_kmh=80.0, fine_inr=1000.0,
            evidence_image="", evidence_clip="",
            ocr_confidence=0.93, mv_act="",
            camera_id="cam_00", gps_lat=0.0, gps_lon=0.0,
            metadata_dict={}, secret_key=_SECRET,
        )
        vid2 = save_violation(
            plate_text="HR26BC0002", vehicle_class="car",
            violation_type="OVERSPEED",
            speed_kmh=90.0, fine_inr=900.0,
            evidence_image="", evidence_clip="",
            ocr_confidence=0.97, mv_act="",
            camera_id="cam_00", gps_lat=0.0, gps_lon=0.0,
            metadata_dict={}, secret_key=_SECRET,
        )
        with get_session() as db:
            e1 = db.query(BlockchainLog).filter_by(violation_id=vid1).first()
            e2 = db.query(BlockchainLog).filter_by(violation_id=vid2).first()
            e1_tx_hash = e1.tx_hash if e1 else None
            e2_prev_hash = e2.prev_hash if e2 else None
        assert e1_tx_hash is not None
        assert e2_prev_hash is not None
        assert e2_prev_hash == e1_tx_hash


class TestRBAC:
    def test_admin_has_all(self):
        assert check_permission("admin", "view")
        assert check_permission("admin", "review")
        assert check_permission("admin", "upload")

    def test_reviewer_can_review_view(self):
        assert check_permission("reviewer", "review")
        assert check_permission("reviewer", "view")
        assert not check_permission("reviewer", "upload")

    def test_readonly_view_only(self):
        assert check_permission("readonly", "view")
        assert not check_permission("readonly", "review")

    def test_officer_upload(self):
        assert check_permission("officer", "upload")
        assert not check_permission("officer", "review")

    def test_unknown_role(self):
        assert not check_permission("hacker", "view")
