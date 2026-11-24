import numpy as np
import pytest

from charliecam.database import FaceDatabase
from charliecam.types import UNKNOWN


def vec(*xs):
    return np.array(xs, dtype=np.float32)


def test_empty_database_returns_unknown(db):
    result = db.match(vec(1, 0, 0))
    assert result.name == UNKNOWN
    assert not result.is_known


def test_exact_match_is_recognised(db):
    db.add("alice", vec(1, 0, 0))
    result = db.match(vec(1, 0, 0))
    assert result.name == "alice"
    assert result.similarity == pytest.approx(1.0)


def test_match_is_scale_invariant(db):
    db.add("alice", vec(1, 0, 0))
    assert db.match(vec(42, 0, 0)).name == "alice"


def test_dissimilar_face_is_unknown(db):
    db.add("alice", vec(1, 0, 0))
    result = db.match(vec(0, 1, 0))
    assert result.name == UNKNOWN
    assert result.similarity == pytest.approx(0.0)


def test_threshold_boundary():
    db = FaceDatabase(threshold=0.5)
    db.add("alice", vec(1, 0))
    # cos(60 deg) = 0.5 -> exactly at threshold counts as a match
    assert db.match(vec(0.5, np.sqrt(3) / 2)).name == "alice"
    assert db.match(vec(0.49, np.sqrt(1 - 0.49 ** 2))).name == UNKNOWN


def test_best_of_multiple_people(db):
    db.add("alice", vec(1, 0, 0))
    db.add("bob", vec(0, 1, 0))
    assert db.match(vec(0.1, 1, 0)).name == "bob"


def test_multiple_samples_per_person(db):
    db.add("alice", vec(1, 0, 0))
    db.add("alice", vec(0, 0, 1))
    assert db.match(vec(0, 0, 1)).name == "alice"
    assert db.people() == {"alice": 2}
    assert len(db) == 2


def test_remove_person(db):
    db.add("alice", vec(1, 0, 0))
    db.add("bob", vec(0, 1, 0))
    db.add("alice", vec(0, 0, 1))
    assert db.remove("alice") == 2
    assert "alice" not in db
    assert db.match(vec(1, 0, 0)).name == UNKNOWN
    assert db.match(vec(0, 1, 0)).name == "bob"


def test_remove_last_person_leaves_usable_db(db):
    db.add("alice", vec(1, 0, 0))
    db.remove("alice")
    assert len(db) == 0
    db.add("bob", vec(0, 1))  # different dimension is fine once empty
    assert db.match(vec(0, 1)).name == "bob"


def test_remove_missing_person_returns_zero(db):
    assert db.remove("nobody") == 0


@pytest.mark.parametrize("name", ["", "   ", UNKNOWN])
def test_invalid_names_rejected(db, name):
    with pytest.raises(ValueError):
        db.add(name, vec(1, 0, 0))


def test_zero_embedding_rejected(db):
    with pytest.raises(ValueError):
        db.add("alice", vec(0, 0, 0))


def test_dimension_mismatch_rejected(db):
    db.add("alice", vec(1, 0, 0))
    with pytest.raises(ValueError):
        db.add("bob", vec(1, 0))


def test_save_and_load_roundtrip(db, tmp_path):
    db.add("alice", vec(1, 0, 0))
    db.add("bob", vec(0, 1, 0))
    path = tmp_path / "sub" / "faces.npz"
    db.save(path)
    loaded = FaceDatabase.load(path, threshold=0.9)
    assert loaded.people() == {"alice": 1, "bob": 1}
    assert loaded.match(vec(0, 1, 0)).name == "bob"


def test_load_missing_file_gives_empty_db(tmp_path):
    assert len(FaceDatabase.load(tmp_path / "nope.npz", threshold=0.5)) == 0


def test_save_empty_database_roundtrip(db, tmp_path):
    db.save(tmp_path / "faces.npz")
    assert len(FaceDatabase.load(tmp_path / "faces.npz", 0.5)) == 0
