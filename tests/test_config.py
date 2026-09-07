"""Tests for src/config.py — shared paths/constants the whole pipeline depends on, and the
`require()` guard clause every pipeline stage now uses to fail cleanly on a missing input."""

import pytest

import config as C


def test_paths_are_nested_correctly():
    assert C.RAW == C.DATA / "raw"
    assert C.PROC == C.DATA / "processed"
    assert C.JOINT_RAW == C.PROC / "joint_raw.h5ad"


def test_human_and_mouse_file_lists_are_disjoint_and_nonempty():
    """A raw filename accidentally listed under both species would silently double-count
    or cross-contaminate a donor between species during loading."""
    human = set(C.HUMAN_FILES)
    mouse = set(C.MOUSE_FILES)
    assert human, "no human files configured"
    assert mouse, "no mouse files configured"
    assert human.isdisjoint(mouse), "a raw file is listed for both species"


def test_shared_celltypes_has_no_duplicates():
    assert len(C.SHARED_CELLTYPES) == len(set(C.SHARED_CELLTYPES))


def test_require_passes_through_existing_path(tmp_path):
    f = tmp_path / "exists.txt"
    f.write_text("x")
    assert C.require(f, "should not matter") == f


def test_require_raises_actionable_message_for_missing_path(tmp_path):
    missing = tmp_path / "does_not_exist.h5ad"
    with pytest.raises(SystemExit) as exc:
        C.require(missing, "run src/00_download.py first")
    msg = str(exc.value)
    # the message must name the missing path AND the fix -- that's the whole point of
    # this helper over letting pandas/h5py raise several frames deep.
    assert str(missing) in msg
    assert "run src/00_download.py first" in msg
