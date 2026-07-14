"""Tests for the 1:1 ortholog mapping — the project's silent-failure hotspot.

These exercise the two traps called out in CLAUDE.md: many-to-one homology
groups masquerading as usable pairs, and symbols that only look 1:1 until you
intersect with the genes actually measured in each dataset.
"""

import pandas as pd

from orthologs import one_to_one_orthologs, restrict_to_measured

# A tiny MGI-style homology table exercising every case. Columns match the real
# HOM_MouseHumanSequence.rpt; "Common Organism Name" is matched by prefix.
_ROWS = [
    # group, organism, symbol
    ("100", "human", "INS"),
    ("100", "mouse, laboratory", "Ins2"),  # clean 1:1
    ("200", "human", "GCG"),
    ("200", "mouse, laboratory", "Gcg"),  # clean 1:1
    ("300", "human", "SST"),
    ("300", "mouse, laboratory", "Sst1"),
    ("300", "mouse, laboratory", "Sst2"),  # 1 human : 2 mouse -> excluded
    ("400", "human", "PPY"),
    ("400", "human", "PPY2"),
    ("400", "mouse, laboratory", "Ppy"),  # 2 human : 1 mouse -> excluded
    ("500", "human", "HONLY"),  # human only -> excluded
    ("600", "mouse, laboratory", "Monly"),  # mouse only -> excluded
    ("700", "human", "KEEP"),
    ("700", "mouse, laboratory", "Keep"),
    ("700", "rat", "Keep_rat"),  # non-human/mouse row must be ignored, 700 stays 1:1
]


def _write_hom(path):
    df = pd.DataFrame(_ROWS, columns=["DB Class Key", "Common Organism Name", "Symbol"])
    df.to_csv(path, sep="\t", index=False)
    return path


def test_only_strict_one_to_one_pairs_are_returned(tmp_path):
    pairs = one_to_one_orthologs(_write_hom(tmp_path / "hom.rpt"))

    got = set(map(tuple, pairs.to_numpy()))
    assert got == {("INS", "Ins2"), ("GCG", "Gcg"), ("KEEP", "Keep")}
    # the many-to-one groups never appear
    assert "SST" not in pairs["human"].values
    assert "PPY" not in pairs["human"].values


def test_non_human_mouse_rows_are_ignored(tmp_path):
    """The rat row in group 700 must not break its human/mouse 1:1 status."""
    pairs = one_to_one_orthologs(_write_hom(tmp_path / "hom.rpt"))
    assert ("KEEP", "Keep") in set(map(tuple, pairs.to_numpy()))


def test_symbols_are_case_sensitive_not_blindly_matched(tmp_path):
    """Guards the case trap: mapping comes from the table, not symbol equality.
    Human INS and mouse Ins2 are paired despite never being string-equal."""
    pairs = one_to_one_orthologs(_write_hom(tmp_path / "hom.rpt"))
    row = pairs[pairs["human"] == "INS"]
    assert row["mouse"].item() == "Ins2"


def test_restrict_to_measured_drops_unmeasured_and_multi_partner():
    pairs = pd.DataFrame(
        {"human": ["INS", "GCG", "GCG", "SST"], "mouse": ["Ins2", "Gcg", "Gcg2", "Sst"]}
    )
    # SST/Sst aren't measured; GCG has two mouse partners once intersected.
    keep = restrict_to_measured(pairs, human_genes={"INS", "GCG"}, mouse_genes={"Ins2", "Gcg", "Gcg2"})

    assert set(map(tuple, keep.to_numpy())) == {("INS", "Ins2")}


def test_restrict_to_measured_keeps_clean_pairs():
    pairs = pd.DataFrame({"human": ["INS", "GCG"], "mouse": ["Ins2", "Gcg"]})
    keep = restrict_to_measured(pairs, human_genes={"INS", "GCG"}, mouse_genes={"Ins2", "Gcg"})
    assert len(keep) == 2
