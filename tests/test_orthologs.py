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
    # Mouse-specific gene-family expansion: each mouse paralog gets its OWN class key,
    # each internally a clean 1 human : 1 mouse group -- but AADACL4 ends up with SIX
    # mouse partners overall. Modeled on the real MGI table (18 class keys, same
    # pattern; see AADACL4/Aadacl4fm1-5). Every one of these six groups individually
    # passes the per-group "exactly one human, exactly one mouse" check.
    ("810", "human", "AADACL4"),
    ("810", "mouse, laboratory", "Aadacl4"),
    ("811", "human", "AADACL4"),
    ("811", "mouse, laboratory", "Aadacl4fm1"),
    ("812", "human", "AADACL4"),
    ("812", "mouse, laboratory", "Aadacl4fm2"),
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


def test_one_to_one_orthologs_rejects_a_symbol_repeated_across_groups(tmp_path):
    """Each of AADACL4's three synthetic class keys (810/811/812) is, on its own,
    a clean "exactly one human, exactly one mouse" group -- the per-group check alone
    would let all three through, leaving AADACL4 paired with THREE different mouse
    symbols in a table that claims to be 1:1. This is not a corner case: on the real
    MGI table, 1,173 of 18,782 raw per-group-1:1 pairs (~6.2%) share a human symbol
    with at least one other pair, almost entirely from mouse-specific paralog
    expansions like this one."""
    pairs = one_to_one_orthologs(_write_hom(tmp_path / "hom.rpt"))

    assert "AADACL4" not in pairs["human"].values
    assert not pairs["mouse"].isin(["Aadacl4", "Aadacl4fm1", "Aadacl4fm2"]).any()
    # every symbol that DOES appear is globally unique, both sides -- the function's
    # own claimed contract ("Groups... are excluded -- a symbol with multiple
    # partners is not one-to-one"), checked directly rather than trusting the
    # per-group construction to have implied it.
    assert not pairs["human"].duplicated().any()
    assert not pairs["mouse"].duplicated().any()


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


def test_restrict_to_measured_catches_a_duplicate_the_wrong_pass_order_would_miss():
    """A sequential "drop human dupes, then drop mouse dupes on what's left" can under-drop:
    row A=(X,Y) and row B=(X,Z) share human X, so a human-first pass removes both -- taking
    row C=(W,Y) 's ONLY competitor for mouse symbol Y down with them. Checked sequentially,
    Y then looks unique by the time the mouse pass runs, and C wrongly survives even though
    Y was never actually W's exclusive partner. Both counts must come from the SAME,
    unfiltered set."""
    pairs = pd.DataFrame({
        "human": ["X", "X", "W"],
        "mouse": ["Y", "Z", "Y"],
    })
    keep = restrict_to_measured(
        pairs, human_genes={"X", "W"}, mouse_genes={"Y", "Z"}
    )
    assert len(keep) == 0, f"expected every row dropped (X and Y both ambiguous), got {keep}"
