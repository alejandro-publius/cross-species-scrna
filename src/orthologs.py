"""One-to-one human/mouse ortholog mapping — the crux of the whole project.

Human and mouse do not share a gene list, and their symbols use different casing
(human ``INS`` vs mouse ``Ins1``), so a case-blind ``set(human) & set(mouse)``
intersection silently returns almost nothing and raises no error. We instead go
through a curated homology table (MGI ``HOM_MouseHumanSequence.rpt``) and keep
only genes that are strictly **1:1** orthologs — exactly one human gene paired
with exactly one mouse gene.

This module is deliberately dependency-light (pandas + numpy only) and free of
any project config, so the mapping can be reused or unit-tested on its own. The
downstream pipeline (``02_orthologs.py``) imports it; the ``__main__`` guard
below also makes it runnable as a CLI to sanity-check a homology table.
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# MGI's homology export uses these columns; "Common Organism Name" is e.g.
# "human" or "mouse, laboratory", so we match by prefix.
_HOM_COLUMNS = ["DB Class Key", "Common Organism Name", "Symbol"]


def _drop_ambiguous_partners(pairs: pd.DataFrame) -> pd.DataFrame:
    """Drop every row whose human OR mouse symbol is not globally unique in `pairs`.

    Both counts are computed on the SAME, unfiltered `pairs` and applied as one
    combined mask — not two sequential ``drop_duplicates`` calls. Sequential
    dedup (human pass, then mouse pass on what's left) can under-drop: if
    removing a human-side duplicate happens to leave a mouse symbol that was
    ALSO duplicated with a now-removed row looking unique, the second pass
    never sees it. Computing both counts up front avoids that ordering bug.
    """
    h_counts = pairs["human"].value_counts()
    m_counts = pairs["mouse"].value_counts()
    mask = pairs["human"].map(h_counts).eq(1) & pairs["mouse"].map(m_counts).eq(1)
    return pairs[mask]


def one_to_one_orthologs(hom_file: str | Path) -> pd.DataFrame:
    """Read a homology table and return strict 1:1 pairs.

    A homology group with exactly one human member and exactly one mouse
    member is necessary but NOT sufficient for that pair to be one-to-one:
    mouse-specific gene-family expansions routinely give several mouse
    paralogs their OWN DB Class Key, each independently paired 1:1 with the
    SAME human symbol (e.g. human ``AADACL4`` <-> mouse ``Aadacl4``,
    ``Aadacl4fm1``, ``Aadacl4fm2``, ... via six different class keys — each
    key on its own looks like a clean 1:1 group). Left unguarded this makes
    the "1:1" table contain human (and occasionally mouse) symbols with
    several partners: on the real MGI table this is not a rare corner case —
    1,173 of 18,782 raw pairs (~6.2%) share a human symbol with at least one
    other pair. This function now also drops any symbol that repeats ACROSS
    groups, so its output is 1:1 on its own, independent of whatever the
    caller does downstream.

    Returns a DataFrame with columns ``[human, mouse]``: for every homology
    group that has exactly one human member and exactly one mouse member, AND
    whose human and mouse symbols each appear in exactly one such group, the
    corresponding symbol pair. Groups with zero or many members on either
    side, or whose symbol is shared with another group, are excluded — a
    symbol with multiple partners is not one-to-one.
    """
    df = pd.read_csv(hom_file, sep="\t", dtype=str)
    df = df[_HOM_COLUMNS].dropna()
    df["Symbol"] = df["Symbol"].str.strip()
    is_h = df["Common Organism Name"].str.startswith("human")
    is_m = df["Common Organism Name"].str.startswith("mouse")

    # Label each row's species and drop anything that isn't human or mouse.
    grp = df.assign(sp=np.where(is_h, "human", np.where(is_m, "mouse", "other")))
    grp = grp[grp["sp"] != "other"]

    # Keep only homology groups with exactly one member per species.
    counts = grp.groupby(["DB Class Key", "sp"]).size().unstack(fill_value=0)
    one2one_keys = counts[(counts.get("human", 0) == 1) & (counts.get("mouse", 0) == 1)].index

    sub = grp[grp["DB Class Key"].isin(one2one_keys)]
    pivot = sub.pivot_table(index="DB Class Key", columns="sp", values="Symbol", aggfunc="first")
    pairs = pivot.dropna().reset_index(drop=True)[["human", "mouse"]]
    # Per-group 1:1 is not global 1:1 -- see docstring. Enforce it here too, so
    # this guarantee does not depend on every caller remembering to also call
    # restrict_to_measured.
    return _drop_ambiguous_partners(pairs).reset_index(drop=True)


def restrict_to_measured(
    pairs: pd.DataFrame, human_genes: Iterable[str], mouse_genes: Iterable[str]
) -> pd.DataFrame:
    """Keep pairs whose genes are present in BOTH datasets, staying strictly 1:1.

    A symbol that ends up with multiple partners after intersecting with the
    measured genes is dropped entirely rather than resolved by arbitrarily
    taking the first — that would quietly reintroduce non-1:1 links. Both
    sides are checked in one combined mask (see `_drop_ambiguous_partners`),
    not two sequential passes, for the same ordering reason.
    """
    hset, mset = set(human_genes), set(mouse_genes)
    keep = pairs[pairs["human"].isin(hset) & pairs["mouse"].isin(mset)].copy()
    return _drop_ambiguous_partners(keep).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count 1:1 orthologs in a homology table.")
    parser.add_argument("--hom-file", required=True, type=Path, help="MGI HOM_*.rpt file")
    args = parser.parse_args()

    pairs = one_to_one_orthologs(args.hom_file)
    print(f"1:1 ortholog pairs in homology table: {len(pairs)}")


if __name__ == "__main__":
    main()
