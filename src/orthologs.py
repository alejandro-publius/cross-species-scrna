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


def one_to_one_orthologs(hom_file: str | Path) -> pd.DataFrame:
    """Read a homology table and return strict 1:1 pairs.

    Returns a DataFrame with columns ``[human, mouse]``: for every homology
    group that has exactly one human member and exactly one mouse member, the
    corresponding symbol pair. Groups with zero or many members on either side
    are excluded — a symbol with multiple partners is not one-to-one.
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
    return pivot.dropna().reset_index(drop=True)[["human", "mouse"]]


def restrict_to_measured(
    pairs: pd.DataFrame, human_genes: Iterable[str], mouse_genes: Iterable[str]
) -> pd.DataFrame:
    """Keep pairs whose genes are present in BOTH datasets, staying strictly 1:1.

    A symbol that ends up with multiple partners after intersecting with the
    measured genes is dropped entirely (``keep=False``) rather than resolved by
    arbitrarily taking the first — that would quietly reintroduce non-1:1 links.
    """
    hset, mset = set(human_genes), set(mouse_genes)
    keep = pairs[pairs["human"].isin(hset) & pairs["mouse"].isin(mset)].copy()
    return keep.drop_duplicates(subset="human", keep=False).drop_duplicates(
        subset="mouse", keep=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Count 1:1 orthologs in a homology table.")
    parser.add_argument("--hom-file", required=True, type=Path, help="MGI HOM_*.rpt file")
    args = parser.parse_args()

    pairs = one_to_one_orthologs(args.hom_file)
    print(f"1:1 ortholog pairs in homology table: {len(pairs)}")


if __name__ == "__main__":
    main()
