"""
Phase 1b: map human and mouse onto a shared ONE-TO-ONE ortholog gene space.

This is the crux of the whole project. Human and mouse do not share a gene list, and their
symbols use different casing (human INS vs mouse Ins1) -- a case-blind intersection silently
returns almost nothing. We instead use a curated homology table (MGI HOM_MouseHumanSequence)
and keep only genes that are 1:1 orthologs: exactly one human gene <-> exactly one mouse gene.

We rename mouse genes to their human ortholog symbol so both species share the SAME feature
names, subset both to the shared genes in the same order, concatenate, and keep only the cell
types present in both species. Raw counts are preserved (scVI needs them downstream).

Robustness: the homology table is read from a local cached file (static fallback). A live
biomart query is NOT required and the pipeline never hangs on a flaky network call.
"""
import numpy as np
import pandas as pd
import anndata as ad

import config as C

HOM_FILE = C.RAW / "HOM_MouseHumanSequence.rpt"


def one_to_one_orthologs():
    """Return a DataFrame with columns [human, mouse]: 1:1 ortholog symbol pairs."""
    df = pd.read_csv(HOM_FILE, sep="\t", dtype=str)
    df = df[["DB Class Key", "Common Organism Name", "Symbol"]].dropna()
    df["Symbol"] = df["Symbol"].str.strip()
    is_h = df["Common Organism Name"].str.startswith("human")
    is_m = df["Common Organism Name"].str.startswith("mouse")

    # count members per homology group per species
    grp = df.assign(sp=np.where(is_h, "human", np.where(is_m, "mouse", "other")))
    grp = grp[grp["sp"] != "other"]
    counts = grp.groupby(["DB Class Key", "sp"]).size().unstack(fill_value=0)
    one2one_keys = counts[(counts.get("human", 0) == 1) & (counts.get("mouse", 0) == 1)].index

    sub = grp[grp["DB Class Key"].isin(one2one_keys)]
    pivot = sub.pivot_table(index="DB Class Key", columns="sp", values="Symbol", aggfunc="first")
    pairs = pivot.dropna().reset_index(drop=True)[["human", "mouse"]]
    return pairs


def main():
    pairs = one_to_one_orthologs()
    print(f"1:1 ortholog pairs in homology table: {len(pairs)}")

    human = ad.read_h5ad(C.PROC / "human_raw.h5ad")
    mouse = ad.read_h5ad(C.PROC / "mouse_raw.h5ad")

    hset, mset = set(human.var_names), set(mouse.var_names)
    keep = pairs[pairs["human"].isin(hset) & pairs["mouse"].isin(mset)].copy()
    # guard against duplicate symbols collapsing the space
    keep = keep.drop_duplicates(subset="human").drop_duplicates(subset="mouse")
    print(f"1:1 orthologs measured in BOTH datasets: {len(keep)}")

    # --- the case-trap sanity guard: a real ortholog space is thousands, not tens ---
    assert len(keep) > 1000, (
        f"Only {len(keep)} shared genes -- ortholog mapping likely failed "
        f"(case mismatch or wrong table). Refusing to proceed."
    )

    human_shared = human[:, keep["human"].values].copy()
    mouse_shared = mouse[:, keep["mouse"].values].copy()
    # rename mouse genes to the human ortholog symbol -> shared feature names
    mouse_shared.var_names = keep["human"].values
    human_shared.var_names = keep["human"].values

    joint = ad.concat([human_shared, mouse_shared], join="inner", label="batch_src")
    joint.obs_names_make_unique()

    # keep only cell types present in both species
    before = joint.n_obs
    joint = joint[joint.obs["cell_type"].isin(C.SHARED_CELLTYPES)].copy()
    joint.obs["cell_type"] = joint.obs["cell_type"].cat.remove_unused_categories()
    print(f"Kept {joint.n_obs}/{before} cells in shared cell types")

    print("\nCells per species x cell_type:")
    print(pd.crosstab(joint.obs["species"], joint.obs["cell_type"]))

    joint.write(C.JOINT_RAW)
    print(f"\nWrote {C.JOINT_RAW}  ({joint.n_obs} cells x {joint.n_vars} genes)")


if __name__ == "__main__":
    main()
