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
import pandas as pd
import anndata as ad

import config as C
from orthologs import one_to_one_orthologs, restrict_to_measured

HOM_FILE = C.RAW / "HOM_MouseHumanSequence.rpt"


def main():
    pairs = one_to_one_orthologs(HOM_FILE)
    print(f"1:1 ortholog pairs in homology table: {len(pairs)}")

    human = ad.read_h5ad(C.PROC / "human_raw.h5ad")
    mouse = ad.read_h5ad(C.PROC / "mouse_raw.h5ad")

    # STRICT 1:1 -- restrict to genes measured in BOTH datasets, dropping any
    # symbol that ends up with multiple partners rather than silently resolving it.
    keep = restrict_to_measured(pairs, human.var_names, mouse.var_names)
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
