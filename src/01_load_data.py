"""
Phase 1a: load the Baron human + mouse islet CSVs into per-species AnnData objects.

Each Baron CSV is cells x (index, barcode, assigned_cluster, gene1, gene2, ...) with raw UMI
counts. We build one AnnData per species holding raw counts in X, with per-cell metadata
(cell_type, species, donor). Gene symbols are kept in their native casing (human UPPER, mouse
Title) -- the ortholog step (02) is where we reconcile them, NOT here.
"""
import gzip
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse

import config as C


def _load_one(path, species, donor):
    df = pd.read_csv(path, index_col=0)
    labels = df["assigned_cluster"].astype(str).values
    genes = df.drop(columns=["barcode", "assigned_cluster"])
    X = sparse.csr_matrix(genes.values.astype(np.float32))
    a = ad.AnnData(X=X)
    a.var_names = genes.columns.astype(str)
    a.obs_names = [f"{donor}_{i}" for i in range(a.n_obs)]
    a.obs["cell_type_raw"] = labels
    a.obs["species"] = species
    a.obs["donor"] = donor
    return a


def load_species(files, species):
    parts = []
    for f in files:
        donor = f.split("_")[1]  # e.g. "human1"
        p = C.RAW / f
        a = _load_one(p, species, donor)
        print(f"  {donor}: {a.n_obs} cells x {a.n_vars} genes")
        parts.append(a)
    joined = ad.concat(parts, join="outer", fill_value=0)
    joined.obs_names_make_unique()
    # harmonize cell-type labels
    joined.obs["cell_type"] = (
        joined.obs["cell_type_raw"].map(lambda s: C.CELLTYPE_CANON.get(s, s)).astype("category")
    )
    return joined


def main():
    print("HUMAN:")
    human = load_species(C.HUMAN_FILES, "human")
    print("MOUSE:")
    mouse = load_species(C.MOUSE_FILES, "mouse")

    print(f"\nHuman total: {human.n_obs} cells x {human.n_vars} genes")
    print(f"Mouse total: {mouse.n_obs} cells x {mouse.n_vars} genes")

    human.write(C.PROC / "human_raw.h5ad")
    mouse.write(C.PROC / "mouse_raw.h5ad")
    print(f"\nWrote {C.PROC/'human_raw.h5ad'} and {C.PROC/'mouse_raw.h5ad'}")


if __name__ == "__main__":
    main()
