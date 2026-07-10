"""
Shared preprocessing used by BOTH the scratch VAE (03) and scVI (04), so they train on an
identical feature set and the comparison is fair.

- keep raw counts in layers['counts'] (models with a count likelihood need these)
- log-normalize into X (used as the ENCODER input; a network trains better on log-scaled input)
- select 2000 highly variable genes with batch_key='species' so the HVG choice is not biased
  toward the species with more cells
"""
import numpy as np
import scanpy as sc
import anndata as ad

import config as C

N_HVG = 2000


def get_data():
    a = ad.read_h5ad(C.JOINT_RAW)
    a.layers["counts"] = a.X.copy()  # raw UMI counts preserved
    # HVG on raw counts (seurat_v3 expects counts), fair across species
    sc.pp.highly_variable_genes(
        a, flavor="seurat_v3", n_top_genes=N_HVG, batch_key="species", layer="counts"
    )
    a = a[:, a.var["highly_variable"]].copy()
    # log-normalize X for encoder input; counts layer stays raw
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a
