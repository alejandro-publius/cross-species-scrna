"""
Phase 3: production cross-species integration with scVI, species as the batch covariate.

Why species-as-batch produces cross-species alignment:
scVI models each cell's counts with an NB likelihood whose rate depends on (a) a latent
biological state z and (b) a batch-specific offset. By declaring species the batch, we tell
the model "differences that track perfectly with species are nuisance, factor them into the
batch term, NOT into z." What's left in z is variation that is NOT explained by species --
i.e. shared biological state. So a human beta cell and a mouse beta cell, stripped of their
species offset, land in the same region of z. That is the alignment.

Output: adata.obsm['X_scVI'] (the shared latent space) + UMAPs colored by species and cell type.
"""
import numpy as np
import scanpy as sc
import scvi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import prep

scvi.settings.seed = C.SEED
MAX_EPOCHS = 200


def main():
    adata = prep.get_data()
    print(f"scVI on {adata.n_obs} cells x {adata.n_vars} genes (CPU, ~3-6 min)")

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="species")
    model = scvi.model.SCVI(adata, n_latent=10)
    model.train(max_epochs=MAX_EPOCHS, accelerator="cpu", enable_progress_bar=False)

    adata.obsm["X_scVI"] = model.get_latent_representation()

    # UMAP on the integrated latent space
    sc.pp.neighbors(adata, use_rep="X_scVI", random_state=C.SEED)
    sc.tl.umap(adata, random_state=C.SEED)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sc.pl.umap(adata, color="species", ax=ax[0], show=False, title="scVI latent — by species\n(want: well mixed)")
    sc.pl.umap(adata, color="cell_type", ax=ax[1], show=False, title="scVI latent — by cell type\n(want: distinct)")
    fig.tight_layout(); fig.savefig(C.RESULTS / "scvi_umap.png", dpi=120)

    model.save(str(C.PROC / "scvi_model"), overwrite=True)
    adata.write(C.PROC / "joint_scvi.h5ad")
    print(f"Wrote {C.RESULTS/'scvi_umap.png'} and {C.PROC/'joint_scvi.h5ad'}")


if __name__ == "__main__":
    main()
