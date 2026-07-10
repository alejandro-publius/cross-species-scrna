# Cross-Species Single-Cell RNA-seq Integration

A learning project: build, from scratch and then with production tooling, a model that
integrates **human and mouse** pancreatic-islet single-cell RNA-seq into a shared latent
space, then uses that space to separate **conserved** from **species-specific** cell types
and gene programs.

> This is a personal learning project. Nothing here implies affiliation with, or work
> performed for, any organization. Results are framed as association, not mechanism.

## Why this exists
I come from classical ML (XGBoost, random forest, SHAP) on microbiome and bulk expression
data. This repo is me learning single-cell + deep learning end to end: the anndata data
model, a variational autoencoder implemented in raw PyTorch before any library, then
scvi-tools for the real cross-species integration, evaluated with scib-metrics and a
human->mouse label-transfer test.

## Roadmap
- **Phase 0** - environment + single-cell literacy (anndata, dropout, batch effects)
- **Phase 1** - Baron human + mouse islet data; QC, normalization, one-to-one ortholog mapping to a shared gene space
- **Phase 2** - VAE from scratch in raw PyTorch (encoder, reparameterization, decoder, ELBO)
- **Phase 3** - scVI with species as the batch covariate; UMAP of the shared latent space
- **Phase 4** - honest baseline (PCA + kNN label transfer) the deep model must beat; scib-metrics integration scoring; human->mouse label transfer; conserved vs species-specific gene programs; label-shuffle negative control
- **Phase 5 (stretch)** - supervised contrastive alignment; fine-tune a foundation model (Geneformer/scGPT)
- **Phase 6** - package, pin, write up

## Environment
Managed with uv. Python 3.11, Apple Silicon, CPU training (MPS is flaky for this stack), no CUDA required.
\`\`\`bash
uv sync          # install pinned dependencies
uv run python -c "import scanpy, scvi; print('ok')"
\`\`\`

## Layout
\`\`\`
data/      raw + processed datasets (gitignored)
src/       pipeline scripts (run from scratch, not notebooks)
results/   figures, metrics, latent embeddings
notes/     working notes and teaching write-ups
INTERVIEW.md   the real deliverable - every design decision + my own answer
\`\`\`

## Reproducibility
Seeds set in every script. Compute stated per run. Dependencies pinned in uv.lock.
