# Cross-Species Single-Cell Integration of Human and Mouse Pancreatic Islets
### A two-page technical summary

> Personal learning project. Results are framed as **association, not mechanism**. Nothing here
> implies affiliation with any organization.

## Problem
Preclinical mouse models are only useful where mouse cells behave like their human
counterparts. This project builds a shared representation of **human and mouse pancreatic
islet** single-cell RNA-seq, then uses it to (a) **map human cell-type phenotypes onto mouse**
and (b) separate **conserved** from **species-specific** gene programs — the core task of
cross-species foundational modeling.

## Data
- **Baron et al. 2016 (GSE84133):** 4 human donors (8,569 cells), 2 mouse (1,886 cells), raw UMI counts with author cell-type labels.
- **Shared feature space:** human and mouse do not share a gene list, and symbols differ by
  case (`INS` vs `Ins1`). Genes were mapped through the curated **MGI mouse↔human homology
  table**, restricted to **one-to-one orthologs** → **12,147 shared genes** (a case-blind
  symbol match would have returned almost none — this is guarded with an assertion).
- After keeping cell types present in both species: **10,380 cells**; top **2,000 highly
  variable genes** (selected with `batch_key=species` so the choice isn't biased toward human).
- **10 cell types are shared across species**; `acinar` is human-only (Baron's mouse islets
  were FACS-enriched) — a built-in species-specific example.

## Method
1. **VAE from scratch (raw PyTorch).** A negative-binomial VAE (encoder → latent `q(z|x)` →
   reparameterized sample → NB decoder scaled by library size; loss = NB reconstruction + KL).
   Built first to prove the architecture is understood, not just called. −ELBO improved
   720 → 697 over 120 epochs. This model has **no** species correction — it is the naive
   baseline.
2. **scVI with species as the batch covariate.** The NB likelihood's rate depends on a latent
   biological state `z` and a species-specific offset; declaring species the batch pushes
   species-tracking variation into the offset, leaving shared biology in `z`. 10-dim latent.
3. **Classical baseline.** PCA (10 comps) on the log-normalized shared-ortholog matrix — the
   unintegrated reference the deep model must beat.

## Results (real numbers)
**Integration quality (scib-metrics):**

| Representation | Batch correction ↑ | Bio-conservation ↑ | Total |
|---|---|---|---|
| PCA (unintegrated) | 0.284 | 0.769 | 0.575 |
| **scVI (species as batch)** | **0.553** | **0.773** | **0.685** |

scVI nearly doubles species mixing while holding cell-type separation — the desired trade.

**Human → mouse cell-type label transfer** (kNN trained on human labels, evaluated on mouse):

| Representation | Accuracy | Balanced accuracy |
|---|---|---|
| PCA (baseline) | 0.848 | 0.632 |
| **scVI** | **0.942** | **0.860** |

**Negative control (label shuffle):** accuracy collapses to 0.279 / 0.110 balanced
(≈ chance for 11 types) — confirms the signal is real, not leakage.

**Supervised contrastive alignment (Phase 5a, stretch).** A SupCon encoder trained on **human
labels only** (mouse held out, to avoid leakage) transfers to mouse at **0.937 / 0.919
balanced** — above scVI, as expected for a supervised method, though scVI reaches 0.86 with
*no* labels and is also generative. An initial all-cells-supervised run scored a suspicious
1.00 (label leakage); catching and correcting that is documented in `CHEATSHEET.md`.

**Conserved vs species-specific programs:** per shared cell type, marker genes (Wilcoxon,
one-vs-rest) were computed within each species and compared. Conserved sets recover canonical
biology — beta: `ABCC8, G6PC2, IAPP, NKX6-1, NEUROD1, INSM1, CHGA`; alpha: `GCG, ARX, MAFB,
PCSK2`. Marker-set Jaccard ranged ~0.12–0.30 across cell types (top canonical markers conserved;
long tail species-specific).

## Honest limitations
- **Balanced accuracy (0.86) < raw accuracy (0.94):** rare types (schwann, macrophage) transfer
  worse; the headline number is buoyed by abundant beta/alpha cells.
- **One-to-one ortholog restriction** discards many-to-many families (gene duplications), which
  can *bias the conserved-vs-specific call*: a gene dropped for lack of a 1:1 partner is invisible,
  not "species-specific."
- **Conserved/specific is descriptive, not causal** — it reflects marker-gene overlap in this
  dataset, not a mechanistic claim.
- **Baron is small and clean.** On a real atlas (millions of cells, many donors/technologies)
  the batch structure is nested and far harder; this pipeline would need scaling and stronger
  batch modeling.

## Reproduce
`./run_all.sh` (uv, Python 3.11, CPU, ~10 min). Seeds fixed; deps pinned in `uv.lock`.
