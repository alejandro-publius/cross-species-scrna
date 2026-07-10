# Cross-Species Single-Cell Integration of Human and Mouse Pancreatic Islets
### A two-page technical summary

> Personal learning project. Results are framed as **association, not mechanism**. Nothing here
> implies affiliation with any organization.

## Problem
Preclinical mouse models are only useful where mouse cells behave like their human
counterparts. This project builds a shared representation of **human and mouse pancreatic
islet** single-cell RNA-seq, then uses it to (a) **map human cell-type phenotypes onto mouse**
and (b) separate **conserved** from **species-specific** cell types and gene programs.

## Data
- **Baron et al. 2016 (GSE84133):** 4 human donors (8,569 cells), 2 mouse (1,886 cells), raw UMI counts with author cell-type labels.
- **Shared feature space:** human and mouse do not share a gene list, and symbols differ by
  case (`INS` vs `Ins1`). Genes were mapped through the curated **MGI mouse↔human homology
  table**, restricted to **strict one-to-one orthologs** → **12,067 shared genes** (a case-blind
  symbol match would have returned almost none — guarded with an assertion).
- **What 1:1 restriction throws away, concretely:** `INS` is dropped, because human insulin is a
  one-to-many co-ortholog of mouse `Ins1`/`Ins2`. Enforcing strict 1:1 honestly excludes it
  rather than arbitrarily keeping one paralog — a real example of the bias this restriction introduces.
- After keeping cell types present in both species: **10,380 cells**; top **2,000 highly
  variable genes** (selected with `batch_key=species` so the choice isn't biased toward human).
- **10 cell types are shared across species**; `acinar` is human-only (Baron's mouse islets
  were FACS-enriched) — a built-in species-specific example.

## Method
1. **VAE from scratch (raw PyTorch).** A negative-binomial VAE (encoder → latent `q(z|x)` →
   reparameterized sample → NB decoder scaled by library size; loss = NB reconstruction + KL).
   Built first to prove the architecture is understood, not just called. −ELBO decreased steadily
   over 120 epochs to ≈680, with KL stable at ≈18 nats (no posterior collapse). This model has
   **no** species correction — it is the naive baseline.
2. **scVI with species as the batch covariate.** The NB likelihood's rate depends on a latent
   biological state `z` and a species-specific offset; declaring species the batch pushes
   species-tracking variation into the offset, leaving shared biology in `z`. 10-dim latent.
3. **Classical baseline.** PCA (10 comps) on the log-normalized shared-ortholog matrix — the
   unintegrated reference the deep model must beat.

## Results (real numbers, re-run after a code-review pass)
**Integration quality (scib-metrics):**

| Representation | Batch-correction aggregate ↑ | Bio-conservation ↑ | Total |
|---|---|---|---|
| PCA (unintegrated) | 0.285 | 0.754 | 0.566 |
| **scVI (species as batch)** | **0.561** | 0.742 | **0.670** |

**Read this honestly:** the batch-correction *aggregate* roughly doubles, but that gain comes
from graph-connectivity (0.78→0.97) and PCR (0.00→0.92); the **direct local-mixing metrics stay
modest** (iLISI 0.00→0.003, kBET 0.13→0.16). So species are integrated at the cluster/global
level but **not perfectly interspersed at the finest local scale** — consistent with genuine
species differences, and not something to oversell as "fully mixed." Bio-conservation dips
slightly (0.754→0.742), the expected small cost of integration.

**Human → mouse cell-type label transfer** (kNN trained on human labels, evaluated on mouse):

| Representation | Accuracy | Balanced accuracy |
|---|---|---|
| PCA (baseline) | 0.877 | 0.752 |
| **scVI** | **0.965** | **0.903** |
| Supervised contrastive (human-labeled, mouse held out) | 0.940 | 0.911 |

**Negative control (label shuffle):** balanced accuracy collapses to ~0.08–0.14
(≈ chance; 10 cell types present in mouse) — confirms the signal is real, not leakage.

**Conserved vs species-specific programs:** per shared cell type, marker genes (Wilcoxon,
one-vs-rest) were computed within each species and compared. Conserved sets recover canonical
biology — beta: `ABCC8, G6PC2, IAPP, INSM1, NEUROD1, CHGA, HADH`; alpha: `GCG, ARX, MAFB, FEV`.
*Caveat:* the per-species "specific" **counts** are an artifact of taking a fixed top-50 markers
(they are forced equal); only the conserved gene **lists** are interpreted, not the specific counts.

## Honest limitations
- **Local mixing is modest** (iLISI/kBET low): integration is global, not fine-grained.
- **Balanced accuracy < raw accuracy:** rare types (schwann, macrophage) transfer worse.
- **Strict 1:1 orthologs** discard many-to-many families (e.g. `INS`), which can *bias the
  conserved-vs-specific call*: a gene dropped for lack of a 1:1 partner is invisible, not "specific".
- **Contrastive uses human labels; scVI uses none** — the 0.91 vs 0.90 is not a like-for-like win.
- **Conserved/specific is descriptive, not causal**, and the negative control is a single permutation.
- **Baron is small and clean.** On a real atlas the batch structure is nested and confounded;
  this pipeline would need scaling and stronger batch modeling.

## Reproduce
`./run_all.sh` (uv, Python 3.11, CPU, ~10 min). Seeds fixed; deps pinned in `uv.lock`.
