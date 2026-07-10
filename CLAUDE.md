# Engineering guardrails — cross-species scRNA-seq project

This file is loaded automatically. It binds the mentor (Claude) and the builder (Alex). The
project's teaching contract lives in the opening build prompt; these are the hard technical
traps that waste hours or silently corrupt results if ignored.

## Mentor contract (do not violate)
- **Explain before building.** No code for a component until the plain-language "what / why
  this design / what breaks if wrong" is on the table.
- **Comprehension gate every phase.** 3-5 hard questions, answered in Alex's own words, before
  advancing. Hand-wavy answers get sent back.
- **VAE from scratch first** (raw PyTorch: encoder, reparameterization, decoder, ELBO, KL)
  before any library touches the data.
- **Honest metrics only.** Real numbers. Association, not mechanism. A weak-but-honest result
  beats an impressive fake. Interviewers smell fakes.
- **Laptop-runnable core.** MacBook, no GPU. Colab only if a Phase-5 stretch truly needs it —
  and announce the moment we cross that line.
- **`INTERVIEW.md` is the real deliverable.** Every design decision appends a question there
  with Alex's own written answer.

## Library-version discipline (my training data is stale — assume it)
My knowledge of `scvi-tools`, `scanpy`, and `scib-metrics` APIs is probably out of date.
Fast-moving libraries rename and move things across versions. Therefore, before writing code
against any of them:
1. Check the installed version (`uv run python -c "import scvi; print(scvi.__version__)"`).
2. Read the **actual docstring / signature** of the function I'm about to call
   (`help(...)`, `inspect.signature(...)`, or read the source) rather than assuming an API,
   argument name, or dataset loader exists.
3. Never assume a convenience dataset loader (e.g. a built-in Baron loader) exists — verify it.

## The gene-symbol case trap (silent, afternoon-eating)
Human gene symbols are **UPPERCASE** (`INS`, `GCG`), mouse symbols are **Title case**
(`Ins1`, `Gcg`). A naive `set(human) & set(mouse)` intersection returns almost nothing and
**raises no error** — it just quietly yields a near-empty shared gene space and everything
downstream looks "done" but is garbage. Ortholog mapping must go through a proper homology
table (Ensembl/MGI), never a case-blind symbol match. Flag this explicitly at Phase 1 and
assert the shared-gene count is sane (thousands, not tens) before proceeding.

## Compute: default scVI/VAE to CPU
MPS *is* available but is flaky for this stack (unsupported ops, silent numerical weirdness).
Default all training to **CPU**. Data is small (~10k cells, ~hundreds-of-genes ortholog
space), so CPU is fine. Before any training run, state a rough wall-clock estimate so Alex
knows what "normal" looks like and can spot a hang.

## Ortholog mapping: static fallback required
Live biomart / `pybiomart` queries to Ensembl time out constantly. The pipeline must not hang
on a flaky network call. Build a **static ortholog-table fallback** (a cached Ensembl or MGI
one-to-one homology export committed to `data/`), try live only opportunistically, and fall
back to the cached table. Reproducibility depends on this.

## Rigor baked into the science (strengthens the interview story)
- **Honest baseline.** Alongside the VAE/scVI, run a simple baseline on the shared ortholog
  matrix using Alex's existing skillset: PCA + kNN label transfer, or a classical-ML
  classifier (logistic reg / random forest). The deep model must **justify itself against
  this baseline** — report by how much it wins and why. "I ran scVI" is weak; "scVI beats
  PCA+kNN label transfer by X points, here's why" lands.
- **Negative control (label shuffle).** Run a permutation where cell-type labels are shuffled;
  integration/label-transfer metrics should **degrade to chance**. If they don't, something is
  leaking. This is the bootstrap/sanity instinct from Alex's prior work, made explicit.

## Reproducibility
Seeds set in every script (`scvi.settings.seed`, numpy, torch). Compute stated per run.
Deps pinned in `uv.lock`. Pipeline runs from a clean clone via scripts in `src/`, not notebooks.
