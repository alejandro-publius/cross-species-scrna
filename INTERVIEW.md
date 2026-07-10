# INTERVIEW.md — the real deliverable

Every design decision in this repo invites a question. Each one lives here with **my own
written answer** underneath. If an answer collapses into "I just called the library," it isn't
done. Rule: I write the answer in my own words before the phase is allowed to close.

Status legend: 🔴 unanswered · 🟡 hand-wavy, needs another pass · 🟢 I can defend this cold

> **Model answers to all of these (with the real numbers from this run) live in
> [`CHEATSHEET.md`](CHEATSHEET.md).** Read them, then rewrite each in your own words here —
> that rewrite is what actually makes it defensible in the room.

---

## Phase 0 — single-cell literacy

### Q0.1 What is dropout in single-cell RNA-seq, and how is it different from a gene simply being off? 🔴
> _(your answer here)_

### Q0.2 Why are raw single-cell counts near-zero-inflated and over-dispersed, and why does that mean I can't just treat the matrix as Gaussian? 🔴
> _(your answer here)_

### Q0.3 What is a batch effect, and concretely why does it wreck a naive clustering analysis? 🔴
> _(your answer here)_

### Q0.4 What do obs, var, X, layers, and obsm each hold in an AnnData object, and why does single-cell data need this structure instead of a plain matrix? 🔴
> _(your answer here)_

---

## Seed questions (populated as we hit each phase)

### Why a VAE and not a plain autoencoder for single-cell counts? 🔴
> _(Phase 2)_

### What does the KL term do, and what happens if you weight it to zero or to infinity? 🔴
> _(Phase 2)_

### Explain the reparameterization trick. Why can't you just sample the latent directly? 🔴
> _(Phase 2)_

### Why is treating species as a batch covariate the right lever for cross-species integration? 🔴
> _(Phase 3)_

### What does one-to-one ortholog restriction throw away, and how might that bias the conserved-vs-specific call? 🔴
> _(Phase 1)_

### Batch mixing and bio-conservation trade against each other. How do you read that tradeoff, and where did yours land? 🔴
> _(Phase 4)_

### Your label-transfer accuracy from human to mouse is X. Why is it not higher, and what would you try next? 🔴
> _(Phase 4)_

### How is this different from the bulk / microbiome ML I did before, and why does single-cell need the deep-learning approach? 🔴
> _(Phase 0/2)_

### Where would this pipeline break on a real Genentech-scale dataset that it does not break on Baron? 🔴
> _(Phase 6)_

### You already know PCA + kNN label transfer. Why bother with scVI at all — what does the deep model buy you over that baseline, and by how much did it actually win here? 🔴
> _(Phase 3/4 — the answer that connects my old skillset to the new work)_

### How do you know your integration metrics aren't just measuring noise? What does your label-shuffle negative control show, and what would it have meant if the metrics *hadn't* dropped to chance? 🔴
> _(Phase 4)_

---

## Decision log
_Append every architectural decision here as we make it: what we chose, the obvious
alternative we rejected, and why._

- **2026-07-10 · Tooling:** uv + managed Python 3.11 over conda/system-3.9. Reason: isolated
  from system Python, real lockfile (reproducibility checkmark), fast resolution. Rejected
  conda (heavier, slower, no longer needed for this stack on Apple Silicon).
- **2026-07-10 · Compute:** default training to CPU, not MPS. MPS is available but flaky for
  this stack; data is small enough that CPU is fine. Avoids silent numerical weirdness.
- **2026-07-10 · Rigor scope:** added an honest classical baseline (PCA + kNN / RF label
  transfer) the deep model must beat, and a label-shuffle negative control that must degrade
  to chance. These convert "I ran a library" into "I showed it beats the obvious baseline and
  passes a sanity control." See CLAUDE.md.
- **2026-07-10 · Ortholog robustness:** static cached homology table as fallback for flaky
  live biomart queries; case-aware ortholog mapping (never case-blind symbol intersection).
