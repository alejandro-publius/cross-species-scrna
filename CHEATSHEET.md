# Interview cheat sheet

Model answers to the questions this project invites. Numbers are the real ones from this repo.
Adapt to your own voice — and read `SUMMARY.md` once so the story is yours, not recited.

**Elevator version (say this first if asked "walk me through your project"):**
> I integrated human and mouse pancreatic-islet single-cell RNA-seq into a shared latent space
> so I could map cell types across species and separate conserved from species-specific gene
> programs. I mapped the two genomes to a 12k one-to-one ortholog space, built a
> negative-binomial VAE from scratch in PyTorch to make sure I understood the architecture,
> then used scVI with species as the batch covariate for the real integration. It lifted
> human→mouse label-transfer balanced accuracy from 0.75 with a PCA baseline to 0.90, roughly
> doubled the scib batch-correction aggregate (0.28→0.56) while holding cell-type separation, and
> a label-shuffle control collapsed it to chance — so the signal is real.

---

### Q: Why a VAE and not a plain autoencoder for single-cell counts?
A plain autoencoder learns one point per cell with no structure on the latent space — it can
memorize and the space between points is meaningless. A VAE forces each cell to a *distribution*
and regularizes it toward a standard normal (the KL term), so the latent is smooth and
continuous — nearby points are biologically similar, which is exactly what I need for aligning
cell types across species. And I use a **negative-binomial** likelihood, not MSE, because raw
counts are non-negative, integer, and over-dispersed (variance > mean). MSE assumes Gaussian
noise and would mishandle the huge pile of zeros; NB models the count noise directly.

### Q: What does the KL term do; what if you weight it to 0 or ∞?
KL is a regularizer measuring how far each cell's latent distribution is from N(0,I). Weight it
to **0** and you get a plain autoencoder — great reconstruction, ragged unusable latent
(overfits, no smoothness). Weight it to **∞** and every cell collapses to N(0,I) — the latent
carries no information ("posterior collapse") and reconstruction dies. The ELBO is the balance:
reconstruct well *while* keeping the latent well-behaved. In my run KL settled around 18 nats —
not collapsed to 0, not exploding — which is the healthy regime.

### Q: Explain the reparameterization trick. Why not sample directly?
I need to backprop through a random sample of z, but sampling is non-differentiable — there's no
gradient through "draw from a distribution." The trick rewrites `z = μ + σ·ε` where `ε ~ N(0,I)`
is a parameter-free noise draw. Now the randomness lives in ε (no gradient needed), and μ and σ
are deterministic functions of the network, so gradients flow through them normally. Same
distribution, differentiable path.

### Q: Why is treating species as a batch covariate the right lever for cross-species integration?
scVI's likelihood makes each cell's expression a function of a latent biological state z **plus**
a batch-specific offset. When I declare species the batch, I'm telling the model: variation that
tracks perfectly with species is nuisance — absorb it into the batch offset, not into z. What
survives in z is variation *not* explained by species, i.e. shared biology. So a human beta cell
and a mouse beta cell, stripped of their species offset, land in the same region of z. The scib
batch-correction aggregate jumped 0.28→0.56 while bio-conservation held (~0.75→0.74). Caveat I'd
volunteer: the *local* mixing metrics (iLISI, kBET) stayed low, so integration is global, not
perfectly interspersed cell-by-cell — honest, and consistent with real species differences.

### Q: How would you know from the UMAP whether integration worked or failed?
Two-color test. Color by **species**: I want the two colors interleaved *within* each cluster —
if human and mouse form separate blobs, integration failed. Color by **cell type**: I want
clusters to stay distinct — if everything smears into one ball, I over-corrected and destroyed
biology. Success is both at once: species mixed, cell types separated. I don't eyeball only —
scib-metrics quantifies exactly that tension (batch correction vs bio-conservation).

### Q: What does one-to-one ortholog restriction throw away, and how does it bias conserved-vs-specific?
It throws away many-to-many families — gene duplications where one human gene maps to several
mouse genes or vice versa. **The killer concrete example in my own data: `INS` (insulin).** Human
INS is a one-to-many co-ortholog of mouse `Ins1`/`Ins2`, so strict 1:1 *drops insulin entirely* —
the single most important beta-cell gene is absent from my shared space. The bias: a gene with no
1:1 partner is invisible, so I can never call it species-specific — it's excluded, not classified.
So my "species-specific" list is conditioned on "had a clean 1:1 ortholog and was measured in
both." The honest fix if I needed INS is to *aggregate* the mouse paralogs (sum Ins1+Ins2) into
the human ortholog rather than pick one arbitrarily — I chose to exclude, and I'd say so plainly.

### Q: Batch-mixing vs bio-conservation trade off. How do you read it, where did yours land?
They pull against each other: maximize mixing and you can blend distinct cell types together
(false conservation); maximize bio-conservation and you can leave species unmixed (no
integration). I read them jointly, never one alone. The local-mixing metrics (iLISI/kBET) stayed modest, so I wouldn't claim "perfectly mixed."
Mine: PCA baseline was 0.28 batch-correction / 0.75
bio; scVI moved to 0.56 batch-correction / 0.74 bio. So I roughly doubled the aggregate at only a
tiny bio cost — the good corner of the trade. That aggregate gain is driven by graph-connectivity
and PCR, not by local mixing. If mixing had climbed while bio fell sharply, I'd suspect over-correction.

### Q: Your human→mouse label-transfer balanced accuracy is 0.90. Why not higher, what next?
Two reasons it's not higher. First, **rare types** — schwann and macrophage have a handful of
mouse cells, and balanced accuracy weights them equally with beta, so a few misses hurt a lot
(raw accuracy is 0.965; the gap *is* the rare-type penalty). Second, genuine **species
divergence** in some programs — not everything is conserved, which is the whole point of the
project. Next I'd: (1) add a **supervised contrastive** loss to explicitly pull homologous
types together, (2) use scANVI to inject the human labels during training, (3) get more mouse
cells for the rare types. I'd also report per-type accuracy, not just the aggregate.

### Q: You already know PCA + kNN. Why bother with scVI — what did it buy you, and by how much?
This is the honest baseline I ran precisely so scVI had to earn its place. On the *same* shared
ortholog matrix, PCA+kNN got 0.75 balanced label-transfer accuracy and only 0.28 batch-correction.
scVI took those to 0.90 and 0.56. So the deep model bought +0.15 balanced accuracy and ~2× the
batch-correction aggregate — because PCA is linear and models no batch structure, so species differences sit right
in the top components; scVI has an explicit batch term and a count-appropriate likelihood. If
scVI *hadn't* beaten PCA I'd have said so and used PCA — "I ran a library" isn't a result.

### Q: How do you know the metrics aren't measuring noise? (negative control)
I ran a label-shuffle control: permute the human training labels, breaking the
expression↔label link, and redo label transfer. If my pipeline were leaking or the metric were
gameable, accuracy would stay high. It didn't — it collapsed to ~0.10 balanced (≈ chance for the
10 cell types present in mouse). That tells me the 0.90 is coming from real structure, not from a
bug. It's the same sanity instinct as a permutation test in my prior bootstrap work.

### Q: How is this different from your bulk / microbiome ML, and why does single-cell need deep learning?
Bulk gives one deep, stable averaged vector per sample — tabular, low-noise, and tree ensembles
on engineered features work great. Single-cell is the opposite: per cell you get a sparse,
90%+-zero, over-dispersed count vector where a zero is ambiguous (gene off vs **dropout** — not
captured). You can't average the noise away because resolving individual cells is the point. So
you need a model that (a) has the right **noise model** (negative binomial, not Gaussian) and
(b) learns a **structured latent** you can align across batches/species — that's what the VAE /
scVI family gives you, and what an XGBoost on raw counts cannot.

### Q: Where would this break at Genentech scale that it doesn't on Baron?
Baron is ~10k clean cells, 6 samples, one technology. At scale you have millions of cells,
hundreds of donors, multiple platforms (10x versions, protocols) — the batch structure is
**nested and confounded** (species × donor × technology), so "species as batch" alone is too
coarse; I'd need multiple/hierarchical covariates and probably a foundation model (Geneformer/
scGPT) pretrained across tissues. Memory and training time stop being trivial — this ran on CPU;
that wouldn't. And ortholog coverage matters more: rare/duplicated gene families I dropped could
be exactly the species-specific drug-relevant biology.

### Q: You added supervised contrastive learning. How does it differ from the VAE, and what did you find?
The VAE is generative and unsupervised — it reconstructs counts and regularizes the latent;
cross-species alignment is a *side effect* of removing the species batch term. Supervised
contrastive is discriminative metric learning — for each cell it pulls same-cell-type cells
together and pushes other types apart, directly shaping the geometry by label. On human→mouse
transfer the ladder was: PCA 0.75 → scVI 0.90 (unsupervised) → contrastive 0.91 balanced
accuracy. But I'd stress the asymmetry: contrastive *used human labels*, scVI used none — so
the 0.91-vs-0.90 edge is only fair given labels; scVI's 0.90 with zero supervision is arguably the
more impressive result, and it's also generative (I can sample/impute), which contrastive isn't.

### Q: (the one to volunteer) Tell me about a bug or mistake in your own analysis.
My first contrastive run scored a *perfect* 1.00 on human→mouse label transfer. That's a red
flag, not a win — I'd trained the contrastive loss on every cell's label including the mouse
cells, then "evaluated" on those same mouse labels. Textbook **label leakage**: the model was
shown the answers. I caught it because 1.00 is implausible, retrained supervising on **human
cells only** with mouse fully held out, and got an honest 0.91 balanced. The negative control
(label shuffle → chance) is exactly the tripwire that would have caught it if I hadn't. I'd
rather report 0.91 I can defend than 1.00 I can't.

---

### Eligibility
You're an enrolled Master's student at UC Berkeley — you **meet** the JD's Master's/PhD
requirement. No workaround needed: state it plainly and let the project carry the conversation.
