"""
Phase 5a: supervised contrastive alignment (runs on CPU, no GPU needed).

Goal: explicitly PULL same-cell-type cells together in a shared embedding, then test whether
that structure transfers ACROSS species. This is a different objective from the VAE:
  - the VAE is GENERATIVE + unsupervised: reconstruct counts, regularize the latent to N(0,I).
    Cross-species alignment there is a *side effect* of removing the species batch term.
  - supervised contrastive is DISCRIMINATIVE metric learning: for each cell (anchor), pull
    same-cell-type cells closer and push different types away.

HONESTY NOTE (why we train on HUMAN cells only):
The tempting version trains SupCon on *all* cells' labels -- human AND mouse -- and then
"evaluates" human->mouse label transfer. That is LABEL LEAKAGE: the embedding was shown the
mouse labels during training, so a perfect transfer score is circular and meaningless. To get
an honest, non-leaky number that is directly comparable to scVI's, we train the encoder using
HUMAN labels ONLY (mouse is never seen with a label), embed both species, and only then ask:
does the human-supervised structure generalize to mouse? That is a real held-out test.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import config as C
import prep

DEVICE = "cpu"
EMB = 10
HIDDEN = 128
EPOCHS = 100
BATCH = 256
LR = 1e-3
TAU = 0.1


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


class Encoder(nn.Module):
    def __init__(self, n_genes, hidden=HIDDEN, emb=EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, emb),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)   # embeddings live on the unit sphere


def supcon_loss(z, labels, tau=TAU):
    """Supervised contrastive loss. Positives = same label (anywhere in the batch)."""
    sim = z @ z.t() / tau                          # cosine sim / temperature
    n = z.shape[0]
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(logits)
    mask_self = torch.eye(n, dtype=torch.bool, device=z.device)
    exp = exp.masked_fill(mask_self, 0)
    denom = exp.sum(dim=1, keepdim=True)
    log_prob = logits - torch.log(denom + 1e-9)
    pos = (labels[:, None] == labels[None, :]) & ~mask_self
    # mean log-prob over positives, averaged over anchors that have >=1 positive
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    mean_pos = (log_prob * pos).sum(1)[valid] / pos_cnt[valid]
    return -mean_pos.mean()


def label_transfer(emb, species, ct, shuffle=False, rng=None):
    is_h = (species == "human")
    y_train = ct[is_h].copy()
    if shuffle:
        y_train = rng.permutation(y_train)
    clf = KNeighborsClassifier(n_neighbors=15)
    clf.fit(emb[is_h], y_train)
    pred = clf.predict(emb[~is_h])
    truth = ct[~is_h]
    return accuracy_score(truth, pred), balanced_accuracy_score(truth, pred)


def main():
    set_seed(C.SEED)
    adata = prep.get_data()
    X = torch.tensor(np.asarray(adata.X.todense()), dtype=torch.float32)
    ct_str = adata.obs["cell_type"].astype(str).values
    species = adata.obs["species"].astype(str).values
    classes = sorted(set(ct_str))
    y = torch.tensor([classes.index(c) for c in ct_str], dtype=torch.long)

    # --- train on HUMAN cells only (mouse is unlabeled) to avoid label leakage ---
    is_h = torch.tensor(species == "human")
    X_train, y_train_t = X[is_h], y[is_h]
    print(f"SupCon trained on {X_train.shape[0]} HUMAN cells only (mouse held out, unlabeled) "
          f"-> {EMB}d (CPU, ~1-2 min)")

    loader = DataLoader(TensorDataset(X_train, y_train_t), batch_size=BATCH, shuffle=True, drop_last=True)
    enc = Encoder(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        enc.train(); tot = n = 0
        for xb, yb in loader:
            z = enc(xb)
            loss = supcon_loss(z, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.shape[0]; n += xb.shape[0]
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1:3d}  supcon_loss={tot/n:.4f}")

    enc.eval()
    with torch.no_grad():
        emb = enc(X).cpu().numpy()
    adata.obsm["X_contrastive"] = emb
    adata.write(C.PROC / "joint_contrastive.h5ad")

    rng = np.random.default_rng(C.SEED)
    acc, bacc = label_transfer(emb, species, ct_str)
    acc_s, bacc_s = label_transfer(emb, species, ct_str, shuffle=True, rng=rng)
    print("\n=== Contrastive: human -> mouse label transfer ===")
    print(f"  real:     acc={acc:.3f}  balanced_acc={bacc:.3f}")
    print(f"  shuffled: acc={acc_s:.3f}  balanced_acc={bacc_s:.3f}")

    out = {"contrastive_label_transfer": {
        "accuracy": round(acc, 3), "balanced_acc": round(bacc, 3),
        "shuffled_accuracy": round(acc_s, 3), "shuffled_balanced_acc": round(bacc_s, 3)}}
    with open(C.RESULTS / "contrastive_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {C.RESULTS/'contrastive_report.json'} and {C.PROC/'joint_contrastive.h5ad'}")


if __name__ == "__main__":
    main()
