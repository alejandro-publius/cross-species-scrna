"""
Phase 2: a variational autoencoder built FROM SCRATCH in raw PyTorch. No scvi-tools here.
The point is comprehension: every piece of the ELBO is visible.

Architecture (a hand-rolled mini-scVI):
  encoder:  log-normalized expression -> hidden -> (mu, logvar) of q(z|x)
  sample:   z = mu + sigma * eps        <- the reparameterization trick
  decoder:  z -> gene proportions (softmax); mean count = library_size * proportion
  likelihood: Negative Binomial (counts are over-dispersed; Gaussian/Poisson are both wrong)
  loss:     -ELBO = -E[log p(x|z)]  +  KL( q(z|x) || N(0, I) )
                     \____reconstruction____/    \_____regularizer_____/

Why NB and not MSE: raw counts are non-negative, integer, and over-dispersed (variance > mean).
An MSE loss silently assumes Gaussian noise and mishandles the pile of zeros. NB models the
count noise directly, so the latent space captures biology instead of fighting the noise model.

Why the reparameterization trick: we need to backprop THROUGH a random sample. Sampling z
directly is non-differentiable. Writing z = mu + sigma*eps moves the randomness into eps (a
fixed N(0,I) draw with no parameters), leaving a differentiable path through mu and sigma.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import prep

DEVICE = "cpu"          # MPS is flaky for this stack; data is small, CPU is fine
LATENT = 10
HIDDEN = 128
EPOCHS = 120
BATCH = 128
LR = 1e-3


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


class VAE(nn.Module):
    def __init__(self, n_genes, hidden=HIDDEN, latent=LATENT):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_genes, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, n_genes),
        )
        # per-gene dispersion (theta) of the negative binomial, learned in log space
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)      # the only source of randomness; no grad needed
        return mu + eps * std

    def decode(self, z, library):
        logits = self.dec(z)
        prop = F.softmax(logits, dim=-1)         # gene fractions, sum to 1
        mu = library * prop                      # expected counts scaled by cell's library size
        return mu

    def forward(self, x_log, x_counts):
        mu_z, logvar_z = self.encode(x_log)
        z = self.reparameterize(mu_z, logvar_z)
        library = x_counts.sum(1, keepdim=True)
        mu = self.decode(z, library)
        return mu, mu_z, logvar_z


def nb_nll(x, mu, theta, eps=1e-8):
    """Negative binomial negative log-likelihood (mean-dispersion parameterization)."""
    theta = theta.clamp(min=eps)
    log_theta_mu = torch.log(theta + mu + eps)
    ll = (
        theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return -ll.sum(-1)


def kl_standard_normal(mu, logvar):
    """KL( N(mu, sigma^2) || N(0, I) ), closed form, summed over latent dims."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)


def main():
    set_seed(C.SEED)
    adata = prep.get_data()
    x_log = torch.tensor(np.asarray(adata.X.todense()), dtype=torch.float32)
    x_cnt = torch.tensor(np.asarray(adata.layers["counts"].todense()), dtype=torch.float32)
    n_genes = x_log.shape[1]
    print(f"VAE on {x_log.shape[0]} cells x {n_genes} genes | latent={LATENT} | {EPOCHS} epochs (CPU, ~1-2 min)")

    loader = DataLoader(TensorDataset(x_log, x_cnt), batch_size=BATCH, shuffle=True)
    model = VAE(n_genes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"elbo": [], "recon": [], "kl": []}
    for epoch in range(EPOCHS):
        model.train()
        tot_r = tot_k = n = 0
        for xb_log, xb_cnt in loader:
            mu, mu_z, logvar_z = model(xb_log, xb_cnt)
            theta = torch.exp(model.log_theta)
            recon = nb_nll(xb_cnt, mu, theta).mean()
            kl = kl_standard_normal(mu_z, logvar_z).mean()
            loss = recon + kl                     # = -ELBO
            opt.zero_grad(); loss.backward(); opt.step()
            bs = xb_log.shape[0]
            tot_r += recon.item() * bs; tot_k += kl.item() * bs; n += bs
        history["recon"].append(tot_r / n)
        history["kl"].append(tot_k / n)
        history["elbo"].append(-(tot_r + tot_k) / n)
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1:3d}  -ELBO={tot_r/n + tot_k/n:9.1f}  recon={tot_r/n:9.1f}  KL={tot_k/n:6.2f}")

    # save latent embedding
    model.eval()
    with torch.no_grad():
        mu_z, _ = model.encode(x_log)
    adata.obsm["X_vae"] = mu_z.cpu().numpy()
    adata.write(C.PROC / "joint_vae.h5ad")

    # loss curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["recon"], label="reconstruction (NB NLL)")
    ax[0].plot(history["kl"], label="KL")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("VAE loss terms")
    ax[1].plot(history["elbo"], color="k"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("ELBO")
    ax[1].set_title("ELBO (higher = better)")
    fig.tight_layout(); fig.savefig(C.RESULTS / "vae_loss_curve.png", dpi=120)
    print(f"\nWrote {C.RESULTS/'vae_loss_curve.png'} and {C.PROC/'joint_vae.h5ad'}")


if __name__ == "__main__":
    main()
