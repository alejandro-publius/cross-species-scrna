"""Shared configuration and paths. Single source of truth for the pipeline."""
from pathlib import Path

# ---- reproducibility ----
SEED = 0

# ---- paths ----
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
RESULTS = ROOT / "results"
for _d in (PROC, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---- raw Baron (GSE84133) files ----
HUMAN_FILES = [
    "GSM2230757_human1_umifm_counts.csv.gz",
    "GSM2230758_human2_umifm_counts.csv.gz",
    "GSM2230759_human3_umifm_counts.csv.gz",
    "GSM2230760_human4_umifm_counts.csv.gz",
]
MOUSE_FILES = [
    "GSM2230761_mouse1_umifm_counts.csv.gz",
    "GSM2230762_mouse2_umifm_counts.csv.gz",
]

# Cell-type label harmonization (raw label -> canonical). Baron uses slightly different
# casing/names across species; we normalize so "the same cell type" really is the same string.
CELLTYPE_CANON = {
    "t_cell": "t_cell", "T_cell": "t_cell",
    "b_cell": "b_cell", "B_cell": "b_cell",
    "activated_stellate": "activated_stellate",
    "quiescent_stellate": "quiescent_stellate",
}

# Cell types present in BOTH species with enough cells to be meaningful. Fixed here so the
# analysis is reproducible and we don't silently include a type with 2 cells.
SHARED_CELLTYPES = [
    "beta", "alpha", "delta", "gamma",
    "ductal", "acinar", "endothelial",
    "quiescent_stellate", "activated_stellate", "macrophage", "schwann",
]

# ---- processed outputs ----
JOINT_RAW = PROC / "joint_raw.h5ad"          # human+mouse, shared ortholog gene space, raw counts
