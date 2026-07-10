#!/usr/bin/env bash
# Reproduce the whole pipeline from a clean clone.
#   ./run_all.sh
# Requires: uv (https://docs.astral.sh/uv/). Everything else is pinned in uv.lock.
set -euo pipefail
cd "$(dirname "$0")"

echo ">> installing pinned dependencies"
uv sync

echo ">> [0/5] fetching raw data (Baron GSE84133 + MGI homology table)"
uv run python src/00_download.py

echo ">> [1/5] loading human + mouse islet data"
uv run python src/01_load_data.py

echo ">> [2/5] mapping to shared one-to-one ortholog gene space"
uv run python src/02_orthologs.py

echo ">> [3/5] VAE from scratch (raw PyTorch)"
uv run python src/03_vae_scratch.py

echo ">> [4/5] scVI cross-species integration (species as batch)"
uv run python src/04_scvi.py

echo ">> [5/5] evaluation: metrics, label transfer, negative control, conserved genes"
uv run python src/05_eval.py

echo ">> done. See results/ for figures, metrics, and eval_report.json"
