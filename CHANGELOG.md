# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `config.require()` guard clauses so `01_load_data.py`, `02_orthologs.py`, `prep.get_data()`
  (used by `03`/`04`/`06`), and `05_eval.py` fail with a one-line actionable message pointing
  to the missing upstream script, instead of a raw traceback, when run before their required
  inputs exist.
- `ruff` lint configuration in `pyproject.toml`, a `ruff check` step in CI, and a
  `.pre-commit-config.yaml`.
- Tests for `src/config.py` and `src/prep.py`.
- README quickstart, results section, and screenshot.

### Changed
- `00_download.py` now fails with a clear message (and cleans up any partial download) on a
  network error instead of raising deep inside `urllib`.
- Removed unused imports (`gzip`, `scanpy`, `numpy`) flagged by `ruff` in `src/01_load_data.py`,
  `src/04_scvi.py`, and `src/prep.py`.
- Split several multi-statement (`;`-joined) lines in `src/03_vae_scratch.py`,
  `src/04_scvi.py`, and `src/06_contrastive.py` to satisfy `ruff`'s `E702`.

## [0.1.0] - 2026-09-04

### Added
- Dependabot config for monthly GitHub Actions updates (`cb09f36`).
- CI badge in the README (`530acd3`).
- CI workflow running the ortholog unit tests on every push, without installing
  scvi-tools/torch (`146c8dd`).
- `src/orthologs.py`: a reusable, unit-tested 1:1 human/mouse ortholog mapping module, factored
  out of the pipeline script (`5422ed1`, merged via #1).
- Full pipeline: raw data download, per-species loading, ortholog mapping, a from-scratch
  PyTorch VAE, scVI cross-species integration, evaluation (scib-metrics, label transfer,
  negative control, conserved-vs-specific gene programs), and a supervised-contrastive stretch
  phase (`6a87d69`, `5999218`).

[Unreleased]: https://github.com/alejandro-publius/cross-species-scrna/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alejandro-publius/cross-species-scrna/releases/tag/v0.1.0
