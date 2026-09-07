"""Tests for src/prep.py -- the shared preprocessing both the scratch VAE and scVI train on.

The main thing worth guarding is exactly what prep.py's own docstring calls out: HVG gene
subsetting must happen AFTER normalize_total() runs on the FULL gene set, not before --
otherwise each cell's size factor is silently computed from only the genes that happen to
survive HVG selection, biasing every downstream normalized value. A synthetic dataset small
enough to be readable, with two genes plainly containing more counts than the rest, is enough
to make that ordering bug produce a numerically detectable difference.
"""
import pytest

np = pytest.importorskip("numpy")
sparse = pytest.importorskip("scipy.sparse")
ad = pytest.importorskip("anndata")
pytest.importorskip(
    "scanpy",
    reason="scanpy is not installed in the fast CI lane (see .github/workflows/ci.yml); "
           "run `uv sync && uv run pytest` for full coverage.",
)

import config as C  # noqa: E402
import prep  # noqa: E402


def _make_joint_raw(path, n_cells=40, n_genes=30, seed=0):
    """Write a tiny synthetic joint_raw.h5ad and return the raw counts as a dense array."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(lam=3, size=(n_cells, n_genes)).astype(np.float32)
    X[:, :5] = rng.negative_binomial(5, 0.3, size=(n_cells, 5)).astype(np.float32)
    a = ad.AnnData(X=sparse.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    half = n_cells // 2
    a.obs["species"] = (["human"] * half + ["mouse"] * (n_cells - half))
    a.write(path)
    return X


def test_get_data_selects_requested_number_of_hvgs(tmp_path, monkeypatch):
    joint = tmp_path / "joint_raw.h5ad"
    _make_joint_raw(joint)
    monkeypatch.setattr(C, "JOINT_RAW", joint)
    monkeypatch.setattr(prep, "N_HVG", 10)

    out = prep.get_data()
    assert out.n_vars == 10
    assert out.n_obs == 40


def test_get_data_preserves_raw_counts_layer(tmp_path, monkeypatch):
    joint = tmp_path / "joint_raw.h5ad"
    raw = _make_joint_raw(joint)
    monkeypatch.setattr(C, "JOINT_RAW", joint)
    monkeypatch.setattr(prep, "N_HVG", 10)

    out = prep.get_data()
    got = out.layers["counts"]
    got = got.toarray() if sparse.issparse(got) else got
    hvg_genes = [int(g[1:]) for g in out.var_names]
    np.testing.assert_array_equal(got, raw[:, hvg_genes])


def test_get_data_normalizes_using_full_gene_set_before_subsetting(tmp_path, monkeypatch):
    """The size factor for log-normalization must come from each cell's total count across
    ALL genes, computed before HVG subsetting -- not from the total restricted to whichever
    HVG genes survive. Get this order wrong and every X value in the output is silently off."""
    joint = tmp_path / "joint_raw.h5ad"
    raw = _make_joint_raw(joint)
    monkeypatch.setattr(C, "JOINT_RAW", joint)
    monkeypatch.setattr(prep, "N_HVG", 10)

    out = prep.get_data()
    got = out.X.toarray() if sparse.issparse(out.X) else out.X

    hvg_genes = [int(g[1:]) for g in out.var_names]
    raw_hvg = raw[:, hvg_genes]
    full_total_per_cell = raw.sum(axis=1)  # size factor basis: ALL 30 genes, not just the 10 kept

    expected = np.log1p(1e4 * raw_hvg / full_total_per_cell[:, None])
    np.testing.assert_allclose(got, expected, atol=1e-4)
