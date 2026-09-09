"""Does SHARED_CELLTYPES (config.py) actually mean "shared"?

config.py's own comment says the list exists "so we don't silently include a type
with 2 cells" -- i.e. it is supposed to guarantee both species have a meaningful
population of every listed type. This file checks that guarantee against the real
Baron GSE84133 data (not synthetic): it reads `data/processed/{human,mouse}_raw.h5ad`,
which `uv run python src/01_load_data.py` produces, and SKIPS if they are not present
-- CI does not download data (see .github/workflows/ci.yml and CLAUDE.md), so this
only runs locally after the raw pipeline stage has been run once.

    uv run python src/00_download.py   # public NCBI GEO + MGI downloads, ~44 MB total
    uv run python src/01_load_data.py
    uv run pytest tests/test_shared_celltypes.py -v
"""

import pytest

ad = pytest.importorskip("anndata")

import config as C  # noqa: E402

MIN_CELLS = 1  # the absolute floor: a type absent from a species (0 cells) is not
# "shared" under ANY reading of config.py's own comment. The stricter >=10 threshold
# used downstream (05_eval.py's conserved_analysis, per-species) is a higher bar for
# a different purpose (enough cells for a stable marker-gene test); this file checks
# the weaker claim that the type exists in both species at all.


def _counts_by_species():
    human_path = C.PROC / "human_raw.h5ad"
    mouse_path = C.PROC / "mouse_raw.h5ad"
    if not (human_path.exists() and mouse_path.exists()):
        pytest.skip(
            "data/processed/{human,mouse}_raw.h5ad not present -- run "
            "`uv run python src/00_download.py && uv run python src/01_load_data.py` "
            "first. Not run in CI (no data download there); see ci.yml."
        )
    human = ad.read_h5ad(human_path)
    mouse = ad.read_h5ad(mouse_path)
    return human.obs["cell_type"].value_counts(), mouse.obs["cell_type"].value_counts()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN, NOT FIXED: the real Baron mouse islet sample has ZERO cells labelled "
        "'acinar' -- confirmed by loading data/processed/mouse_raw.h5ad and checking "
        "cell_type_raw.value_counts(); 'acinar' never appears among mouse1/mouse2's "
        "13 raw cluster labels, only human's. config.py's SHARED_CELLTYPES lists "
        "'acinar' anyway, so 958 human-only acinar cells flow into joint_raw.h5ad, HVG "
        "selection, VAE/scVI training, and the human->mouse label-transfer classifier's "
        "label space as if it were a genuinely shared type -- with nothing on the mouse "
        "side to integrate against. Removing 'acinar' from SHARED_CELLTYPES (the "
        "straightforward fix) changes the shared gene panel size (12,063 -> a different "
        "number once genes are re-selected against a smaller/different cell population), "
        "which changes the VAE/scVI training inputs, and therefore every number "
        "README.md/results/ currently report (label-transfer accuracy, scib-metrics, "
        "conserved-vs-specific counts) -- none of which this audit regenerates without "
        "re-running the full (expensive) training pipeline. Left as strict xfail rather "
        "than fixed or silently passing: if SHARED_CELLTYPES or the underlying data ever "
        "changes so this test starts passing, strict=True turns that into a loud failure "
        "(XPASS) instead of a silent no-op, which is exactly the signal the repo owner "
        "should get to know it's now safe to remove this marker."
    ),
)
def test_every_shared_celltype_has_cells_in_both_species():
    human_counts, mouse_counts = _counts_by_species()
    missing_in_mouse = sorted(
        ct for ct in C.SHARED_CELLTYPES
        if human_counts.get(ct, 0) >= MIN_CELLS and mouse_counts.get(ct, 0) < MIN_CELLS
    )
    missing_in_human = sorted(
        ct for ct in C.SHARED_CELLTYPES
        if mouse_counts.get(ct, 0) >= MIN_CELLS and human_counts.get(ct, 0) < MIN_CELLS
    )
    assert not missing_in_mouse and not missing_in_human, (
        f"SHARED_CELLTYPES contains type(s) with zero cells in one species -- not "
        f"actually shared: missing_in_mouse={missing_in_mouse} "
        f"missing_in_human={missing_in_human}"
    )
