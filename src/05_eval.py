"""
Phase 4: evaluation and the conserved vs species-specific story.

Four things, each a line item in the JD:
  1. Integration metrics (scib-metrics): batch mixing (does species mix?) balanced against
     bio-conservation (do cell types stay distinct?). We score scVI AND an unintegrated PCA
     baseline so scVI has to justify itself.
  2. Human -> mouse label transfer: train a kNN classifier on HUMAN cell-type labels in a
     representation, predict MOUSE labels, report accuracy. This is literally "map human
     phenotypes to mouse phenotypes." Run on both PCA (baseline) and scVI.
  3. Negative control: shuffle the labels and repeat -- accuracy must collapse to chance. If
     it doesn't, we're leaking and the headline number is meaningless.
  4. Conserved vs species-specific gene programs: for each shared cell type, find marker genes
     within each species and compare. Markers in BOTH = conserved; in ONE = species-specific.
"""
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import config as C

RNG = np.random.default_rng(C.SEED)


# ---------- 2 & 3: cross-species label transfer + negative control ----------
def label_transfer(adata, rep, shuffle=False):
    """Train kNN on human labels in `rep`, predict mouse. Return (accuracy, balanced_acc)."""
    is_h = (adata.obs["species"] == "human").values
    is_m = ~is_h
    y = adata.obs["cell_type"].astype(str).values
    y_train = y[is_h].copy()
    if shuffle:
        y_train = RNG.permutation(y_train)  # break the label<->expression link
    X = adata.obsm[rep]
    clf = KNeighborsClassifier(n_neighbors=15)
    clf.fit(X[is_h], y_train)
    pred = clf.predict(X[is_m])
    truth = y[is_m]
    return accuracy_score(truth, pred), balanced_accuracy_score(truth, pred)


# ---------- 4: conserved vs species-specific gene programs ----------
def species_markers(adata, species, cell_type, n=50):
    sub = adata[adata.obs["species"] == species].copy()
    if cell_type not in sub.obs["cell_type"].unique() or (sub.obs["cell_type"] == cell_type).sum() < 10:
        return None
    sub.obs["grp"] = np.where(sub.obs["cell_type"] == cell_type, cell_type, "rest")
    sc.tl.rank_genes_groups(sub, "grp", groups=[cell_type], reference="rest", method="wilcoxon")
    names = sub.uns["rank_genes_groups"]["names"][cell_type][:n]
    return set(map(str, names))


def conserved_analysis(adata):
    shared = [ct for ct in C.SHARED_CELLTYPES
              if ((adata.obs["species"] == "mouse") & (adata.obs["cell_type"] == ct)).sum() >= 10]
    rows = []
    examples = {}
    for ct in shared:
        h = species_markers(adata, "human", ct)
        m = species_markers(adata, "mouse", ct)
        if h is None or m is None:
            continue
        conserved = h & m
        hn = h - m
        mn = m - h
        rows.append({
            "cell_type": ct,
            "n_conserved": len(conserved),
            "n_human_specific": len(hn),
            "n_mouse_specific": len(mn),
            "jaccard": round(len(conserved) / len(h | m), 3),
        })
        if ct in ("beta", "alpha", "delta"):
            examples[ct] = {
                "conserved": sorted(conserved)[:12],
                "human_specific": sorted(hn)[:12],
                "mouse_specific": sorted(mn)[:12],
            }
    return pd.DataFrame(rows), examples


def main():
    adata = ad.read_h5ad(C.PROC / "joint_scvi.h5ad")
    # unintegrated baseline representation: PCA on the log-normalized data
    sc.pp.pca(adata, n_comps=10, random_state=C.SEED)
    reps = {"PCA (unintegrated baseline)": "X_pca", "scVI (integrated)": "X_scVI"}

    report = {}

    # --- label transfer ---
    print("=== Human -> mouse label transfer ===")
    lt = {}
    for name, rep in reps.items():
        acc, bacc = label_transfer(adata, rep)
        acc_s, bacc_s = label_transfer(adata, rep, shuffle=True)
        lt[name] = {"accuracy": round(acc, 3), "balanced_acc": round(bacc, 3),
                    "shuffled_accuracy": round(acc_s, 3), "shuffled_balanced_acc": round(bacc_s, 3)}
        print(f"  {name:30s} acc={acc:.3f} bal_acc={bacc:.3f}  | shuffled acc={acc_s:.3f} bal_acc={bacc_s:.3f}")
    report["label_transfer"] = lt
    n_types = adata.obs["cell_type"].nunique()
    print(f"  (chance level ~ {1/n_types:.3f} for {n_types} types; shuffled should be near chance)")

    # --- scib integration metrics ---
    print("\n=== scib-metrics integration (scVI vs baseline) ===")
    try:
        from scib_metrics.benchmark import Benchmarker
        bm = Benchmarker(adata, batch_key="species", label_key="cell_type",
                         embedding_obsm_keys=["X_pca", "X_scVI"], n_jobs=1)
        bm.benchmark()
        res = bm.get_results(min_max_scale=False)
        print(res.to_string())
        report["scib"] = json.loads(res.reset_index().to_json(orient="records"))
    except Exception as e:
        print(f"  scib-metrics failed: {e}")
        report["scib_error"] = str(e)

    # --- conserved vs species-specific ---
    print("\n=== Conserved vs species-specific gene programs ===")
    df, examples = conserved_analysis(adata)
    print(df.to_string(index=False))
    report["conserved_table"] = json.loads(df.to_json(orient="records"))
    report["conserved_examples"] = examples

    with open(C.RESULTS / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    df.to_csv(C.RESULTS / "conserved_vs_specific.csv", index=False)
    print(f"\nWrote {C.RESULTS/'eval_report.json'} and {C.RESULTS/'conserved_vs_specific.csv'}")


if __name__ == "__main__":
    main()
