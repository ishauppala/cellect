"""Train and evaluate an XGBoost model that predicts LN_IC50 from gene expression.

Example:
    python src/train.py            # uses data/raw/GDSC2-dataset.csv (Kaggle) + Cell_Lines_Details.xlsx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from data import build_dataset  # noqa: E402
from demo import build_demo  # noqa: E402

# Literature-motivated (drug, gene) expectations used as a sanity check, NOT as ground truth.
# Higher expression of the target gene is expected to go with lower LN_IC50 (more sensitive).
BENCHMARKS = [("Lapatinib", "ERBB2"), ("Erlotinib", "EGFR"), ("Gefitinib", "EGFR"), ("Afatinib", "EGFR")]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--details", default="data/raw/Cell_Lines_Details.xlsx")
    ap.add_argument("--dose", default="data/raw/GDSC2-dataset.csv", help="GDSC dose-response file (.xlsx/.csv)")
    ap.add_argument("--expr", default=None, help="optional GDSC basal expression file (.txt/.csv)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max_genes", type=int, default=500, help="top-variance genes to use")
    ap.add_argument("--drugs", nargs="*", help="restrict to these drug names")
    ap.add_argument("--max_drugs", type=int, help="keep only the N most-screened drugs")
    ap.add_argument("--split", choices=["group", "random"], default="group",
                    help="group = hold out whole cell lines (honest); random = random drug-cell pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="fast, lower-accuracy run for debugging")
    return ap.parse_args()


def make_splits(df, split, seed, val_frac=0.10):
    """80/20 train/test. A further 10% of train is used only for early stopping."""
    idx = np.arange(len(df))
    if split == "group":
        g = df["COSMIC_ID"].values
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(idx, groups=g))
        tr_in, va = next(GroupShuffleSplit(1, test_size=val_frac, random_state=seed).split(tr, groups=g[tr]))
        return tr[tr_in], tr[va], te
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
    tr, va = train_test_split(tr, test_size=val_frac, random_state=seed)
    return tr, va, te


def reg_metrics(y, p):
    mse = mean_squared_error(y, p)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)),
        "Pearson": float(pearsonr(y, p)[0]) if np.ptp(p) > 1e-12 else 0.0,
        "Spearman": float(spearmanr(y, p)[0]) if np.ptp(p) > 1e-12 else 0.0,
    }


def within_drug_corr(test_df, pred, min_n=15):
    """Median Pearson across drugs: can the model rank cell lines *within* a drug?"""
    d = test_df[["DRUG_NAME", "LN_IC50"]].assign(pred=pred)
    rs = []
    for _, g in d.groupby("DRUG_NAME", observed=True):
        if len(g) >= min_n and g.LN_IC50.std() > 0 and g.pred.std() > 0:
            rs.append(pearsonr(g.LN_IC50, g.pred)[0])
    return float(np.median(rs)) if rs else 0.0


def sensitivity_auc(train_df, test_df, pred):
    """Sensitive = LN_IC50 below that drug's median in the training set."""
    med = train_df.assign(DRUG=train_df["DRUG"].astype(str)).groupby("DRUG")["LN_IC50"].median()
    thr = test_df["DRUG"].astype(str).map(med).fillna(train_df.LN_IC50.median()).values
    y_sens = (test_df.LN_IC50.values < thr).astype(int)
    if y_sens.min() == y_sens.max():
        return float("nan")
    return float(roc_auc_score(y_sens, thr - pred))


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading and joining data...")
    df, feats, genes = build_dataset(a.details, a.dose, a.expr, a.max_genes, a.drugs, a.max_drugs)
    print(f"  {len(df):,} cell-line/drug pairs | {df.COSMIC_ID.nunique()} cell lines | "
          f"{df.DRUG_NAME.nunique()} drugs | {len(genes)} genes")

    tr, va, te = make_splits(df, a.split, a.seed)
    X, y = df[feats], df["LN_IC50"].values
    print(f"  split={a.split}: train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    params = dict(
        n_estimators=300 if a.quick else 3000,
        learning_rate=0.1 if a.quick else 0.05,
        max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.4,
        reg_lambda=5.0, tree_method="hist", enable_categorical=True,
        max_cat_to_onehot=1, objective="reg:squarederror", eval_metric="rmse",
        early_stopping_rounds=50, n_jobs=-1, random_state=a.seed,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=100)
    print(f"  best iteration: {model.best_iteration}")

    pred = model.predict(X.iloc[te])
    test_df, train_df = df.iloc[te].reset_index(drop=True), df.iloc[tr]

    # Baselines the model must beat
    drug_mean = train_df.assign(DRUG=train_df["DRUG"].astype(str)).groupby("DRUG")["LN_IC50"].mean()
    base_drug = test_df["DRUG"].astype(str).map(drug_mean).fillna(y[tr].mean()).values
    base_global = np.full(len(te), y[tr].mean())

    results = {
        "XGBoost": reg_metrics(y[te], pred),
        "Baseline: per-drug mean": reg_metrics(y[te], base_drug),
        "Baseline: global mean": reg_metrics(y[te], base_global),
    }
    results["XGBoost"]["within_drug_pearson_median"] = within_drug_corr(test_df, pred)
    results["XGBoost"]["sensitive_vs_resistant_AUROC"] = sensitivity_auc(train_df, test_df, pred)
    results["Baseline: per-drug mean"]["within_drug_pearson_median"] = 0.0
    results["Baseline: per-drug mean"]["sensitive_vs_resistant_AUROC"] = sensitivity_auc(train_df, test_df, base_drug)

    summary = pd.DataFrame(results).T
    print("\n=== Held-out test results ===")
    print(summary.round(4).to_string())

    # Error by tissue
    tissue = (test_df.assign(se=(y[te] - pred) ** 2).groupby("TISSUE", observed=True)
              .agg(n=("se", "size"), MSE=("se", "mean")).sort_values("MSE"))
    tissue.to_csv(out / "mse_by_tissue.csv")

    # Feature importance via exact tree contributions (no extra dependency)
    rng = np.random.default_rng(a.seed)
    sub = rng.choice(len(te), size=min(2000, len(te)), replace=False)
    booster = model.get_booster()
    dm = xgb.DMatrix(X.iloc[te].iloc[sub], enable_categorical=True)
    contrib = booster.predict(dm, pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=feats).sort_values(ascending=False)
    imp.rename("mean_abs_contribution").to_csv(out / "feature_importance.csv")

    # Clinical-benchmark sanity check
    rows = []
    for drug, gene in BENCHMARKS:
        col = "GE_" + gene
        d_all = df[df.DRUG_NAME.str.lower() == drug.lower()]
        if d_all.empty or col not in df.columns:
            continue
        rho = spearmanr(d_all[col], d_all.LN_IC50, nan_policy="omit")[0]
        d_te = X.iloc[te][(test_df.DRUG_NAME.str.lower() == drug.lower()).values]
        rank = np.nan
        if len(d_te) >= 5:
            c = booster.predict(xgb.DMatrix(d_te.iloc[:500], enable_categorical=True), pred_contribs=True)[:, :-1]
            r = pd.Series(np.abs(c).mean(0), index=feats)[genes].rank(ascending=False)
            rank = int(r[col])
        rows.append({"drug": drug, "gene": gene, "n_lines": len(d_all),
                     "spearman_expr_vs_LN_IC50": round(float(rho), 3),
                     "model_importance_rank": rank, "of_genes": len(genes)})
    bench = pd.DataFrame(rows)
    bench.to_csv(out / "biomarker_sanity_check.csv", index=False)
    if not bench.empty:
        print("\n=== Biomarker sanity check (negative rho = high expression -> more sensitive) ===")
        print(bench.to_string(index=False))

    # Save artifacts
    model.save_model(out / "xgb_model.json")
    test_df.assign(pred_LN_IC50=pred)[["COSMIC_ID", "CELL_LINE_NAME", "TISSUE", "DRUG_NAME", "LN_IC50", "pred_LN_IC50"]] \
        .to_csv(out / "test_predictions.csv", index=False)
    summary.to_csv(out / "metrics.csv")
    (out / "run_config.json").write_text(json.dumps({**vars(a), "best_iteration": int(model.best_iteration)}, indent=2))

    # Figures
    fig, ax = plt.subplots(figsize=(5.5, 5))
    hb = ax.hexbin(y[te], pred, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    lim = [min(y[te].min(), pred.min()), max(y[te].max(), pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(xlabel="Actual LN_IC50", ylabel="Predicted LN_IC50",
           title=f"Test set (MSE {results['XGBoost']['MSE']:.2f})")
    fig.colorbar(hb, label="log10(count)")
    fig.tight_layout(); fig.savefig(out / "pred_vs_actual.png", dpi=200); plt.close(fig)

    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(top.index, top.values, color="#2a7ab0")
    ax.set(xlabel="Mean |contribution| to predicted LN_IC50", title="Top 20 features")
    fig.tight_layout(); fig.savefig(out / "top_features.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(tissue.index, tissue.MSE, color="#d98e3a")
    ax.axvline(results["XGBoost"]["MSE"], color="k", ls="--", lw=1, label="overall")
    ax.set(xlabel="Test MSE", title="Error by tissue"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "mse_by_tissue.png", dpi=200); plt.close(fig)

    demo_path = build_demo(model, df, feats, results, out, results["XGBoost"]["RMSE"])
    print(f"\nSaved results to {out.resolve()}")
    print(f"Demo page: {demo_path.resolve()}  (double-click to open in your browser)")


if __name__ == "__main__":
    main()
