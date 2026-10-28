"""Generate SYNTHETIC GDSC-style files so you can test the pipeline before downloading real data.

Uses the real cell line IDs/tissues from Cell_Lines_Details.xlsx, but the expression values
and IC50s are simulated. Never report results from this data as real findings.

    python src/make_synthetic.py
    python src/train.py --dose data/synthetic/dose_response.csv --expr data/synthetic/basalExp.txt --quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import load_cell_lines  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", default="data/raw/Cell_Lines_Details.xlsx")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n_genes", type=int, default=1500)
    ap.add_argument("--n_drugs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cl = load_cell_lines(a.details)
    cl = cl[cl.has_expression & cl.has_drug_response].reset_index(drop=True)
    n = len(cl)

    genes = ["ERBB2", "EGFR", "ESR1", "MDM2"] + [f"GENE{i:04d}" for i in range(a.n_genes - 4)]
    tissues = cl.TISSUE.unique()
    shift = {t: rng.normal(0, 0.6, len(genes)) * (rng.random(len(genes)) < 0.05) for t in tissues}
    expr = 6 + rng.normal(0, 1, (n, len(genes))) + np.vstack([shift[t] for t in cl.TISSUE])

    expr_df = pd.DataFrame(expr.T, columns=[f"DATA.{c}" for c in cl.COSMIC_ID])
    expr_df.insert(0, "GENE_title", "synthetic")
    expr_df.insert(0, "GENE_SYMBOLS", genes)
    expr_df.to_csv(out / "basalExp.txt", sep="\t", index=False)

    gidx = {g: i for i, g in enumerate(genes)}
    names = ["Lapatinib", "Erlotinib", "Gefitinib"] + [f"Drug{i}" for i in range(a.n_drugs - 3)]
    pathways = ["EGFR signaling", "PI3K/MTOR signaling", "Apoptosis regulation", "Cell cycle", "Other"]
    tissue_eff = {t: rng.normal(0, 0.8) for t in tissues}
    rows = []
    for k, name in enumerate(names):
        if name == "Lapatinib":
            w = {"ERBB2": -1.0}
        elif name in ("Erlotinib", "Gefitinib"):
            w = {"EGFR": -0.8}
        else:
            w = {genes[j]: rng.normal(0, 0.7) for j in rng.choice(len(genes), 3, replace=False)}
        base = rng.normal(2, 1.5)
        y = base + sum(c * (expr[:, gidx[g]] - 6) for g, c in w.items())
        y = y + np.array([tissue_eff[t] for t in cl.TISSUE]) + rng.normal(0, 0.8, n)
        keep = rng.random(n) > 0.10
        pw = pathways[k % len(pathways)] if k >= 3 else pathways[0]
        for i in np.where(keep)[0]:
            rows.append((cl.COSMIC_ID[i], 1000 + k, name, "synthetic", pw, y[i]))
    pd.DataFrame(rows, columns=["COSMIC_ID", "DRUG_ID", "DRUG_NAME", "PUTATIVE_TARGET",
                                "PATHWAY_NAME", "LN_IC50"]).to_csv(out / "dose_response.csv", index=False)
    print(f"Wrote synthetic data for {n} cell lines, {len(names)} drugs, {len(genes)} genes to {out}/")


if __name__ == "__main__":
    main()
