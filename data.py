"""Data loading and feature building for the BioHack drug-sensitivity model.

Three GDSC inputs are joined on COSMIC_ID:
  1. Cell_Lines_Details.xlsx   -> cell line metadata (tissue, MSI, growth) + data-availability flags
  2. GDSC dose-response file   -> target: LN_IC50 per (cell line, drug), plus drug -> pathway/target map
  3. GDSC basal expression     -> gene-expression features per cell line

Each row of the final table is one (cell line, drug) pair.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CAT_COLS = ["TISSUE", "TCGA", "MSI", "GROWTH", "MEDIUM", "DRUG", "PATHWAY", "TARGET"]
NUM_COLS = ["MAX_CONC"]  # highest tested concentration of the drug (a drug property)

# Genes tied to well-known drug biomarkers; always kept in the feature set so the
# clinical-benchmark check in train.py can look at them (see BENCHMARKS there).
BENCHMARK_GENES = ["ERBB2", "EGFR", "ESR1", "MDM2"]


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def load_cell_lines(path) -> pd.DataFrame:
    """Read the 'Cell line details' sheet and return clean per-cell-line metadata."""
    df = _clean_cols(pd.read_excel(path, sheet_name="Cell line details"))
    df = df.rename(
        columns={
            "COSMIC identifier": "COSMIC_ID",
            "Sample Name": "CELL_LINE_NAME",
            "GDSC Tissue descriptor 1": "TISSUE",
            "Cancer Type (matching TCGA label)": "TCGA",
            "Microsatellite instability Status (MSI)": "MSI",
            "Growth Properties": "GROWTH",
            "Screen Medium": "MEDIUM",
        }
    )
    df = df.dropna(subset=["COSMIC_ID"]).copy()  # drops the 'TOTAL:' summary row
    df["COSMIC_ID"] = df["COSMIC_ID"].astype(int)
    df["has_expression"] = df["Gene Expression"].astype(str).str.upper().eq("Y")
    df["has_drug_response"] = df["Drug Response"].astype(str).str.upper().eq("Y")
    for c in ["TISSUE", "TCGA", "MSI", "GROWTH", "MEDIUM"]:
        df[c] = df[c].fillna("unknown").astype(str)
    return df[
        ["COSMIC_ID", "CELL_LINE_NAME", "TISSUE", "TCGA", "MSI", "GROWTH", "MEDIUM",
         "has_expression", "has_drug_response"]
    ]


def load_dose_response(path) -> pd.DataFrame:
    """Read a GDSC fitted dose-response table (xlsx or csv). Target column: LN_IC50."""
    p = Path(path)
    raw = pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)
    df = _clean_cols(raw)
    need = {"COSMIC_ID", "DRUG_ID", "DRUG_NAME", "LN_IC50"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{p.name} is missing required columns: {sorted(missing)}")
    if "PATHWAY_NAME" not in df.columns:
        df["PATHWAY_NAME"] = "unknown"
    if "PUTATIVE_TARGET" not in df.columns:
        df["PUTATIVE_TARGET"] = "unknown"
    if "MAX_CONC" not in df.columns:
        df["MAX_CONC"] = np.nan
    df = df.dropna(subset=["LN_IC50"])
    # Some (cell line, drug) pairs are screened more than once -> average them.
    df = df.groupby(["COSMIC_ID", "DRUG_ID"], as_index=False).agg(
        DRUG_NAME=("DRUG_NAME", "first"),
        PUTATIVE_TARGET=("PUTATIVE_TARGET", "first"),
        PATHWAY=("PATHWAY_NAME", "first"),
        LN_IC50=("LN_IC50", "mean"),
        MAX_CONC=("MAX_CONC", "first"),
    )
    df["COSMIC_ID"] = df["COSMIC_ID"].astype(int)
    df["PATHWAY"] = df["PATHWAY"].fillna("unknown").astype(str)
    df["PATHWAY"] = df["PATHWAY"].fillna("unknown").astype(str)
    df["TARGET"] = df["PUTATIVE_TARGET"].fillna("unknown").astype(str)
    df["DRUG"] = df["DRUG_ID"].astype(str)
    return df


def _sanitize(name: str) -> str:
    return "GE_" + re.sub(r"[^0-9A-Za-z_]", "_", str(name))


def load_expression(path, max_genes: int = 500, keep_genes=BENCHMARK_GENES) -> pd.DataFrame:
    """Return a (COSMIC_ID x gene) float32 matrix with the top-variance genes.

    Accepts the GDSC 'Cell_line_RMA_proc_basalExp.txt' layout (GENE_SYMBOLS column,
    one DATA.<COSMIC_ID> column per cell line) or a wide table with a COSMIC_ID column.
    Gene selection uses variance across cell lines only (no labels), so it does not leak
    information from the test set about drug response.
    """
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".txt", ".tsv"} else ","
    raw = pd.read_csv(p, sep=sep)
    data_cols = [c for c in raw.columns if str(c).startswith("DATA.")]

    if data_cols and "GENE_SYMBOLS" in raw.columns:
        mat = raw.dropna(subset=["GENE_SYMBOLS"])[["GENE_SYMBOLS"] + data_cols].copy()
        mat.columns = ["GENE"] + [int(str(c).split(".")[1]) for c in data_cols]
        mat = mat.set_index("GENE")
        order = mat.var(axis=1).sort_values(ascending=False).index
        mat = mat.loc[order]
        mat = mat[~mat.index.duplicated(keep="first")]
        expr = mat.T
    elif "COSMIC_ID" in raw.columns:
        expr = raw.set_index("COSMIC_ID")
    else:
        raise ValueError("Unrecognised expression layout (need GDSC format or a COSMIC_ID column).")

    expr.index = expr.index.astype(int)
    expr.index.name = "COSMIC_ID"
    expr = expr.apply(pd.to_numeric, errors="coerce")

    top = list(expr.var().nlargest(max_genes).index)
    extra = [g for g in (keep_genes or []) if g in expr.columns and g not in top]
    expr = expr[top + extra].astype("float32")

    names = [_sanitize(g) for g in expr.columns]
    if len(set(names)) != len(names):  # extremely rare after sanitising
        names = [f"{n}_{i}" for i, n in enumerate(names)]
    expr.columns = names
    return expr


def build_dataset(details_path, dose_path, expr_path=None, max_genes=500, drugs=None, max_drugs=None):
    """Join everything. Returns (df, feature_cols, gene_cols).

    expr_path is optional: the Kaggle GDSC2-dataset.csv has no expression values, so without it the
    features are cell line metadata + drug information only (gene_cols is then empty).
    """
    cells = load_cell_lines(details_path)
    dose = load_dose_response(dose_path)

    if drugs:
        wanted = {d.lower() for d in drugs}
        dose = dose[dose.DRUG_NAME.str.lower().isin(wanted)]
    if max_drugs:
        keep = dose.DRUG_NAME.value_counts().head(max_drugs).index
        dose = dose[dose.DRUG_NAME.isin(keep)]

    cell_tbl = cells.drop(columns=["has_expression", "has_drug_response"])
    gene_cols = []
    if expr_path:
        expr = load_expression(expr_path, max_genes=max_genes)
        cell_tbl = cell_tbl[cells.has_expression.values].merge(expr.reset_index(), on="COSMIC_ID", how="inner")
        gene_cols = list(expr.columns)

    df = dose.merge(cell_tbl, on="COSMIC_ID", how="inner").reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows after joining. Check that COSMIC IDs match across the files.")

    for c in CAT_COLS:
        df[c] = df[c].astype("category")  # categories fixed on the full table -> train/test consistent
    feature_cols = gene_cols + NUM_COLS + CAT_COLS
    return df, feature_cols, gene_cols
