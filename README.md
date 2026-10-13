# BioHack: Predicting Cancer Drug Sensitivity from Gene Expression (XGBoost model)

**Question:** can gene expression predict how sensitive a cancer cell line is to an anticancer drug?
**Target:** `LN_IC50` (log of the drug concentration that inhibits 50% of cell growth; lower = more sensitive).
**Model:** one pan-drug XGBoost regressor. Each row is a (cell line, drug) pair.
**Split / metric:** 80-20 train-test, evaluated with Mean Squared Error.

## Data (Kaggle GDSC dataset)

Both files go in `data/raw/`:

| File | What it gives |
|---|---|
| `GDSC2-dataset.csv` (242,036 rows) | `LN_IC50` target, drug name, putative target, pathway, min/max tested concentration |
| `Cell_Lines_Details.xlsx` | cell line tissue, TCGA type, MSI status, growth properties, screen medium |

**Heads-up:** neither file contains actual gene expression values. The "Gene Expression" column in the details
file is just a Y/N flag saying whether expression data exists for that line. So the default model uses cell line
metadata + drug information. To add real expression features, get a basal expression matrix (for example from the
official GDSC site) and pass it with `--expr`; the loader in `src/data.py` supports it.

## Setup in VS Code

```bash
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/train.py            # full Kaggle data, honest split by cell line
python src/train.py --quick    # fast check
python src/train.py --max_drugs 50   # if your laptop is slow
```
Other flags: `--split random`, `--drugs Lapatinib Erlotinib`, `--expr path/to/expression.txt`, `--max_genes 1000`.

Results land in `outputs/`: `metrics.csv`, `pred_vs_actual.png`, `top_features.png`, `mse_by_tissue.png`,
`feature_importance.csv`, `test_predictions.csv`, `xgb_model.json`.

(`src/make_synthetic.py` makes fake data for pipeline tests only. Never report synthetic results.)

## Design choices (good answers for Q&A)

- **Features:** cell line tissue / TCGA type / MSI / growth / screen medium, drug identity, drug target and pathway, and max tested concentration. If `--expr` is given, top-variance genes are added (selected without labels, so no leakage).
- **Honest split:** the 80-20 split holds out **whole cell lines** by default. A random split puts the same cell
  line in both train and test (with different drugs), which inflates scores. `--split random` is available to compare.
- **No test peeking:** early stopping uses a 10% validation slice carved out of the training 80%.
- **Baselines:** XGBoost must beat the per-drug mean. The key extra metric is `within_drug_pearson_median`,
  which asks whether the model can rank cell lines for a given drug, which the per-drug baseline cannot do (0 by construction).
- **Sensitive vs resistant:** a line is "sensitive" if its LN_IC50 is below that drug's training-set median.
  The AUROC reported is for that classification, computed from the regression output.
- **Biomarker sanity check (only when `--expr` is used):** compares expression/IC50 correlation and model importance for literature-motivated pairs
  (Lapatinib-ERBB2, EGFR inhibitors-EGFR). This is a plausibility check, not a clinical validation.

## Limitations to state openly

- Cell lines are not patients; GDSC IC50s are in vitro. "Clinical benchmarks" here means agreement with known biomarker biology.
- No true expression features in the Kaggle files, a single data source, and no external validation cohort yet.
- Roadmap candidates: hyperparameter search, adding mutation/CNA features (the details sheet shows WES/CNA exist),
  external validation on CCLE/CTRP, per-drug models, and a small web demo.

## Rubric map

| Criterion | Where it is covered |
|---|---|
| Innovation & Impact | Pan-drug model with drug-pathway features; drug repurposing / patient stratification angle |
| Technical Execution | Leakage-aware split, baselines, early stopping on a separate slice, multiple metrics |
| Prototype & Demo | One command produces metrics + figures; use `pred_vs_actual.png` and `top_features.png` live |
| Pitch (3-4 min) | Problem -> data -> model -> result vs baseline -> biomarker check -> limits (about 30 s each) |
| Q&A / Roadmap | Limitations and roadmap sections above |
