"""Build a self-contained demo page (outputs/demo.html) from the trained model.

Needs nothing beyond the packages already installed. The page has no server and no internet
dependency: double-click it (or right-click > Open with > browser) and it works.
Predictions for every drug x tissue x MSI x growth combination are computed here by the real
XGBoost model and embedded in the page, so the page itself only looks them up.
"""
from __future__ import annotations

import base64
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

QS = np.arange(0, 101, 5)
MSIS = ["MSS/MSI-L", "MSI-H"]
GROWTHS = ["Adherent", "Suspension"]


def _b64(path):
    p = Path(path)
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def build_demo(model, df, feats, results, out_dir, rmse):
    out_dir = Path(out_dir)
    cat = lambda s: s.astype(str)
    mode = lambda s: cat(s).mode().iat[0]
    cell = df.drop_duplicates("COSMIC_ID")
    tissues = sorted(cat(df.TISSUE).unique())
    genes = [c for c in feats if c.startswith("GE_")]

    prof = {}
    for t in tissues:
        sub = cell[cat(cell.TISSUE) == t]
        prof[t] = {"TCGA": mode(sub.TCGA), "MEDIUM": mode(sub.MEDIUM)}
        for g in genes:
            prof[t][g] = float(sub[g].mean())

    drugs = df.drop_duplicates("DRUG_ID").sort_values("DRUG_NAME").reset_index(drop=True)
    q = df.groupby("DRUG_ID").LN_IC50.quantile(QS / 100).unstack()

    rows = []
    for di, t, m, g in itertools.product(range(len(drugs)), tissues, MSIS, GROWTHS):
        d = drugs.loc[di]
        r = {"TISSUE": t, "MSI": m, "GROWTH": g, "DRUG": str(d.DRUG), "PATHWAY": str(d.PATHWAY),
             "TARGET": str(d.TARGET), "MAX_CONC": d.MAX_CONC, **prof[t]}
        rows.append(r)
    grid = pd.DataFrame(rows)
    for c in ["TISSUE", "TCGA", "MSI", "GROWTH", "MEDIUM", "DRUG", "PATHWAY", "TARGET"]:
        grid[c] = pd.Categorical(grid[c].astype(str), categories=df[c].cat.categories)
    pred = np.asarray(model.predict(grid[feats]), dtype=float).reshape(len(drugs), -1)

    dup = drugs.DRUG_NAME.duplicated(keep=False)
    payload = {
        "rmse": float(rmse),
        "tissues": tissues, "msis": MSIS, "growths": GROWTHS,
        "drugs": [
            {"n": f"{r.DRUG_NAME} ({r.DRUG_ID})" if dup[i] else str(r.DRUG_NAME),
             "t": str(r.TARGET), "p": str(r.PATHWAY),
             "q": [round(float(x), 3) for x in q.loc[r.DRUG_ID].values],
             "v": [round(float(x), 3) for x in pred[i]]}
            for i, r in drugs.iterrows()
        ],
        "results": {k: {m: (None if (isinstance(x, float) and np.isnan(x)) else round(float(x), 4))
                        for m, x in v.items()} for k, v in results.items()},
        "stats": {"pairs": int(len(df)), "lines": int(df.COSMIC_ID.nunique()), "drugs": int(df.DRUG_ID.nunique())},
        "imgs": {k: _b64(out_dir / f"{k}.png") for k in ["pred_vs_actual", "top_features", "mse_by_tissue"]},
    }
    html = TEMPLATE.replace("/*DATA*/", json.dumps(payload))
    path = out_dir / "demo.html"
    path.write_text(html, encoding="utf-8")
    return path


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>BioHack: Drug Sensitivity Predictor</title>
<style>
:root{--bg:#f4f7fb;--card:#fff;--fg:#1c2430;--mut:#5b6675;--line:#d9e0e8;--acc:#1f6fb2;--ok:#1d7a4d;--bad:#b3341f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif}
header{background:#12304d;color:#fff;padding:18px 24px}header h1{margin:0;font-size:1.4rem}header p{margin:4px 0 0;opacity:.8}
nav{display:flex;gap:6px;padding:10px 24px;background:#fff;border-bottom:1px solid var(--line);flex-wrap:wrap}
nav button{border:0;background:none;padding:8px 14px;font:inherit;border-radius:8px;cursor:pointer;color:var(--mut)}
nav button.on{background:var(--acc);color:#fff}main{max-width:1000px;margin:0 auto;padding:20px}
.tab{display:none}.tab.on{display:block}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}label{display:block;font-size:.8rem;color:var(--mut);margin-bottom:3px}
select{width:100%;padding:8px;font:inherit;border:1px solid var(--line);border-radius:8px;background:#fff}
.big{font-size:2.2rem;font-weight:700}.pill{display:inline-block;padding:4px 12px;border-radius:99px;color:#fff;font-weight:600}
.bar{position:relative;height:18px;border-radius:9px;background:linear-gradient(90deg,#2e9b63,#e8d44d,#cc4b37);margin:14px 0 4px}
.dot{position:absolute;top:-5px;width:6px;height:28px;background:#12304d;border-radius:3px;transform:translateX(-3px)}
.row{display:flex;justify-content:space-between;font-size:.8rem;color:var(--mut)}table{border-collapse:collapse;width:100%}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;font-size:.9rem}th{background:#eef3f8}.sc{overflow-x:auto}
img{max-width:100%;border:1px solid var(--line);border-radius:8px}.warn{font-size:.82rem;color:var(--mut)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.stat{background:#eef3f8;border-radius:10px;padding:10px}
.stat b{display:block;font-size:1.5rem;color:var(--acc)}
</style></head><body>
<header><h1>Cancer Drug Sensitivity Predictor</h1><p>XGBoost trained on GDSC2 cell line data. Research prototype, not clinical advice.</p></header>
<nav id="nav"></nav><main>
<section class="tab on" id="t0">
<div class="card"><div class="grid">
<div><label>Drug</label><select id="drug"></select></div><div><label>Tissue / cancer site</label><select id="tis"></select></div>
<div><label>MSI status</label><select id="msi"></select></div><div><label>Growth type</label><select id="gro"></select></div></div></div>
<div class="card" id="out"></div>
<div class="card"><h3 style="margin-top:0">Which drugs look most promising for this cell line profile?</h3>
<p class="warn">Ranked by predicted response relative to each drug's own typical cell line, so potent and weak drugs are compared fairly.</p>
<div class="sc"><table id="rank"></table></div></div></section>
<section class="tab" id="t1"><div class="card"><h3 style="margin-top:0">Held-out test results (cell lines never seen in training)</h3>
<div class="sc"><table id="res"></table></div><p class="warn" id="resnote"></p></div></section>
<section class="tab" id="t2"><div id="imgs"></div></section>
<section class="tab" id="t3"><div class="card"><h3 style="margin-top:0">About this prototype</h3><div class="stats" id="stats"></div>
<ul><li>Target: LN_IC50, the natural log of the drug concentration that inhibits 50% of cell growth. Lower means more sensitive.</li>
<li>Inputs: tissue, MSI status, growth type, screen medium and cancer type (typical for the chosen tissue), plus drug, target, pathway and max tested concentration.</li>
<li>80/20 train-test split, grouped by cell line. Metric: Mean Squared Error.</li>
<li>Limitation: the data has no per-cell-line gene expression, so predictions describe a tissue-level profile, not an individual line.</li>
<li>Cell lines are laboratory models. These outputs are research estimates, not treatment advice.</li></ul></div></section></main>
<script>
const D=/*DATA*/;
const $=id=>document.getElementById(id);
["Predict","Results","Charts","About"].forEach((n,i)=>{const b=document.createElement("button");b.textContent=n;if(!i)b.className="on";
b.onclick=()=>{[...$("nav").children].forEach((x,j)=>{x.classList.toggle("on",i==j);$("t"+j).classList.toggle("on",i==j)})};$("nav").appendChild(b)});
const fill=(id,arr)=>{$(id).innerHTML=arr.map((x,i)=>`<option value="${i}">${x}</option>`).join("")};
fill("drug",D.drugs.map(d=>d.n));fill("tis",D.tissues);fill("msi",D.msis);fill("gro",D.growths);
const idx=()=>(+$("tis").value*D.msis.length+ +$("msi").value)*D.growths.length+ +$("gro").value;
function pct(d,x){const q=d.q;if(x<=q[0])return 0;if(x>=q[q.length-1])return 100;
for(let i=1;i<q.length;i++)if(x<=q[i]){const f=q[i]==q[i-1]?0:(x-q[i-1])/(q[i]-q[i-1]);return (i-1+f)*5}return 100}
function update(){const d=D.drugs[+$("drug").value],k=idx(),x=d.v[k],p=pct(d,x),r=D.rmse;
const lab=p<33?["More sensitive than typical","#1d7a4d"]:p>67?["More resistant than typical","#b3341f"]:["Typical sensitivity","#8a7a14"];
$("out").innerHTML=`<div class="grid"><div><label>Predicted LN_IC50</label><div class="big">${x.toFixed(2)}</div>
<div class="warn">IC50 about ${Math.exp(x).toPrecision(2)} &micro;M (likely range ${Math.exp(x-r).toPrecision(2)} to ${Math.exp(x+r).toPrecision(2)} &micro;M, using the model's typical error)</div></div>
<div><label>Verdict</label><span class="pill" style="background:${lab[1]}">${lab[0]}</span>
<div class="warn" style="margin-top:8px">Target: ${d.t} &middot; Pathway: ${d.p}</div></div></div>
<div class="bar"><div class="dot" style="left:${p}%"></div></div><div class="row"><span>More sensitive</span><span>${Math.round(p)}th percentile among this drug's cell lines</span><span>More resistant</span></div>`;
const rk=D.drugs.map(e=>({e,p:pct(e,e.v[k]),x:e.v[k]})).sort((a,b)=>a.p-b.p).slice(0,10);
$("rank").innerHTML="<tr><th>#</th><th>Drug</th><th>Target</th><th>Pathway</th><th>Predicted LN_IC50</th><th>Percentile (lower = more sensitive)</th></tr>"+
rk.map((o,i)=>`<tr><td>${i+1}</td><td>${o.e.n}</td><td>${o.e.t}</td><td>${o.e.p}</td><td>${o.x.toFixed(2)}</td><td>${Math.round(o.p)}</td></tr>`).join("")}
["drug","tis","msi","gro"].forEach(i=>$(i).onchange=update);update();
const R=D.results,cols=Object.keys(R.XGBoost),f=v=>v==null?"n/a":v.toFixed(3);
$("res").innerHTML="<tr><th>Model</th>"+cols.map(c=>`<th>${c.replace(/_/g," ")}</th>`).join("")+"</tr>"+
Object.keys(R).map(m=>`<tr><td><b>${m}</b></td>${cols.map(c=>`<td>${f(R[m][c])}</td>`).join("")}</tr>`).join("");
const b=R["Baseline: per-drug mean"];$("resnote").textContent=`XGBoost cuts MSE by ${(100*(1-R.XGBoost.MSE/b.MSE)).toFixed(1)}% versus the per-drug-mean baseline.`;
$("imgs").innerHTML=Object.entries({pred_vs_actual:"Predicted vs actual",top_features:"Top features",mse_by_tissue:"Error by tissue"}).map(([k,t])=>D.imgs[k]?`<div class="card"><h3 style="margin-top:0">${t}</h3><img src="${D.imgs[k]}" alt="${t}"></div>`:"").join("");
$("stats").innerHTML=[[D.stats.pairs.toLocaleString(),"cell line / drug pairs"],[D.stats.lines,"cell lines"],[D.stats.drugs,"drugs"]].map(s=>`<div class="stat"><b>${s[0]}</b>${s[1]}</div>`).join("");
</script></body></html>
"""
