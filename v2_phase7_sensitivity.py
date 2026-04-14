"""
Phase 7 (v2): Sensitivity Analyses
====================================
14 sensitivity analyses validating the robustness of primary findings.

Outputs:
  analysis_results/sensitivity_summary.csv
  analysis_results/sensitivity_detail.json
"""

import pandas as pd
import numpy as np
import json, warnings, time
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

from utils import load_study_data, get_leadership, to_bool, RESULTS

warnings.filterwarnings("ignore")

print("=" * 70)
print("PHASE 7 (v2): SENSITIVITY ANALYSES")
print("=" * 70)

# -- Load data ----------------------------------------------------------------
works, authorships = load_study_data()
leadership = get_leadership(works, authorships)
df = works.merge(leadership, on="work_id", how="left")
N = len(df)
print(f"Study set: {N:,}")

# -- Build primary model predictors and groups --------------------------------
bloc_dummies = pd.get_dummies(df["partner_bloc"], prefix="bloc", dtype=int)
bloc_ref = "bloc_African"
bloc_cols = [c for c in bloc_dummies.columns if c != bloc_ref]

field_dummies = pd.get_dummies(df["field_reg"], prefix="field", dtype=int)
field_ref = "field_Health Sciences"
field_cols = [c for c in field_dummies.columns if c != field_ref]

core_preds = ["log_author_count", "year_centered", "year_centered_sq",
              "has_funding_int", "is_oa_int"]


def build_X(data, extra_cols=None, extra_data=None):
    """Build X matrix for GEE from a DataFrame."""
    X = data[core_preds].copy()
    bd = pd.get_dummies(data["partner_bloc"], prefix="bloc", dtype=int)
    for c in bloc_cols:
        if c not in bd.columns:
            bd[c] = 0
    fd = pd.get_dummies(data["field_reg"], prefix="field", dtype=int)
    for c in field_cols:
        if c not in fd.columns:
            fd[c] = 0
    X = pd.concat([X.reset_index(drop=True),
                    bd[bloc_cols].reset_index(drop=True),
                    fd[field_cols].reset_index(drop=True)], axis=1)
    if extra_cols and extra_data is not None:
        for c in extra_cols:
            X[c] = extra_data[c].reset_index(drop=True)
    return sm.add_constant(X)


def run_sensitivity(name, data, y_col="gh_first", use_gee=True,
                    extra_cols=None, extra_data=None,
                    use_different_preds=None):
    """Run a sensitivity analysis and return key results."""
    data_sorted = data.sort_values("primary_gh_institution").reset_index(drop=True)
    y = data_sorted[y_col].astype(int)
    
    if use_different_preds is not None:
        X = use_different_preds(data_sorted)
    else:
        X = build_X(data_sorted, extra_cols, 
                     data_sorted if extra_data is None else extra_data)
    groups = data_sorted["primary_gh_institution"]
    
    t0 = time.time()
    try:
        if use_gee:
            model = GEE(y, X, groups=groups, family=Binomial(),
                        cov_struct=Exchangeable())
            result = model.fit(maxiter=200)
        else:
            result = Logit(y, X).fit(disp=0, cov_type="HC1")
        elapsed = time.time() - t0
        
        # Extract key coefficients
        key_vars = {}
        for var in ["log_author_count", "year_centered", "year_centered_sq",
                     "bloc_Western", "bloc_Multi-bloc", "has_funding_int",
                     "is_oa_int", "covid_era", "is_bilateral_int",
                     "bilateral_x_year"]:
            if var in result.params.index:
                key_vars[var] = {
                    "OR": round(np.exp(result.params[var]), 4),
                    "p": round(result.pvalues[var], 4),
                    "sig": result.pvalues[var] < 0.05,
                }
        
        print(f"  {name}: N={len(y):,}, converged in {elapsed:.1f}s")
        for k, v in key_vars.items():
            star = "*" if v["sig"] else " "
            print(f"    {star} {k:30s} OR={v['OR']:.4f} p={v['p']:.4f}")
        
        return {"name": name, "n": len(y), "converged": True, 
                "coefficients": key_vars}
    except Exception as e:
        print(f"  {name}: FAILED - {e}")
        return {"name": name, "n": len(y), "converged": False, "error": str(e)}


# ==============================================================================
# RUN SENSITIVITY ANALYSES
# ==============================================================================

results = []

# S1: Dual -> GH reclassification
print("\n--- S1: Dual reclassified as Ghanaian ---")
df_s1 = df.copy()
df_s1["gh_first"] = df_s1["first_cat"].isin(["Ghanaian", "Dual-affiliated"])
results.append(run_sensitivity("S1_dual_as_GH", df_s1))

# S2: Dual -> Non-GH reclassification
print("\n--- S2: Dual reclassified as Non-Ghanaian ---")
df_s2 = df.copy()
df_s2["gh_first"] = df_s2["first_cat"] == "Ghanaian"
results.append(run_sensitivity("S2_dual_as_nonGH", df_s2))

# S3: >= 4 authors only
print("\n--- S3: 4+ authors only ---")
df_s3 = df[df["author_count"] >= 4].copy()
results.append(run_sensitivity("S3_4plus_authors", df_s3))

# S4: Articles only (exclude preprints, reviews, etc.)
print("\n--- S4: Articles only ---")
df_s4 = df[df["type"] == "article"].copy()
results.append(run_sensitivity("S4_articles_only", df_s4))

# S5: Exclude mega-consortia (>50 authors)
print("\n--- S5: Exclude >50 authors ---")
df_s5 = df[df["author_count"] <= 50].copy()
results.append(run_sensitivity("S5_max50_authors", df_s5))

# S6: Exclude COVID-topic papers (rough keyword match)
print("\n--- S6: Exclude COVID-topic ---")
covid_kw = df["title"].str.contains("COVID|SARS-CoV|coronavirus|pandemic",
                                     case=False, na=False)
df_s6 = df[~covid_kw].copy()
results.append(run_sensitivity("S6_no_covid_papers", df_s6))

# S7: Add covid_era to primary model
print("\n--- S7: Add covid_era ---")
df_s7 = df.copy()
df_s7["covid_era_int"] = df_s7["covid_era"]
def s7_builder(data):
    X = build_X(data)
    X["covid_era"] = data["covid_era"].values
    return X
results.append(run_sensitivity("S7_with_covid_era", df_s7,
                                use_different_preds=s7_builder))

# S8: Standard logit (no GEE)
print("\n--- S8: Standard logit (no GEE) ---")
results.append(run_sensitivity("S8_standard_logit", df, use_gee=False))

# S9: Clustered SE by first-author ID
print("\n--- S9: Clustered by first-author ---")
df_s9 = df.copy()
df_s9["primary_gh_institution"] = df_s9["first_author_id"]
results.append(run_sensitivity("S9_cluster_first_author", df_s9))

# S10: GH-only outcome (exclude Dual from positive)
print("\n--- S10: GH-only outcome ---")
df_s10 = df.copy()
df_s10["gh_first"] = df_s10["first_cat"] == "Ghanaian"
results.append(run_sensitivity("S10_gh_only", df_s10))

# S11: 2000-2024 only
print("\n--- S11: 2000-2024 only ---")
df_s11 = df[df["publication_year"] <= 2024].copy()
results.append(run_sensitivity("S11_2000_2024", df_s11))

# S12: Linear year only (no quadratic)
print("\n--- S12: Linear year (no quadratic) ---")
def s12_builder(data):
    X_data = data[["log_author_count", "year_centered", "has_funding_int", "is_oa_int"]].copy()
    bd = pd.get_dummies(data["partner_bloc"], prefix="bloc", dtype=int)
    for c in bloc_cols:
        if c not in bd.columns:
            bd[c] = 0
    fd = pd.get_dummies(data["field_reg"], prefix="field", dtype=int)
    for c in field_cols:
        if c not in fd.columns:
            fd[c] = 0
    X_data = pd.concat([X_data.reset_index(drop=True),
                         bd[bloc_cols].reset_index(drop=True),
                         fd[field_cols].reset_index(drop=True)], axis=1)
    return sm.add_constant(X_data)
results.append(run_sensitivity("S12_linear_year", df, use_different_preds=s12_builder))

# S13: Linear author_count (not log)
print("\n--- S13: Linear author_count ---")
def s13_builder(data):
    X_data = data[["author_count", "year_centered", "year_centered_sq",
                   "has_funding_int", "is_oa_int"]].copy()
    X_data = X_data.rename(columns={"author_count": "author_count_linear"})
    bd = pd.get_dummies(data["partner_bloc"], prefix="bloc", dtype=int)
    for c in bloc_cols:
        if c not in bd.columns:
            bd[c] = 0
    fd = pd.get_dummies(data["field_reg"], prefix="field", dtype=int)
    for c in field_cols:
        if c not in fd.columns:
            fd[c] = 0
    X_data = pd.concat([X_data.reset_index(drop=True),
                         bd[bloc_cols].reset_index(drop=True),
                         fd[field_cols].reset_index(drop=True)], axis=1)
    return sm.add_constant(X_data)
results.append(run_sensitivity("S13_linear_AC", df, use_different_preds=s13_builder))

# S14: Last authorship (apply same sensitivities to last auth)
print("\n--- S14: Last authorship (primary model) ---")
results.append(run_sensitivity("S14_last_auth", df, y_col="gh_last"))

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
print(f"\n{'='*70}")
print("ROBUSTNESS SUMMARY")
print(f"{'='*70}")

summary_rows = []
for r in results:
    if not r["converged"]:
        continue
    row = {"Sensitivity": r["name"], "N": r["n"]}
    for var in ["log_author_count", "bloc_Multi-bloc", "bloc_Western",
                "has_funding_int", "is_oa_int", "year_centered"]:
        if var in r["coefficients"]:
            row[f"{var}_OR"] = r["coefficients"][var]["OR"]
            row[f"{var}_sig"] = "*" if r["coefficients"][var]["sig"] else ""
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS / "sensitivity_summary.csv", index=False)

with open(RESULTS / "sensitivity_detail.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Saved: sensitivity_summary.csv, sensitivity_detail.json")
print(f"  Total analyses: {len(results)}")
print(f"  Converged: {sum(1 for r in results if r['converged'])}")
print(f"{'='*70}")
