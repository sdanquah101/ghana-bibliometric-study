"""
Phase 6 (v2): Primary Regression Analysis â€” GEE Logistic Regression
=====================================================================
Primary model: GEE with exchangeable correlation, clustered by
primary Ghanaian institution (109 clusters, median 25 papers each).

Changes from original phase6_regression.py:
  1. GEE replaces standard logistic regression
  2. log(author_count) replaces linear author_count
  3. year_centered + year_centered^2 (quadratic temporal trend)
  4. covid_era removed from primary model
  5. Average Marginal Effects (AMEs) computed
  6. ROC/AUC and calibration plots
  7. Honest VIF computation
  8. Formal interaction test for Simpson's paradox
  9. FDR correction on primary model p-values
  10. Only first and last authorship as primary outcomes

Outputs:
  analysis_results/v2_regression_results.csv
  analysis_results/v2_model_diagnostics.json
  analysis_results/v2_marginal_effects.csv
  analysis_results/v2_vif_results.csv
  Charts: v2_chart_forest_*.png, v2_chart_calibration_*.png, v2_chart_roc_*.png
"""

import pandas as pd
import numpy as np
import json, warnings, time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as sp_stats

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests

from utils import (
    setup_plot_style, save_chart, get_leadership, load_study_data,
    to_bool, wilson_ci, RESULTS
)

warnings.filterwarnings("ignore")
setup_plot_style()

print("=" * 70)
print("PHASE 6 (v2): PRIMARY REGRESSION ANALYSIS")
print("=" * 70)

# â”€â”€ Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
works, authorships = load_study_data()
leadership = get_leadership(works, authorships)
reg = works.merge(leadership, on="work_id", how="left")
print(f"Study set: {len(reg):,} works")

# â”€â”€ Prepare regression matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n1. Preparing regression matrix...")

# Partner bloc dummies (ref: African)
bloc_dummies = pd.get_dummies(reg["partner_bloc"], prefix="bloc", dtype=int)
bloc_ref = "bloc_African"
bloc_cols = [c for c in bloc_dummies.columns if c != bloc_ref]

# Field dummies (ref: Health Sciences)
field_dummies = pd.get_dummies(reg["field_reg"], prefix="field", dtype=int)
field_ref = "field_Health Sciences"
field_cols = [c for c in field_dummies.columns if c != field_ref]

# Core predictors (NO covid_era)
core_predictors = ["log_author_count", "year_centered", "year_centered_sq",
                   "has_funding_int", "is_oa_int"]
all_predictors = core_predictors + bloc_cols + field_cols

# Build X matrix
X_data = reg[core_predictors].copy()
X_data = pd.concat([X_data, bloc_dummies[bloc_cols], field_dummies[field_cols]], axis=1)
X_full = sm.add_constant(X_data)

# Sort by cluster for GEE
sort_idx = reg["primary_gh_institution"].argsort()
X_full = X_full.iloc[sort_idx].reset_index(drop=True)
reg_sorted = reg.iloc[sort_idx].reset_index(drop=True)
groups = reg_sorted["primary_gh_institution"]

print(f"   Predictors: {len(all_predictors)}")
print(f"   Clusters (GH institutions): {groups.nunique()}")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODEL FITTING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

all_results = []
diagnostics = {}

def fit_gee(y, X, groups, model_name):
    """Fit GEE logistic regression and return results dict."""
    t0 = time.time()
    gee = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable())
    result = gee.fit(maxiter=200)
    elapsed = time.time() - t0
    
    print(f"\n   {model_name}: converged in {elapsed:.1f}s")
    
    # Extract results
    rows = []
    for var in X.columns:
        if var == "const":
            continue
        rows.append({
            "Variable": var,
            "Coef": result.params[var],
            "OR": np.exp(result.params[var]),
            "CI_lo": np.exp(result.params[var] - 1.96 * result.bse[var]),
            "CI_hi": np.exp(result.params[var] + 1.96 * result.bse[var]),
            "SE": result.bse[var],
            "z": result.tvalues[var],
            "p_value": result.pvalues[var],
            "Model": model_name,
        })
    
    return result, pd.DataFrame(rows)


def fit_logit_for_diagnostics(y, X, model_name):
    """Fit standard logit for AUC/calibration (GEE doesn't provide predicted probs easily)."""
    result = Logit(y, X).fit(disp=0)
    y_pred = result.predict(X)
    auc = roc_auc_score(y, y_pred)
    
    # Brier score
    brier = np.mean((y - y_pred) ** 2)
    
    # Pseudo-R2
    pseudo_r2 = 1 - result.llf / result.llnull
    
    return result, y_pred, auc, brier, pseudo_r2


# â”€â”€ Primary outcomes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
outcomes = {
    "gh_first": ("First Authorship", reg_sorted["gh_first"].astype(int)),
    "gh_last": ("Last Authorship", reg_sorted["gh_last"].astype(int)),
    "gh_corr": ("Corresponding Authorship (supplementary)",
                reg_sorted["gh_corr"].astype(int)),
}

for outcome_key, (outcome_label, y) in outcomes.items():
    print(f"\n{'â”€'*60}")
    print(f"  Model A: {outcome_label}")
    print(f"{'â”€'*60}")
    
    # Fit GEE (primary)
    gee_result, gee_df = fit_gee(y, X_full, groups, f"{outcome_key}_A")
    all_results.append(gee_df)
    
    # Fit standard logit for diagnostics
    logit_result, y_pred, auc, brier, pseudo_r2 = fit_logit_for_diagnostics(
        y, X_full, f"{outcome_key}_A")
    
    n_positive = y.sum()
    n_total = len(y)
    print(f"   N = {n_total:,}, positive = {n_positive:,} ({n_positive/n_total*100:.1f}%)")
    print(f"   AUC = {auc:.3f}")
    print(f"   Brier score = {brier:.4f}")
    print(f"   Pseudo-R2 (McFadden) = {pseudo_r2:.4f}")
    print(f"   AIC = {logit_result.aic:.1f}")
    
    diagnostics[f"{outcome_key}_A"] = {
        "n": n_total,
        "n_positive": int(n_positive),
        "auc": round(auc, 4),
        "brier": round(brier, 4),
        "pseudo_r2": round(pseudo_r2, 4),
        "aic": round(logit_result.aic, 1),
    }
    
    # â”€â”€ ROC Curve â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fpr, tpr, _ = roc_curve(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: {outcome_label}")
    ax.legend(loc="lower right")
    save_chart(fig, f"v2_chart_roc_{outcome_key}")
    
    # â”€â”€ Calibration Plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, ax = plt.subplots(figsize=(6, 6))
    # Decile calibration
    df_cal = pd.DataFrame({"pred": y_pred, "actual": y})
    df_cal["decile"] = pd.qcut(df_cal["pred"], 10, duplicates="drop")
    cal_summary = df_cal.groupby("decile", observed=True).agg(
        mean_pred=("pred", "mean"),
        mean_actual=("actual", "mean"),
        n=("actual", "size")
    ).reset_index()
    
    ax.scatter(cal_summary["mean_pred"], cal_summary["mean_actual"],
               s=cal_summary["n"] / 20, color="#2563eb", alpha=0.7, zorder=5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Mean Observed Proportion")
    ax.set_title(f"Calibration Plot: {outcome_label}")
    ax.legend()
    save_chart(fig, f"v2_chart_calibration_{outcome_key}")

# â”€â”€ Model B (country_count replaces bloc) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'â”€'*60}")
print(f"  Models B (country_count) and C (both)")
print(f"{'â”€'*60}")

for outcome_key, (outcome_label, y) in list(outcomes.items())[:2]:  # first + last only
    # Model B
    X_b_data = reg_sorted[["log_author_count", "year_centered", "year_centered_sq",
                           "has_funding_int", "is_oa_int", "country_count"]].copy()
    X_b_data = pd.concat([X_b_data, field_dummies[field_cols].iloc[sort_idx].reset_index(drop=True)], axis=1)
    X_b = sm.add_constant(X_b_data)
    _, gee_b_df = fit_gee(y, X_b, groups, f"{outcome_key}_B")
    all_results.append(gee_b_df)
    
    # Model C (both bloc dummies + country_count)
    X_c_data = reg_sorted[["log_author_count", "year_centered", "year_centered_sq",
                           "has_funding_int", "is_oa_int", "country_count"]].copy()
    X_c_data = pd.concat([
        X_c_data,
        bloc_dummies[bloc_cols].iloc[sort_idx].reset_index(drop=True),
        field_dummies[field_cols].iloc[sort_idx].reset_index(drop=True)
    ], axis=1)
    X_c = sm.add_constant(X_c_data)
    _, gee_c_df = fit_gee(y, X_c, groups, f"{outcome_key}_C")
    all_results.append(gee_c_df)

# â”€â”€ Model D (add covid_era) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'â”€'*60}")
print(f"  Model D (primary + covid_era)")
print(f"{'â”€'*60}")

for outcome_key, (outcome_label, y) in list(outcomes.items())[:2]:
    X_d_data = reg_sorted[core_predictors + ["covid_era"]].copy()
    X_d_data = pd.concat([
        X_d_data,
        bloc_dummies[bloc_cols].iloc[sort_idx].reset_index(drop=True),
        field_dummies[field_cols].iloc[sort_idx].reset_index(drop=True)
    ], axis=1)
    X_d = sm.add_constant(X_d_data)
    _, gee_d_df = fit_gee(y, X_d, groups, f"{outcome_key}_D")
    all_results.append(gee_d_df)

# â”€â”€ Model E (interaction: bilateral x year) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'â”€'*60}")
print(f"  Model E (bilateral x year interaction)")
print(f"{'â”€'*60}")

for outcome_key, (outcome_label, y) in list(outcomes.items())[:2]:
    # Model E: replace bloc dummies with is_bilateral + interaction
    # (is_bilateral is a linear function of bloc dummies, so including both
    # creates near-perfect collinearity and singular Hessian)
    X_e_data = reg_sorted[["log_author_count", "year_centered", "year_centered_sq",
                           "has_funding_int", "is_oa_int"]].copy()
    X_e_data["is_bilateral_int"] = reg_sorted["is_bilateral"].astype(int)
    X_e_data["bilateral_x_year"] = (
        X_e_data["is_bilateral_int"] * X_e_data["year_centered"])
    X_e_data = pd.concat([
        X_e_data,
        field_dummies[field_cols].iloc[sort_idx].reset_index(drop=True)
    ], axis=1)
    X_e = sm.add_constant(X_e_data)
    _, gee_e_df = fit_gee(y, X_e, groups, f"{outcome_key}_E")
    all_results.append(gee_e_df)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# POST-HOC ANALYSES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# â”€â”€ VIF (on Model A predictor matrix, excluding constant) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'â”€'*60}")
print(f"  VIF Computation")
print(f"{'â”€'*60}")

X_vif = X_data.copy()
vif_data = []
for i, col in enumerate(X_vif.columns):
    vif_val = variance_inflation_factor(X_vif.values, i)
    vif_data.append({"Variable": col, "VIF": round(vif_val, 2)})

vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
print("   VIF results:")
for _, row in vif_df.head(10).iterrows():
    flag = " *** HIGH" if row["VIF"] > 5 else ""
    print(f"     {row['Variable']:45s} VIF = {row['VIF']:.2f}{flag}")
vif_df.to_csv(RESULTS / "v2_vif_results.csv", index=False)

# â”€â”€ Average Marginal Effects (AMEs) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'â”€'*60}")
print(f"  Average Marginal Effects (AMEs)")
print(f"{'â”€'*60}")

ame_rows = []
for outcome_key, (outcome_label, y) in list(outcomes.items())[:2]:
    logit_res = Logit(y, X_full).fit(disp=0)
    mfx = logit_res.get_margeff(at="overall")
    
    for i, var in enumerate(mfx.summary_frame().index):
        ame_rows.append({
            "Outcome": outcome_key,
            "Variable": var,
            "AME": mfx.margeff[i],
            "AME_pct": mfx.margeff[i] * 100,  # percentage points
            "SE": mfx.margeff_se[i],
            "p_value": mfx.pvalues[i],
        })
    
    print(f"\n   {outcome_label}:")
    for _, r in pd.DataFrame(ame_rows).query(f"Outcome == '{outcome_key}'").iterrows():
        if r["p_value"] < 0.05:
            print(f"     {r['Variable']:40s} AME = {r['AME_pct']:+.2f} pp (p={r['p_value']:.4f})")

ame_df = pd.DataFrame(ame_rows)
ame_df.to_csv(RESULTS / "v2_marginal_effects.csv", index=False)

# â”€â”€ FDR Correction on Primary Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'â”€'*60}")
print(f"  FDR Correction (Primary Model: gh_first_A)")
print(f"{'â”€'*60}")

results_combined = pd.concat(all_results, ignore_index=True)
primary = results_combined[results_combined["Model"] == "gh_first_A"].copy()
reject, pvals_corrected, _, _ = multipletests(
    primary["p_value"], alpha=0.05, method="fdr_bh")
primary["p_fdr"] = pvals_corrected
primary["sig_fdr"] = reject

print("   After FDR correction:")
for _, r in primary.iterrows():
    star = "*" if r["sig_fdr"] else " "
    print(f"   {star} {r['Variable']:40s} OR={r['OR']:.3f} p={r['p_value']:.4f} p_fdr={r['p_fdr']:.4f}")

# Add FDR to results
results_combined = results_combined.merge(
    primary[["Variable", "Model", "p_fdr", "sig_fdr"]],
    on=["Variable", "Model"], how="left")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# FOREST PLOTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

print(f"\n{'â”€'*60}")
print(f"  Generating Forest Plots")
print(f"{'â”€'*60}")

def plot_forest(model_name, title, filename):
    """Create a forest plot for a single model."""
    df = results_combined[results_combined["Model"] == model_name].copy()
    # Exclude constant
    df = df[df["Variable"] != "const"]
    
    # Clean variable names for display
    rename = {
        "log_author_count": "Log(Author Count)",
        "year_centered": "Year (linear)",
        "year_centered_sq": "Year (quadratic)",
        "has_funding_int": "Has Funding",
        "is_oa_int": "Open Access",
        "bloc_East Asian": "East Asian bloc",
        "bloc_MENA": "MENA bloc",
        "bloc_Multi-bloc": "Multi-bloc",
        "bloc_Other": "Other bloc",
        "bloc_South Asian": "South Asian bloc",
        "bloc_Western": "Western bloc",
        "bloc_Latin American": "Latin American bloc",
        "country_count": "Country Count",
        "covid_era": "COVID Era (2020+)",
        "is_bilateral_int": "Bilateral",
        "bilateral_x_year": "Bilateral x Year",
    }
    for col in df["Variable"]:
        if col.startswith("field_"):
            rename[col] = col.replace("field_", "").replace("_", " ")
    
    df["Label"] = df["Variable"].map(rename).fillna(df["Variable"])
    df = df.sort_values("OR")
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    y_pos = range(len(df))
    
    colors = []
    for _, row in df.iterrows():
        if row["p_value"] >= 0.05:
            colors.append("#9ca3af")  # grey for ns
        elif row["OR"] > 1:
            colors.append("#dc2626")  # red for OR > 1
        else:
            colors.append("#2563eb")  # blue for OR < 1
    
    ax.scatter(df["OR"], y_pos, c=colors, s=60, zorder=5)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([row["CI_lo"], row["CI_hi"]], [i, i],
                color=colors[i], alpha=0.6, lw=2)
    
    ax.axvline(1, color="black", linestyle="--", alpha=0.3, lw=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["Label"])
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_title(title, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    
    plt.tight_layout()
    save_chart(fig, filename)

for outcome_key, (outcome_label, _) in list(outcomes.items())[:2]:
    plot_forest(f"{outcome_key}_A",
                f"Model A: Predictors of Ghanaian {outcome_label} (GEE)",
                f"v2_chart_forest_{outcome_key}")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SAVE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

results_combined.to_csv(RESULTS / "v2_regression_results.csv", index=False)
with open(RESULTS / "v2_model_diagnostics.json", "w") as f:
    json.dump(diagnostics, f, indent=2)

print(f"\n{'='*70}")
print("PHASE 6 (v2) COMPLETE")
print(f"  Models fitted: {results_combined['Model'].nunique()}")
print(f"  Results saved: v2_regression_results.csv")
print(f"  Diagnostics: v2_model_diagnostics.json")
print(f"  AMEs: v2_marginal_effects.csv")
print(f"  VIF: v2_vif_results.csv")
print(f"{'='*70}")

