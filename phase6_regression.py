"""
PHASE 6: REGRESSION ANALYSIS + CHARTS 14-16, 22
Ghana Bibliometric Study
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
import pymannkendall as mk
import warnings
warnings.filterwarnings("ignore")

# ---- Setup ----
works = pd.read_parquet("analysis_results/intermediate/works.parquet")
authorships = pd.read_parquet("analysis_results/intermediate/authorships.parquet")

OUTPUT_DIR = Path("analysis_results")

COLORS = {
    "Ghanaian": "#2E7D32", "Dual-affiliated": "#F9A825",
    "Non-Ghanaian": "#1565C0", "total": "#616161",
}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
})

def save_chart(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {name}.png / .svg")

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.5)
    ax.set_axisbelow(True)

def to_bool(val):
    if pd.isna(val): return False
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() == "true"
    return bool(val)

# ================================================================
# 6.1: Prepare regression dataset
# ================================================================
print("=" * 80)
print("PHASE 6: REGRESSION ANALYSIS")
print("=" * 80)

# Get first/last/corresponding author affiliations
first = authorships[authorships["author_position"] == "first"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
first.columns = ["work_id", "first_cat"]

last = authorships[authorships["author_position"] == "last"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
last.columns = ["work_id", "last_cat"]

corr = authorships[authorships["is_corresponding_combined"] == True].drop_duplicates("work_id")[["work_id", "affiliation_category"]]
corr.columns = ["work_id", "corr_cat"]

reg = works.merge(first, on="work_id", how="left")
reg = reg.merge(last, on="work_id", how="left")
reg = reg.merge(corr, on="work_id", how="left")

# Outcome variables: GH or Dual = 1
reg["gh_first"] = reg["first_cat"].isin(["Ghanaian", "Dual-affiliated"]).astype(int)
reg["gh_last"] = reg["last_cat"].isin(["Ghanaian", "Dual-affiliated"]).astype(int)
reg["gh_corr"] = reg["corr_cat"].isin(["Ghanaian", "Dual-affiliated"]).astype(int)

print(f"Outcome prevalence:")
for out in ["gh_first", "gh_last", "gh_corr"]:
    print(f"  {out}: {reg[out].sum():,}/{len(reg):,} ({100*reg[out].mean():.1f}%)")

# ================================================================
# 6.2: Collapse small bloc categories
# ================================================================
print("\n--- 6.2 Collapse Small Blocs ---")
bloc_counts = reg["partner_bloc"].value_counts()
print("Bloc counts:")
print(bloc_counts.to_string())

small_blocs = bloc_counts[bloc_counts < 50].index.tolist()
print(f"Collapsing blocs with <50 works: {small_blocs}")
reg["partner_bloc_reg"] = reg["partner_bloc"].replace({b: "Other" for b in small_blocs})
print(reg["partner_bloc_reg"].value_counts().to_string())

# ================================================================
# 6.3: Build model matrices and fit models
# ================================================================
print("\n--- 6.3 Model Fitting ---")

# Create dummies
bloc_dummies = pd.get_dummies(reg["partner_bloc_reg"], prefix="bloc", drop_first=False)
field_dummies = pd.get_dummies(reg["field_reg"], prefix="field", drop_first=False)

# Reference categories
bloc_ref = "bloc_African"
field_ref = "field_Health Sciences"

# Drop reference categories
bloc_cols = [c for c in bloc_dummies.columns if c != bloc_ref]
field_cols = [c for c in field_dummies.columns if c != field_ref]

# Numeric predictors
reg["has_funding_int"] = reg["has_funding_bool"]
reg["is_oa_int"] = reg["is_oa_bool"]

common_predictors = ["author_count", "year_centered", "covid_era", "has_funding_int", "is_oa_int"]

# Model A: partner_bloc + author_count + year + covid + funding + oa + field (NO country_count)
X_A_parts = [reg[common_predictors], bloc_dummies[bloc_cols], field_dummies[field_cols]]
X_A = pd.concat(X_A_parts, axis=1).astype(float)
X_A = add_constant(X_A)

# Model B: country_count + author_count + year + covid + funding + oa + field (NO partner_bloc)
X_B_parts = [reg[common_predictors + ["country_count"]], field_dummies[field_cols]]
X_B = pd.concat(X_B_parts, axis=1).astype(float)
X_B = add_constant(X_B)

# Model C: Both partner_bloc AND country_count
X_C_parts = [reg[common_predictors + ["country_count"]], bloc_dummies[bloc_cols], field_dummies[field_cols]]
X_C = pd.concat(X_C_parts, axis=1).astype(float)
X_C = add_constant(X_C)

def hosmer_lemeshow(y, y_pred, g=10):
    """Hosmer-Lemeshow test."""
    try:
        data = pd.DataFrame({"y": y, "p": y_pred})
        data["group"] = pd.qcut(data["p"], g, duplicates="drop")
        obs = data.groupby("group")["y"].sum()
        exp = data.groupby("group")["p"].sum()
        n = data.groupby("group").size()
        hl_stat = (((obs - exp) ** 2) / (exp * (1 - exp / n))).sum()
        p_val = 1 - stats.chi2.cdf(hl_stat, g - 2)
        return hl_stat, p_val
    except:
        return np.nan, np.nan

def fit_and_report(y, X, label, cluster_ids=None):
    """Fit logistic regression and report results."""
    # Drop rows with NaN
    mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[mask]
    y_clean = y[mask]
    
    n_dropped = len(X) - len(X_clean)
    
    try:
        if cluster_ids is not None:
            cluster_clean = cluster_ids[mask]
            model = Logit(y_clean, X_clean).fit(disp=0, cov_type='cluster', 
                                                 cov_kwds={'groups': cluster_clean})
        else:
            model = Logit(y_clean, X_clean).fit(disp=0, cov_type='HC1')
    except Exception as e:
        print(f"  Clustered SE failed ({e}), falling back to HC1")
        model = Logit(y_clean, X_clean).fit(disp=0, cov_type='HC1')
    
    # Results
    coefs = model.params
    ci = model.conf_int()
    pvalues = model.pvalues
    
    results = pd.DataFrame({
        "Variable": coefs.index,
        "Coef": coefs.values,
        "OR": np.exp(coefs.values),
        "CI_lo": np.exp(ci.iloc[:, 0].values),
        "CI_hi": np.exp(ci.iloc[:, 1].values),
        "p_value": pvalues.values,
    })
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  N = {len(y_clean):,}, Dropped = {n_dropped}")
    print(f"  AIC = {model.aic:.1f}, Pseudo-R2 = {model.prsquared:.4f}")
    
    # Hosmer-Lemeshow
    y_pred = model.predict(X_clean)
    hl_chi2, hl_p = hosmer_lemeshow(y_clean.values, y_pred.values)
    print(f"  Hosmer-Lemeshow: chi2={hl_chi2:.2f}, p={hl_p:.4f}" if not np.isnan(hl_chi2) else "  Hosmer-Lemeshow: could not compute")
    
    print(f"{'='*60}")
    for _, row in results.iterrows():
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else ""))
        print(f"  {row['Variable']:40s}  OR={row['OR']:.3f}  [{row['CI_lo']:.3f}, {row['CI_hi']:.3f}]  p={row['p_value']:.4f} {sig}")
    
    return model, results

# Fit all 9 main models (3 specs x 3 outcomes)
outcomes = {"gh_first": "First Author", "gh_last": "Last Author", "gh_corr": "Corresponding Author"}
model_specs = {"A": X_A, "B": X_B, "C": X_C}

all_results = {}
all_models = {}

for out_var, out_label in outcomes.items():
    for spec, X_mat in model_specs.items():
        label = f"Model {spec} - {out_label}"
        model, results = fit_and_report(reg[out_var], X_mat, label, 
                                         cluster_ids=reg["first_author_id"])
        all_results[f"{out_var}_{spec}"] = results
        all_models[f"{out_var}_{spec}"] = model

# Save all results
all_reg_data = []
for key, df in all_results.items():
    df_copy = df.copy()
    df_copy["Model"] = key
    all_reg_data.append(df_copy)
pd.concat(all_reg_data).to_csv(OUTPUT_DIR / "regression_results.csv", index=False)

# ================================================================
# 6.4: Year-only and COVID-only variants
# ================================================================
print("\n--- 6.4 Year-Only and COVID-Only Variants ---")

# Year only (no covid)
X_year_only = X_A.drop(columns=["covid_era"], errors="ignore")
# COVID only (no year)
X_covid_only = X_A.drop(columns=["year_centered"], errors="ignore")

for out_var, out_label in outcomes.items():
    fit_and_report(reg[out_var], X_year_only, f"Year-Only - {out_label}", reg["first_author_id"])
    fit_and_report(reg[out_var], X_covid_only, f"COVID-Only - {out_label}", reg["first_author_id"])

# ================================================================
# 6.7: VIF check
# ================================================================
print("\n--- 6.7 VIF Check ---")

# VIF on Model A predictors (excluding constant)
X_vif = X_A.drop(columns=["const"], errors="ignore")
mask_vif = X_vif.notna().all(axis=1)
X_vif_clean = X_vif[mask_vif].astype(float)

vif_data = []
for i, col in enumerate(X_vif_clean.columns):
    try:
        vif_val = variance_inflation_factor(X_vif_clean.values, i)
        vif_data.append({"Variable": col, "VIF": round(vif_val, 2)})
    except:
        vif_data.append({"Variable": col, "VIF": np.nan})

vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
print(vif_df.to_string(index=False))
vif_df.to_csv(OUTPUT_DIR / "vif_results.csv", index=False)

# ================================================================
# 6.8: Time-stratified regression  
# ================================================================
print("\n--- 6.8 Time-Stratified Regression ---")

time_strata = [("2000-2009", 2000, 2009), ("2010-2019", 2010, 2019), ("2020-2025", 2020, 2025)]

for stratum_name, start, end in time_strata:
    mask = (reg["publication_year"] >= start) & (reg["publication_year"] <= end)
    reg_sub = reg[mask].copy()
    print(f"\n  Stratum {stratum_name}: N={len(reg_sub):,}")
    
    # Rebuild model matrix for subset
    bloc_d = pd.get_dummies(reg_sub["partner_bloc_reg"], prefix="bloc", drop_first=False)
    field_d = pd.get_dummies(reg_sub["field_reg"], prefix="field", drop_first=False)
    
    # Ensure all columns exist
    for c in bloc_cols:
        if c not in bloc_d.columns:
            bloc_d[c] = 0
    for c in field_cols:
        if c not in field_d.columns:
            field_d[c] = 0
    
    # Select predictors - drop covid_era if stratum is entirely pre or post COVID
    preds = [p for p in common_predictors]
    if reg_sub["covid_era"].nunique() <= 1:
        preds = [p for p in preds if p != "covid_era"]
        print(f"  Dropped covid_era (no variance in this stratum)")
    
    X_sub = pd.concat([reg_sub[preds], bloc_d[bloc_cols], field_d[field_cols]], axis=1).astype(float)
    
    # Drop zero-variance columns
    zero_var = [c for c in X_sub.columns if X_sub[c].nunique() <= 1]
    if zero_var:
        print(f"  Dropped zero-variance columns: {zero_var}")
        X_sub = X_sub.drop(columns=zero_var)
    
    X_sub = add_constant(X_sub)
    
    for out_var, out_label in outcomes.items():
        try:
            fit_and_report(reg_sub[out_var], X_sub, f"Time-Stratified {stratum_name} - {out_label}",
                           cluster_ids=reg_sub["first_author_id"])
        except Exception as e:
            print(f"  FAILED: {stratum_name} - {out_label}: {e}")

# ================================================================
# 6.9: Trend Analysis
# ================================================================
print("\n--- 6.9 Trend Analysis ---")

annual = pd.read_csv(OUTPUT_DIR / "annual_leadership.csv")
cats_combined = ["Ghanaian", "Dual-affiliated"]

for pos in ["first", "last", "corresponding"]:
    print(f"\n  === {pos.title()} Authorship ===")
    
    # Annual GH+Dual %
    pos_data = annual[annual["Position"] == pos]
    gh_annual = pos_data[pos_data["Category"] == "Ghanaian"].sort_values("Year")
    dual_annual = pos_data[pos_data["Category"] == "Dual-affiliated"].sort_values("Year")
    
    gh_pcts = gh_annual["Pct"].values
    dual_pcts = dual_annual["Pct"].values
    combined_pcts = gh_pcts + dual_pcts
    years = gh_annual["Year"].values
    
    # Bivariate logistic regression (year only)
    y_biv = reg[f"gh_{pos}"] if pos != "corresponding" else reg["gh_corr"]
    X_biv = add_constant(reg[["year_centered"]].astype(float))
    mask_biv = X_biv.notna().all(axis=1) & y_biv.notna()
    model_biv = Logit(y_biv[mask_biv], X_biv[mask_biv]).fit(disp=0)
    or_year = np.exp(model_biv.params["year_centered"])
    ci_biv = np.exp(model_biv.conf_int().loc["year_centered"])
    p_biv = model_biv.pvalues['year_centered']
    p_str = '<0.001' if p_biv < 0.001 else f'{p_biv:.4f}'
    print(f"  Bivariate logistic (year): OR={or_year:.4f} [{ci_biv.iloc[0]:.4f}, {ci_biv.iloc[1]:.4f}], p={p_str}")
    
    # Mann-Kendall test
    try:
        mk_result = mk.original_test(combined_pcts)
        print(f"  Mann-Kendall: tau={mk_result.Tau:.4f}, p={mk_result.p:.4f}, trend={mk_result.trend}")
        print(f"  Sen's slope: {mk_result.slope:.4f}")
    except Exception as e:
        print(f"  Mann-Kendall error: {e}")

# ================================================================
# 6.10: Bonferroni Pairwise Comparisons
# ================================================================
print("\n--- 6.10 Bonferroni Pairwise Comparisons ---")

pairs = [("Western", "African"), ("Western", "East Asian"), ("Western", "South Asian")]
alpha_bonf = 0.05 / len(pairs)
print(f"  Pre-specified pairs: {pairs}")
print(f"  Bonferroni-corrected alpha: {alpha_bonf:.4f}")

for out_var, out_label in outcomes.items():
    print(f"\n  === {out_label} ===")
    for b1, b2 in pairs:
        g1 = reg[reg["partner_bloc_reg"] == b1][out_var]
        g2 = reg[reg["partner_bloc_reg"] == b2][out_var]
        # Chi-square test
        ct = pd.DataFrame({
            b1: [g1.sum(), len(g1) - g1.sum()],
            b2: [g2.sum(), len(g2) - g2.sum()],
        }, index=["GH+Dual", "Non-GH"])
        chi2, p_val, dof, exp = stats.chi2_contingency(ct)
        sig = "SIG" if p_val < alpha_bonf else "NS"
        print(f"    {b1} vs {b2}: chi2={chi2:.1f}, p={p_val:.4f} [{sig}] "
              f"(rates: {100*g1.mean():.1f}% vs {100*g2.mean():.1f}%)")

# ================================================================
# CHARTS 14-16: Forest Plots
# ================================================================
print("\n--- CHARTS 14-16: Forest Plots ---")

def draw_forest_plot(results, title, chart_name):
    """Draw a forest plot from regression results."""
    # Exclude constant
    df = results[results["Variable"] != "const"].copy()
    df = df.iloc[::-1]  # Reverse for top-down display
    
    n_vars = len(df)
    fig_height = max(4, n_vars * 0.4 + 1)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    
    y_pos = np.arange(n_vars)
    
    for i, (_, row) in enumerate(df.iterrows()):
        color = "#E53935" if row["OR"] > 1 else "#1565C0"
        alpha = 1.0 if row["p_value"] < 0.05 else 0.4
        
        # Alternating row shading
        if i % 2 == 0:
            ax.axhspan(i - 0.4, i + 0.4, color="#F5F5F5", zorder=0)
        
        ax.plot(row["OR"], i, "o", color=color, alpha=alpha, markersize=7, zorder=3)
        ax.plot([row["CI_lo"], row["CI_hi"]], [i, i], "-", color=color, alpha=alpha, linewidth=2, zorder=2)
    
    ax.axvline(x=1.0, color="#333333", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Variable"].values, fontsize=8)
    ax.set_xlabel("Odds Ratio (95% CI)")
    ax.set_xscale("log")
    ax.set_title(title, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    save_chart(fig, chart_name)

for chart_num, (out_var, out_label) in [(14, ("gh_first", "First")), 
                                         (15, ("gh_last", "Last")),
                                         (16, ("gh_corr", "Corresponding"))]:
    results = all_results[f"{out_var}_A"]
    draw_forest_plot(results, f"Predictors of Ghanaian {out_label.lower()} authorship (Model A)",
                     f"chart{chart_num}_forest_{out_label.lower()}")

# ================================================================
# CHART 22: Robustness Comparison — Year Coefficient
# ================================================================
print("\n--- CHART 22: Robustness Comparison ---")

fig, ax = plt.subplots(figsize=(6, 3))
model_labels = []
ors = []
ci_los = []
ci_his = []

for spec in ["A", "B", "C"]:
    key = f"gh_first_{spec}"
    r = all_results[key]
    year_row = r[r["Variable"] == "year_centered"]
    if len(year_row) > 0:
        model_labels.append(f"Model {spec}")
        ors.append(year_row["OR"].values[0])
        ci_los.append(year_row["CI_lo"].values[0])
        ci_his.append(year_row["CI_hi"].values[0])

y_pos = np.arange(len(model_labels))
for i, (label, or_val, lo, hi) in enumerate(zip(model_labels, ors, ci_los, ci_his)):
    color = "#E53935" if or_val > 1 else "#1565C0"
    ax.plot(or_val, i, "o", color=color, markersize=8)
    ax.plot([lo, hi], [i, i], "-", color=color, linewidth=2)

ax.axvline(x=1.0, color="#333333", linestyle="--", linewidth=1, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(model_labels)
ax.set_xlabel("Odds Ratio (95% CI) for year_centered")
ax.set_title("Year coefficient stability across model specifications\n(first authorship)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save_chart(fig, "chart22_robustness_year")

print("\nPhase 6 complete.")
