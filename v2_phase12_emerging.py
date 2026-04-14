"""
Phase 12 (v2): Western vs Emerging Powers — Complete Analysis
==============================================================
Full pipeline: descriptive, regression, advanced, and ML with the
Traditional Western vs Emerging Powers (BRICS+) angle throughout.
"""

import pandas as pd
import numpy as np
import json, warnings, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from utils import (
    setup_plot_style, save_chart, get_leadership, load_study_data,
    to_bool, wilson_ci, cohens_h, RESULTS
)

warnings.filterwarnings("ignore")
setup_plot_style()

print("=" * 70)
print("PHASE 12: WESTERN vs EMERGING POWERS — COMPLETE ANALYSIS")
print("=" * 70)

# -- Load data ----------------------------------------------------------------
works, authorships = load_study_data()
leadership = get_leadership(works, authorships)
df = works.merge(leadership, on="work_id", how="left")
N = len(df)
print(f"Study set: {N:,}")

# Country names
COUNTRY_NAMES = {
    "US": "United States", "GB": "United Kingdom", "DE": "Germany",
    "FR": "France", "NL": "Netherlands", "CA": "Canada", "AU": "Australia",
    "CH": "Switzerland", "SE": "Sweden", "BE": "Belgium", "DK": "Denmark",
    "NO": "Norway", "IT": "Italy", "ES": "Spain",
    "CN": "China", "IN": "India", "BR": "Brazil", "ZA": "South Africa",
    "NG": "Nigeria", "KE": "Kenya", "TZ": "Tanzania", "ET": "Ethiopia",
    "SN": "Senegal", "UG": "Uganda", "BF": "Burkina Faso", "MW": "Malawi",
    "JP": "Japan", "KR": "South Korea",
}

# ==============================================================================
# SECTION 1: DESCRIPTIVE — Country-Level Authorship Equity
# ==============================================================================
print(f"\n{'='*70}")
print("SECTION 1: COUNTRY-LEVEL DESCRIPTIVE ANALYSIS")
print(f"{'='*70}")

# Get partner country lists
def get_partners(x):
    if isinstance(x, list):
        return [c for c in x if c != "GH"]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    return [c.strip() for c in str(x).split("|") if c.strip() != "GH"]

df["partner_list"] = df["partner_countries"].apply(get_partners)

# Country-level equity table (for individual partner countries)
country_rows = []
all_partner_codes = set()
for plist in df["partner_list"]:
    all_partner_codes.update(plist)

for cc in all_partner_codes:
    mask = df["partner_list"].apply(lambda x: cc in x)
    sub = df[mask]
    if len(sub) < 30:
        continue
    country_rows.append({
        "Country": COUNTRY_NAMES.get(cc, cc),
        "Code": cc,
        "N": len(sub),
        "GH_first_pct": round(sub["gh_first"].mean() * 100, 1),
        "GH_last_pct": round(sub["gh_last"].mean() * 100, 1),
        "Median_team": int(sub["author_count"].median()),
        "Mean_countries": round(sub["country_count"].mean(), 1),
        "Pct_funded": round(sub["has_funding_int"].mean() * 100, 1),
    })

country_df = pd.DataFrame(country_rows).sort_values("N", ascending=False)
country_df.to_csv(RESULTS / "country_level_equity.csv", index=False)

# Top 20 partner countries
top20 = country_df.head(20)
print(f"\nTop 20 Partner Countries by Volume:")
print(f"{'Country':<20s} {'N':>6s} {'GH 1st%':>8s} {'GH Last%':>8s} {'Team':>5s} {'Countries':>10s}")
print("-" * 60)
for _, r in top20.iterrows():
    print(f"{r['Country']:<20s} {r['N']:>6,d} {r['GH_first_pct']:>7.1f}% {r['GH_last_pct']:>7.1f}% {r['Median_team']:>5d} {r['Mean_countries']:>10.1f}")

# Chart: Top 20 partner countries by GH first authorship
fig, ax = plt.subplots(figsize=(12, 8))
top20_sorted = top20.sort_values("GH_first_pct", ascending=True)

# Colour by type
EMERGING = {"China", "India", "Brazil", "South Africa"}
AFRICAN_NAMES = {"Nigeria", "Kenya", "Tanzania", "Ethiopia", "Senegal",
                  "Uganda", "Burkina Faso", "Malawi", "Cameroon", "Rwanda"}
colors = []
for _, r in top20_sorted.iterrows():
    if r["Country"] in EMERGING:
        colors.append("#f59e0b")  # amber
    elif r["Country"] in AFRICAN_NAMES:
        colors.append("#059669")  # green
    else:
        colors.append("#2563eb")  # blue

y_pos = range(len(top20_sorted))
bars = ax.barh(y_pos, top20_sorted["GH_first_pct"], color=colors, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['Country']} (n={r['N']:,})" for _, r in top20_sorted.iterrows()],
                   fontsize=9)
ax.set_xlabel("GH/Dual First Authorship Rate (%)")
ax.set_title("Ghanaian First Authorship by Partner Country\n"
             "(Blue=Western, Amber=Emerging Power, Green=African)")
ax.axvline(50, color="black", ls="--", lw=0.8, alpha=0.5,
           label="50% equity line")
for bar, val in zip(bars, top20_sorted["GH_first_pct"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=8)
ax.legend(fontsize=9)
plt.tight_layout()
save_chart(fig, "country_equity_first")

# Same for last authorship
fig, ax = plt.subplots(figsize=(12, 8))
top20_sorted_last = top20.sort_values("GH_last_pct", ascending=True)
colors_last = []
for _, r in top20_sorted_last.iterrows():
    if r["Country"] in EMERGING:
        colors_last.append("#f59e0b")
    elif r["Country"] in AFRICAN_NAMES:
        colors_last.append("#059669")
    else:
        colors_last.append("#2563eb")

y_pos = range(len(top20_sorted_last))
bars = ax.barh(y_pos, top20_sorted_last["GH_last_pct"], color=colors_last, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['Country']} (n={r['N']:,})" for _, r in top20_sorted_last.iterrows()],
                   fontsize=9)
ax.set_xlabel("GH/Dual Last Authorship Rate (%)")
ax.set_title("Ghanaian Last Authorship by Partner Country\n"
             "(Blue=Western, Amber=Emerging Power, Green=African)")
ax.axvline(50, color="black", ls="--", lw=0.8, alpha=0.5)
for bar, val in zip(bars, top20_sorted_last["GH_last_pct"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=8)
plt.tight_layout()
save_chart(fig, "country_equity_last")

# ==============================================================================
# SECTION 2: PARTNER TYPE COMPARISON
# ==============================================================================
print(f"\n{'='*70}")
print("SECTION 2: PARTNER TYPE COMPARISON (Western vs Emerging vs African)")
print(f"{'='*70}")

# partner_type: Traditional Western, Emerging Power, African, Mixed, Other
pt_summary = []
for pt in ["Traditional Western", "Emerging Power", "African", "Mixed", "Other"]:
    sub = df[df["partner_type"] == pt]
    if len(sub) < 10:
        continue
    first_lo, first_hi = wilson_ci(sub["gh_first"].sum(), len(sub))
    last_lo, last_hi = wilson_ci(sub["gh_last"].sum(), len(sub))
    pt_summary.append({
        "Partner_type": pt,
        "N": len(sub),
        "Pct_of_total": round(len(sub) / N * 100, 1),
        "GH_first_pct": round(sub["gh_first"].mean() * 100, 1),
        "GH_first_CI": f"[{first_lo:.1f}, {first_hi:.1f}]",
        "GH_last_pct": round(sub["gh_last"].mean() * 100, 1),
        "GH_last_CI": f"[{last_lo:.1f}, {last_hi:.1f}]",
        "Median_team": int(sub["author_count"].median()),
        "Mean_countries": round(sub["country_count"].mean(), 1),
        "Pct_funded": round(sub["has_funding_int"].mean() * 100, 1),
        "Pct_OA": round(sub["is_oa_int"].mean() * 100, 1),
    })

pt_df = pd.DataFrame(pt_summary)
pt_df.to_csv(RESULTS / "partner_type_equity.csv", index=False)
print(pt_df.to_string(index=False))

# Effect sizes: Western vs Emerging
western_sub = df[df["partner_type"] == "Traditional Western"]
emerging_sub = df[df["partner_type"] == "Emerging Power"]
african_sub = df[df["partner_type"] == "African"]

if len(emerging_sub) > 0 and len(western_sub) > 0:
    h_first_we = cohens_h(emerging_sub["gh_first"].mean(), western_sub["gh_first"].mean())
    h_last_we = cohens_h(emerging_sub["gh_last"].mean(), western_sub["gh_last"].mean())
    h_first_wa = cohens_h(african_sub["gh_first"].mean(), western_sub["gh_first"].mean())
    h_last_wa = cohens_h(african_sub["gh_last"].mean(), western_sub["gh_last"].mean())
    
    print(f"\nEffect sizes (Cohen's h):")
    print(f"  Emerging vs Western (first): {h_first_we:.3f}")
    print(f"  Emerging vs Western (last):  {h_last_we:.3f}")
    print(f"  African vs Western (first):  {h_first_wa:.3f}")
    print(f"  African vs Western (last):   {h_last_wa:.3f}")

# Chart: partner type comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
pt_plot = pt_df.sort_values("GH_first_pct", ascending=True)
type_colors = {
    "Traditional Western": "#2563eb",
    "Emerging Power": "#f59e0b",
    "African": "#059669",
    "Mixed": "#6b7280",
    "Other": "#9ca3af",
}
for i, (col, title) in enumerate([("GH_first_pct", "First Authorship"),
                                   ("GH_last_pct", "Last Authorship")]):
    ax = axes[i]
    pt_plot_sorted = pt_df.sort_values(col, ascending=True)
    colors_pt = [type_colors.get(r["Partner_type"], "#ddd") for _, r in pt_plot_sorted.iterrows()]
    y = range(len(pt_plot_sorted))
    bars = ax.barh(y, pt_plot_sorted[col], color=colors_pt, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['Partner_type']} (n={r['N']:,})"
                        for _, r in pt_plot_sorted.iterrows()], fontsize=9)
    ax.set_xlabel(f"GH {title} Rate (%)")
    ax.set_title(f"GH {title} by Partner Type")
    ax.axvline(50, color="black", ls="--", lw=0.8, alpha=0.5)
    for bar, val in zip(bars, pt_plot_sorted[col]):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
plt.tight_layout()
save_chart(fig, "partner_type_equity")

# ==============================================================================
# SECTION 3: TEMPORAL TRENDS BY PARTNER TYPE
# ==============================================================================
print(f"\n{'='*70}")
print("SECTION 3: TEMPORAL TRENDS BY PARTNER TYPE")
print(f"{'='*70}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, (col, title) in enumerate([("gh_first", "First Authorship"),
                                   ("gh_last", "Last Authorship")]):
    ax = axes[i]
    for pt, color, ls in [("Traditional Western", "#2563eb", "-"),
                           ("Emerging Power", "#f59e0b", "-"),
                           ("African", "#059669", "-"),
                           ("Mixed", "#6b7280", "--")]:
        sub = df[df["partner_type"] == pt]
        if len(sub) < 100:
            continue
        annual = sub.groupby("publication_year").agg(
            rate=(col, "mean"),
            n=("work_id", "size"),
        ).reset_index()
        annual = annual[annual["n"] >= 10]
        ax.plot(annual["publication_year"], annual["rate"] * 100,
                f"o{ls}", color=color, label=pt, markersize=3, lw=1.5)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"GH {title} Rate (%)")
    ax.set_title(f"GH {title} Over Time by Partner Type")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
save_chart(fig, "partner_type_trends")

# Mann-Kendall for each partner type
print("\nMann-Kendall trend tests:")
for pt in ["Traditional Western", "Emerging Power", "African", "Mixed"]:
    sub = df[df["partner_type"] == pt]
    if len(sub) < 100:
        continue
    annual = sub.groupby("publication_year")["gh_first"].mean()
    if len(annual) < 8:
        continue
    tau, p = stats.kendalltau(annual.index, annual.values)
    trend = "increasing" if p < 0.05 and tau > 0 else "decreasing" if p < 0.05 and tau < 0 else "no trend"
    print(f"  {pt}: tau={tau:.3f}, p={p:.4f} ({trend})")

# ==============================================================================
# SECTION 4: GEE REGRESSION WITH PARTNER TYPE
# ==============================================================================
print(f"\n{'='*70}")
print("SECTION 4: GEE REGRESSION WITH PARTNER TYPE VARIABLE")
print(f"{'='*70}")

# Add partner_type dummies (reference = Traditional Western)
pt_dummies = pd.get_dummies(df["partner_type"], prefix="pt", dtype=int)
pt_ref = "pt_Traditional Western"
pt_cols = [c for c in pt_dummies.columns if c != pt_ref]

field_dummies = pd.get_dummies(df["field_reg"], prefix="field", dtype=int)
field_ref = "field_Health Sciences"
field_cols = [c for c in field_dummies.columns if c != field_ref]

# Build X matrix
X_cols = ["log_author_count", "year_centered", "year_centered_sq",
          "has_funding_int", "is_oa_int"]

reg_df = df.sort_values("primary_gh_institution").reset_index(drop=True)
X = reg_df[X_cols].copy()
for c in pt_cols:
    X[c] = pt_dummies[c].iloc[reg_df.index.values].values if len(pt_dummies) == len(df) else pt_dummies.loc[reg_df.index, c].values
for c in field_cols:
    X[c] = field_dummies[c].iloc[reg_df.index.values].values if len(field_dummies) == len(df) else field_dummies.loc[reg_df.index, c].values
X = sm.add_constant(X)
groups = reg_df["primary_gh_institution"]

pt_regression_results = []

for outcome, label in [("gh_first", "First"), ("gh_last", "Last")]:
    y = reg_df[outcome].astype(int)
    
    gee = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable())
    result = gee.fit(maxiter=200)
    
    print(f"\n  --- GEE: GH {label} Authorship (partner_type model) ---")
    for var in pt_cols + ["log_author_count", "has_funding_int", "is_oa_int"]:
        if var in result.params.index:
            or_val = np.exp(result.params[var])
            ci_lo = np.exp(result.conf_int().loc[var, 0])
            ci_hi = np.exp(result.conf_int().loc[var, 1])
            p = result.pvalues[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {var:30s} OR={or_val:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] p={p:.4f} {sig}")
            
            pt_regression_results.append({
                "Outcome": label,
                "Variable": var,
                "OR": round(or_val, 4),
                "CI_lo": round(ci_lo, 4),
                "CI_hi": round(ci_hi, 4),
                "p_value": round(p, 6),
            })

pt_reg_df = pd.DataFrame(pt_regression_results)
pt_reg_df.to_csv(RESULTS / "partner_type_regression.csv", index=False)

# Pairwise comparisons: Emerging vs Western, African vs Western
print(f"\n  Pairwise comparisons (from model coefficients):")
for var, label in [("pt_Emerging Power", "Emerging vs Western"),
                    ("pt_African", "African vs Western"),
                    ("pt_Mixed", "Mixed vs Western")]:
    if var in result.params.index:
        row = pt_reg_df[(pt_reg_df["Variable"] == var) & (pt_reg_df["Outcome"] == "First")]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"    {label} (first): OR={r['OR']:.3f} [{r['CI_lo']:.3f}, {r['CI_hi']:.3f}] p={r['p_value']:.4f}")

# ==============================================================================
# SECTION 5: HEAD-TO-HEAD BILATERAL COMPARISONS
# ==============================================================================
print(f"\n{'='*70}")
print("SECTION 5: HEAD-TO-HEAD BILATERAL COMPARISONS")
print(f"{'='*70}")

# Bilateral papers only — compare by dominant partner country
bilateral = df[df["is_bilateral"] == True].copy()
print(f"Bilateral papers: {len(bilateral):,}")

bilateral_country_results = []
for cc in ["US", "GB", "DE", "NL", "CN", "IN", "ZA", "NG", "KE", "AU", "CA", "FR", "BR", "JP"]:
    col = f"has_{cc}"
    if col not in bilateral.columns:
        continue
    sub = bilateral[bilateral[col] == 1]
    if len(sub) < 20:
        continue
    bilateral_country_results.append({
        "Country": COUNTRY_NAMES.get(cc, cc),
        "Code": cc,
        "N": len(sub),
        "GH_first_pct": round(sub["gh_first"].mean() * 100, 1),
        "GH_last_pct": round(sub["gh_last"].mean() * 100, 1),
        "Median_team": int(sub["author_count"].median()),
    })

bil_country_df = pd.DataFrame(bilateral_country_results).sort_values("N", ascending=False)
bil_country_df.to_csv(RESULTS / "bilateral_country_equity.csv", index=False)

print(f"\nBilateral Equity by Partner Country:")
print(bil_country_df.to_string(index=False))

# Chart: bilateral head-to-head
fig, ax = plt.subplots(figsize=(12, 7))
bcd_sorted = bil_country_df.sort_values("GH_first_pct", ascending=True)
colors_bil = []
for _, r in bcd_sorted.iterrows():
    if r["Country"] in EMERGING:
        colors_bil.append("#f59e0b")
    elif r["Country"] in AFRICAN_NAMES:
        colors_bil.append("#059669")
    else:
        colors_bil.append("#2563eb")

y = range(len(bcd_sorted))
ax.barh(y, bcd_sorted["GH_first_pct"], color=colors_bil, alpha=0.85, height=0.6,
        label="First Author")
ax.barh([i + 0.3 for i in y], bcd_sorted["GH_last_pct"], color=colors_bil,
        alpha=0.45, height=0.3, label="Last Author")
ax.set_yticks([i + 0.15 for i in y])
ax.set_yticklabels([f"{r['Country']} (n={r['N']:,})"
                    for _, r in bcd_sorted.iterrows()], fontsize=9)
ax.set_xlabel("GH Authorship Rate (%)")
ax.set_title("Bilateral Partnerships: GH Authorship by Partner Country\n"
             "(Blue=Western, Amber=Emerging, Green=African)")
ax.axvline(50, color="black", ls="--", lw=0.8, alpha=0.5, label="Equity line")
ax.legend(fontsize=8)
plt.tight_layout()
save_chart(fig, "bilateral_country_equity")

# ==============================================================================
# SECTION 6: WESTERN vs EMERGING REGRESSION (bilateral only)
# ==============================================================================
print(f"\n{'='*70}")
print("SECTION 6: GEE — WESTERN vs EMERGING (bilateral only)")
print(f"{'='*70}")

# Bilateral-only GEE comparing Western vs Emerging vs African
bil_for_reg = bilateral[bilateral["partner_type"].isin(
    ["Traditional Western", "Emerging Power", "African"])].copy()
bil_for_reg = bil_for_reg.sort_values("primary_gh_institution").reset_index(drop=True)
print(f"Bilateral (Western/Emerging/African): {len(bil_for_reg):,}")

X_bil = bil_for_reg[["log_author_count", "year_centered", "year_centered_sq",
                      "has_funding_int", "is_oa_int"]].copy()
X_bil["is_emerging"] = (bil_for_reg["partner_type"] == "Emerging Power").astype(int)
X_bil["is_african"] = (bil_for_reg["partner_type"] == "African").astype(int)
# Add field dummies
for c in field_cols:
    if c in field_dummies.columns:
        X_bil[c] = field_dummies.loc[bil_for_reg.index, c].values if c in field_dummies.columns else 0
X_bil = sm.add_constant(X_bil)
groups_bil = bil_for_reg["primary_gh_institution"]

bil_reg_results = []
for outcome, label in [("gh_first", "First"), ("gh_last", "Last")]:
    y = bil_for_reg[outcome].astype(int)
    try:
        gee = GEE(y, X_bil, groups=groups_bil, family=Binomial(),
                  cov_struct=Exchangeable())
        result = gee.fit(maxiter=200)
        
        print(f"\n  --- Bilateral GEE: {label} ---")
        for var in ["is_emerging", "is_african", "log_author_count",
                     "has_funding_int", "is_oa_int"]:
            if var in result.params.index:
                or_val = np.exp(result.params[var])
                ci_lo = np.exp(result.conf_int().loc[var, 0])
                ci_hi = np.exp(result.conf_int().loc[var, 1])
                p = result.pvalues[var]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {var:20s} OR={or_val:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] p={p:.4f} {sig}")
                bil_reg_results.append({
                    "Outcome": label,
                    "Variable": var,
                    "OR": round(or_val, 4),
                    "CI_lo": round(ci_lo, 4),
                    "CI_hi": round(ci_hi, 4),
                    "p_value": round(p, 6),
                })
    except Exception as e:
        print(f"  {label}: FAILED - {e}")

bil_reg_df = pd.DataFrame(bil_reg_results)
bil_reg_df.to_csv(RESULTS / "bilateral_type_regression.csv", index=False)

# ==============================================================================
# SECTION 7: XGBoost + SHAP WITH PARTNER TYPE
# ==============================================================================
print(f"\n{'='*70}")
print("SECTION 7: XGBoost + SHAP WITH PARTNER TYPE FEATURES")
print(f"{'='*70}")

import xgboost as xgb
import shap

# Build ML features with partner type
ml_features = ["log_author_count", "year_centered", "author_count",
               "country_count", "has_funding_int", "is_oa_int",
               "is_bilateral"]

# Add has_XX country flags
for cc in ["US", "GB", "CN", "IN", "ZA", "NG", "KE", "DE", "NL"]:
    col = f"has_{cc}"
    if col in df.columns:
        ml_features.append(col)

# Add partner_type encoded
from sklearn.preprocessing import LabelEncoder
le_pt = LabelEncoder()
df["pt_encoded"] = le_pt.fit_transform(df["partner_type"].fillna("Unknown"))
ml_features.append("pt_encoded")

le_field = LabelEncoder()
df["field_encoded"] = le_field.fit_transform(df["field_reg"].fillna("Other"))
ml_features.append("field_encoded")

X_ml = df[ml_features].copy()
y_first = df["gh_first"].astype(int).values
y_last = df["gh_last"].astype(int).values

xgb_params = {
    "n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
    "random_state": 42, "n_jobs": -1, "eval_metric": "auc",
}

ml_results = {}
for outcome_name, y in [("first", y_first), ("last", y_last)]:
    print(f"\n  --- {outcome_name.upper()} AUTHORSHIP ---")
    
    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(
        xgb.XGBClassifier(**xgb_params), X_ml, y, cv=cv, method="predict_proba"
    )[:, 1]
    cv_auc = roc_auc_score(y, cv_probs)
    print(f"  5-fold CV AUC: {cv_auc:.4f}")
    
    # Train final model for SHAP
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_ml, y, verbose=False)
    
    # SHAP
    print(f"  Computing SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_ml)
    
    # SHAP summary
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_ml, feature_names=ml_features,
                      show=False, max_display=15)
    plt.title(f"SHAP: GH {outcome_name.title()} Authorship (with partner countries)")
    plt.tight_layout()
    save_chart(plt.gcf(), f"ml_shap_partners_{outcome_name}")
    plt.close("all")
    
    # SHAP bar
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_ml, feature_names=ml_features,
                      plot_type="bar", show=False, max_display=15)
    plt.title(f"SHAP Mean |Impact|: {outcome_name.title()} (with partners)")
    plt.tight_layout()
    save_chart(plt.gcf(), f"ml_shap_bar_partners_{outcome_name}")
    plt.close("all")
    
    # Mean absolute SHAP
    mean_shap = pd.DataFrame({
        "Feature": ml_features,
        "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0),
        "Mean_SHAP": shap_values.mean(axis=0),
    }).sort_values("Mean_abs_SHAP", ascending=False)
    mean_shap.to_csv(RESULTS / f"ml_shap_partners_{outcome_name}.csv", index=False)
    
    print(f"  Top SHAP features:")
    for _, r in mean_shap.head(8).iterrows():
        direction = "+" if r["Mean_SHAP"] > 0 else "-"
        print(f"    {r['Feature']:20s} |SHAP|={r['Mean_abs_SHAP']:.4f} ({direction})")
    
    ml_results[outcome_name] = {
        "cv_auc": round(cv_auc, 4),
        "top_features": mean_shap.head(5)["Feature"].tolist(),
    }

# ==============================================================================
# SECTION 8: EMERGING POWERS DEEP DIVE
# ==============================================================================
print(f"\n{'='*70}")
print("SECTION 8: EMERGING POWERS DEEP DIVE")
print(f"{'='*70}")

# China, India, Brazil, South Africa — individual analysis
emerging_detail = []
for cc, name in [("CN", "China"), ("IN", "India"), ("ZA", "South Africa"), ("BR", "Brazil")]:
    col = f"has_{cc}"
    if col not in df.columns:
        continue
    sub = df[df[col] == 1]
    if len(sub) < 20:
        print(f"  {name}: N={len(sub)} (too few)")
        continue
    
    # Bilateral only papers with this country
    bil_sub = bilateral[bilateral[col] == 1] if col in bilateral.columns else pd.DataFrame()
    
    detail = {
        "Country": name,
        "N_total": len(sub),
        "N_bilateral": len(bil_sub) if len(bil_sub) > 0 else 0,
        "GH_first_all": round(sub["gh_first"].mean() * 100, 1),
        "GH_last_all": round(sub["gh_last"].mean() * 100, 1),
        "GH_first_bilateral": round(bil_sub["gh_first"].mean() * 100, 1) if len(bil_sub) > 0 else None,
        "GH_last_bilateral": round(bil_sub["gh_last"].mean() * 100, 1) if len(bil_sub) > 0 else None,
        "Median_team": int(sub["author_count"].median()),
        "Pct_funded": round(sub["has_funding_int"].mean() * 100, 1),
        "Pct_OA": round(sub["is_oa_int"].mean() * 100, 1),
    }
    emerging_detail.append(detail)
    
    print(f"\n  {name}:")
    print(f"    Total papers: {detail['N_total']:,}, Bilateral: {detail['N_bilateral']:,}")
    print(f"    All papers -> GH first: {detail['GH_first_all']}%, last: {detail['GH_last_all']}%")
    if detail["GH_first_bilateral"]:
        print(f"    Bilateral  -> GH first: {detail['GH_first_bilateral']}%, last: {detail['GH_last_bilateral']}%")
    print(f"    Team: {detail['Median_team']}, Funded: {detail['Pct_funded']}%, OA: {detail['Pct_OA']}%")

# Compare to top Western partners
for cc, name in [("US", "United States"), ("GB", "United Kingdom")]:
    col = f"has_{cc}"
    sub = df[df[col] == 1]
    bil_sub = bilateral[bilateral[col] == 1]
    print(f"\n  {name} (comparison):")
    print(f"    Total: {len(sub):,}, Bilateral: {len(bil_sub):,}")
    print(f"    All papers -> GH first: {sub['gh_first'].mean()*100:.1f}%, last: {sub['gh_last'].mean()*100:.1f}%")
    print(f"    Bilateral  -> GH first: {bil_sub['gh_first'].mean()*100:.1f}%, last: {bil_sub['gh_last'].mean()*100:.1f}%")

emerging_df = pd.DataFrame(emerging_detail)
emerging_df.to_csv(RESULTS / "emerging_powers_detail.csv", index=False)

# ==============================================================================
# SECTION 9: TEMPORAL GROWTH OF EMERGING PARTNERSHIPS
# ==============================================================================
print(f"\n{'='*70}")
print("SECTION 9: GROWTH OF EMERGING VS WESTERN PARTNERSHIPS")
print(f"{'='*70}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Volume over time
for cc, name, color in [("US", "US", "#2563eb"), ("GB", "UK", "#60a5fa"),
                          ("CN", "China", "#f59e0b"), ("IN", "India", "#fb923c"),
                          ("ZA", "S. Africa", "#059669"), ("NG", "Nigeria", "#34d399")]:
    col = f"has_{cc}"
    if col not in df.columns:
        continue
    annual = df[df[col] == 1].groupby("publication_year").size()
    axes[0].plot(annual.index, annual.values, "o-", color=color,
                 label=name, markersize=3, lw=1.5)

axes[0].set_xlabel("Year")
axes[0].set_ylabel("Number of Papers")
axes[0].set_title("Growth of Partnerships by Country")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# GH first authorship rate over time
for cc, name, color in [("US", "US", "#2563eb"), ("GB", "UK", "#60a5fa"),
                          ("CN", "China", "#f59e0b"), ("IN", "India", "#fb923c"),
                          ("ZA", "S. Africa", "#059669")]:
    col = f"has_{cc}"
    if col not in df.columns:
        continue
    sub = df[df[col] == 1]
    annual = sub.groupby("publication_year").agg(
        rate=("gh_first", "mean"), n=("work_id", "size")).reset_index()
    annual = annual[annual["n"] >= 10]
    axes[1].plot(annual["publication_year"], annual["rate"] * 100,
                 "o-", color=color, label=name, markersize=3, lw=1.5)

axes[1].set_xlabel("Year")
axes[1].set_ylabel("GH First Authorship (%)")
axes[1].set_title("GH First Authorship Trend by Partner")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
plt.tight_layout()
save_chart(fig, "emerging_growth_trends")

# ==============================================================================
# SAVE ALL RESULTS
# ==============================================================================
all_results = {
    "partner_type_summary": pt_summary,
    "ml_results": ml_results,
    "emerging_detail": emerging_detail,
}
with open(RESULTS / "emerging_powers_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n{'='*70}")
print("PHASE 12 COMPLETE -- WESTERN vs EMERGING POWERS")
print(f"{'='*70}")
print("CSVs: country_level_equity, partner_type_equity, partner_type_regression,")
print("       bilateral_country_equity, bilateral_type_regression,")
print("       emerging_powers_detail, ml_shap_partners_*")
print("Charts: ~10 new charts")
print("JSON: emerging_powers_results.json")
