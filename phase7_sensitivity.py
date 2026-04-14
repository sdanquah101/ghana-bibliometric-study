"""
PHASE 7: SENSITIVITY ANALYSES + CHART 20
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
from statsmodels.tools import add_constant
from statsmodels.stats.proportion import proportion_confint
import pymannkendall as mk
import warnings
warnings.filterwarnings("ignore")

works = pd.read_parquet("analysis_results/intermediate/works.parquet")
authorships = pd.read_parquet("analysis_results/intermediate/authorships.parquet")

OUTPUT_DIR = Path("analysis_results")

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

def wilson_ci(count, total, alpha=0.05):
    if total == 0: return 0, 0, 0
    pct = count / total
    lo, hi = proportion_confint(count, total, alpha=alpha, method='wilson')
    return pct * 100, lo * 100, hi * 100

def get_leadership(works_df, auths_df):
    first = auths_df[auths_df["author_position"] == "first"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
    first.columns = ["work_id", "first_cat"]
    last = auths_df[auths_df["author_position"] == "last"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
    last.columns = ["work_id", "last_cat"]
    corr = auths_df[auths_df["is_corresponding_combined"] == True].drop_duplicates("work_id")[["work_id", "affiliation_category"]]
    corr.columns = ["work_id", "corr_cat"]
    result = works_df[["work_id"]].merge(first, on="work_id", how="left")
    result = result.merge(last, on="work_id", how="left")
    result = result.merge(corr, on="work_id", how="left")
    return result

def compute_leadership_pcts(works_df, auths_df, label=""):
    lead = get_leadership(works_df, auths_df)
    n = len(works_df)
    results = {}
    for pos in ["first", "last", "corresponding"]:
        col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
        valid = lead[col].dropna()
        gh_dual = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
        pct, lo, hi = wilson_ci(gh_dual, len(valid))
        results[pos] = {"pct": pct, "ci_lo": lo, "ci_hi": hi, "n": len(valid), "gh_dual": gh_dual}
    if label:
        print(f"  {label} (N={n:,}):")
        for pos, r in results.items():
            print(f"    {pos.title()}: {r['pct']:.1f}% [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]")
    return results

print("=" * 80)
print("PHASE 7: SENSITIVITY ANALYSES")
print("=" * 80)

# Get baseline leadership
baseline = compute_leadership_pcts(works, authorships, "BASELINE (full study set)")

# ================================================================
# 7.1: Dual-affiliation reclassification
# ================================================================
print("\n--- 7.1 Dual Reclassification ---")

# Generous: Dual -> Ghanaian
auths_gen = authorships.copy()
auths_gen.loc[auths_gen["affiliation_category"] == "Dual-affiliated", "affiliation_category"] = "Ghanaian"
compute_leadership_pcts(works, auths_gen, "GENEROUS (Dual -> Ghanaian)")

# Conservative: Dual -> Non-Ghanaian
auths_con = authorships.copy()
auths_con.loc[auths_con["affiliation_category"] == "Dual-affiliated", "affiliation_category"] = "Non-Ghanaian"
compute_leadership_pcts(works, auths_con, "CONSERVATIVE (Dual -> Non-GH)")

# ================================================================
# 7.2: Articles-only
# ================================================================
print("\n--- 7.2 Articles-Only ---")
articles = works[works["type"] == "article"]
articles_a = authorships[authorships["work_id"].isin(set(articles["work_id"]))]
compute_leadership_pcts(articles, articles_a, "ARTICLES ONLY")

# ================================================================
# 7.3: Exclude >50-author papers
# ================================================================
print("\n--- 7.3 Exclude >50 Authors ---")
small = works[works["author_count"] <= 50]
small_a = authorships[authorships["work_id"].isin(set(small["work_id"]))]
compute_leadership_pcts(small, small_a, "EXCLUDE >50 AUTHORS")

# ================================================================
# 7.4: Exclude COVID-topic papers
# ================================================================
print("\n--- 7.4 Exclude COVID-Topic Papers ---")

covid_kw = ["covid", "sars-cov", "coronavirus", "pandemic", "covid-19", "sars-cov-2"]
def is_covid_topic(row):
    text = ""
    for col in ["title", "abstract"]:
        if pd.notna(row.get(col)):
            text += " " + str(row[col])
    text_lower = text.lower()
    return any(kw in text_lower for kw in covid_kw)

works["is_covid_topic"] = works.apply(is_covid_topic, axis=1)
n_covid = works["is_covid_topic"].sum()
print(f"  COVID-topic papers: {n_covid:,}")

non_covid = works[~works["is_covid_topic"]]
non_covid_a = authorships[authorships["work_id"].isin(set(non_covid["work_id"]))]

# Pre/post comparison without COVID papers
print("  Pre/Post COVID (excluding COVID-topic papers):")
for era, label in [(0, "Pre-COVID"), (1, "Post-COVID")]:
    sub = non_covid[non_covid["covid_era"] == era]
    sub_a = non_covid_a[non_covid_a["work_id"].isin(set(sub["work_id"]))]
    compute_leadership_pcts(sub, sub_a, f"{label} (no COVID papers)")

# ================================================================
# 7.5: Minimum team size >= 4 authors
# ================================================================
print("\n--- 7.5 Minimum Team Size >= 4 ---")
big_teams = works[works["author_count"] >= 4]
big_a = authorships[authorships["work_id"].isin(set(big_teams["work_id"]))]
compute_leadership_pcts(big_teams, big_a, ">= 4 AUTHORS")

# Regression with >=4 authors
print("  Running Model A on >=4 author subset...")
first = big_a[big_a["author_position"] == "first"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
first.columns = ["work_id", "first_cat"]
last = big_a[big_a["author_position"] == "last"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
last.columns = ["work_id", "last_cat"]
corr = big_a[big_a["is_corresponding_combined"] == True].drop_duplicates("work_id")[["work_id", "affiliation_category"]]
corr.columns = ["work_id", "corr_cat"]

reg_big = big_teams.merge(first, on="work_id", how="left")
reg_big = reg_big.merge(last, on="work_id", how="left")
reg_big = reg_big.merge(corr, on="work_id", how="left")

for pos, col in [("first", "first_cat"), ("last", "last_cat"), ("corresponding", "corr_cat")]:
    y = reg_big[col].isin(["Ghanaian", "Dual-affiliated"]).astype(int)
    
    bloc_d = pd.get_dummies(reg_big["partner_bloc"].replace(
        {b: "Other" for b in reg_big["partner_bloc"].value_counts()[reg_big["partner_bloc"].value_counts() < 50].index}
    ), prefix="bloc", drop_first=False)
    field_d = pd.get_dummies(reg_big["field_reg"], prefix="field", drop_first=False)
    
    bloc_ref = "bloc_African" if "bloc_African" in bloc_d.columns else bloc_d.columns[0]
    field_ref = "field_Health Sciences" if "field_Health Sciences" in field_d.columns else field_d.columns[0]
    
    X = pd.concat([
        reg_big[["author_count", "year_centered", "covid_era", "has_funding_bool", "is_oa_bool"]].astype(float),
        bloc_d[[c for c in bloc_d.columns if c != bloc_ref]],
        field_d[[c for c in field_d.columns if c != field_ref]],
    ], axis=1).astype(float)
    X = add_constant(X)
    
    mask = X.notna().all(axis=1) & y.notna()
    try:
        model = Logit(y[mask], X[mask]).fit(disp=0, cov_type='HC1')
        yr = model.params.get("year_centered", np.nan)
        covid = model.params.get("covid_era", np.nan)
        print(f"    {pos.title()}: year OR={np.exp(yr):.4f}, covid OR={np.exp(covid):.4f}")
    except Exception as e:
        print(f"    {pos.title()}: regression failed: {e}")

# Mann-Kendall on >=4 subset
print("  Mann-Kendall on >=4 author subset:")
for pos in ["first", "last", "corresponding"]:
    annual = []
    for year in range(2000, 2026):
        yr_w = big_teams[big_teams["publication_year"] == year]
        yr_a = big_a[big_a["work_id"].isin(set(yr_w["work_id"]))]
        lead = get_leadership(yr_w, yr_a)
        col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
        valid = lead[col].dropna()
        gh = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
        pct = 100 * gh / len(valid) if len(valid) > 0 else np.nan
        annual.append(pct)
    
    series = [x for x in annual if not np.isnan(x)]
    try:
        mk_result = mk.original_test(series)
        print(f"    {pos.title()}: tau={mk_result.Tau:.4f}, p={mk_result.p:.4f}")
    except:
        pass

# ================================================================
# 7.6: Corresponding author non-overlap
# ================================================================
print("\n--- 7.6 Corresponding Author Non-Overlap ---")
non_overlap = works[works["first_is_corr"] == False]
non_overlap_a = authorships[authorships["work_id"].isin(set(non_overlap["work_id"]))]
compute_leadership_pcts(non_overlap, non_overlap_a, "NON-OVERLAP (first != corr)")

# ================================================================
# 7.7: Simpson's paradox validation
# ================================================================
print("\n--- 7.7 Simpson's Paradox Validation ---")

bilateral = works[works["is_bilateral"] == 1]
multibloc = works[works["is_bilateral"] == 0]

bilat_annual = []
multi_annual = []
agg_annual = []

for year in range(2000, 2026):
    # Bilateral
    yr_b = bilateral[bilateral["publication_year"] == year]
    yr_ba = authorships[authorships["work_id"].isin(set(yr_b["work_id"]))]
    lead_b = get_leadership(yr_b, yr_ba)
    valid_b = lead_b["first_cat"].dropna()
    gh_b = ((valid_b == "Ghanaian") | (valid_b == "Dual-affiliated")).sum()
    pct_b = 100 * gh_b / len(valid_b) if len(valid_b) >= 5 else np.nan
    bilat_annual.append(pct_b)
    
    # Multi-bloc
    yr_m = multibloc[multibloc["publication_year"] == year]
    yr_ma = authorships[authorships["work_id"].isin(set(yr_m["work_id"]))]
    lead_m = get_leadership(yr_m, yr_ma)
    valid_m = lead_m["first_cat"].dropna()
    gh_m = ((valid_m == "Ghanaian") | (valid_m == "Dual-affiliated")).sum()
    pct_m = 100 * gh_m / len(valid_m) if len(valid_m) >= 5 else np.nan
    multi_annual.append(pct_m)
    
    # Aggregate
    yr_a = works[works["publication_year"] == year]
    yr_aa = authorships[authorships["work_id"].isin(set(yr_a["work_id"]))]
    lead_a = get_leadership(yr_a, yr_aa)
    valid_a = lead_a["first_cat"].dropna()
    gh_a = ((valid_a == "Ghanaian") | (valid_a == "Dual-affiliated")).sum()
    pct_a = 100 * gh_a / len(valid_a) if len(valid_a) >= 5 else np.nan
    agg_annual.append(pct_a)

years = list(range(2000, 2026))

# Mann-Kendall on each stratum
bilat_clean = [x for x in bilat_annual if not np.isnan(x)]
multi_clean = [x for x in multi_annual if not np.isnan(x)]
agg_clean = [x for x in agg_annual if not np.isnan(x)]

print("  Mann-Kendall on bilateral-only first authorship:")
try:
    mk_b = mk.original_test(bilat_clean)
    print(f"    tau={mk_b.Tau:.4f}, p={mk_b.p:.4f}, trend={mk_b.trend}")
except Exception as e:
    print(f"    Error: {e}")

print("  Mann-Kendall on multi-bloc-only first authorship:")
try:
    mk_m = mk.original_test(multi_clean)
    print(f"    tau={mk_m.Tau:.4f}, p={mk_m.p:.4f}, trend={mk_m.trend}")
except Exception as e:
    print(f"    Error: {e}")

print("  Mann-Kendall on aggregate first authorship:")
try:
    mk_a = mk.original_test(agg_clean)
    print(f"    tau={mk_a.Tau:.4f}, p={mk_a.p:.4f}, trend={mk_a.trend}")
except Exception as e:
    print(f"    Error: {e}")

# Multi-bloc share change
early = works[(works["publication_year"] >= 2000) & (works["publication_year"] <= 2005)]
late = works[(works["publication_year"] >= 2020) & (works["publication_year"] <= 2025)]
early_multi_pct = 100 * (early["is_bilateral"] == 0).sum() / len(early)
late_multi_pct = 100 * (late["is_bilateral"] == 0).sum() / len(late)
print(f"\n  Multi-bloc share: 2000-2005={early_multi_pct:.1f}%, 2020-2025={late_multi_pct:.1f}%")

# ================================================================
# CHART 20: Simpson's Paradox Validation
# ================================================================
print("\n--- CHART 20: Simpson's Paradox ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, data, title, mk_str in [
    (ax1, bilat_annual, "Bilateral only", f"tau={mk_b.Tau:.3f}, p={mk_b.p:.3f}" if 'mk_b' in dir() else ""),
    (ax2, multi_annual, "Multi-bloc only", f"tau={mk_m.Tau:.3f}, p={mk_m.p:.3f}" if 'mk_m' in dir() else ""),
]:
    valid_mask = [not np.isnan(x) for x in data]
    valid_years = [y for y, m in zip(years, valid_mask) if m]
    valid_data = [d for d, m in zip(data, valid_mask) if m]
    
    ax.plot(valid_years, valid_data, "o-", color="#2E7D32", markersize=5, linewidth=1.5, label="Within-stratum")
    
    # Aggregate trend (faint dashed)
    agg_valid_mask = [not np.isnan(x) for x in agg_annual]
    agg_v_years = [y for y, m in zip(years, agg_valid_mask) if m]
    agg_v_data = [d for d, m in zip(agg_annual, agg_valid_mask) if m]
    ax.plot(agg_v_years, agg_v_data, "--", color="#9E9E9E", linewidth=1, alpha=0.6, label="Aggregate")
    
    # Linear trend line
    if valid_data:
        z = np.polyfit(valid_years, valid_data, 1)
        p = np.poly1d(z)
        ax.plot(valid_years, p(valid_years), "--", color="#E53935", linewidth=1.5, alpha=0.7)
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Year")
    if mk_str:
        ax.text(0.05, 0.95, mk_str, transform=ax.transAxes, fontsize=9,
                va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(fontsize=8, loc="lower right")
    clean_axes(ax)

ax1.set_ylabel("GH + Dual First Author (%)")
fig.suptitle("Within-stratum temporal trends in Ghanaian first authorship", fontsize=14, fontweight="bold")
plt.tight_layout()
save_chart(fig, "chart20_simpsons_paradox")

# ================================================================
# 7.8: Bilateral-only country comparison
# ================================================================
print("\n--- 7.8 Bilateral-Only Country Comparison ---")

focus_countries = {"CN": "China", "IN": "India", "ZA": "South Africa", "BR": "Brazil"}
bilat_western = works[(works["is_bilateral"] == 1) & (works["partner_bloc"] == "Western")]
bilat_western_a = authorships[authorships["work_id"].isin(set(bilat_western["work_id"]))]
western_rates = compute_leadership_pcts(bilat_western, bilat_western_a, "BILATERAL Western")

for code, name in focus_countries.items():
    bilat_country = bilateral[bilateral["countries"].str.contains(code, na=False)]
    bilat_country_a = authorships[authorships["work_id"].isin(set(bilat_country["work_id"]))]
    compute_leadership_pcts(bilat_country, bilat_country_a, f"BILATERAL {name}")

# ================================================================
# 7.9: Ghanaian-only regression
# ================================================================
print("\n--- 7.9 Ghanaian-Only Regression ---")

first = authorships[authorships["author_position"] == "first"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
first.columns = ["work_id", "first_cat"]
last = authorships[authorships["author_position"] == "last"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
last.columns = ["work_id", "last_cat"]
corr = authorships[authorships["is_corresponding_combined"] == True].drop_duplicates("work_id")[["work_id", "affiliation_category"]]
corr.columns = ["work_id", "corr_cat"]

reg_gh = works.merge(first, on="work_id", how="left")
reg_gh = reg_gh.merge(last, on="work_id", how="left")
reg_gh = reg_gh.merge(corr, on="work_id", how="left")

# Collapse small blocs
small_blocs = reg_gh["partner_bloc"].value_counts()[reg_gh["partner_bloc"].value_counts() < 50].index
reg_gh["partner_bloc_reg"] = reg_gh["partner_bloc"].replace({b: "Other" for b in small_blocs})

bloc_d = pd.get_dummies(reg_gh["partner_bloc_reg"], prefix="bloc", drop_first=False)
field_d = pd.get_dummies(reg_gh["field_reg"], prefix="field", drop_first=False)

bloc_ref = "bloc_African" if "bloc_African" in bloc_d.columns else bloc_d.columns[0]
field_ref = "field_Health Sciences" if "field_Health Sciences" in field_d.columns else field_d.columns[0]

X_gh = pd.concat([
    reg_gh[["author_count", "year_centered", "covid_era", "has_funding_bool", "is_oa_bool"]].astype(float),
    bloc_d[[c for c in bloc_d.columns if c != bloc_ref]],
    field_d[[c for c in field_d.columns if c != field_ref]],
], axis=1).astype(float)
X_gh = add_constant(X_gh)

for pos, col in [("first", "first_cat"), ("last", "last_cat"), ("corresponding", "corr_cat")]:
    y = (reg_gh[col] == "Ghanaian").astype(int)  # Ghanaian ONLY, not Dual
    mask = X_gh.notna().all(axis=1) & y.notna()
    try:
        model = Logit(y[mask], X_gh[mask]).fit(disp=0, cov_type='HC1')
        print(f"\n  Ghanaian-Only {pos.title()} Authorship:")
        print(f"    N={mask.sum():,}, AIC={model.aic:.1f}, Pseudo-R2={model.prsquared:.4f}")
        for var in ["year_centered", "covid_era", "has_funding_bool", "is_oa_bool"]:
            if var in model.params.index:
                or_val = np.exp(model.params[var])
                ci = np.exp(model.conf_int().loc[var])
                p = model.pvalues[var]
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                print(f"    {var}: OR={or_val:.4f} [{ci.iloc[0]:.4f}, {ci.iloc[1]:.4f}] p={p:.4f} {sig}")
    except Exception as e:
        print(f"  {pos.title()}: regression failed: {e}")

# ================================================================
# 7.10: Robustness table
# ================================================================
print("\n--- 7.10 Robustness Summary Table ---")

try:
    all_reg = pd.read_csv(OUTPUT_DIR / "regression_results.csv")
    
    key_vars = ["year_centered", "covid_era", "has_funding_int", "is_oa_int"]
    robustness_rows = []
    
    for model_key in all_reg["Model"].unique():
        model_data = all_reg[all_reg["Model"] == model_key]
        for var in key_vars:
            row = model_data[model_data["Variable"] == var]
            if len(row) > 0:
                robustness_rows.append({
                    "Model": model_key,
                    "Variable": var,
                    "OR": round(row["OR"].values[0], 4),
                    "CI_lo": round(row["CI_lo"].values[0], 4),
                    "CI_hi": round(row["CI_hi"].values[0], 4),
                    "p_value": round(row["p_value"].values[0], 4),
                })
    
    rob_df = pd.DataFrame(robustness_rows)
    rob_pivot = rob_df.pivot_table(index="Variable", columns="Model", values="OR", aggfunc="first")
    print(rob_pivot.to_string())
    rob_df.to_csv(OUTPUT_DIR / "robustness_summary.csv", index=False)
except Exception as e:
    print(f"  Error: {e}")

print("\nPhase 7 complete.")
