"""
PHASE 8: ADDITIONAL ANALYSES
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

print("=" * 80)
print("PHASE 8: ADDITIONAL ANALYSES")
print("=" * 80)

# ================================================================
# 8.1: OA Decomposition
# ================================================================
print("\n--- 8.1 OA Decomposition ---")

first = authorships[authorships["author_position"] == "first"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
first.columns = ["work_id", "first_cat"]
works_w_first = works.merge(first, on="work_id", how="left")
works_w_first["is_oa_clean"] = works_w_first["is_oa"].apply(to_bool)

cats = ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]
print("  OA rate by first-author affiliation:")
for cat in cats:
    sub = works_w_first[works_w_first["first_cat"] == cat]
    oa_rate = 100 * sub["is_oa_clean"].sum() / len(sub)
    print(f"    {cat}: {oa_rate:.1f}% OA (N={len(sub):,})")

# Chi-square for OA by affiliation
ct = pd.crosstab(works_w_first["first_cat"], works_w_first["is_oa_clean"])
chi2, p_val, dof, exp = stats.chi2_contingency(ct)
print(f"  Chi-square: chi2={chi2:.1f}, p={'<0.001' if p_val < 0.001 else f'{p_val:.4f}'}")

# OA status breakdown
print("\n  OA status by first-author affiliation:")
for cat in cats:
    sub = works_w_first[works_w_first["first_cat"] == cat]
    print(f"    {cat}:")
    print(f"      {sub['oa_status'].value_counts(normalize=True).head(5).to_string()}")

# ================================================================
# 8.2: Dual-Affiliated Composition
# ================================================================
print("\n--- 8.2 Dual-Affiliated Composition ---")

dual_first = authorships[
    (authorships["author_position"] == "first") &
    (authorships["affiliation_category"] == "Dual-affiliated") &
    (authorships["work_id"].isin(set(works["work_id"])))
]

dual_last = authorships[
    (authorships["author_position"] == "last") &
    (authorships["affiliation_category"] == "Dual-affiliated") &
    (authorships["work_id"].isin(set(works["work_id"])))
]

print("  Non-GH countries for Dual-affiliated FIRST authors:")
dual_countries_first = {}
for countries in dual_first["non_gh_institution_names"].dropna():
    # Use author_countries or all_institution_countries instead
    pass

# Better approach: use all_institution_countries
for countries_str in dual_first["all_institution_countries"].dropna():
    for code in str(countries_str).split("|"):
        code = code.strip()
        if code and code != "GH":
            dual_countries_first[code] = dual_countries_first.get(code, 0) + 1

top10_first = sorted(dual_countries_first.items(), key=lambda x: -x[1])[:10]
print("  Top 10 non-GH countries for Dual first authors:")
for code, count in top10_first:
    print(f"    {code}: {count}")

# Pre/post COVID comparison
dual_first_pre = dual_first[dual_first["publication_year"] < 2020]
dual_first_post = dual_first[dual_first["publication_year"] >= 2020]

print("\n  Dual first authors - Pre-COVID vs Post-COVID:")
for label, sub in [("Pre-COVID", dual_first_pre), ("Post-COVID", dual_first_post)]:
    countries = {}
    for cs in sub["all_institution_countries"].dropna():
        for code in str(cs).split("|"):
            code = code.strip()
            if code and code != "GH":
                countries[code] = countries.get(code, 0) + 1
    top5 = sorted(countries.items(), key=lambda x: -x[1])[:5]
    print(f"  {label} (N={len(sub):,}): {', '.join([f'{c}({n})' for c, n in top5])}")

# Same for last authors
print("\n  Non-GH countries for Dual-affiliated LAST authors:")
dual_countries_last = {}
for countries_str in dual_last["all_institution_countries"].dropna():
    for code in str(countries_str).split("|"):
        code = code.strip()
        if code and code != "GH":
            dual_countries_last[code] = dual_countries_last.get(code, 0) + 1

top10_last = sorted(dual_countries_last.items(), key=lambda x: -x[1])[:10]
for code, count in top10_last:
    print(f"    {code}: {count}")

# ================================================================
# 8.3: Corresponding Author Coverage by Year
# ================================================================
print("\n--- 8.3 Corresponding Author Coverage by Year ---")

corr_coverage = []
for year in range(2000, 2026):
    yr_w = works[works["publication_year"] == year]
    yr_a = authorships[authorships["work_id"].isin(set(yr_w["work_id"]))]
    has_corr = yr_a[yr_a["is_corresponding_combined"] == True]["work_id"].nunique()
    total_yr = len(yr_w)
    pct = 100 * has_corr / total_yr if total_yr > 0 else 0
    corr_coverage.append({"Year": year, "N_with_corr": has_corr, "N_total": total_yr, "Pct": round(pct, 1)})
    flag = " *** LOW ***" if pct < 50 else ""
    print(f"  {year}: {has_corr}/{total_yr} ({pct:.1f}%){flag}")

corr_cov_df = pd.DataFrame(corr_coverage)
corr_cov_df.to_csv(OUTPUT_DIR / "corresponding_coverage_by_year.csv", index=False)

# Check if any year below 50%
low_years = corr_cov_df[corr_cov_df["Pct"] < 50]
if len(low_years) > 0:
    print(f"\n  WARNING: {len(low_years)} years have <50% corresponding author coverage!")
    print(low_years.to_string(index=False))
else:
    print("\n  All years have >=50% corresponding author coverage.")

print("\nPhase 8 complete.")
