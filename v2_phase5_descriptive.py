"""
Phase 5 (v2): Descriptive Analysis
====================================
Generates all descriptive statistics, charts, and tables.

Key improvements:
  1. Paper-level random-assignment null model (replaces chi-square)
  2. Effect sizes (Cohen's h, Cramer's V) alongside all p-values
  3. Reduced to 8 primary charts + supplementary
  4. Corrected PRISMA flow (includes retracted/paratext exclusion)

Outputs:
  analysis_results/ - all charts and CSVs
"""

import pandas as pd
import numpy as np
import json, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy import stats
import pymannkendall as mk

from utils import (
    setup_plot_style, save_chart, get_leadership, load_study_data,
    to_bool, wilson_ci, cohens_h, cramers_v, random_assignment_test,
    clean_fwci, RESULTS
)

warnings.filterwarnings("ignore")
setup_plot_style()

print("=" * 70)
print("PHASE 5 (v2): DESCRIPTIVE ANALYSIS")
print("=" * 70)

# -- Load data -----------------------------------------------------------------
works, authorships = load_study_data()
leadership = get_leadership(works, authorships)
df = works.merge(leadership, on="work_id", how="left")
N = len(df)

prisma = json.load(open(RESULTS / "v2_prisma_numbers.json"))
print(f"Study set: {N:,} works")

# ==============================================================================
# CHART 1: PRISMA Flow Diagram
# ==============================================================================
print("\n1. PRISMA Flow Diagram...")

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

boxes = [
    (0.5, 0.95, f"Records identified\nvia OpenAlex\n(n = {prisma['total_openalex']:,})"),
    (0.5, 0.82, f"Biomedical records\n(n = {prisma['total_biomedical']:,})"),
    (0.5, 0.69, f"Within study period\n2000-2025\n(n = {prisma['within_study_period']:,})"),
    (0.5, 0.56, f"International\ncollaborations\n(n = {prisma['international_collabs']:,})"),
    (0.5, 0.43, f"Multi-authored\npapers\n(n = {prisma['after_multi_author']:,})"),
    (0.5, 0.30, f"After quality\nexclusions\n(n = {prisma['after_multi_author'] - prisma['excluded_retracted'] - prisma['excluded_paratext']:,})"),
    (0.5, 0.12, f"FINAL STUDY SET\n(n = {prisma['final_study_set']:,})"),
]

exclusions = [
    (0.87, 0.88, f"Excluded:\nnon-biomedical\n(n = {prisma['total_openalex'] - prisma['total_biomedical']:,})"),
    (0.87, 0.75, f"Excluded:\noutside 2000-2025\n(n = {prisma['excluded_outside_years']:,})"),
    (0.87, 0.62, f"Excluded:\ndomestic only\n(n = {prisma['excluded_domestic_only']:,})"),
    (0.87, 0.49, f"Excluded:\nsingle author\n(n = {prisma['excluded_single_author']:,})"),
    (0.87, 0.36, f"Excluded:\nretracted ({prisma['excluded_retracted']})\nparatext ({prisma['excluded_paratext']})"),
]

for x, y, text in boxes:
    color = "#1e40af" if "FINAL" in text else "#f0f4ff"
    tc = "white" if "FINAL" in text else "black"
    ax.add_patch(plt.Rectangle((x - 0.18, y - 0.045), 0.36, 0.09,
                                facecolor=color, edgecolor="#1e40af",
                                linewidth=1.5, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5,
            fontweight="bold" if "FINAL" in text else "normal", color=tc, zorder=3)

for x, y, text in exclusions:
    ax.add_patch(plt.Rectangle((x - 0.12, y - 0.035), 0.24, 0.07,
                                facecolor="#fef2f2", edgecolor="#dc2626",
                                linewidth=1, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
            color="#991b1b", zorder=3)

# Arrows
for i in range(len(boxes) - 1):
    ax.annotate("", xy=(0.5, boxes[i+1][1] + 0.045),
                xytext=(0.5, boxes[i][1] - 0.045),
                arrowprops=dict(arrowstyle="->", color="#1e40af", lw=1.5))

for i, (_, ey, _) in enumerate(exclusions):
    by = boxes[i][1]
    ax.annotate("", xy=(0.75, ey), xytext=(0.68, by),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))

ax.set_title("PRISMA Flow Diagram", fontsize=14, fontweight="bold", pad=10)
save_chart(fig, "chart01_prisma_flow")

# ==============================================================================
# CHART 2: Publications by Year
# ==============================================================================
print("2. Publications by Year...")

yearly = df.groupby("publication_year").size().reset_index(name="count")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(yearly["publication_year"], yearly["count"], color="#2563eb", alpha=0.85, width=0.8)
ax.set_xlabel("Publication Year")
ax.set_ylabel("Number of Publications")
ax.set_title("International Biomedical Collaborations Involving Ghana, 2000-2025")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
save_chart(fig, "chart02_publications_by_year")

# ==============================================================================
# CHART 3: Overall Leadership Proportions
# ==============================================================================
print("3. Overall Leadership Proportions...")

# Compute proportions with Wilson CIs
positions = ["first", "last", "corr"]
categories = ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]
table1_rows = []

for pos in positions:
    col = f"{pos}_cat"
    for cat in categories:
        count = (df[col] == cat).sum()
        total = df[col].notna().sum()
        pct = count / total * 100
        lo, hi = wilson_ci(count, total)
        table1_rows.append({
            "Position": pos.capitalize() if pos != "corr" else "Corresponding",
            "Category": cat,
            "Count": count,
            "Total": total,
            "Percentage": round(pct, 1),
            "CI_lower": round(lo * 100, 1),
            "CI_upper": round(hi * 100, 1),
        })

table1 = pd.DataFrame(table1_rows)
table1.to_csv(RESULTS / "table1_leadership_proportions.csv", index=False)

# Random-assignment null model
print("   Computing paper-level random-assignment null model...")
for pos in ["first", "last", "corr"]:
    ra = random_assignment_test(works, authorships, pos if pos != "corr" else "corresponding")
    print(f"   {pos}: expected={ra['expected']:.1f}, observed={ra['observed']}, "
          f"ratio={ra['ratio']:.3f}, z={ra['z_stat']:.2f}, p={ra['p_value']:.4f} "
          f"({ra['direction']}-represented)")
    table1_rows.append({
        "Position": pos.capitalize() if pos != "corr" else "Corresponding",
        "Category": "Random assignment null",
        "Count": ra["observed"],
        "Total": ra["n_papers"],
        "Percentage": round(ra["ratio"] * 100, 1),
        "CI_lower": round(ra["expected"], 1),
        "CI_upper": round(ra["z_stat"], 2),
    })

# Bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
pos_labels = {"first": "First Author", "last": "Last Author", "corr": "Corresponding Author"}
colors = {"Ghanaian": "#059669", "Dual-affiliated": "#2563eb", "Non-Ghanaian": "#dc2626"}

for i, pos in enumerate(positions):
    ax = axes[i]
    col = f"{pos}_cat"
    data = df[col].value_counts(normalize=True) * 100
    for cat in categories:
        val = data.get(cat, 0)
        bar = ax.bar(cat, val, color=colors[cat], alpha=0.85)
        ax.text(bar[0].get_x() + bar[0].get_width()/2, val + 1,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(pos_labels[pos], fontweight="bold")
    ax.set_ylabel("Percentage (%)" if i == 0 else "")
    ax.set_ylim(0, 85)
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("Authorship Position by Affiliation Category", fontsize=14, fontweight="bold")
plt.tight_layout()
save_chart(fig, "chart03_leadership_proportions")

# ==============================================================================
# CHART 4: Bilateral vs Multi-bloc
# ==============================================================================
print("4. Bilateral vs Multi-bloc Comparison...")

bilateral = df[df["is_bilateral"] == True]
multibloc = df[df["partner_bloc"] == "Multi-bloc"]

table2_rows = []
for pos in positions:
    col = f"gh_{pos}"
    for label, subset in [("Bilateral", bilateral), ("Multi-bloc", multibloc)]:
        n_gh = subset[col].sum()
        n_tot = len(subset)
        pct = n_gh / n_tot * 100
        lo, hi = wilson_ci(n_gh, n_tot)
        table2_rows.append({
            "Position": pos.capitalize() if pos != "corr" else "Corresponding",
            "Partnership": label,
            "GH_count": n_gh,
            "Total": n_tot,
            "Percentage": round(pct, 1),
            "CI_lower": round(lo * 100, 1),
            "CI_upper": round(hi * 100, 1),
        })

    # Effect size
    p1 = bilateral[f"gh_{pos}"].mean()
    p2 = multibloc[f"gh_{pos}"].mean()
    h = cohens_h(p1, p2)
    table2_rows.append({
        "Position": pos.capitalize() if pos != "corr" else "Corresponding",
        "Partnership": "Effect size (Cohen's h)",
        "GH_count": 0, "Total": 0,
        "Percentage": round(h, 3),
        "CI_lower": 0, "CI_upper": 0,
    })

table2 = pd.DataFrame(table2_rows)
table2.to_csv(RESULTS / "table2_bilateral_consortium.csv", index=False)

# Chart
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(3)
w = 0.35
bil_vals = [bilateral[f"gh_{p}"].mean() * 100 for p in positions]
mul_vals = [multibloc[f"gh_{p}"].mean() * 100 for p in positions]
b1 = ax.bar(x - w/2, bil_vals, w, color="#059669", alpha=0.85, label="Bilateral")
b2 = ax.bar(x + w/2, mul_vals, w, color="#dc2626", alpha=0.85, label="Multi-bloc")
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["First Author", "Last Author", "Corresponding"])
ax.set_ylabel("GH/Dual Leadership (%)")
ax.set_title("Ghanaian Leadership: Bilateral vs Multi-bloc Partnerships")
ax.legend()
ax.set_ylim(0, 70)
save_chart(fig, "chart04_bilateral_multibloc")

# ==============================================================================
# CHART 5: Annual Trends (First Authorship) + Mann-Kendall
# ==============================================================================
print("5. Annual Trends...")

annual = df.groupby("publication_year").agg(
    n=("work_id", "size"),
    gh_first=("gh_first", "sum"),
    gh_last=("gh_last", "sum"),
).reset_index()
annual["first_pct"] = annual["gh_first"] / annual["n"] * 100
annual["last_pct"] = annual["gh_last"] / annual["n"] * 100

# Mann-Kendall
mk_first = mk.original_test(annual["first_pct"])
mk_last = mk.original_test(annual["last_pct"])
print(f"   First auth MK: trend={mk_first.trend}, tau={mk_first.Tau:.3f}, p={mk_first.p:.4f}")
print(f"   Last auth MK:  trend={mk_last.trend}, tau={mk_last.Tau:.3f}, p={mk_last.p:.4f}")

annual.to_csv(RESULTS / "annual_leadership.csv", index=False)

# Bilateral vs Multi-bloc trends (for Simpson's paradox)
bil_annual = bilateral.groupby("publication_year").agg(
    n=("work_id", "size"), gh_first=("gh_first", "sum")).reset_index()
bil_annual["pct"] = bil_annual["gh_first"] / bil_annual["n"] * 100

mul_annual = multibloc.groupby("publication_year").agg(
    n=("work_id", "size"), gh_first=("gh_first", "sum")).reset_index()
mul_annual["pct"] = mul_annual["gh_first"] / mul_annual["n"] * 100

mk_bil = mk.original_test(bil_annual["pct"])
mk_mul = mk.original_test(mul_annual["pct"])
print(f"   Bilateral MK:  trend={mk_bil.trend}, tau={mk_bil.Tau:.3f}, p={mk_bil.p:.4f}")
print(f"   Multi-bloc MK: trend={mk_mul.trend}, tau={mk_mul.Tau:.3f}, p={mk_mul.p:.4f}")

# Chart: 3-panel
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Overall
axes[0].plot(annual["publication_year"], annual["first_pct"], "o-",
             color="#2563eb", markersize=4, lw=1.5)
z = np.polyfit(annual["publication_year"], annual["first_pct"], 1)
axes[0].plot(annual["publication_year"],
             np.polyval(z, annual["publication_year"]), "--", color="#dc2626", alpha=0.5)
axes[0].set_title(f"Overall (MK: {mk_first.trend}, p={mk_first.p:.3f})")
axes[0].set_ylabel("GH/Dual First Authorship (%)")

# Bilateral
axes[1].plot(bil_annual["publication_year"], bil_annual["pct"], "o-",
             color="#059669", markersize=4, lw=1.5)
z = np.polyfit(bil_annual["publication_year"], bil_annual["pct"], 1)
axes[1].plot(bil_annual["publication_year"],
             np.polyval(z, bil_annual["publication_year"]), "--", color="#dc2626", alpha=0.5)
axes[1].set_title(f"Bilateral (MK: {mk_bil.trend}, p={mk_bil.p:.3f})")

# Multi-bloc
axes[2].plot(mul_annual["publication_year"], mul_annual["pct"], "o-",
             color="#dc2626", markersize=4, lw=1.5)
z = np.polyfit(mul_annual["publication_year"], mul_annual["pct"], 1)
axes[2].plot(mul_annual["publication_year"],
             np.polyval(z, mul_annual["publication_year"]), "--", color="#dc2626", alpha=0.5)
axes[2].set_title(f"Multi-bloc (MK: {mk_mul.trend}, p={mk_mul.p:.3f})")

for ax in axes:
    ax.set_xlabel("Year")
    ax.set_ylim(0, 75)
fig.suptitle("Temporal Trends in GH/Dual First Authorship", fontsize=14, fontweight="bold")
plt.tight_layout()
save_chart(fig, "chart05_annual_trends")

# ==============================================================================
# CHART 6: Simpson's Paradox Visualization
# ==============================================================================
print("6. Simpson's Paradox...")

# Composition shift
comp = df.groupby("publication_year")["is_bilateral"].mean().reset_index()
comp.columns = ["year", "bilateral_share"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Trends by type
axes[0].plot(bil_annual["publication_year"], bil_annual["pct"], "o-",
             color="#059669", markersize=4, lw=1.5, label="Bilateral")
axes[0].plot(mul_annual["publication_year"], mul_annual["pct"], "s-",
             color="#dc2626", markersize=4, lw=1.5, label="Multi-bloc")
axes[0].plot(annual["publication_year"], annual["first_pct"], "^--",
             color="#6b7280", markersize=4, lw=1, label="Overall (aggregate)")
axes[0].set_ylabel("GH/Dual First Authorship (%)")
axes[0].set_xlabel("Year")
axes[0].set_title("A: Trends by Partnership Type")
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, 75)

# Panel B: Composition shift
axes[1].stackplot(comp["year"],
                  comp["bilateral_share"] * 100,
                  (1 - comp["bilateral_share"]) * 100,
                  labels=["Bilateral", "Multi-bloc"],
                  colors=["#059669", "#dc2626"], alpha=0.7)
axes[1].set_ylabel("Share of Publications (%)")
axes[1].set_xlabel("Year")
axes[1].set_title("B: Compositional Shift over Time")
axes[1].legend(loc="center right", fontsize=9)
axes[1].set_ylim(0, 100)

fig.suptitle("Simpson's Paradox: Compositional Masking of Temporal Gains",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_chart(fig, "chart06_simpsons_paradox")

# ==============================================================================
# SUPPLEMENTARY CHARTS (saved with supp_ prefix)
# ==============================================================================
print("7. Supplementary charts...")

# -- Leadership by partner bloc --
print("   7a. Leadership by bloc...")
bloc_stats = []
for bloc in df["partner_bloc"].unique():
    sub = df[df["partner_bloc"] == bloc]
    if len(sub) < 10:
        continue
    for pos in ["first", "last"]:
        n_gh = sub[f"gh_{pos}"].sum()
        pct = n_gh / len(sub) * 100
        lo, hi = wilson_ci(n_gh, len(sub))
        bloc_stats.append({
            "Bloc": bloc, "Position": pos, "N": len(sub),
            "GH_pct": pct, "CI_lo": lo*100, "CI_hi": hi*100
        })
bloc_df = pd.DataFrame(bloc_stats)
bloc_df.to_csv(RESULTS / "leadership_by_bloc.csv", index=False)

# -- Leadership by funding --
print("   7b. Leadership by funding...")
fund_stats = []
for fund in df["funder_category"].unique():
    sub = df[df["funder_category"] == fund]
    for pos in ["first", "last"]:
        n_gh = sub[f"gh_{pos}"].sum()
        pct = n_gh / len(sub) * 100
        lo, hi = wilson_ci(n_gh, len(sub))
        fund_stats.append({
            "Funding": fund, "Position": pos, "N": len(sub),
            "GH_pct": pct, "CI_lo": lo*100, "CI_hi": hi*100
        })
fund_df = pd.DataFrame(fund_stats)
fund_df.to_csv(RESULTS / "leadership_by_funding.csv", index=False)

# -- Leadership by field --
print("   7c. Leadership by field...")
field_stats = []
for field in df["field_reg"].unique():
    sub = df[df["field_reg"] == field]
    for pos in ["first", "last"]:
        n_gh = sub[f"gh_{pos}"].sum()
        pct = n_gh / len(sub) * 100
        lo, hi = wilson_ci(n_gh, len(sub))
        field_stats.append({
            "Field": field, "Position": pos, "N": len(sub),
            "GH_pct": pct, "CI_lo": lo*100, "CI_hi": hi*100
        })
field_df = pd.DataFrame(field_stats)
field_df.to_csv(RESULTS / "leadership_by_field.csv", index=False)

# -- Time period leadership --
print("   7d. Time period leadership...")
period_stats = []
for period in df["time_period"].dropna().unique():
    sub = df[df["time_period"] == period]
    for pos in ["first", "last"]:
        n_gh = sub[f"gh_{pos}"].sum()
        pct = n_gh / len(sub) * 100
        lo, hi = wilson_ci(n_gh, len(sub))
        period_stats.append({
            "Period": period, "Position": pos, "N": len(sub),
            "GH_pct": pct, "CI_lo": lo*100, "CI_hi": hi*100
        })
period_df = pd.DataFrame(period_stats)
period_df.to_csv(RESULTS / "time_period_leadership.csv", index=False)

# -- FWCI analysis --
print("   7e. FWCI analysis...")
fwci = clean_fwci(df["fwci"])
df["fwci_clean"] = fwci
fwci_valid = df[df["fwci_clean"].notna()]
print(f"   FWCI valid: {len(fwci_valid):,} / {N:,}")
print(f"   FWCI median: {fwci_valid['fwci_clean'].median():.2f}")
print(f"   FWCI IQR: {fwci_valid['fwci_clean'].quantile(0.25):.2f} - {fwci_valid['fwci_clean'].quantile(0.75):.2f}")

# -- Descriptive summary JSON --
print("   7f. Saving descriptive summary...")
desc_summary = {
    "n_works": N,
    "n_authorships": int(prisma["total_authorships"]),
    "n_unique_authors": int(prisma["unique_authors"]),
    "n_countries": int(prisma["unique_countries"]),
    "median_team_size": int(df["author_count"].median()),
    "iqr_team_size_lo": int(df["author_count"].quantile(0.25)),
    "iqr_team_size_hi": int(df["author_count"].quantile(0.75)),
    "mean_team_size": round(df["author_count"].mean(), 1),
    "year_range": f"{int(df['publication_year'].min())}-{int(df['publication_year'].max())}",
    "pct_articles": round((df["type"] == "article").mean() * 100, 1),
    "pct_reviews": round((df["type"] == "review").mean() * 100, 1),
    "pct_preprints": round((df["type"] == "preprint").mean() * 100, 1),
    "pct_oa": round(df["is_oa_bool"].mean() * 100, 1),
    "pct_funded": round(df["has_funding_bool"].mean() * 100, 1),
    "fwci_median": round(fwci_valid["fwci_clean"].median(), 2),
    "fwci_iqr_lo": round(fwci_valid["fwci_clean"].quantile(0.25), 2),
    "fwci_iqr_hi": round(fwci_valid["fwci_clean"].quantile(0.75), 2),
    "fwci_missing_pct": round((1 - len(fwci_valid) / N) * 100, 1),
    "mk_first_trend": mk_first.trend,
    "mk_first_tau": round(mk_first.Tau, 3),
    "mk_first_p": round(mk_first.p, 4),
    "mk_last_trend": mk_last.trend,
    "mk_last_tau": round(mk_last.Tau, 3),
    "mk_last_p": round(mk_last.p, 4),
    "mk_bilateral_trend": mk_bil.trend,
    "mk_bilateral_tau": round(mk_bil.Tau, 3),
    "mk_bilateral_p": round(mk_bil.p, 4),
    "mk_multibloc_trend": mk_mul.trend,
    "mk_multibloc_tau": round(mk_mul.Tau, 3),
    "mk_multibloc_p": round(mk_mul.p, 4),
    "first_is_corr_pct": prisma["first_is_corr_pct"],
}
with open(RESULTS / "descriptive_summary.json", "w") as f:
    json.dump(desc_summary, f, indent=2)

print(f"\n{'='*70}")
print("PHASE 5 (v2) COMPLETE")
print(f"  Charts: 6 primary + supplementary CSVs")
print(f"  Tables: table1, table2, bloc, funding, field, period")
print(f"{'='*70}")
