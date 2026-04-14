"""
PHASE 5: DESCRIPTIVE ANALYSIS + CHARTS 1-13, 17-19
Ghana Bibliometric Study
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

# ---- Setup ----
works = pd.read_parquet("analysis_results/intermediate/works.parquet")
authorships = pd.read_parquet("analysis_results/intermediate/authorships.parquet")

with open("analysis_results/prisma_numbers.json") as f:
    prisma = json.load(f)

COLORS = {
    "Ghanaian": "#2E7D32",
    "Dual-affiliated": "#F9A825",
    "Non-Ghanaian": "#1565C0",
    "total": "#616161",
}
MARKERS = {"Ghanaian": "o", "Dual-affiliated": "^", "Non-Ghanaian": "s"}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = Path("analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

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
    if total == 0:
        return 0, 0, 0
    pct = count / total
    lo, hi = proportion_confint(count, total, alpha=alpha, method='wilson')
    return pct * 100, lo * 100, hi * 100

# ---- Helper: get leadership flags ----
def get_leadership_data(works_df, auths_df):
    """Returns work-level leadership flags."""
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

leadership = get_leadership_data(works, authorships)
works = works.merge(leadership[["work_id", "first_cat", "last_cat", "corr_cat"]], on="work_id", how="left")

N = len(works)

# ================================================================
# CHART 1: PRISMA Flow Diagram
# ================================================================
print("\n--- CHART 1: PRISMA Flow Diagram ---")

import matplotlib.path as mpath

fig, ax = plt.subplots(figsize=(10, 11))
ax.set_xlim(0, 12)
ax.set_ylim(-1, 13)
ax.axis("off")

def draw_box(ax, x, y, w, h, text, color="white", edge="black", fontsize=10, weight="normal"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="square,pad=0.1", facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, wrap=True)

def draw_arrow(ax, pnt_from, pnt_to):
    ax.annotate("", xy=pnt_to, xytext=pnt_from,
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5))

def draw_line(ax, pnt_from, pnt_to):
    path_data = [
        (mpath.Path.MOVETO, pnt_from),
        (mpath.Path.LINETO, pnt_to)
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor='#555555', lw=1.5)
    ax.add_patch(patch)

def draw_branching_arrows(ax, y_top, y_bot, x_main, x_exc, w_exc_box):
    y_mid = (y_top + y_bot) / 2
    # Vertical line top to mid
    draw_line(ax, (x_main, y_top), (x_main, y_mid))
    # Vertical arrow mid to bot
    draw_arrow(ax, (x_main, y_mid), (x_main, y_bot))
    # Horizontal arrow to right exclusion box
    draw_arrow(ax, (x_main, y_mid), (x_exc - w_exc_box/2, y_mid))

# Phase boxes on the left
phase_x = 1.0
phase_w = 1.5
draw_box(ax, phase_x, 10.5, phase_w, 2.0, "Identification", color="#e8f4f8", edge="#2874A6", weight="bold")
draw_box(ax, phase_x, 7.5, phase_w, 3.0, "Screening", color="#e8f4f8", edge="#2874A6", weight="bold")
draw_box(ax, phase_x, 3.5, phase_w, 4.0, "Eligibility", color="#e8f4f8", edge="#2874A6", weight="bold")
draw_box(ax, phase_x, 0.5, phase_w, 1.0, "Included", color="#e8f4f8", edge="#2874A6", weight="bold")

# Main boxes
main_x = 4.5
main_w = 4.0
main_h = 1.2
main_boxes = [
    (10.5, f"Records identified from\nOpenAlex API\n(n = {prisma['total_openalex']:,})"),
    (8.5, f"Biomedical records\nafter applying domain filter\n(n = {prisma['total_biomedical']:,})"),
    (6.5, f"Records published within\nstudy period (2000-2025)\n(n = {prisma['within_study_period']:,})"),
    (4.5, f"Records featuring international\ncollaboration\n(n = {prisma['international_collabs']:,})"),
    (2.5, f"Records with \n≥2 authors\n(n = {prisma['international_collabs'] - prisma['excluded_single_author']:,})"),
    (0.5, f"Final study set\n(n = {prisma['final_study_set']:,})")
]

# Exclusion boxes
exc_x = 9.5
exc_w = 3.5
exc_h = 1.0
exc_boxes = [
    (9.5, f"Records excluded:\nNon-biomedical fields\n(n = {prisma['excluded_non_biomedical']:,})"),
    (7.5, f"Records excluded:\nPublished outside 2000-2025\n(n = {prisma['excluded_outside_years']:,})"),
    (5.5, f"Records excluded:\nDomestic-only affiliations\n(n = {prisma['excluded_domestic_only']:,})"),
    (3.5, f"Records excluded:\nSingle-author works\n(n = {prisma['excluded_single_author']:,})")
]

for y, text in main_boxes[:-1]:
    draw_box(ax, main_x, y, main_w, main_h, text, color="#EBF5FB", edge="#2874A6")
draw_box(ax, main_x, main_boxes[-1][0], main_w, main_h, main_boxes[-1][1], color="#D5F5E3", edge="#1B7A3D", weight="bold")

for y, text in exc_boxes:
    draw_box(ax, exc_x, y, exc_w, exc_h, text, color="#FDEDEC", edge="#C0392B", fontsize=9)

# Draw branching
for i in range(len(main_boxes)-2): # First 4 transitions have exclusions
    y_top = main_boxes[i][0] - main_h/2
    y_bot = main_boxes[i+1][0] + main_h/2
    draw_branching_arrows(ax, y_top, y_bot, main_x, exc_x, exc_w)

# The last step (to Final study set) is a straight downward arrow
y_top = main_boxes[4][0] - main_h/2
y_bot = main_boxes[5][0] + main_h/2
draw_arrow(ax, (main_x, y_top), (main_x, y_bot))

ax.set_title("PRISMA Flow Diagram", fontsize=16, fontweight="bold", y=1.02)
save_chart(fig, "chart01_prisma_flow")

# ================================================================
# CHART 2: Publications by Year
# ================================================================
print("\n--- CHART 2: Publications by Year ---")

yearly = works.groupby("publication_year").size()
years = yearly.index.values
counts = yearly.values

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(years, counts, color="#616161", alpha=0.85, width=0.8)
ax.axvline(x=2019.5, color="#B71C1C", linestyle="--", linewidth=1.5, alpha=0.7, label="COVID-19 onset")

# Polynomial trend
z = np.polyfit(years, counts, 3)
p = np.poly1d(z)
ax.plot(years, p(years), color="#B71C1C", linestyle="--", linewidth=2, alpha=0.6)

ax.set_xlabel("Publication Year")
ax.set_ylabel("Number of Publications")
ax.set_title(f"International collaborative biomedical publications\ninvolving Ghanaian researchers, 2000-2025 (N={N:,})")
ax.legend(loc="upper left")
clean_axes(ax)
save_chart(fig, "chart02_publications_by_year")

# ================================================================
# 5.1: Table 1 — Overall Leadership Proportions  
# ================================================================
print("\n--- 5.1 Table 1: Overall Leadership Proportions ---")

cats = ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]
positions = ["first", "last", "corresponding"]

table1_data = []
for pos in positions:
    col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
    valid = works[col].dropna()
    total_valid = len(valid)
    for cat in cats:
        count = (valid == cat).sum()
        pct, lo, hi = wilson_ci(count, total_valid)
        table1_data.append({
            "Position": pos.title(),
            "Category": cat,
            "N": count,
            "Total": total_valid,
            "Pct": round(pct, 1),
            "CI_lo": round(lo, 1),
            "CI_hi": round(hi, 1),
        })

table1 = pd.DataFrame(table1_data)
print(table1.to_string(index=False))

# Chi-square goodness-of-fit
print("\nChi-square goodness-of-fit tests:")
overall_dist = authorships["affiliation_category"].value_counts(normalize=True)
for pos in positions:
    col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
    valid = works[col].dropna()
    observed = [int((valid == cat).sum()) for cat in cats]
    expected_pcts = [overall_dist.get(cat, 0) for cat in cats]
    expected = [p * sum(observed) for p in expected_pcts]
    chi2, p_val = stats.chisquare(observed, expected)
    print(f"  {pos.title()}: chi2={chi2:.1f}, p={'<0.001' if p_val < 0.001 else f'{p_val:.4f}'}")

table1.to_csv(OUTPUT_DIR / "table1_leadership_proportions.csv", index=False)

# ================================================================
# CHART 3: Overall Leadership Proportions (grouped bar)
# ================================================================
print("\n--- CHART 3: Overall Leadership Proportions ---")

fig, ax = plt.subplots(figsize=(8, 5))
y_positions_chart = np.arange(len(positions))
bar_height = 0.25

for i, cat in enumerate(cats):
    subset = table1[table1["Category"] == cat]
    values = subset["Pct"].values
    bars = ax.barh(y_positions_chart + i * bar_height, values, bar_height,
                   label=cat, color=COLORS[cat], alpha=0.9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8)

ax.set_yticks(y_positions_chart + bar_height)
ax.set_yticklabels([p.title() for p in positions])
ax.set_xlabel("Percentage (%)")
ax.set_title(f"Research leadership by authorship position (N={N:,})")
ax.legend(loc="lower right")
clean_axes(ax)
save_chart(fig, "chart03_leadership_proportions")

# ================================================================
# 5.2: Team size by first-author category
# ================================================================
print("\n--- 5.2 Team Size by First-Author Category ---")
for cat in cats:
    subset = works[works["first_cat"] == cat]
    print(f"  {cat} as first author: N={len(subset):,}, Mean team={subset['author_count'].mean():.1f}, "
          f"Median={subset['author_count'].median():.0f}")

for cat in cats:
    subset = works[works["last_cat"] == cat]
    print(f"  {cat} as last author: N={len(subset):,}, Mean team={subset['author_count'].mean():.1f}, "
          f"Median={subset['author_count'].median():.0f}")

# ================================================================  
# 5.3: Table 2 — Bilateral vs Consortium  
# ================================================================
print("\n--- 5.3 Table 2: Bilateral vs Consortium ---")

table2_data = []
for structure, label in [(1, "Bilateral"), (0, "Multi-bloc")]:
    subset_w = works[works["is_bilateral"] == structure]
    subset_a = authorships[authorships["work_id"].isin(set(subset_w["work_id"]))]
    lead = get_leadership_data(subset_w, subset_a)
    n_sub = len(subset_w)
    
    for pos in positions:
        col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
        valid = lead[col].dropna()
        gh_dual = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
        pct, lo, hi = wilson_ci(gh_dual, len(valid))
        table2_data.append({
            "Structure": label,
            "Position": pos.title(),
            "N_total": n_sub,
            "GH_Dual_N": gh_dual,
            "Pct": round(pct, 1),
            "CI_lo": round(lo, 1),
            "CI_hi": round(hi, 1),
        })

table2 = pd.DataFrame(table2_data)
print(table2.to_string(index=False))
table2.to_csv(OUTPUT_DIR / "table2_bilateral_consortium.csv", index=False)

# Bilateral/consortium share over time
print("\n  Bilateral vs Multi-bloc share over time:")
TIME_PERIODS = {"2000-2005": (2000,2005), "2006-2010": (2006,2010), 
                "2011-2015": (2011,2015), "2016-2019": (2016,2019), "2020-2025": (2020,2025)}
for period, (s, e) in TIME_PERIODS.items():
    sub = works[(works["publication_year"] >= s) & (works["publication_year"] <= e)]
    bilat = (sub["is_bilateral"] == 1).sum()
    multi = (sub["is_bilateral"] == 0).sum()
    total_p = len(sub)
    print(f"  {period}: Bilateral={bilat} ({100*bilat/total_p:.1f}%), Multi-bloc={multi} ({100*multi/total_p:.1f}%)")

# ================================================================
# CHART 7: Bilateral vs Consortium  
# ================================================================
print("\n--- CHART 7: Bilateral vs Consortium ---")
fig, ax = plt.subplots(figsize=(8, 5))
structures = ["Bilateral", "Multi-bloc"]
pos_colors = {"First": "#2E7D32", "Last": "#1565C0", "Corresponding": "#F9A825"}
x = np.arange(len(structures))
w = 0.25

for i, pos in enumerate(["First", "Last", "Corresponding"]):
    vals = table2[table2["Position"] == pos]["Pct"].values
    bars = ax.bar(x + i*w, vals, w, label=pos, color=list(pos_colors.values())[i], alpha=0.9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=8)

ax.set_xticks(x + w)
ax.set_xticklabels(structures)
ax.set_ylabel("GH + Dual-affiliated (%)")
ax.set_title("Ghanaian leadership by partnership structure")
ax.legend()
clean_axes(ax)
save_chart(fig, "chart07_bilateral_consortium")

# ================================================================
# 5.4: Leadership by Partner Bloc
# ================================================================
print("\n--- 5.4 Leadership by Partner Bloc ---")

bloc_data = []
for bloc in works["partner_bloc"].value_counts().index:
    subset_w = works[works["partner_bloc"] == bloc]
    subset_a = authorships[authorships["work_id"].isin(set(subset_w["work_id"]))]
    lead = get_leadership_data(subset_w, subset_a)
    n_bloc = len(subset_w)
    
    for pos in positions:
        col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
        valid = lead[col].dropna()
        gh_dual = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
        pct, lo, hi = wilson_ci(gh_dual, len(valid))
        bloc_data.append({
            "Bloc": bloc, "N": n_bloc, "Position": pos.title(),
            "GH_Dual_N": gh_dual, "Pct": round(pct, 1),
            "CI_lo": round(lo, 1), "CI_hi": round(hi, 1),
        })

bloc_df = pd.DataFrame(bloc_data)
print(bloc_df.to_string(index=False))
bloc_df.to_csv(OUTPUT_DIR / "leadership_by_bloc.csv", index=False)

# ================================================================
# CHART 8: Leadership by Partner Bloc
# ================================================================
print("\n--- CHART 8: Leadership by Partner Bloc ---")

blocs_ordered = works["partner_bloc"].value_counts().index.tolist()
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(blocs_ordered))
w = 0.25

for i, pos in enumerate(["First", "Last", "Corresponding"]):
    vals = [bloc_df[(bloc_df["Bloc"] == b) & (bloc_df["Position"] == pos)]["Pct"].values[0] for b in blocs_ordered]
    bars = ax.bar(x + i*w, vals, w, label=pos, color=list(pos_colors.values())[i], alpha=0.9)

ax.set_xticks(x + w)
ax.set_xticklabels(blocs_ordered, rotation=30, ha="right")
ax.set_ylabel("GH + Dual-affiliated (%)")
ax.set_title("Ghanaian research leadership by partner bloc")
ax.legend()
clean_axes(ax)
save_chart(fig, "chart08_leadership_by_bloc")

# ================================================================
# 5.5: Country-level analysis
# ================================================================
print("\n--- 5.5 Country-Level Analysis ---")

focus_countries = {"CN": "China", "IN": "India", "ZA": "South Africa", "BR": "Brazil"}
country_data = []

for code, name in focus_countries.items():
    # All papers with at least one co-author from this country
    all_papers = works[works["countries"].str.contains(code, na=False)]
    all_a = authorships[authorships["work_id"].isin(set(all_papers["work_id"]))]
    all_lead = get_leadership_data(all_papers, all_a)
    
    # Bilateral-only
    bilat_papers = all_papers[all_papers["is_bilateral"] == 1]
    bilat_a = authorships[authorships["work_id"].isin(set(bilat_papers["work_id"]))]
    bilat_lead = get_leadership_data(bilat_papers, bilat_a)
    
    for method, papers, lead in [("All papers", all_papers, all_lead), 
                                  ("Bilateral-only", bilat_papers, bilat_lead)]:
        for pos in positions:
            col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
            valid = lead[col].dropna()
            gh_dual = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
            pct, lo, hi = wilson_ci(gh_dual, len(valid))
            country_data.append({
                "Country": name, "Method": method, "N": len(papers),
                "Position": pos.title(), "GH_Dual_Pct": round(pct, 1),
                "CI_lo": round(lo, 1), "CI_hi": round(hi, 1),
            })

country_df = pd.DataFrame(country_data)
print(country_df.to_string(index=False))
country_df.to_csv(OUTPUT_DIR / "country_level_analysis.csv", index=False)

# ================================================================
# 5.6: Pre/Post COVID
# ================================================================
print("\n--- 5.6 Pre/Post COVID ---")

covid_data = []
for era, label in [(0, "Pre-COVID (2000-2019)"), (1, "Post-COVID (2020-2025)")]:
    subset_w = works[works["covid_era"] == era]
    subset_a = authorships[authorships["work_id"].isin(set(subset_w["work_id"]))]
    lead = get_leadership_data(subset_w, subset_a)
    n_era = len(subset_w)
    
    for pos in positions:
        col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
        valid = lead[col].dropna()
        for cat in cats + ["GH+Dual"]:
            if cat == "GH+Dual":
                count = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
            else:
                count = (valid == cat).sum()
            pct, lo, hi = wilson_ci(count, len(valid))
            covid_data.append({
                "Era": label, "Position": pos.title(), "Category": cat,
                "N": count, "Total": len(valid), "Pct": round(pct, 1),
                "CI_lo": round(lo, 1), "CI_hi": round(hi, 1),
            })

covid_df = pd.DataFrame(covid_data)
print(covid_df[covid_df["Category"] != "GH+Dual"].to_string(index=False))

# Chi-square pre vs post
print("\n  Chi-square tests (pre vs post):")
for pos in positions:
    col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
    pre = works[works["covid_era"] == 0][col if col in works.columns else "first_cat"].dropna() if col in works.columns else pd.Series()
    # Simpler approach
    pre_w = works[works["covid_era"] == 0]
    post_w = works[works["covid_era"] == 1]
    pre_lead = get_leadership_data(pre_w, authorships[authorships["work_id"].isin(set(pre_w["work_id"]))])
    post_lead = get_leadership_data(post_w, authorships[authorships["work_id"].isin(set(post_w["work_id"]))])
    
    pre_vals = pre_lead[col].dropna()
    post_vals = post_lead[col].dropna()
    
    ct = pd.DataFrame({
        "Pre": [int((pre_vals == c).sum()) for c in cats],
        "Post": [int((post_vals == c).sum()) for c in cats],
    }, index=cats)
    chi2, p_val, dof, exp = stats.chi2_contingency(ct)
    print(f"  {pos.title()}: chi2={chi2:.1f}, p={'<0.001' if p_val < 0.001 else f'{p_val:.4f}'}")

covid_df.to_csv(OUTPUT_DIR / "pre_post_covid.csv", index=False)

# ================================================================
# CHART 11: Pre vs Post COVID (3-panel)
# ================================================================
print("\n--- CHART 11: Pre vs Post COVID ---")

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
for idx, pos in enumerate(["First", "Last", "Corresponding"]):
    ax = axes[idx]
    sub = covid_df[(covid_df["Position"] == pos) & (covid_df["Category"] != "GH+Dual")]
    x = np.arange(len(cats))
    w_bar = 0.35
    
    pre_vals = sub[sub["Era"].str.contains("Pre")]["Pct"].values
    post_vals = sub[sub["Era"].str.contains("Post")]["Pct"].values
    
    bars1 = ax.bar(x - w_bar/2, pre_vals, w_bar, label="Pre-COVID", color="#9E9E9E", alpha=0.85)
    bars2 = ax.bar(x + w_bar/2, post_vals, w_bar, label="Post-COVID", color="#E53935", alpha=0.85)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", fontsize=7)
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("-")[0] if "-" in c else c[:3] for c in cats], fontsize=8)
    ax.set_title(pos, fontsize=12)
    clean_axes(ax)
    if idx == 0:
        ax.set_ylabel("Percentage (%)")
    if idx == 2:
        ax.legend(fontsize=8)

fig.suptitle("Research leadership before and after COVID-19", fontsize=14, fontweight="bold", y=1.02)
save_chart(fig, "chart11_pre_post_covid")

# ================================================================
# 5.7: Funding and Leadership
# ================================================================
print("\n--- 5.7 Funding and Leadership ---")

fund_data = []
for fcat in ["International (Northern)", "Ghanaian", "Other/Unclassified", "No funding data"]:
    subset_w = works[works["funder_category"] == fcat]
    subset_a = authorships[authorships["work_id"].isin(set(subset_w["work_id"]))]
    lead = get_leadership_data(subset_w, subset_a)
    n_f = len(subset_w)
    
    for pos in positions:
        col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
        valid = lead[col].dropna()
        gh_dual = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
        pct, lo, hi = wilson_ci(gh_dual, len(valid))
        fund_data.append({
            "Funder": fcat, "N": n_f, "Position": pos.title(),
            "GH_Dual_Pct": round(pct, 1), "CI_lo": round(lo, 1), "CI_hi": round(hi, 1),
        })

fund_df = pd.DataFrame(fund_data)
print(fund_df.to_string(index=False))
fund_df.to_csv(OUTPUT_DIR / "leadership_by_funding.csv", index=False)

# ================================================================
# CHART 12: Leadership by Funding Source
# ================================================================
print("\n--- CHART 12: Leadership by Funding ---")

fig, ax = plt.subplots(figsize=(8, 5))
funder_labels = ["International\n(Northern)", "Ghanaian", "Other/\nUnclassified", "No funding\ndata"]
funder_cats = ["International (Northern)", "Ghanaian", "Other/Unclassified", "No funding data"]
x = np.arange(len(funder_cats))
w = 0.25

for i, pos in enumerate(["First", "Last", "Corresponding"]):
    vals = [fund_df[(fund_df["Funder"] == f) & (fund_df["Position"] == pos)]["GH_Dual_Pct"].values[0] for f in funder_cats]
    ax.bar(x + i*w, vals, w, label=pos, color=list(pos_colors.values())[i], alpha=0.9)

ax.set_xticks(x + w)
ax.set_xticklabels(funder_labels, fontsize=9)
ax.set_ylabel("GH + Dual-affiliated (%)")
ax.set_title("Research leadership by funding source")
ax.legend()
clean_axes(ax)
save_chart(fig, "chart12_leadership_by_funding")

# ================================================================
# 5.8: Citation Impact
# ================================================================
print("\n--- 5.8 Citation Impact ---")

works["fwci_clean"] = pd.to_numeric(works["fwci"], errors="coerce")

for cat in cats:
    subset = works[works["first_cat"] == cat]["fwci_clean"].dropna()
    print(f"  {cat} first-authored: Median FWCI={subset.median():.2f}, "
          f"Mean={subset.mean():.2f}, N={len(subset):,}")

# Mann-Whitney U tests
print("\n  Mann-Whitney U tests (FWCI):")
for c1, c2 in [("Ghanaian", "Non-Ghanaian"), ("Dual-affiliated", "Non-Ghanaian"), ("Ghanaian", "Dual-affiliated")]:
    v1 = works[works["first_cat"] == c1]["fwci_clean"].dropna()
    v2 = works[works["first_cat"] == c2]["fwci_clean"].dropna()
    u, p_val = stats.mannwhitneyu(v1, v2, alternative="two-sided")
    print(f"  {c1} vs {c2}: U={u:.0f}, p={'<0.001' if p_val < 0.001 else f'{p_val:.4f}'}")

# Stratified
print("\n  Stratified by bilateral/multi-bloc:")
for structure, label in [(1, "Bilateral"), (0, "Multi-bloc")]:
    sub = works[works["is_bilateral"] == structure]
    for cat in cats:
        subset = sub[sub["first_cat"] == cat]["fwci_clean"].dropna()
        if len(subset) > 0:
            print(f"    {label} - {cat}: Median FWCI={subset.median():.2f}, N={len(subset):,}")

# ================================================================
# CHART 13: Citation Impact Box Plot
# ================================================================
print("\n--- CHART 13: Citation Impact Box Plot ---")

fig, ax = plt.subplots(figsize=(8, 5))
box_data = []
box_labels = []
box_colors_list = []

for cat in cats:
    vals = works[works["first_cat"] == cat]["fwci_clean"].dropna()
    # Cap at 99th percentile for visualization
    cap = vals.quantile(0.99)
    box_data.append(vals[vals <= cap].values)
    box_labels.append(cat)
    box_colors_list.append(COLORS[cat])

bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
for patch, color in zip(bp["boxes"], box_colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for i, cat in enumerate(cats):
    med = works[works["first_cat"] == cat]["fwci_clean"].dropna().median()
    ax.text(i+1, med + 0.1, f"Median: {med:.2f}", ha="center", fontsize=8, fontweight="bold")

ax.axhline(y=1.0, color="#B71C1C", linestyle="--", alpha=0.5, label="World average (FWCI=1.0)")
ax.set_ylabel("Field-Weighted Citation Impact (FWCI)")
ax.set_title("Citation impact by first author affiliation category")
ax.legend(fontsize=8)
clean_axes(ax)
save_chart(fig, "chart13_citation_impact")

# ================================================================
# 5.9: Institutional Variation
# ================================================================
print("\n--- 5.9 Institutional Variation ---")

# Get Ghanaian institution names for first authors
first_auths = authorships[(authorships["author_position"] == "first") & 
                          (authorships["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]))]
first_auths_study = first_auths[first_auths["work_id"].isin(set(works["work_id"]))]

# Parse institution names
inst_counts = {}
for names in first_auths_study["gh_institution_names"].dropna():
    for name in str(names).split("|"):
        name = name.strip()
        if name:
            inst_counts[name] = inst_counts.get(name, 0) + 1

top_insts = sorted(inst_counts.items(), key=lambda x: -x[1])[:10]
print("  Top 10 Ghanaian institutions by first-author volume:")
for inst, count in top_insts:
    print(f"    {inst}: {count}")

# Leadership rates for top 5
inst_leadership = []
for inst_name, _ in top_insts[:10]:
    # Find works where this institution appears in GH author affiliations
    inst_works_ids = set(first_auths_study[first_auths_study["gh_institution_names"].str.contains(inst_name, na=False, regex=False)]["work_id"])
    inst_w = works[works["work_id"].isin(inst_works_ids)]
    inst_a = authorships[authorships["work_id"].isin(inst_works_ids)]
    lead = get_leadership_data(inst_w, inst_a)
    
    for pos in positions:
        col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
        valid = lead[col].dropna()
        gh_dual = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
        pct, lo, hi = wilson_ci(gh_dual, len(valid))
        inst_leadership.append({
            "Institution": inst_name[:40], "N": len(inst_w),
            "Position": pos.title(), "GH_Dual_Pct": round(pct, 1),
        })

inst_df = pd.DataFrame(inst_leadership)
print(inst_df.to_string(index=False))
inst_df.to_csv(OUTPUT_DIR / "institutional_leadership.csv", index=False)

# ================================================================
# CHART 19: Top Institutions
# ================================================================
print("\n--- CHART 19: Top Institutions ---")

top5_names = [n[:35] for n, _ in top_insts[:10]]
fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(top5_names))
h = 0.25

for i, pos in enumerate(["First", "Last", "Corresponding"]):
    vals = []
    for full_name, _ in top_insts[:10]:
        row = inst_df[(inst_df["Institution"] == full_name[:40]) & (inst_df["Position"] == pos)]
        vals.append(row["GH_Dual_Pct"].values[0] if len(row) > 0 else 0)
    ax.barh(y + i*h, vals, h, label=pos, color=list(pos_colors.values())[i], alpha=0.9)

ax.set_yticks(y + h)
ax.set_yticklabels(top5_names, fontsize=8)
ax.set_xlabel("GH + Dual-affiliated (%)")
ax.set_title("Research leadership at top Ghanaian institutions")
ax.legend(loc="lower right")
clean_axes(ax)
save_chart(fig, "chart19_top_institutions")

# ================================================================
# 5.10: Time-Period Stratified Leadership
# ================================================================
print("\n--- 5.10 Time-Period Stratified Leadership ---")

period_data = []
period_order = ["2000-2005", "2006-2010", "2011-2015", "2016-2019", "2020-2025"]
for period in period_order:
    subset_w = works[works["time_period"] == period]
    subset_a = authorships[authorships["work_id"].isin(set(subset_w["work_id"]))]
    lead = get_leadership_data(subset_w, subset_a)
    n_p = len(subset_w)
    
    for pos in positions:
        col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
        valid = lead[col].dropna()
        gh_dual = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
        pct, lo, hi = wilson_ci(gh_dual, len(valid))
        period_data.append({
            "Period": period, "N": n_p, "Position": pos.title(),
            "GH_Dual_Pct": round(pct, 1), "CI_lo": round(lo, 1), "CI_hi": round(hi, 1),
        })

period_df = pd.DataFrame(period_data)
print(period_df.to_string(index=False))
period_df.to_csv(OUTPUT_DIR / "time_period_leadership.csv", index=False)

# ================================================================
# CHARTS 4-6: Temporal Trends (annual line charts)
# ================================================================
print("\n--- CHARTS 4-6: Temporal Trends ---")

years_range = range(2000, 2026)
annual_data = []
for year in years_range:
    yr_w = works[works["publication_year"] == year]
    yr_a = authorships[authorships["work_id"].isin(set(yr_w["work_id"]))]
    lead = get_leadership_data(yr_w, yr_a)
    n_yr = len(yr_w)
    
    for pos in positions:
        col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
        valid = lead[col].dropna()
        for cat in cats:
            count = (valid == cat).sum()
            pct = 100 * count / len(valid) if len(valid) > 0 else 0
            annual_data.append({
                "Year": year, "Position": pos, "Category": cat,
                "Count": count, "Total": len(valid), "Pct": pct,
            })

annual_df = pd.DataFrame(annual_data)
annual_df.to_csv(OUTPUT_DIR / "annual_leadership.csv", index=False)

for chart_num, pos in [(4, "first"), (5, "last"), (6, "corresponding")]:
    fig, ax = plt.subplots(figsize=(8, 5))
    pos_data = annual_df[annual_df["Position"] == pos]
    
    for cat in cats:
        cat_data = pos_data[pos_data["Category"] == cat].sort_values("Year")
        ax.plot(cat_data["Year"], cat_data["Pct"], 
                color=COLORS[cat], marker=MARKERS[cat], markersize=4,
                linewidth=1.5, label=cat, alpha=0.85)
    
    ax.axvline(x=2019.5, color="#B71C1C", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(2019.7, ax.get_ylim()[1]*0.95, "COVID-19", fontsize=8, color="#B71C1C", alpha=0.7)
    
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"{pos.title()} authorship in international collaborations\nby affiliation category, 2000-2025")
    ax.legend(loc="best", fontsize=9)
    clean_axes(ax)
    save_chart(fig, f"chart{chart_num:02d}_{pos}_authorship_trends")

# ================================================================
# CHART 9: Multi-Bloc Growth (stacked area)
# ================================================================
print("\n--- CHART 9: Multi-Bloc Growth ---")

bloc_yearly = works.groupby(["publication_year", "partner_bloc"]).size().unstack(fill_value=0)
bloc_yearly = bloc_yearly.reindex(range(2000, 2026), fill_value=0)

# Order blocs by total volume
bloc_order = bloc_yearly.sum().sort_values(ascending=False).index.tolist()
bloc_colors = {
    "Western": "#1565C0", "Multi-bloc": "#616161", "African": "#2E7D32",
    "East Asian": "#E53935", "South Asian": "#F9A825", "MENA": "#7B1FA2",
    "Latin American": "#FF6F00", "Other": "#90A4AE",
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.stackplot(bloc_yearly.index, [bloc_yearly[b].values for b in bloc_order],
             labels=bloc_order, colors=[bloc_colors.get(b, "#90A4AE") for b in bloc_order],
             alpha=0.85)
ax.set_xlabel("Publication Year")
ax.set_ylabel("Number of Publications")
ax.set_title("Growth of Ghanaian international collaborations by partner bloc, 2000-2025")
ax.legend(loc="upper left", fontsize=8)
clean_axes(ax)
save_chart(fig, "chart09_bloc_growth_stacked")

# ================================================================
# CHART 10: First Author Trends by Bloc (moving average)
# ================================================================
print("\n--- CHART 10: First Author Trends by Bloc ---")

fig, ax = plt.subplots(figsize=(8, 5))
focus_blocs = ["Western", "African", "East Asian", "South Asian"]
bloc_markers = {"Western": "o", "African": "s", "East Asian": "^", "South Asian": "D"}

for bloc in focus_blocs:
    bloc_w = works[works["partner_bloc"] == bloc]
    annual_bloc = []
    for year in years_range:
        yr_w = bloc_w[bloc_w["publication_year"] == year]
        yr_a = authorships[authorships["work_id"].isin(set(yr_w["work_id"]))]
        lead = get_leadership_data(yr_w, yr_a)
        valid = lead["first_cat"].dropna()
        gh_dual = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
        pct = 100 * gh_dual / len(valid) if len(valid) >= 5 else np.nan
        annual_bloc.append(pct)
    
    series = pd.Series(annual_bloc, index=list(years_range))
    ma = series.rolling(3, min_periods=1, center=True).mean()
    ax.plot(ma.index, ma.values, color=bloc_colors.get(bloc, "#616161"),
            marker=bloc_markers.get(bloc, "o"), markersize=4, linewidth=1.5,
            label=bloc, alpha=0.85)

ax.set_xlabel("Publication Year")
ax.set_ylabel("GH + Dual-affiliated First Author (%)")
ax.set_title("Trends in Ghanaian first authorship by partner bloc\n2000-2025 (3-year moving average)")
ax.legend(fontsize=9)
clean_axes(ax)
save_chart(fig, "chart10_first_author_by_bloc")

# ================================================================
# CHART 17: Composition of First Authorship (stacked area)
# ================================================================
print("\n--- CHART 17: Composition Stacked Area ---")

fig, ax = plt.subplots(figsize=(8, 5))
comp_data = {}
for cat in cats:
    cat_annual = annual_df[(annual_df["Position"] == "first") & (annual_df["Category"] == cat)].sort_values("Year")
    comp_data[cat] = cat_annual["Pct"].values

ax.stackplot(list(years_range), 
             [comp_data["Ghanaian"], comp_data["Dual-affiliated"], comp_data["Non-Ghanaian"]],
             labels=cats, colors=[COLORS[c] for c in cats], alpha=0.85)
ax.set_xlabel("Publication Year")
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 100)
ax.set_title("Composition of first authorship in Ghanaian\ninternational collaborations, 2000-2025")
ax.legend(loc="center right", fontsize=9)
clean_axes(ax)
save_chart(fig, "chart17_first_author_composition")

# ================================================================
# CHART 18: Leadership by Field
# ================================================================
print("\n--- CHART 18: Leadership by Field ---")

field_data = []
for field in works["field_reg"].value_counts().index:
    subset_w = works[works["field_reg"] == field]
    subset_a = authorships[authorships["work_id"].isin(set(subset_w["work_id"]))]
    lead = get_leadership_data(subset_w, subset_a)
    n_f = len(subset_w)
    
    for pos in positions:
        col = {"first": "first_cat", "last": "last_cat", "corresponding": "corr_cat"}[pos]
        valid = lead[col].dropna()
        gh_dual = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
        pct, lo, hi = wilson_ci(gh_dual, len(valid))
        field_data.append({
            "Field": field, "N": n_f, "Position": pos.title(),
            "GH_Dual_Pct": round(pct, 1),
        })

field_df = pd.DataFrame(field_data)
print(field_df.to_string(index=False))
field_df.to_csv(OUTPUT_DIR / "leadership_by_field.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 7))
fields_list = works["field_reg"].value_counts().index.tolist()
y = np.arange(len(fields_list))
h = 0.25

for i, pos in enumerate(["First", "Last", "Corresponding"]):
    vals = [field_df[(field_df["Field"] == f) & (field_df["Position"] == pos)]["GH_Dual_Pct"].values[0] for f in fields_list]
    ax.barh(y + i*h, vals, h, label=pos, color=list(pos_colors.values())[i], alpha=0.9)

ax.set_yticks(y + h)
ax.set_yticklabels(fields_list, fontsize=9)
ax.set_xlabel("GH + Dual-affiliated (%)")
ax.set_title("Ghanaian leadership rates by field")
ax.legend(loc="lower right")
clean_axes(ax)
save_chart(fig, "chart18_leadership_by_field")

# ================================================================
# CHART 21: 5-Period Leadership Trajectory
# ================================================================
print("\n--- CHART 21: 5-Period Trajectory ---")

fig, ax = plt.subplots(figsize=(8, 5))
period_midpoints = [2002.5, 2008, 2013, 2017.5, 2022.5]
trajectory_colors = {"First": "#2E7D32", "Last": "#1565C0", "Corresponding": "#F9A825"}

for pos in ["First", "Last", "Corresponding"]:
    vals = period_df[period_df["Position"] == pos].sort_values("Period")["GH_Dual_Pct"].values
    ax.plot(period_midpoints, vals, color=trajectory_colors[pos], marker="o",
            linewidth=2, markersize=8, label=pos)
    for x, v in zip(period_midpoints, vals):
        ax.text(x, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(period_midpoints)
ax.set_xticklabels(period_order, rotation=20, fontsize=9)
ax.set_ylabel("GH + Dual-affiliated (%)")
ax.set_title("Ghanaian leadership trajectory across five periods, 2000-2025")
ax.legend(fontsize=10)
clean_axes(ax)
save_chart(fig, "chart21_period_trajectory")

print("\nPhase 5 complete. All descriptive analyses and charts saved.")
