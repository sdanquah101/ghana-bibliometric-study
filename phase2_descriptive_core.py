"""
Phase 2: Descriptive Analysis & Core Leadership
=================================================
Produces Tables 1-6 and Charts 1-12.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from config import *

# ── Chart style setup ──
plt.rcParams.update({
    "font.family": CHART_FONT,
    "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": CHART_DPI,
    "savefig.bbox": "tight",
})

CAT_ORDER = ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]

def save_chart(fig, name):
    """Save chart as both PNG and SVG."""
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=CHART_DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png and .svg")

def wilson_ci(count, total, alpha=0.05):
    """Wilson confidence interval for a proportion."""
    if total == 0:
        return 0, 0
    lo, hi = proportion_confint(count, total, alpha=alpha, method="wilson")
    return lo, hi

# ═══════════════════════════════════════════════════════════
# LOAD PREPARED DATA
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 2: DESCRIPTIVE ANALYSIS & CORE LEADERSHIP")
print("=" * 60)

print("\nLoading prepared data...")
works = pd.read_parquet(INTERMEDIATE_DIR / "works.parquet")
authorships = pd.read_parquet(INTERMEDIATE_DIR / "authorships.parquet")
try:
    funding = pd.read_parquet(INTERMEDIATE_DIR / "funding.parquet")
except FileNotFoundError:
    funding = pd.DataFrame()
try:
    topics = pd.read_parquet(INTERMEDIATE_DIR / "topics.parquet")
except FileNotFoundError:
    topics = pd.DataFrame()

N = len(works)
print(f"  Works: {N:,}, Authorships: {len(authorships):,}")

# Quick helper
def to_bool(val):
    if pd.isna(val): return False
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() == "true"
    return bool(val)

# ═══════════════════════════════════════════════════════════
# 4.1 TABLE 1: STUDY CHARACTERISTICS
# ═══════════════════════════════════════════════════════════
print("\n--- Table 1: Study Characteristics ---")

rows = []

# Overall
n_auth = len(authorships)
n_uniq_auth = authorships["author_id"].nunique()
n_uniq_inst = len(set(
    inst.strip()
    for col in ["gh_institution_names", "non_gh_institution_names"]
    if col in authorships.columns
    for val in authorships[col].dropna()
    for inst in str(val).split("|") if inst.strip()
))
n_countries = len(set(
    c.strip() for val in works["countries"].dropna()
    for c in str(val).split("|") if c.strip()
))

med_auth = works["author_count"].median()
iqr_auth = f"{works['author_count'].quantile(0.25):.0f}-{works['author_count'].quantile(0.75):.0f}"
med_ctry = works["country_count"].median()
iqr_ctry = f"{works['country_count'].quantile(0.25):.0f}-{works['country_count'].quantile(0.75):.0f}"

rows.append({"Characteristic": "Total international collaborative works", "Overall": f"{N:,}"})
rows.append({"Characteristic": "Total authorships", "Overall": f"{n_auth:,}"})
rows.append({"Characteristic": "Unique authors", "Overall": f"{n_uniq_auth:,}"})
rows.append({"Characteristic": "Unique institutions", "Overall": f"{n_uniq_inst:,}"})
rows.append({"Characteristic": "Unique countries", "Overall": f"{n_countries:,}"})
rows.append({"Characteristic": "Median authors per paper (IQR)", "Overall": f"{med_auth:.0f} ({iqr_auth})"})
rows.append({"Characteristic": "Median countries per paper (IQR)", "Overall": f"{med_ctry:.0f} ({iqr_ctry})"})

# Domain distribution
if "domain_name" in works.columns:
    for dom in ["Health Sciences", "Life Sciences", "Physical Sciences"]:
        n_dom = (works["domain_name"] == dom).sum()
        rows.append({"Characteristic": f"Works by domain: {dom}", "Overall": f"{n_dom:,} ({n_dom/N*100:.1f}%)"})

# Funding, corresponding author, OA
fund_n = works["has_funding"].apply(to_bool).sum() if "has_funding" in works.columns else 0
rows.append({"Characteristic": "Works with funding data", "Overall": f"{fund_n:,} ({fund_n/N*100:.1f}%)"})

corr_n = authorships[authorships["is_corresponding_combined"]]["work_id"].nunique()
rows.append({"Characteristic": "Works with corresponding author data", "Overall": f"{corr_n:,} ({corr_n/N*100:.1f}%)"})

oa_n = works["is_oa"].apply(to_bool).sum() if "is_oa" in works.columns else 0
rows.append({"Characteristic": "Open access works", "Overall": f"{oa_n:,} ({oa_n/N*100:.1f}%)"})

med_cite = works["cited_by_count"].median() if "cited_by_count" in works.columns else 0
iqr_cite = f"{works['cited_by_count'].quantile(0.25):.0f}-{works['cited_by_count'].quantile(0.75):.0f}" if "cited_by_count" in works.columns else "N/A"
rows.append({"Characteristic": "Median citations per work (IQR)", "Overall": f"{med_cite:.0f} ({iqr_cite})"})

# By time period
for period_name, (yr_start, yr_end) in TIME_PERIODS.items():
    pw = works[(works["publication_year"] >= yr_start) & (works["publication_year"] <= yr_end)]
    for r in rows:
        char = r["Characteristic"]
        if char == "Total international collaborative works":
            r[period_name] = f"{len(pw):,}"
        elif char == "Total authorships":
            pa = authorships[authorships["work_id"].isin(pw["work_id"])]
            r[period_name] = f"{len(pa):,}"
        elif char == "Median authors per paper (IQR)":
            if len(pw) > 0:
                r[period_name] = f"{pw['author_count'].median():.0f} ({pw['author_count'].quantile(0.25):.0f}-{pw['author_count'].quantile(0.75):.0f})"
            else:
                r[period_name] = "N/A"
        elif char == "Median citations per work (IQR)" and "cited_by_count" in pw.columns:
            if len(pw) > 0:
                r[period_name] = f"{pw['cited_by_count'].median():.0f} ({pw['cited_by_count'].quantile(0.25):.0f}-{pw['cited_by_count'].quantile(0.75):.0f})"
            else:
                r[period_name] = "N/A"

table1 = pd.DataFrame(rows)
table1.to_csv(OUTPUT_DIR / "table1_study_characteristics.csv", index=False)
print("  Saved table1_study_characteristics.csv")

# ═══════════════════════════════════════════════════════════
# 4.2 CHART 1: Publications by Year (bar chart)
# ═══════════════════════════════════════════════════════════
print("\n--- Chart 1: Publications by Year ---")

yearly = works.groupby("publication_year").size().reset_index(name="count")

fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
ax.bar(yearly["publication_year"], yearly["count"], color=COLORS["total"], width=0.8, edgecolor="white", linewidth=0.5)

# LOESS-like trend using polynomial
z = np.polyfit(yearly["publication_year"], yearly["count"], 3)
p = np.poly1d(z)
x_smooth = np.linspace(yearly["publication_year"].min(), yearly["publication_year"].max(), 100)
ax.plot(x_smooth, p(x_smooth), color="#B71C1C", linewidth=2, linestyle="--", label="Polynomial trend")

ax.set_xlabel("Year")
ax.set_ylabel("Number of Publications")
ax.set_title("International collaborative publications involving Ghanaian\nresearchers in biomedical science and engineering, 2000-2025")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_chart(fig, "chart_01_publications_by_year")

# ═══════════════════════════════════════════════════════════
# 4.2 CHART 2: Field composition over time (stacked area)
# ═══════════════════════════════════════════════════════════
print("\n--- Chart 2: Field Composition Over Time ---")

if "field_name" in works.columns:
    top_fields = works["field_name"].value_counts().head(6).index.tolist()
    works["field_plot"] = works["field_name"].apply(lambda x: x if x in top_fields else "Other")
    field_yearly = works.groupby(["publication_year", "field_plot"]).size().unstack(fill_value=0)
    # Reorder columns to put top fields first
    cols_order = [f for f in top_fields if f in field_yearly.columns]
    if "Other" in field_yearly.columns:
        cols_order.append("Other")
    field_yearly = field_yearly[[c for c in cols_order if c in field_yearly.columns]]

    field_colors = sns.color_palette("colorblind", n_colors=len(field_yearly.columns))

    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
    ax.stackplot(field_yearly.index, field_yearly.values.T, labels=field_yearly.columns,
                 colors=field_colors, alpha=0.85)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Publications")
    ax.set_title("Field composition of Ghanaian international collaborations, 2000-2025")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_chart(fig, "chart_02_field_composition")
    works.drop(columns=["field_plot"], inplace=True)

# ═══════════════════════════════════════════════════════════
# 4.3 CHART 3: Top 20 Partner Countries (horizontal bar)
# ═══════════════════════════════════════════════════════════
print("\n--- Chart 3: Top 20 Partner Countries ---")

all_partner_codes = []
for val in works["partner_countries"].dropna():
    for c in str(val).split("|"):
        if c.strip():
            all_partner_codes.append(c.strip())

partner_counts = pd.Series(all_partner_codes).value_counts().head(20)

fig, ax = plt.subplots(figsize=(8, 7))
bars = ax.barh(range(len(partner_counts)), partner_counts.values, color=COLORS["total"],
               edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(partner_counts)))
ax.set_yticklabels(partner_counts.index)
ax.invert_yaxis()
for i, v in enumerate(partner_counts.values):
    ax.text(v + max(partner_counts.values)*0.01, i, f"{v:,}", va="center", fontsize=9)
ax.set_xlabel("Number of Collaborations")
ax.set_title("Top 20 international partner countries for Ghanaian\nbiomedical and engineering research, 2000-2025")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_chart(fig, "chart_03_top20_partner_countries")

# ── Partner country leadership table ──
print("  Computing partner country leadership table...")
top20_codes = partner_counts.index.tolist()

# For each partner country, find works where that country is involved
partner_leadership_rows = []
for code in top20_codes:
    mask = works["partner_countries"].fillna("").str.contains(f"(^|\\|){code}(\\||$)", regex=True)
    code_works = works[mask]
    code_work_ids = set(code_works["work_id"])
    code_auths = authorships[authorships["work_id"].isin(code_work_ids)]

    n_collabs = len(code_works)
    pct_all = n_collabs / N * 100

    # First author stats
    first = code_auths[code_auths["author_position"] == "first"]
    gh_first = (first["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"])).sum()
    pct_gh_first = gh_first / len(first) * 100 if len(first) > 0 else 0

    # Last author stats
    last = code_auths[code_auths["author_position"] == "last"]
    gh_last = (last["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"])).sum()
    pct_gh_last = gh_last / len(last) * 100 if len(last) > 0 else 0

    # Corresponding author
    corr = code_auths[code_auths["is_corresponding_combined"] == True]
    gh_corr = (corr["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"])).sum()
    pct_gh_corr = gh_corr / len(corr) * 100 if len(corr) > 0 else 0

    partner_leadership_rows.append({
        "Country": code,
        "Total collabs": n_collabs,
        "% of all collabs": round(pct_all, 1),
        "% Ghanaian first author": round(pct_gh_first, 1),
        "% Ghanaian last author": round(pct_gh_last, 1),
        "% Ghanaian corresponding author": round(pct_gh_corr, 1),
    })

pd.DataFrame(partner_leadership_rows).to_csv(OUTPUT_DIR / "table_s5_partner_country_leadership_full.csv", index=False)
print("  Saved table_s5_partner_country_leadership_full.csv")

# ═══════════════════════════════════════════════════════════
# 5.1 OVERALL LEADERSHIP PROPORTIONS
# ═══════════════════════════════════════════════════════════
print("\n--- Section 5.1: Overall Leadership Proportions ---")

leadership_rows = []
for pos_name, pos_filter in [("First author", "first"), ("Last author", "last")]:
    pos_data = authorships[authorships["author_position"] == pos_filter]
    total = len(pos_data)
    row = {"Position": pos_name, "Total N": total}
    for cat in CAT_ORDER:
        n_cat = (pos_data["affiliation_category"] == cat).sum()
        pct = n_cat / total * 100 if total > 0 else 0
        lo, hi = wilson_ci(n_cat, total)
        row[f"{cat} N"] = n_cat
        row[f"{cat} %"] = round(pct, 1)
        row[f"{cat} 95% CI"] = f"{lo*100:.1f}-{hi*100:.1f}"
    leadership_rows.append(row)

# Corresponding author
corr_auths = authorships[authorships["is_corresponding_combined"] == True]
total_corr = len(corr_auths)
row = {"Position": "Corresponding author", "Total N": total_corr}
for cat in CAT_ORDER:
    n_cat = (corr_auths["affiliation_category"] == cat).sum()
    pct = n_cat / total_corr * 100 if total_corr > 0 else 0
    lo, hi = wilson_ci(n_cat, total_corr)
    row[f"{cat} N"] = n_cat
    row[f"{cat} %"] = round(pct, 1)
    row[f"{cat} 95% CI"] = f"{lo*100:.1f}-{hi*100:.1f}"
leadership_rows.append(row)

table2 = pd.DataFrame(leadership_rows)
table2.to_csv(OUTPUT_DIR / "table2_leadership_overall.csv", index=False)
print("  Saved table2_leadership_overall.csv")
print(table2[["Position"] + [c for c in table2.columns if "%" in c]].to_string(index=False))

# Chi-square goodness-of-fit test
print("\n  Chi-square goodness-of-fit tests:")
overall_props = authorships["affiliation_category"].value_counts(normalize=True)
for pos_name, pos_filter in [("First author", "first"), ("Last author", "last"), ("Corresponding author", None)]:
    if pos_filter:
        pos_data = authorships[authorships["author_position"] == pos_filter]
    else:
        pos_data = authorships[authorships["is_corresponding_combined"] == True]
    observed = np.array([
        (pos_data["affiliation_category"] == cat).sum() for cat in CAT_ORDER
    ])
    expected = np.array([overall_props.get(cat, 0) for cat in CAT_ORDER]) * observed.sum()
    chi2, p = stats.chisquare(observed, expected)
    print(f"    {pos_name}: chi2({len(CAT_ORDER)-1}) = {chi2:.2f}, p = {p:.3f}" if p >= 0.001 else f"    {pos_name}: chi2({len(CAT_ORDER)-1}) = {chi2:.2f}, p < 0.001")

# ═══════════════════════════════════════════════════════════
# 5.1 CHART 4: Leadership by Position (grouped bar with CIs)
# ═══════════════════════════════════════════════════════════
print("\n--- Chart 4: Leadership by Position ---")

positions = ["First author", "Last author", "Corresponding author"]
x = np.arange(len(positions))
bar_width = 0.25

fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)

# Reference line: overall Ghanaian+Dual proportion
gh_dual_pct = (authorships["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"])).mean() * 100
ax.axhline(y=gh_dual_pct, color=COLORS["total"], linestyle="--", linewidth=1,
           label=f"Expected baseline ({gh_dual_pct:.1f}%)")

for i, cat in enumerate(CAT_ORDER):
    vals = []
    errs_lo = []
    errs_hi = []
    for pos_row in leadership_rows:
        pct = pos_row[f"{cat} %"]
        ci = pos_row[f"{cat} 95% CI"].split("-")
        lo, hi = float(ci[0]), float(ci[1])
        vals.append(pct)
        errs_lo.append(pct - lo)
        errs_hi.append(hi - pct)
    ax.bar(x + i * bar_width, vals, bar_width, label=cat,
           color=COLORS[cat], edgecolor="white", linewidth=0.5,
           yerr=[errs_lo, errs_hi], capsize=3, error_kw={"linewidth": 1})

ax.set_xticks(x + bar_width)
ax.set_xticklabels(positions)
ax.set_ylabel("Percentage (%)")
ax.set_title("Research leadership in Ghanaian international collaborations\nby authorship position, 2000-2025")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_chart(fig, "chart_04_leadership_overall")

# ═══════════════════════════════════════════════════════════
# 5.2 TEMPORAL TRENDS IN LEADERSHIP
# ═══════════════════════════════════════════════════════════
print("\n--- Section 5.2: Temporal Trends ---")

years = sorted(works["publication_year"].unique())

trend_rows = []
for yr in years:
    yr_auths = authorships[authorships["publication_year"] == yr]
    row = {"Year": yr, "N_works": (works["publication_year"] == yr).sum()}
    for pos_name, pos_filter in [("first", "first"), ("last", "last")]:
        pos_data = yr_auths[yr_auths["author_position"] == pos_filter]
        total = len(pos_data)
        for cat in CAT_ORDER:
            n_cat = (pos_data["affiliation_category"] == cat).sum()
            pct = n_cat / total * 100 if total > 0 else 0
            row[f"{pos_name}_{cat}_pct"] = round(pct, 1)
            row[f"{pos_name}_{cat}_n"] = n_cat
        row[f"{pos_name}_total"] = total

    # Corresponding
    corr_yr = yr_auths[yr_auths["is_corresponding_combined"] == True]
    total_corr_yr = len(corr_yr)
    for cat in CAT_ORDER:
        n_cat = (corr_yr["affiliation_category"] == cat).sum()
        pct = n_cat / total_corr_yr * 100 if total_corr_yr > 0 else 0
        row[f"corresponding_{cat}_pct"] = round(pct, 1)
        row[f"corresponding_{cat}_n"] = n_cat
    row["corresponding_total"] = total_corr_yr

    # Corresponding coverage
    works_yr = works[works["publication_year"] == yr]
    corr_works_yr = corr_yr["work_id"].nunique()
    row["corr_coverage_pct"] = round(corr_works_yr / len(works_yr) * 100, 1) if len(works_yr) > 0 else 0

    trend_rows.append(row)

table3 = pd.DataFrame(trend_rows)
table3.to_csv(OUTPUT_DIR / "table3_leadership_by_year.csv", index=False)
print("  Saved table3_leadership_by_year.csv")

# Helper function for trend charts
def plot_leadership_trend(trend_df, position_prefix, chart_num, chart_name, title, show_coverage=False):
    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.5)
    ax.set_axisbelow(True)

    for cat in CAT_ORDER:
        col = f"{position_prefix}_{cat}_pct"
        ax.plot(trend_df["Year"], trend_df[col], marker=MARKERS[cat], markersize=5,
                color=COLORS[cat], linewidth=2, label=cat)

        # 3-year moving average
        if len(trend_df) >= 3:
            ma = trend_df[col].rolling(3, center=True, min_periods=1).mean()
            ax.plot(trend_df["Year"], ma, color=COLORS[cat], linewidth=1.5,
                    linestyle="--", alpha=0.5)

    # CI band for Ghanaian line
    gh_n_col = f"{position_prefix}_Ghanaian_n"
    gh_total_col = f"{position_prefix}_total"
    if gh_n_col in trend_df.columns and gh_total_col in trend_df.columns:
        ci_lo = []
        ci_hi = []
        for _, r in trend_df.iterrows():
            lo, hi = wilson_ci(int(r[gh_n_col]), int(r[gh_total_col]))
            ci_lo.append(lo * 100)
            ci_hi.append(hi * 100)
        ax.fill_between(trend_df["Year"], ci_lo, ci_hi, color=COLORS["Ghanaian"], alpha=0.15)

    if show_coverage and "corr_coverage_pct" in trend_df.columns:
        ax2 = ax.twinx()
        ax2.bar(trend_df["Year"], trend_df["corr_coverage_pct"], alpha=0.15, color="#9E9E9E",
                width=0.8, label="Coverage %")
        ax2.set_ylabel("Corresponding Author Coverage (%)", color="#9E9E9E")
        ax2.set_ylim(0, 110)
        ax2.tick_params(axis="y", colors="#9E9E9E")

    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    save_chart(fig, chart_name)

# Chart 5: First author trend
plot_leadership_trend(table3, "first", 5, "chart_05_first_author_trend",
    "First authorship in Ghanaian international collaborations\nby affiliation category, 2000-2025")

# Chart 6: Last author trend
plot_leadership_trend(table3, "last", 6, "chart_06_last_author_trend",
    "Last authorship in Ghanaian international collaborations\nby affiliation category, 2000-2025")

# Chart 7: Corresponding author trend with coverage
plot_leadership_trend(table3, "corresponding", 7, "chart_07_corresponding_author_trend",
    "Corresponding authorship in Ghanaian international collaborations\nby affiliation category, 2000-2025", show_coverage=True)

# Cochran-Armitage / logistic trend test
print("\n  Trend tests (logistic regression with year as predictor):")
from statsmodels.api import Logit, add_constant

for pos_name, pos_filter in [("First author", "first"), ("Last author", "last"), ("Corresponding", None)]:
    try:
        if pos_filter:
            pos_data = authorships[authorships["author_position"] == pos_filter].copy()
        else:
            pos_data = authorships[authorships["is_corresponding_combined"] == True].copy()

        pos_data["gh_binary"] = (pos_data["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"])).astype(int)
        X = add_constant(pos_data["publication_year"] - 2000)
        model = Logit(pos_data["gh_binary"], X).fit(disp=0)
        coeff = model.params.iloc[1]
        p_val = model.pvalues.iloc[1]
        direction = "increasing" if coeff > 0 else "decreasing"
        or_val = np.exp(coeff)
        ci_lo, ci_hi = np.exp(model.conf_int().iloc[1])
        p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        print(f"    {pos_name}: OR={or_val:.3f} ({ci_lo:.3f}-{ci_hi:.3f}), {p_str}, {direction}")
    except Exception as e:
        print(f"    {pos_name}: trend test failed - {e}")

# ═══════════════════════════════════════════════════════════
# 5.3 LEADERSHIP BY PARTNER COUNTRY
# ═══════════════════════════════════════════════════════════
print("\n--- Section 5.3: Leadership by Partner Country ---")

# Table 4: Leadership by top 10 partner countries
top10_codes = partner_counts.head(10).index.tolist()
country_period_rows = []

for code in top10_codes:
    mask = works["partner_countries"].fillna("").str.contains(f"(^|\\|){code}(\\||$)", regex=True)
    code_works = works[mask]

    for period_name, (yr_start, yr_end) in TIME_PERIODS.items():
        pw = code_works[(code_works["publication_year"] >= yr_start) & (code_works["publication_year"] <= yr_end)]
        pw_ids = set(pw["work_id"])
        pa = authorships[authorships["work_id"].isin(pw_ids)]

        row_data = {
            "Country": code,
            "Time_period": period_name,
            "N_works": len(pw),
        }
        for pos_name, pos_filter in [("first", "first"), ("last", "last"), ("corresponding", None)]:
            if pos_filter:
                pos = pa[pa["author_position"] == pos_filter]
            else:
                pos = pa[pa["is_corresponding_combined"] == True]
            gh_n = pos["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).sum()
            pct_gh = gh_n / len(pos) * 100 if len(pos) > 0 else 0
            row_data[f"Ghanaian_{pos_name}_author_pct"] = round(pct_gh, 1)
        country_period_rows.append(row_data)

table4 = pd.DataFrame(country_period_rows)
table4.to_csv(OUTPUT_DIR / "table4_leadership_by_partner_country.csv", index=False)
print("  Saved table4_leadership_by_partner_country.csv")

# Chart 8: Heatmap — three panels for first, last, corresponding
print("\n--- Chart 8: Heatmap Country x Period ---")
period_order = list(TIME_PERIODS.keys())
country_order = [c for c in top10_codes]

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
for ax_idx, (pos_label, col_name) in enumerate([
    ("First Author", "Ghanaian_first_author_pct"),
    ("Last Author", "Ghanaian_last_author_pct"),
    ("Corresponding Author", "Ghanaian_corresponding_author_pct")
]):
    hm = table4.pivot(index="Country", columns="Time_period", values=col_name)
    hm = hm[[c for c in period_order if c in hm.columns]]
    hm = hm.loc[[c for c in country_order if c in hm.index]]
    ax = axes[ax_idx]
    sns.heatmap(hm, annot=True, fmt=".0f", cmap="RdYlGn", center=50,
                vmin=0, vmax=100, linewidths=0.5, ax=ax,
                cbar=ax_idx == 2, cbar_kws={"label": "% GH/Dual"})
    ax.set_title(pos_label, fontsize=12)
    ax.set_xlabel("")
    if ax_idx > 0:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Partner Country")
fig.suptitle("Proportion of Ghanaian leadership by partner country and time period",
             fontsize=TITLE_SIZE, fontweight="bold")
fig.tight_layout()
save_chart(fig, "chart_08_heatmap_country_period")

# ═══════════════════════════════════════════════════════════
# 5.4 LEADERSHIP BY PARTNER REGION
# ═══════════════════════════════════════════════════════════
print("\n--- Section 5.4: Leadership by Partner Region ---")

region_rows = []
blocs_to_analyze = ["Western", "African", "East Asian", "South Asian", "MENA", "Latin American", "Multi-bloc"]

for bloc in blocs_to_analyze:
    bw = works[works["partner_bloc"] == bloc]
    bw_ids = set(bw["work_id"])
    ba = authorships[authorships["work_id"].isin(bw_ids)]

    row = {"Region": bloc, "N_works": len(bw)}
    for pos_name, pos_filter in [("first", "first"), ("last", "last")]:
        pos = ba[ba["author_position"] == pos_filter]
        for cat in CAT_ORDER:
            n = (pos["affiliation_category"] == cat).sum()
            pct = n / len(pos) * 100 if len(pos) > 0 else 0
            row[f"{pos_name}_{cat}_pct"] = round(pct, 1)

    corr = ba[ba["is_corresponding_combined"] == True]
    for cat in CAT_ORDER:
        n = (corr["affiliation_category"] == cat).sum()
        pct = n / len(corr) * 100 if len(corr) > 0 else 0
        row[f"corresponding_{cat}_pct"] = round(pct, 1)

    region_rows.append(row)

region_df = pd.DataFrame(region_rows)

# Chart 9: Leadership by Region
fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
x = np.arange(len(blocs_to_analyze))
bar_w = 0.25

for i, (pos_label, pos_prefix) in enumerate([("First Author", "first"), ("Last Author", "last"), ("Corresponding", "corresponding")]):
    vals = []
    for _, r in region_df.iterrows():
        gh_pct = r.get(f"{pos_prefix}_Ghanaian_pct", 0) + r.get(f"{pos_prefix}_Dual-affiliated_pct", 0)
        vals.append(gh_pct)
    offset = (i - 1) * bar_w
    ax.bar(x + offset, vals, bar_w, label=pos_label,
           color=[COLORS["Ghanaian"], COLORS["Dual-affiliated"], COLORS["Non-Ghanaian"]][i],
           edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([r["Region"] for _, r in region_df.iterrows()], rotation=30, ha="right")
ax.set_ylabel("% Ghanaian or Dual-affiliated")
ax.set_title("Ghanaian research leadership by partner region")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_chart(fig, "chart_09_leadership_by_region")

# North-South vs South-South test
print("\n  North-South vs South-South hypothesis test:")
western_works = works[works["partner_bloc"] == "Western"]
african_works = works[works["partner_bloc"] == "African"]

for pos_name, pos_filter in [("First author", "first"), ("Last author", "last"), ("Corresponding author", None)]:
    if pos_filter:
        w_auths = authorships[(authorships["work_id"].isin(western_works["work_id"])) & (authorships["author_position"] == pos_filter)]
        a_auths = authorships[(authorships["work_id"].isin(african_works["work_id"])) & (authorships["author_position"] == pos_filter)]
    else:
        w_auths = authorships[(authorships["work_id"].isin(western_works["work_id"])) & (authorships["is_corresponding_combined"] == True)]
        a_auths = authorships[(authorships["work_id"].isin(african_works["work_id"])) & (authorships["is_corresponding_combined"] == True)]

    w_gh = w_auths["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).sum()
    a_gh = a_auths["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).sum()

    table = np.array([[w_gh, len(w_auths) - w_gh], [a_gh, len(a_auths) - a_gh]])
    try:
        chi2, p, dof, _ = stats.chi2_contingency(table)
        p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
        print(f"    {pos_name}: Western GH%={w_gh/len(w_auths)*100:.1f}%, African GH%={a_gh/len(a_auths)*100:.1f}%, chi2={chi2:.2f}, {p_str}")
    except Exception as e:
        print(f"    {pos_name}: test failed - {e}")

# ═══════════════════════════════════════════════════════════
# 5.5 LEADERSHIP BY FIELD
# ═══════════════════════════════════════════════════════════
print("\n--- Section 5.5: Leadership by Field ---")

field_rows = []
if "field_name" in works.columns:
    top_fields = works["field_name"].value_counts().head(10).index.tolist()
    for field in top_fields:
        fw = works[works["field_name"] == field]
        fw_ids = set(fw["work_id"])
        fa = authorships[authorships["work_id"].isin(fw_ids)]

        row = {"Field": field, "N_works": len(fw)}
        for pos_name, pos_filter in [("first", "first"), ("last", "last"), ("corresponding", None)]:
            if pos_filter:
                pos = fa[fa["author_position"] == pos_filter]
            else:
                pos = fa[fa["is_corresponding_combined"] == True]

            for cat in CAT_ORDER:
                n = (pos["affiliation_category"] == cat).sum()
                pct = n / len(pos) * 100 if len(pos) > 0 else 0
                row[f"{pos_name}_{cat}_pct"] = round(pct, 1)
        field_rows.append(row)

table5 = pd.DataFrame(field_rows)
table5.to_csv(OUTPUT_DIR / "table5_leadership_by_field.csv", index=False)
print("  Saved table5_leadership_by_field.csv")

# Chart 10: Leadership by Field (all 3 positions)
if len(table5) > 0:
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(table5))
    bar_w = 0.25

    pos_colors = [COLORS["Ghanaian"], COLORS["Dual-affiliated"], COLORS["Non-Ghanaian"]]
    for i, (pos_label, pos_prefix) in enumerate([("First Author", "first"), ("Last Author", "last"), ("Corresponding", "corresponding")]):
        vals = []
        for _, r in table5.iterrows():
            gh_pct = r.get(f"{pos_prefix}_Ghanaian_pct", 0) + r.get(f"{pos_prefix}_Dual-affiliated_pct", 0)
            vals.append(gh_pct)
        offset = (i - 1) * bar_w
        ax.barh(y + offset, vals, bar_w, label=pos_label,
                color=pos_colors[i], edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(table5["Field"])
    ax.set_xlabel("% Ghanaian or Dual-affiliated")
    ax.set_title("Ghanaian leadership rates by field")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_chart(fig, "chart_10_leadership_by_field")

# ═══════════════════════════════════════════════════════════
# 5.6 LEADERSHIP BY FUNDING
# ═══════════════════════════════════════════════════════════
print("\n--- Section 5.6: Leadership by Funding ---")

funding_cats = ["International (Northern)", "Ghanaian", "Other/Unclassified", "No funding data"]
funding_rows = []
for fcat in funding_cats:
    fw = works[works["funder_category"] == fcat]
    if len(fw) == 0:
        continue
    fw_ids = set(fw["work_id"])
    fa = authorships[authorships["work_id"].isin(fw_ids)]

    row = {"Funding_category": fcat, "N_works": len(fw)}
    for pos_name, pos_filter in [("first", "first"), ("last", "last"), ("corresponding", None)]:
        if pos_filter:
            pos = fa[fa["author_position"] == pos_filter]
        else:
            pos = fa[fa["is_corresponding_combined"] == True]
        for cat in CAT_ORDER:
            n = (pos["affiliation_category"] == cat).sum()
            pct = n / len(pos) * 100 if len(pos) > 0 else 0
            row[f"{pos_name}_{cat}_pct"] = round(pct, 1)
    funding_rows.append(row)

table6 = pd.DataFrame(funding_rows)
table6.to_csv(OUTPUT_DIR / "table6_leadership_by_funding.csv", index=False)
print("  Saved table6_leadership_by_funding.csv")

# Chart 11: Leadership by Funding
if len(table6) > 0:
    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
    x = np.arange(len(table6))
    bar_w = 0.25

    for i, (pos_label, pos_prefix) in enumerate([("First Author", "first"), ("Last Author", "last"), ("Corresponding", "corresponding")]):
        vals = []
        for _, r in table6.iterrows():
            gh_pct = r.get(f"{pos_prefix}_Ghanaian_pct", 0) + r.get(f"{pos_prefix}_Dual-affiliated_pct", 0)
            vals.append(gh_pct)
        offset = (i - 1) * bar_w
        ax.bar(x + offset, vals, bar_w, label=pos_label,
               color=[COLORS["Ghanaian"], COLORS["Dual-affiliated"], COLORS["Non-Ghanaian"]][i],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([r["Funding_category"] for _, r in table6.iterrows()], rotation=25, ha="right")
    ax.set_ylabel("% Ghanaian or Dual-affiliated")
    ax.set_title("Research leadership by funding source")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_chart(fig, "chart_11_leadership_by_funding")

# Chi-square: funded vs unfunded — all three positions
print("\n  Chi-square: Northern funded vs Ghanaian funded vs No funding:")
for pos_name, pos_filter in [("First author", "first"), ("Last author", "last"), ("Corresponding author", None)]:
    contingency_rows = []
    for fcat in ["International (Northern)", "Ghanaian", "No funding data"]:
        fw = works[works["funder_category"] == fcat]
        if pos_filter:
            fa = authorships[(authorships["work_id"].isin(fw["work_id"])) & (authorships["author_position"] == pos_filter)]
        else:
            fa = authorships[(authorships["work_id"].isin(fw["work_id"])) & (authorships["is_corresponding_combined"] == True)]
        gh = fa["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).sum()
        non_gh = len(fa) - gh
        contingency_rows.append([gh, non_gh])
    table_arr = np.array(contingency_rows)
    try:
        chi2, p, dof, _ = stats.chi2_contingency(table_arr)
        p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
        print(f"    {pos_name}: chi2({dof}) = {chi2:.2f}, {p_str}")
    except Exception as e:
        print(f"    {pos_name}: test failed - {e}")

# ═══════════════════════════════════════════════════════════
# 5.7 CITATION IMPACT
# ═══════════════════════════════════════════════════════════
print("\n--- Section 5.7: Citation Impact ---")

# Merge first-author affiliation category onto works
first_auths = authorships[authorships["author_position"] == "first"][["work_id", "affiliation_category"]].copy()
first_auths.rename(columns={"affiliation_category": "first_author_cat"}, inplace=True)
works_citation = works.merge(first_auths, on="work_id", how="left")

# Use FWCI if coverage > 50%
fwci_coverage = works_citation["fwci"].notna().mean()
use_fwci = fwci_coverage > 0.5
metric_col = "fwci" if use_fwci else "cited_by_count"
metric_label = "FWCI" if use_fwci else "Citations"
print(f"  Using metric: {metric_label} (coverage: {fwci_coverage*100:.1f}%)")

for cat in CAT_ORDER:
    subset = works_citation[works_citation["first_author_cat"] == cat][metric_col].dropna()
    print(f"    {cat}: median={subset.median():.2f}, IQR=({subset.quantile(0.25):.2f}-{subset.quantile(0.75):.2f}), N={len(subset):,}")

# Mann-Whitney U tests
print("\n  Mann-Whitney U tests:")
for cat_a, cat_b in [("Ghanaian", "Non-Ghanaian"), ("Dual-affiliated", "Non-Ghanaian")]:
    a = works_citation[works_citation["first_author_cat"] == cat_a][metric_col].dropna()
    b = works_citation[works_citation["first_author_cat"] == cat_b][metric_col].dropna()
    if len(a) > 0 and len(b) > 0:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
        print(f"    {cat_a} vs {cat_b}: U = {u:,.0f}, {p_str}")

# Top 10% analysis
if "is_in_top_10_percent" in works_citation.columns:
    print("\n  Top 10% papers by first author category:")
    for cat in CAT_ORDER:
        subset = works_citation[works_citation["first_author_cat"] == cat]
        top10 = subset["is_in_top_10_percent"].apply(to_bool).sum()
        pct = top10 / len(subset) * 100 if len(subset) > 0 else 0
        print(f"    {cat}: {top10:,}/{len(subset):,} ({pct:.1f}%)")

# Chart 12: Box plots
fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)

plot_data = []
labels = []
colors_list = []
for cat in CAT_ORDER:
    subset = works_citation[works_citation["first_author_cat"] == cat][metric_col].dropna()
    if use_fwci:
        plot_data.append(subset.values)
    else:
        plot_data.append(np.log10(subset.values + 1))
    labels.append(cat)
    colors_list.append(COLORS[cat])

bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, showfliers=True,
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
for patch, color in zip(bp["boxes"], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

y_label = metric_label if use_fwci else "log10(Citations + 1)"
ax.set_ylabel(y_label)
ax.set_title(f"Citation impact by first author affiliation category")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate with top-10% if available
if "is_in_top_10_percent" in works_citation.columns:
    for i, cat in enumerate(CAT_ORDER):
        subset = works_citation[works_citation["first_author_cat"] == cat]
        top10_pct = subset["is_in_top_10_percent"].apply(to_bool).mean() * 100
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f"Top 10%: {top10_pct:.1f}%",
                ha="center", fontsize=8, style="italic")

fig.tight_layout()
save_chart(fig, "chart_12_citations_by_leader")

print("\n" + "=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)
print(f"  Tables saved: table1-6, table_s5")
print(f"  Charts saved: chart_01 through chart_12 (excluding 13-17)")
