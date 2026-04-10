"""
Phase 3: Advanced Analysis
==========================
Produces Tables 7-15 and Charts 13-22.
Covers: COVID analysis, partner bloc analysis, logistic regression,
trend analysis, dual-affiliation deep dive, institutional analysis.
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
from statsmodels.api import Logit, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import *

plt.rcParams.update({
    "font.family": CHART_FONT, "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE, "axes.titleweight": "bold",
    "axes.labelsize": LABEL_SIZE, "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE, "legend.fontsize": LEGEND_SIZE,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": CHART_DPI,
    "savefig.bbox": "tight",
})

CAT_ORDER = ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]

def save_chart(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=CHART_DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png and .svg")

def to_bool(val):
    if pd.isna(val): return False
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() == "true"
    return bool(val)

def wilson_ci(count, total, alpha=0.05):
    if total == 0: return 0, 0
    lo, hi = proportion_confint(count, total, alpha=alpha, method="wilson")
    return lo, hi

# ── Load Data ──
print("=" * 60)
print("PHASE 3: ADVANCED ANALYSIS")
print("=" * 60)

print("\nLoading data...")
works = pd.read_parquet(INTERMEDIATE_DIR / "works.parquet")
authorships = pd.read_parquet(INTERMEDIATE_DIR / "authorships.parquet")
try:
    funding = pd.read_parquet(INTERMEDIATE_DIR / "funding.parquet")
except: funding = pd.DataFrame()
try:
    topics = pd.read_parquet(INTERMEDIATE_DIR / "topics.parquet")
except: topics = pd.DataFrame()

N = len(works)
print(f"  Works: {N:,}, Authorships: {len(authorships):,}")

# ═══════════════════════════════════════════════════════════
# 5.8 PRE-COVID vs POST-COVID ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n--- Section 5.8: Pre-COVID vs Post-COVID ---")

covid_rows = []
for era_name, era_val in [("Pre-COVID (2000-2019)", 0), ("Post-COVID (2020-2025)", 1)]:
    ew = works[works["covid_era"] == era_val]
    ew_ids = set(ew["work_id"])
    ea = authorships[authorships["work_id"].isin(ew_ids)]

    row = {"Period": era_name, "N_works": len(ew),
           "Mean_authors": round(ew["author_count"].mean(), 1),
           "Mean_countries": round(ew["country_count"].mean(), 1)}

    for pos_name, pos_filter in [("first", "first"), ("last", "last")]:
        pos = ea[ea["author_position"] == pos_filter]
        for cat in CAT_ORDER:
            n = (pos["affiliation_category"] == cat).sum()
            pct = n / len(pos) * 100 if len(pos) > 0 else 0
            row[f"{pos_name}_{cat}_pct"] = round(pct, 1)
            row[f"{pos_name}_{cat}_n"] = n
        row[f"{pos_name}_total"] = len(pos)

    corr = ea[ea["is_corresponding_combined"] == True]
    for cat in CAT_ORDER:
        n = (corr["affiliation_category"] == cat).sum()
        pct = n / len(corr) * 100 if len(corr) > 0 else 0
        row[f"corresponding_{cat}_pct"] = round(pct, 1)
    covid_rows.append(row)

table7 = pd.DataFrame(covid_rows)
table7.to_csv(OUTPUT_DIR / "table7_leadership_pre_post_covid.csv", index=False)
print("  Saved table7_leadership_pre_post_covid.csv")

for pos in ["first", "last", "corresponding"]:
    print(f"  {pos.capitalize()} author:")
    for cat in CAT_ORDER:
        pre_val = table7[table7["Period"].str.contains("Pre")][f"{pos}_{cat}_pct"].values[0]
        post_val = table7[table7["Period"].str.contains("Post")][f"{pos}_{cat}_pct"].values[0]
        print(f"    {cat}: {pre_val}% -> {post_val}%")

# Chi-square: pre vs post
print("\n  Chi-square pre vs post COVID:")
for pos_name, pos_filter in [("First author", "first"), ("Last author", "last"), ("Corresponding author", None)]:
    if pos_filter:
        pre_a = authorships[(authorships["work_id"].isin(works[works["covid_era"]==0]["work_id"])) & (authorships["author_position"]==pos_filter)]
        post_a = authorships[(authorships["work_id"].isin(works[works["covid_era"]==1]["work_id"])) & (authorships["author_position"]==pos_filter)]
    else:
        pre_a = authorships[(authorships["work_id"].isin(works[works["covid_era"]==0]["work_id"])) & (authorships["is_corresponding_combined"]==True)]
        post_a = authorships[(authorships["work_id"].isin(works[works["covid_era"]==1]["work_id"])) & (authorships["is_corresponding_combined"]==True)]
    contingency = []
    for cat in CAT_ORDER:
        contingency.append([(pre_a["affiliation_category"]==cat).sum(), (post_a["affiliation_category"]==cat).sum()])
    table_arr = np.array(contingency)
    try:
        chi2, p, dof, _ = stats.chi2_contingency(table_arr)
        p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
        print(f"    {pos_name}: chi2({dof}) = {chi2:.2f}, {p_str}")
    except Exception as e:
        print(f"    {pos_name}: failed - {e}")

# Chart 18: Pre vs Post COVID (3-panel: first, last, corresponding)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax_idx, (pos_label, pos_prefix) in enumerate([("First Author", "first"), ("Last Author", "last"), ("Corresponding Author", "corresponding")]):
    ax = axes[ax_idx]
    x = np.arange(len(CAT_ORDER))
    w = 0.35
    pre_vals = [table7[table7["Period"].str.contains("Pre")][f"{pos_prefix}_{cat}_pct"].values[0] for cat in CAT_ORDER]
    post_vals = [table7[table7["Period"].str.contains("Post")][f"{pos_prefix}_{cat}_pct"].values[0] for cat in CAT_ORDER]

    bars1 = ax.bar(x - w/2, pre_vals, w, label="Pre-COVID", color="#78909C", edgecolor="white")
    bars2 = ax.bar(x + w/2, post_vals, w, label="Post-COVID", color="#E53935", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(CAT_ORDER, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"{pos_label}")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", fontsize=7)

fig.suptitle("Research leadership before and after COVID-19 (2000-2019 vs 2020-2025)",
             fontsize=TITLE_SIZE, fontweight="bold")
fig.tight_layout()
save_chart(fig, "chart_18_pre_post_covid_comparison")

# Chart 19: Publication volume with COVID marker
yearly = works.groupby("publication_year").size().reset_index(name="count")
fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
ax.plot(yearly["publication_year"], yearly["count"], marker="o", color=COLORS["total"],
        linewidth=2, markersize=5)
ax.axvline(x=2020, color="#E53935", linestyle="--", linewidth=1.5, label="COVID-19 pandemic")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Publications")
ax.set_title("Ghanaian international collaborative output with COVID-19 onset marker")
ax.legend()
ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_chart(fig, "chart_19_covid_marker_timeline")

# ═══════════════════════════════════════════════════════════
# 5.9 PARTNER BLOC ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n--- Section 5.9: Partner Bloc Analysis ---")

blocs_to_analyze = ["Western", "East Asian", "South Asian", "African",
                     "Latin American", "MENA", "Multi-bloc"]

bloc_rows = []
for bloc in blocs_to_analyze:
    bw = works[works["partner_bloc"] == bloc]
    bw_ids = set(bw["work_id"])
    ba = authorships[authorships["work_id"].isin(bw_ids)]
    if len(bw) == 0:
        continue

    row = {"Partner_bloc": bloc, "N_works": len(bw),
           "Median_authors": bw["author_count"].median(),
           "Median_countries": bw["country_count"].median(),
           "Pct_funded": round(bw["has_funding"].apply(to_bool).mean()*100, 1) if "has_funding" in bw.columns else 0}

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

    bloc_rows.append(row)

table8 = pd.DataFrame(bloc_rows)
table8.to_csv(OUTPUT_DIR / "table8_leadership_by_partner_bloc.csv", index=False)
print("  Saved table8_leadership_by_partner_bloc.csv")

for _, r in table8.iterrows():
    gh_first = r.get("first_Ghanaian_pct", 0) + r.get("first_Dual-affiliated_pct", 0)
    print(f"  {r['Partner_bloc']}: N={r['N_works']:,}, GH first author %={gh_first:.1f}")

# Chart 20: Leadership by partner bloc
fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
blocs_in_data = table8["Partner_bloc"].tolist()
x = np.arange(len(blocs_in_data))
bar_w = 0.25

for i, (pos_label, pos_prefix) in enumerate([("First Author", "first"), ("Last Author", "last"), ("Corresponding", "corresponding")]):
    vals = [r.get(f"{pos_prefix}_Ghanaian_pct", 0) + r.get(f"{pos_prefix}_Dual-affiliated_pct", 0) for _, r in table8.iterrows()]
    offset = (i - 1) * bar_w
    ax.bar(x + offset, vals, bar_w, label=pos_label,
           color=[COLORS["Ghanaian"], COLORS["Dual-affiliated"], COLORS["Non-Ghanaian"]][i],
           edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(blocs_in_data, rotation=30, ha="right")
ax.set_ylabel("% Ghanaian or Dual-affiliated")
ax.set_title("Ghanaian research leadership by partner bloc")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_chart(fig, "chart_20_leadership_by_partner_bloc")

# Chart 21: First author trend by bloc (3-year MA)
print("  Computing trends by bloc...")
fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
bloc_colors = {"Western": "#1565C0", "East Asian": "#E53935", "South Asian": "#F9A825", "African": "#2E7D32"}
bloc_markers = {"Western": "s", "East Asian": "D", "South Asian": "^", "African": "o"}

for bloc in ["Western", "East Asian", "South Asian", "African"]:
    bw = works[works["partner_bloc"] == bloc]
    bw_ids = set(bw["work_id"])
    yearly_data = []
    for yr in sorted(works["publication_year"].unique()):
        yr_bw = bw[bw["publication_year"] == yr]
        yr_ba = authorships[(authorships["work_id"].isin(yr_bw["work_id"])) & (authorships["author_position"] == "first")]
        if len(yr_ba) >= 5:
            gh_pct = yr_ba["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).mean() * 100
            yearly_data.append({"Year": yr, "pct": gh_pct})
    if yearly_data:
        yd = pd.DataFrame(yearly_data)
        ma = yd["pct"].rolling(3, center=True, min_periods=1).mean()
        ax.plot(yd["Year"], ma, marker=bloc_markers.get(bloc, "o"), markersize=4,
                color=bloc_colors.get(bloc, "#616161"), linewidth=2, label=bloc)

ax.set_xlabel("Year")
ax.set_ylabel("% Ghanaian or Dual-affiliated First Author")
ax.set_title("Trends in Ghanaian first authorship by partner bloc, 2000-2025\n(3-year moving average)")
ax.legend(fontsize=9)
ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_chart(fig, "chart_21_first_author_trend_by_bloc")

# Chart 22: Growth by bloc (stacked area)
fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
bloc_yearly = works.groupby(["publication_year", "partner_bloc"]).size().unstack(fill_value=0)
plot_blocs = [b for b in ["Western", "Multi-bloc", "African", "East Asian", "South Asian", "MENA", "Latin American"] if b in bloc_yearly.columns]
bloc_plot_colors = {"Western": "#1565C0", "Multi-bloc": "#78909C", "African": "#2E7D32",
                    "East Asian": "#E53935", "South Asian": "#F9A825", "MENA": "#8E24AA", "Latin American": "#FF8F00"}
ax.stackplot(bloc_yearly.index, [bloc_yearly[b] for b in plot_blocs],
             labels=plot_blocs, colors=[bloc_plot_colors.get(b, "#616161") for b in plot_blocs], alpha=0.85)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Collaborations")
ax.set_title("Growth of Ghanaian international collaborations by partner bloc, 2000-2025")
ax.legend(loc="upper left", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_chart(fig, "chart_22_collaboration_growth_by_bloc")

# ── Western vs Non-Western chi-square ──
print("\n  Western vs Non-Western (binary) chi-square:")
for pos_name, pos_filter in [("First author", "first"), ("Last author", "last"), ("Corresponding author", None)]:
    if pos_filter:
        w_a = authorships[(authorships["work_id"].isin(works[works["western_vs_nonwestern"]=="Western"]["work_id"])) & (authorships["author_position"]==pos_filter)]
        nw_a = authorships[(authorships["work_id"].isin(works[works["western_vs_nonwestern"]=="Non-Western"]["work_id"])) & (authorships["author_position"]==pos_filter)]
    else:
        w_a = authorships[(authorships["work_id"].isin(works[works["western_vs_nonwestern"]=="Western"]["work_id"])) & (authorships["is_corresponding_combined"]==True)]
        nw_a = authorships[(authorships["work_id"].isin(works[works["western_vs_nonwestern"]=="Non-Western"]["work_id"])) & (authorships["is_corresponding_combined"]==True)]
    if len(w_a) > 0 and len(nw_a) > 0:
        w_gh = w_a["affiliation_category"].isin(["Ghanaian","Dual-affiliated"]).sum()
        nw_gh = nw_a["affiliation_category"].isin(["Ghanaian","Dual-affiliated"]).sum()
        tbl = np.array([[w_gh, len(w_a)-w_gh],[nw_gh, len(nw_a)-nw_gh]])
        chi2, p, _, _ = stats.chi2_contingency(tbl)
        p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
        print(f"    {pos_name}: Western={w_gh/len(w_a)*100:.1f}%, Non-Western={nw_gh/len(nw_a)*100:.1f}%, chi2={chi2:.2f}, {p_str}")

# ── Country Deep Dives ──
print("\n--- Country Deep Dives ---")

def country_deep_dive(code, country_name, table_name):
    mask = works["partner_countries"].fillna("").str.contains(f"(^|\\|){code}(\\||$)", regex=True)
    cw = works[mask]
    if len(cw) == 0:
        print(f"  {country_name}: No collaborations found. Skipping.")
        return

    cw_ids = set(cw["work_id"])
    ca = authorships[authorships["work_id"].isin(cw_ids)]

    rows_out = []
    rows_out.append({"Metric": "Total collaborations", "Value": len(cw)})

    # Yearly growth
    yr_counts = cw.groupby("publication_year").size()
    rows_out.append({"Metric": "Earliest year", "Value": yr_counts.index.min()})
    rows_out.append({"Metric": "Latest year", "Value": yr_counts.index.max()})
    rows_out.append({"Metric": "Peak year count", "Value": yr_counts.max()})

    # Fields
    if "field_name" in cw.columns:
        top_f = cw["field_name"].value_counts().head(5)
        for f_name, f_count in top_f.items():
            rows_out.append({"Metric": f"Field: {f_name}", "Value": f"{f_count} ({f_count/len(cw)*100:.1f}%)"})

    # Leadership
    for pos_name, pos_filter in [("first", "first"), ("last", "last")]:
        pos = ca[ca["author_position"] == pos_filter]
        for cat in CAT_ORDER:
            n = (pos["affiliation_category"] == cat).sum()
            pct = n / len(pos) * 100 if len(pos) > 0 else 0
            rows_out.append({"Metric": f"{pos_name.capitalize()} author {cat}", "Value": f"{n} ({pct:.1f}%)"})

    corr = ca[ca["is_corresponding_combined"] == True]
    for cat in CAT_ORDER:
        n = (corr["affiliation_category"] == cat).sum()
        pct = n / len(corr) * 100 if len(corr) > 0 else 0
        rows_out.append({"Metric": f"Corresponding author {cat}", "Value": f"{n} ({pct:.1f}%)"})

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(OUTPUT_DIR / table_name, index=False)
    gh_first_pct = ca[ca["author_position"]=="first"]["affiliation_category"].isin(["Ghanaian","Dual-affiliated"]).mean()*100
    print(f"  {country_name}: N={len(cw):,}, GH first author={gh_first_pct:.1f}%. Saved {table_name}")

country_deep_dive("CN", "China", "table9_country_deep_dive_china.csv")
country_deep_dive("IN", "India", "table10_country_deep_dive_india.csv")
country_deep_dive("ZA", "South Africa", "table11_country_deep_dive_south_africa.csv")
# Brazil deep dive
mask_br = works["partner_countries"].fillna("").str.contains("(^|\\|)BR(\\||$)", regex=True)
br_n = mask_br.sum()
if br_n >= 10:
    country_deep_dive("BR", "Brazil", "table_s6_brazil_deep_dive.csv")
else:
    print(f"  Brazil: Only {br_n} papers. Skipping deep dive (too few).")

# ═══════════════════════════════════════════════════════════
# 6.1 LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════
print("\n--- Section 6.1: Logistic Regression ---")

# Prepare regression dataset at work level
first_auths = authorships[authorships["author_position"] == "first"][["work_id", "affiliation_category"]].copy()
first_auths["gh_first"] = first_auths["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).astype(int)
first_auths = first_auths.drop_duplicates("work_id")

last_auths = authorships[authorships["author_position"] == "last"][["work_id", "affiliation_category"]].copy()
last_auths["gh_last"] = last_auths["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).astype(int)
last_auths = last_auths.drop_duplicates("work_id")

corr_auths = authorships[authorships["is_corresponding_combined"] == True][["work_id", "affiliation_category"]].copy()
# For corresponding, group by work and check if any Ghanaian/Dual
corr_work = corr_auths.groupby("work_id").apply(
    lambda x: int(x["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).any())
).reset_index(name="gh_corr")

reg_df = works[["work_id", "publication_year", "covid_era", "field_name",
                "partner_bloc", "country_count", "author_count",
                "has_funding", "is_oa"]].copy()
reg_df["has_funding_bool"] = reg_df["has_funding"].apply(to_bool).astype(int)
reg_df["is_oa_bool"] = reg_df["is_oa"].apply(to_bool).astype(int)
reg_df["year_centered"] = reg_df["publication_year"] - 2000

reg_df = reg_df.merge(first_auths[["work_id", "gh_first"]], on="work_id", how="left")
reg_df = reg_df.merge(last_auths[["work_id", "gh_last"]], on="work_id", how="left")
reg_df = reg_df.merge(corr_work, on="work_id", how="left")

# Collapse small blocs
bloc_counts = reg_df["partner_bloc"].value_counts()
small_blocs = bloc_counts[bloc_counts < 30].index.tolist()
reg_df["partner_bloc_reg"] = reg_df["partner_bloc"].apply(lambda x: "Other" if x in small_blocs else x)

# Create dummies
reg_df["field_reg"] = reg_df["field_name"].fillna("Other")
health_fields = ["Medicine", "Nursing", "Health Professions"]
reg_df["field_reg"] = reg_df["field_reg"].apply(lambda x: "Health Sciences" if x in health_fields else x)

top_fields = reg_df["field_reg"].value_counts().head(5).index.tolist()
if "Biochemistry, Genetics and Molecular Biology" not in top_fields:
    top_fields.append("Biochemistry, Genetics and Molecular Biology")
reg_df["field_reg"] = reg_df["field_reg"].apply(lambda x: x if x in top_fields else "Other")

# VIF check
print("  VIF check for year_centered and covid_era:")
vif_df = reg_df[["year_centered", "covid_era"]].dropna()
vif_df = add_constant(vif_df)
try:
    for i, col in enumerate(["const", "year_centered", "covid_era"]):
        vif = variance_inflation_factor(vif_df.values, i)
        print(f"    {col}: VIF = {vif:.2f}")
    year_vif = variance_inflation_factor(vif_df.values, 1)
    covid_vif = variance_inflation_factor(vif_df.values, 2)
    high_vif = year_vif > 5 or covid_vif > 5
except Exception as e:
    print(f"    VIF check failed: {e}")
    high_vif = True

def run_logistic_model(dep_var, model_name, predictors_label, exclude_col=None):
    """Run a logistic regression and return results."""
    # Get dummies for categorical vars
    model_df = reg_df.dropna(subset=[dep_var]).copy()

    # Numeric predictors
    numeric_cols = ["year_centered", "country_count", "author_count",
                    "has_funding_bool", "is_oa_bool"]
    if exclude_col and exclude_col in numeric_cols:
        numeric_cols.remove(exclude_col)
    if exclude_col != "covid_era":
        numeric_cols.append("covid_era")

    if exclude_col == "covid_era":
        pass  # already not included
    elif exclude_col == "year_centered":
        numeric_cols = [c for c in numeric_cols if c != "year_centered"]

    # Categorical dummies
    bloc_dummies = pd.get_dummies(model_df["partner_bloc_reg"], prefix="bloc", drop_first=True)
    field_dummies = pd.get_dummies(model_df["field_reg"], prefix="field", drop_first=True)

    X = pd.concat([model_df[numeric_cols].reset_index(drop=True),
                    bloc_dummies.reset_index(drop=True),
                    field_dummies.reset_index(drop=True)], axis=1)
    X = add_constant(X)
    y = model_df[dep_var].reset_index(drop=True)

    # Drop rows with any NaN
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    try:
        model = Logit(y, X.astype(float)).fit(disp=0, maxiter=100)

        results = []
        for var in model.params.index:
            or_val = np.exp(model.params[var])
            ci = np.exp(model.conf_int().loc[var])
            p_val = model.pvalues[var]
            p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
            results.append({
                "Variable": var,
                "OR": round(or_val, 3),
                "95% CI Lower": round(ci[0], 3),
                "95% CI Upper": round(ci[1], 3),
                "p-value": p_str,
            })

        result_df = pd.DataFrame(results)
        print(f"  {model_name} ({predictors_label}): AIC={model.aic:.1f}, pseudo-R2={model.prsquared:.4f}")

        # Print key predictors
        for _, r in result_df.iterrows():
            if r["Variable"] not in ["const"] and r["p-value"] in ["< 0.001"] or (r["p-value"] not in ["< 0.001"] and float(r["p-value"]) < 0.05):
                print(f"    {r['Variable']}: OR={r['OR']} ({r['95% CI Lower']}-{r['95% CI Upper']}), p={r['p-value']}")

        return result_df, model
    except Exception as e:
        print(f"  {model_name}: FAILED - {e}")
        return None, None

# Run models
model_results = {}
for dep_var, dep_name, table_num in [("gh_first", "First Author", 12), ("gh_last", "Last Author", 13), ("gh_corr", "Corresponding", 14)]:
    if high_vif:
        # Run two model sets
        print(f"\n  Model {table_num-11}A - {dep_name} (with year, no covid_era):")
        res_a, mod_a = run_logistic_model(dep_var, f"Model {table_num-11}A", "with year", exclude_col="covid_era")
        print(f"  Model {table_num-11}B - {dep_name} (with covid_era, no year):")
        res_b, mod_b = run_logistic_model(dep_var, f"Model {table_num-11}B", "with covid_era", exclude_col="year_centered")

        # Save the year-based model as primary
        if res_a is not None:
            res_a.to_csv(OUTPUT_DIR / f"table{table_num}_logistic_regression_model{table_num-11}.csv", index=False)
        model_results[dep_name] = (res_a, mod_a) if res_a is not None else (res_b, mod_b)
    else:
        print(f"\n  Model {table_num-11} - {dep_name} (all predictors):")
        res, mod = run_logistic_model(dep_var, f"Model {table_num-11}", "all predictors")
        if res is not None:
            res.to_csv(OUTPUT_DIR / f"table{table_num}_logistic_regression_model{table_num-11}.csv", index=False)
        model_results[dep_name] = (res, mod)

# Forest plots (Charts 13-15)
print("\n--- Forest Plots ---")

def plot_forest(result_df, chart_num, chart_name, title):
    if result_df is None:
        print(f"  Skipping {chart_name} (no model results)")
        return
    # Exclude 'const'
    df = result_df[result_df["Variable"] != "const"].copy()
    if len(df) == 0:
        print(f"  Skipping {chart_name} (no variables)")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(df)*0.4 + 1)))

    y_pos = np.arange(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        color = "#E53935" if r["OR"] > 1 else "#1565C0"
        if r["p-value"] == "< 0.001" or (r["p-value"] != "< 0.001" and float(r["p-value"]) < 0.05):
            marker = "o"
            alpha = 1.0
        else:
            marker = "o"
            alpha = 0.4

        ax.plot(r["OR"], i, marker=marker, color=color, markersize=8, alpha=alpha, zorder=3)
        ax.hlines(i, r["95% CI Lower"], r["95% CI Upper"], color=color, linewidth=1.5, alpha=alpha, zorder=2)

        # Alternating row shading
        if i % 2 == 0:
            ax.axhspan(i - 0.4, i + 0.4, color="#F5F5F5", zorder=0)

    ax.axvline(x=1.0, color="#9E9E9E", linestyle="--", linewidth=1, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Variable"], fontsize=9)
    ax.set_xlabel("Odds Ratio (log scale)")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_chart(fig, chart_name)

for dep_name, chart_num, chart_name in [
    ("First Author", 13, "chart_13_forest_plot_model1"),
    ("Last Author", 14, "chart_14_forest_plot_model2"),
    ("Corresponding", 15, "chart_15_forest_plot_model3")
]:
    res, _ = model_results.get(dep_name, (None, None))
    plot_forest(res, chart_num, chart_name,
                f"Predictors of Ghanaian {dep_name.lower()} in international collaborations")

# ═══════════════════════════════════════════════════════════
# 6.2 TIME TREND ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n--- Section 6.2: Time Trend Analysis ---")

# Mann-Kendall trend test — for all three positions
try:
    for mk_name, mk_filter in [("First author", "first"), ("Last author", "last"), ("Corresponding", None)]:
        yearly_gh_pcts = []
        for yr in sorted(works["publication_year"].unique()):
            if mk_filter:
                yr_auths = authorships[(authorships["publication_year"]==yr) & (authorships["author_position"]==mk_filter)]
            else:
                yr_auths = authorships[(authorships["publication_year"]==yr) & (authorships["is_corresponding_combined"]==True)]
            if len(yr_auths) > 0:
                pct = yr_auths["affiliation_category"].isin(["Ghanaian","Dual-affiliated"]).mean()*100
                yearly_gh_pcts.append(pct)

        # Simple Mann-Kendall
        n_mk = len(yearly_gh_pcts)
        s = 0
        for k in range(n_mk - 1):
            for j in range(k + 1, n_mk):
                s += np.sign(yearly_gh_pcts[j] - yearly_gh_pcts[k])
        var_s = n_mk * (n_mk - 1) * (2 * n_mk + 5) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        p_mk = 2 * stats.norm.sf(abs(z))
        tau = 2 * s / (n_mk * (n_mk - 1))

        # Sen's slope
        slopes = []
        for k in range(n_mk - 1):
            for j in range(k + 1, n_mk):
                slopes.append((yearly_gh_pcts[j] - yearly_gh_pcts[k]) / (j - k))
        sens_slope = np.median(slopes)

        direction = "increasing" if tau > 0 else "decreasing"
        p_str = f"p = {p_mk:.3f}" if p_mk >= 0.001 else "p < 0.001"
        print(f"  {mk_name} Mann-Kendall: tau={tau:.4f}, {p_str}, direction={direction}")
        print(f"    Sen's slope: {sens_slope:.4f} pp/year")
except Exception as e:
    print(f"  Mann-Kendall failed: {e}")

# ═══════════════════════════════════════════════════════════
# 6.3 DUAL-AFFILIATION DEEP DIVE
# ═══════════════════════════════════════════════════════════
print("\n--- Section 6.3: Dual-Affiliation Deep Dive ---")

first_a = authorships[authorships["author_position"] == "first"]
dual_first_by_year = first_a.groupby("publication_year")["affiliation_category"].value_counts(normalize=True).unstack(fill_value=0)

print("  Dual-affiliated as % of first authors:")
if "Dual-affiliated" in dual_first_by_year.columns:
    recent = dual_first_by_year.loc[dual_first_by_year.index >= 2020, "Dual-affiliated"].mean() * 100
    early = dual_first_by_year.loc[dual_first_by_year.index <= 2010, "Dual-affiliated"].mean() * 100
    print(f"    2000-2010 avg: {early:.1f}%, 2020-2025 avg: {recent:.1f}%")

# Most common non-GH countries for dual affiliates
dual_auths = authorships[authorships["affiliation_category"] == "Dual-affiliated"]
if "non_gh_institution_names" in dual_auths.columns:
    dual_countries = []
    for val in dual_auths["all_institution_countries"].dropna():
        for c in str(val).split("|"):
            if c.strip() and c.strip() != "GH":
                dual_countries.append(c.strip())
    if dual_countries:
        dual_country_counts = pd.Series(dual_countries).value_counts().head(10)
        print("  Top 10 non-GH countries for dual-affiliated authors:")
        for c, n in dual_country_counts.items():
            print(f"    {c}: {n:,}")

# Chart 16: Composition of first authorship over time (stacked area)
fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
years_sorted = sorted(dual_first_by_year.index)
cat_data = {}
for cat in ["Non-Ghanaian", "Dual-affiliated", "Ghanaian"]:
    if cat in dual_first_by_year.columns:
        cat_data[cat] = [dual_first_by_year.loc[yr, cat]*100 if yr in dual_first_by_year.index else 0 for yr in years_sorted]
    else:
        cat_data[cat] = [0]*len(years_sorted)

ax.stackplot(years_sorted,
             cat_data.get("Ghanaian", []), cat_data.get("Dual-affiliated", []), cat_data.get("Non-Ghanaian", []),
             labels=["Ghanaian", "Dual-affiliated", "Non-Ghanaian"],
             colors=[COLORS["Ghanaian"], COLORS["Dual-affiliated"], COLORS["Non-Ghanaian"]],
             alpha=0.85)
ax.set_xlabel("Year")
ax.set_ylabel("% of First Authors")
ax.set_title("Composition of first authorship in Ghanaian\ninternational collaborations, 2000-2025")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
save_chart(fig, "chart_16_dual_affiliation_trend")

# ═══════════════════════════════════════════════════════════
# 6.4 INSTITUTIONAL ANALYSIS
# ═══════════════════════════════════════════════════════════
print("\n--- Section 6.4: Institutional Analysis ---")

# Extract Ghanaian institution names per work
inst_records = []
for _, row in authorships[authorships["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"])].iterrows():
    if pd.notna(row.get("gh_institution_names")):
        for inst in str(row["gh_institution_names"]).split("|"):
            if inst.strip():
                inst_records.append({
                    "work_id": row["work_id"],
                    "institution": inst.strip(),
                    "author_position": row["author_position"],
                    "is_corresponding": row.get("is_corresponding_combined", False),
                    "affiliation_category": row["affiliation_category"],
                })

inst_df = pd.DataFrame(inst_records)

if len(inst_df) > 0:
    top_inst = inst_df["institution"].value_counts().head(20)
    inst_rows = []
    for inst_name in top_inst.index:
        sub = inst_df[inst_df["institution"] == inst_name]
        n_works = sub["work_id"].nunique()
        first_sub = sub[sub["author_position"] == "first"]
        last_sub = sub[sub["author_position"] == "last"]
        corr_sub = sub[sub["is_corresponding"] == True]

        inst_rows.append({
            "Institution": inst_name,
            "Total_works": n_works,
            "First_author_instances": len(first_sub),
            "Last_author_instances": len(last_sub),
            "Corresponding_author_instances": len(corr_sub),
        })

    table15 = pd.DataFrame(inst_rows)
    table15.to_csv(OUTPUT_DIR / "table15_top20_institutions.csv", index=False)
    print("  Saved table15_top20_institutions.csv")
    print("  Top 5 institutions:")
    for _, r in table15.head(5).iterrows():
        print(f"    {r['Institution']}: {r['Total_works']:,} works")

    # Chart 17: Top 10 institutions (all 3 positions)
    top10_inst = table15.head(10)
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(top10_inst))
    bar_w = 0.25

    ax.barh(y - bar_w, top10_inst["First_author_instances"], bar_w,
            label="First Author", color=COLORS["Ghanaian"], edgecolor="white")
    ax.barh(y, top10_inst["Last_author_instances"], bar_w,
            label="Last Author", color=COLORS["Non-Ghanaian"], edgecolor="white")
    ax.barh(y + bar_w, top10_inst["Corresponding_author_instances"], bar_w,
            label="Corresponding Author", color=COLORS["Dual-affiliated"], edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(top10_inst["Institution"], fontsize=8)
    ax.set_xlabel("Number of Instances")
    ax.set_title("Research leadership of top Ghanaian institutions\nin international collaborations")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_chart(fig, "chart_17_top_institutions")
else:
    print("  No institutional data available for analysis.")

print("\n" + "=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)
print("  Tables saved: table7-15")
print("  Charts saved: chart_13-22, chart_16-17")
