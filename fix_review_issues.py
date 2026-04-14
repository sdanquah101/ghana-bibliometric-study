"""
FIX SCRIPT: Address review issues
1. Fix institutional analysis (non-tautological)
2. Verify VIF computation 
3. Check singleton rate for clustered SEs
4. Compute Hosmer-Lemeshow for all models
"""
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
from statsmodels.stats.proportion import proportion_confint
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

works = pd.read_parquet("analysis_results/intermediate/works.parquet")
authorships = pd.read_parquet("analysis_results/intermediate/authorships.parquet")
OUTPUT_DIR = Path("analysis_results")

# ================================================================
# 1. FIX INSTITUTIONAL ANALYSIS
# ================================================================
print("=" * 80)
print("1. FIXED INSTITUTIONAL ANALYSIS")
print("=" * 80)
print("Selecting ALL papers involving each institution (any position),")
print("then computing % with GH/Dual/Non-GH first, last, corresponding.\n")

# Get first/last/corr author categories for each work
first = authorships[authorships["author_position"] == "first"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
first.columns = ["work_id", "first_cat"]
last = authorships[authorships["author_position"] == "last"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
last.columns = ["work_id", "last_cat"]
corr = authorships[authorships["is_corresponding_combined"] == True].drop_duplicates("work_id")[["work_id", "affiliation_category"]]
corr.columns = ["work_id", "corr_cat"]

works_lead = works[["work_id"]].merge(first, on="work_id", how="left")
works_lead = works_lead.merge(last, on="work_id", how="left")
works_lead = works_lead.merge(corr, on="work_id", how="left")

# Get institution names for GH-affiliated authors
gh_auths = authorships[authorships["has_gh_affiliation"] == True].copy()

# Find top institutions by total paper involvement (any position)
inst_papers = {}
for _, row in gh_auths[["work_id", "gh_institution_names"]].dropna(subset=["gh_institution_names"]).iterrows():
    for inst in str(row["gh_institution_names"]).split("|"):
        inst = inst.strip()
        if inst:
            if inst not in inst_papers:
                inst_papers[inst] = set()
            inst_papers[inst].add(row["work_id"])

# Top 10 by total paper count
top10 = sorted(inst_papers.items(), key=lambda x: -len(x[1]))[:10]

print(f"{'Institution':50s} {'N':>6s} {'First%':>7s} {'Last%':>6s} {'Corr%':>6s}")
print("-" * 80)

inst_results = []
for inst_name, work_ids in top10:
    # ALL papers involving this institution
    sub = works_lead[works_lead["work_id"].isin(work_ids)]
    n = len(sub)
    
    for pos, col in [("First", "first_cat"), ("Last", "last_cat"), ("Corresponding", "corr_cat")]:
        valid = sub[col].dropna()
        gh_dual = ((valid == "Ghanaian") | (valid == "Dual-affiliated")).sum()
        gh_only = (valid == "Ghanaian").sum()
        pct_comb = 100 * gh_dual / len(valid) if len(valid) > 0 else 0
        pct_gh = 100 * gh_only / len(valid) if len(valid) > 0 else 0
        
        inst_results.append({
            "Institution": inst_name,
            "N": n,
            "Position": pos,
            "GH_Dual_Pct": round(pct_comb, 1),
            "GH_Only_Pct": round(pct_gh, 1),
        })
    
    # Print summary row
    first_pct = [r["GH_Dual_Pct"] for r in inst_results if r["Institution"] == inst_name and r["Position"] == "First"][0]
    last_pct = [r["GH_Dual_Pct"] for r in inst_results if r["Institution"] == inst_name and r["Position"] == "Last"][0]
    corr_pct = [r["GH_Dual_Pct"] for r in inst_results if r["Institution"] == inst_name and r["Position"] == "Corresponding"][0]
    print(f"{inst_name[:50]:50s} {n:6d} {first_pct:6.1f}% {last_pct:5.1f}% {corr_pct:5.1f}%")

inst_df = pd.DataFrame(inst_results)
inst_df.to_csv(OUTPUT_DIR / "institutional_leadership_fixed.csv", index=False)
print(f"\nSaved to {OUTPUT_DIR / 'institutional_leadership_fixed.csv'}")


# ================================================================
# 2. VERIFY VIF COMPUTATION
# ================================================================
print("\n" + "=" * 80)
print("2. VIF VERIFICATION")
print("=" * 80)

# Rebuild regression dataset
reg = works.merge(first, on="work_id", how="left")
reg = reg.merge(last, on="work_id", how="left")
reg = reg.merge(corr, on="work_id", how="left")
reg["gh_first"] = reg["first_cat"].isin(["Ghanaian", "Dual-affiliated"]).astype(int)

# Collapse blocs
bloc_counts = reg["partner_bloc"].value_counts()
small_blocs = bloc_counts[bloc_counts < 50].index.tolist()
reg["partner_bloc_reg"] = reg["partner_bloc"].replace({b: "Other" for b in small_blocs})

bloc_dummies = pd.get_dummies(reg["partner_bloc_reg"], prefix="bloc", drop_first=False)
field_dummies = pd.get_dummies(reg["field_reg"], prefix="field", drop_first=False)

bloc_ref = "bloc_African"
field_ref = "field_Health Sciences"
bloc_cols = [c for c in bloc_dummies.columns if c != bloc_ref]
field_cols = [c for c in field_dummies.columns if c != field_ref]

reg["has_funding_int"] = reg["has_funding_bool"]
reg["is_oa_int"] = reg["is_oa_bool"]

common_predictors = ["author_count", "year_centered", "covid_era", "has_funding_int", "is_oa_int"]

# Model A matrix (WITH constant)
X_A = pd.concat([reg[common_predictors], bloc_dummies[bloc_cols], field_dummies[field_cols]], axis=1).astype(float)
X_A_const = add_constant(X_A)

# VIF WITH constant included in matrix (WRONG - this inflates VIF)
print("\n--- VIF computed on matrix INCLUDING constant (current, potentially wrong): ---")
X_vif_with_const = X_A_const.drop(columns=["const"], errors="ignore")
mask = X_vif_with_const.notna().all(axis=1)
X_clean = X_vif_with_const[mask].astype(float)
for col in ["year_centered", "covid_era", "is_oa_int"]:
    i = list(X_clean.columns).index(col)
    vif = variance_inflation_factor(X_clean.values, i)
    print(f"  {col}: VIF = {vif:.2f}")

# VIF WITHOUT constant (CORRECT for centered variables)
print("\n--- VIF computed on predictor matrix WITHOUT constant (standard approach): ---")
X_vif_no_const = X_A.copy()
mask2 = X_vif_no_const.notna().all(axis=1)
X_clean2 = X_vif_no_const[mask2].astype(float)
for col in ["year_centered", "covid_era", "is_oa_int", "author_count", "has_funding_int"]:
    i = list(X_clean2.columns).index(col)
    vif = variance_inflation_factor(X_clean2.values, i)
    print(f"  {col}: VIF = {vif:.2f}")

# VIF with add_constant wrapper (statsmodels standard)
print("\n--- VIF using add_constant wrapper (statsmodels standard): ---")
X_for_vif = add_constant(X_A[mask2])
for j, col in enumerate(X_for_vif.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X_for_vif.values, j)
    if col in ["year_centered", "covid_era", "is_oa_int", "author_count", "has_funding_int"]:
        print(f"  {col}: VIF = {vif:.2f}")

# Year-only model (no covid_era)
print("\n--- VIF for year-only model (dropping covid_era): ---")
X_year_only = X_A.drop(columns=["covid_era"]).copy()
mask3 = X_year_only.notna().all(axis=1)
X_clean3 = add_constant(X_year_only[mask3].astype(float))
for col in ["year_centered", "is_oa_int", "author_count", "has_funding_int"]:
    j = list(X_clean3.columns).index(col)
    vif = variance_inflation_factor(X_clean3.values, j)
    print(f"  {col}: VIF = {vif:.2f}")

# Correlation between year_centered and covid_era
corr_year_covid = reg["year_centered"].corr(reg["covid_era"])
print(f"\n  Pearson correlation(year_centered, covid_era) = {corr_year_covid:.4f}")
print(f"  R² = {corr_year_covid**2:.4f}")
print(f"  Theoretical VIF from R² alone = {1/(1-corr_year_covid**2):.2f}")

# Also check year_centered vs is_oa
corr_year_oa = reg["year_centered"].corr(reg["is_oa_int"])
print(f"\n  Pearson correlation(year_centered, is_oa_int) = {corr_year_oa:.4f}")


# ================================================================
# 3. SINGLETON ANALYSIS FOR CLUSTERED SEs
# ================================================================
print("\n" + "=" * 80)
print("3. SINGLETON ANALYSIS")
print("=" * 80)

# How many first-author IDs appear in only one paper?
first_authors = authorships[
    (authorships["author_position"] == "first") & 
    (authorships["work_id"].isin(set(works["work_id"])))
][["work_id", "author_id"]].drop_duplicates("work_id")

author_counts = first_authors["author_id"].value_counts()
n_singletons = (author_counts == 1).sum()
n_multi = (author_counts > 1).sum()
total_authors = len(author_counts)

print(f"  Total unique first authors: {total_authors:,}")
print(f"  Singleton (1 paper): {n_singletons:,} ({100*n_singletons/total_authors:.1f}%)")
print(f"  Multi-paper (>1): {n_multi:,} ({100*n_multi/total_authors:.1f}%)")
print(f"  Max papers by one first author: {author_counts.max()}")
print(f"  Median papers per first author: {author_counts.median():.0f}")
print(f"  Mean papers per first author: {author_counts.mean():.2f}")

# Distribution
for threshold in [1, 2, 3, 5, 10]:
    n = (author_counts <= threshold).sum()
    print(f"  Authors with <={threshold} papers: {n:,} ({100*n/total_authors:.1f}%)")

# Check if NaN author_ids are the problem
n_null = first_authors["author_id"].isna().sum()
print(f"\n  Null author_id in first authors: {n_null:,}")

# Try clustered SE with string-based author_id
print("\n  Attempting clustered SE fix...")
y = reg["gh_first"]
X = X_A_const.copy()
mask_fit = X.notna().all(axis=1) & y.notna()

# Use author_id from first_authors merge
reg_with_aid = reg.merge(
    first_authors.rename(columns={"author_id": "cluster_id"}), 
    on="work_id", how="left"
)

# Fill NaN cluster IDs with unique values
null_mask = reg_with_aid["cluster_id"].isna()
if null_mask.any():
    reg_with_aid.loc[null_mask, "cluster_id"] = [f"null_{i}" for i in range(null_mask.sum())]

cluster_ids = reg_with_aid["cluster_id"][mask_fit]

try:
    model_cl = Logit(y[mask_fit], X[mask_fit]).fit(disp=0, cov_type='cluster', 
                                                     cov_kwds={'groups': cluster_ids.astype(str)})
    print("  Clustered SE succeeded!")
    print(f"  Year OR = {np.exp(model_cl.params['year_centered']):.4f}")
    ci_cl = np.exp(model_cl.conf_int().loc["year_centered"])
    print(f"  Year CI = [{ci_cl.iloc[0]:.4f}, {ci_cl.iloc[1]:.4f}]")
    print(f"  Year p = {model_cl.pvalues['year_centered']:.6f}")
    
    # Compare with HC1
    model_hc = Logit(y[mask_fit], X[mask_fit]).fit(disp=0, cov_type='HC1')
    print(f"\n  HC1 comparison:")
    print(f"  Year OR = {np.exp(model_hc.params['year_centered']):.4f}")
    ci_hc = np.exp(model_hc.conf_int().loc["year_centered"])
    print(f"  Year CI = [{ci_hc.iloc[0]:.4f}, {ci_hc.iloc[1]:.4f}]")
    print(f"  Year p = {model_hc.pvalues['year_centered']:.6f}")
    
except Exception as e:
    print(f"  Clustered SE failed: {e}")


# ================================================================
# 4. HOSMER-LEMESHOW FOR ALL PRIMARY MODELS
# ================================================================
print("\n" + "=" * 80)
print("4. HOSMER-LEMESHOW TEST RESULTS")
print("=" * 80)

def hosmer_lemeshow(y, y_pred, g=10):
    try:
        data = pd.DataFrame({"y": y, "p": y_pred})
        data["group"] = pd.qcut(data["p"], g, duplicates="drop")
        obs = data.groupby("group")["y"].sum()
        exp = data.groupby("group")["p"].sum()
        n = data.groupby("group").size()
        hl_stat = (((obs - exp) ** 2) / (exp * (1 - exp / n))).sum()
        n_groups = len(obs)
        p_val = 1 - stats.chi2.cdf(hl_stat, n_groups - 2)
        return hl_stat, p_val, n_groups
    except:
        return np.nan, np.nan, 0

outcomes = {
    "gh_first": "First Author",
    "gh_last": reg["last_cat"].isin(["Ghanaian", "Dual-affiliated"]).astype(int),
    "gh_corr": reg["corr_cat"].isin(["Ghanaian", "Dual-affiliated"]).astype(int),
}

reg["gh_last"] = reg["last_cat"].isin(["Ghanaian", "Dual-affiliated"]).astype(int)
reg["gh_corr"] = reg["corr_cat"].isin(["Ghanaian", "Dual-affiliated"]).astype(int)

for out_var, label in [("gh_first", "First Author"), ("gh_last", "Last Author"), ("gh_corr", "Corresponding Author")]:
    y = reg[out_var]
    mask_fit = X_A_const.notna().all(axis=1) & y.notna()
    model = Logit(y[mask_fit], X_A_const[mask_fit]).fit(disp=0, cov_type='HC1')
    y_pred = model.predict(X_A_const[mask_fit])
    hl_chi2, hl_p, n_groups = hosmer_lemeshow(y[mask_fit].values, y_pred.values)
    status = "PASS" if hl_p > 0.05 else "FAIL"
    print(f"  Model A - {label}: chi2={hl_chi2:.2f}, p={hl_p:.4f}, groups={n_groups} [{status}]")
    print(f"    AIC={model.aic:.1f}, Pseudo-R2={model.prsquared:.4f}")

print("\nDone.")
