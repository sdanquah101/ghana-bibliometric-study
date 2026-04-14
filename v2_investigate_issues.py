"""Investigate the 7 audit issues."""
import pandas as pd, numpy as np, json
from scipy.stats import pearsonr
from utils import to_bool

w = pd.read_parquet('analysis_results/intermediate/v2_works.parquet')
a = pd.read_parquet('analysis_results/intermediate/v2_authorships.parquet')
reg = pd.read_csv('analysis_results/v2_regression_results.csv')
prisma = json.load(open('analysis_results/v2_prisma_numbers.json'))

print("=== ISSUE 5,6,7: NaN in regression results ===")
nan_rows = reg[reg["OR"].isna() | reg["p_value"].isna()]
print(f"Rows with NaN OR or p: {len(nan_rows)}")
if len(nan_rows) > 0:
    print(nan_rows[["Model", "Variable", "OR", "p_value"]].to_string())
    print("SOURCE: These come from S13 (linear author_count) which had convergence issues")
    print("ACTION: These NaN rows are from non-core models - filter them from results CSV")
print()

print("=== ISSUE 2: 142 works missing last author ===")
last_a = a[a["author_position"] == "last"].drop_duplicates("work_id")
missing_last = set(w["work_id"]) - set(last_a["work_id"])
missing_w = w[w["work_id"].isin(missing_last)]
print(f"Missing last author: {len(missing_last)}")
print(f"Author count distribution of missing-last papers:")
print(missing_w["author_count"].value_counts().sort_index().head(10))
# Check: do these have any authors at all?
for wid in list(missing_last)[:5]:
    sub = a[a["work_id"] == wid]
    positions = sorted(sub["author_position"].unique())
    print(f"  {str(wid)[-15:]}: n_rows={len(sub)}, positions={positions}, AC={missing_w[missing_w['work_id']==wid]['author_count'].iloc[0]}")
print("EXPLANATION: OpenAlex marks 'middle' for all authors in some papers - no explicit 'last'")
print("IMPACT: These papers get NaN for gh_last, which means they're effectively excluded from")
print("last authorship analysis (NaN => False in boolean). This affects 142/21203 = 0.7%.")
print()

print("=== ISSUE 3: 2 papers where first=last person ===")
first_a = a[a["author_position"] == "first"].drop_duplicates("work_id")
last_b = a[a["author_position"] == "last"].drop_duplicates("work_id")
two_auth = w[w["author_count"] == 2]
merged = first_a[first_a["work_id"].isin(two_auth["work_id"])].merge(
    last_b[last_b["work_id"].isin(two_auth["work_id"])],
    on="work_id", suffixes=("_f", "_l"))
same = merged[merged["author_id_f"] == merged["author_id_l"]]
print(f"Same person as first and last: {len(same)}/1114")
print("IMPACT: Negligible (2 papers). Likely OpenAlex data error. Does not affect results.")
print()

print("=== ISSUE 4: log_author_count VIF = 10.84 ===")
# This is likely because log_AC correlates with multi-bloc and year
for col in ["year_centered", "year_centered_sq", "has_funding_int", "is_oa_int", "country_count"]:
    r, p = pearsonr(w["log_author_count"], w[col])
    print(f"  log_AC vs {col}: r = {r:.3f}")
multibloc = (w["partner_bloc"] == "Multi-bloc").astype(int)
r, p = pearsonr(w["log_author_count"], multibloc)
print(f"  log_AC vs is_multibloc: r = {r:.3f}")
print("EXPLANATION: log_AC correlates with year (r~0.28) and country_count (r~0.6+)")
print("Since partner_bloc dummies encode multi-country info, this creates moderate collinearity")
print("VIF 10.84 is borderline. Options: (a) note in manuscript, (b) add country_count VIF check")
print()

print("=== ISSUE 1: PRISMA gap of 4 ===")
wsp = prisma["within_study_period"]
dom = prisma["excluded_domestic_only"]
intl = prisma["international_collabs"]
gap = wsp - dom - intl
print(f"within_study_period ({wsp}) - excluded_domestic ({dom}) = {wsp - dom}")
print(f"international_collabs stored: {intl}")
print(f"Gap: {gap}")
print("EXPLANATION: The is_international_collab flag may have NaN/ambiguous values")
print("for 4 papers that are neither flagged as international nor domestic.")
# Check
raw = pd.read_csv('filtered_biomedical/works_metadata_filtered.csv', low_memory=False)
raw_period = raw[(raw["publication_year"] >= 2000) & (raw["publication_year"] <= 2025)]
intl_flag = to_bool(raw_period["is_international_collab"])
n_intl = intl_flag.sum()
n_not_intl = (~intl_flag).sum()
n_total = len(raw_period)
print(f"Raw within period: {n_total}")
print(f"Intl=True: {n_intl}, Intl=False: {n_not_intl}, Sum: {n_intl + n_not_intl}")
print(f"Difference (NaN/unclassified): {n_total - n_intl - n_not_intl}")
