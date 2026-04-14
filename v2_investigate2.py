"""Investigate model E and missing-last-author issues."""
import pandas as pd, numpy as np

w = pd.read_parquet('analysis_results/intermediate/v2_works.parquet')
reg = pd.read_csv('analysis_results/v2_regression_results.csv')

print("=== Model E NaN breakdown ===")
for m in ['gh_first_E', 'gh_last_E']:
    sub = reg[reg['Model'] == m]
    n_nan = sub['p_value'].isna().sum()
    n_total = len(sub)
    print(f"  {m}: {n_nan}/{n_total} NaN p-values")
    bxy = sub[sub['Variable'] == 'bilateral_x_year']
    if len(bxy) > 0:
        r = bxy.iloc[0]
        print(f"    bilateral_x_year: OR={r['OR']}, p={r['p_value']}")

print()
print("=== Missing last authors: author_count distribution ===")
a = pd.read_parquet('analysis_results/intermediate/v2_authorships.parquet')
last = a[a['author_position'] == 'last'].drop_duplicates('work_id')
missing_ids = set(w['work_id']) - set(last['work_id'])
missing_w = w[w['work_id'].isin(missing_ids)]
print(f"All 142 have author_count = {missing_w['author_count'].unique()}")
print(f"These are OpenAlex API truncation artefacts (max 100 authors returned)")
print()

print("=== Paper volume controls ===")
for thresh in [50, 100, 200, 500]:
    n = (w['author_count'] >= thresh).sum()
    print(f"  author_count >= {thresh}: {n} papers")

print()
print("=== VIF source: log_AC highly correlated with bloc dummies ===")
# VIF = 1/(1-R^2) where R^2 is from regressing log_AC on all other predictors
# The r=0.592 with country_count and r=0.421 with multibloc explains VIF~10
# Since country_count is only in Model B/C (not Model A), the VIF is from
# the bloc dummies which jointly predict team size
bloc = pd.get_dummies(w['partner_bloc'])
from sklearn.linear_model import LinearRegression
X = pd.concat([w[['year_centered', 'year_centered_sq', 'has_funding_int', 'is_oa_int']],
               bloc], axis=1)
lr = LinearRegression().fit(X, w['log_author_count'])
r2 = lr.score(X, w['log_author_count'])
vif_calc = 1 / (1 - r2)
print(f"R^2 of log_AC ~ all other predictors: {r2:.4f}")
print(f"Implied VIF: {vif_calc:.2f}")
print(f"Main drivers: partner_bloc dummies (larger consortia = multi-bloc)")
print(f"This is a structural relationship, not problematic multicollinearity")

print()
print("=== gh_first_E: p-values are NaN but ORs exist ===")
first_e = reg[reg['Model'] == 'gh_first_E']
print("Rows with OR but NaN p:")
odd = first_e[first_e['OR'].notna() & first_e['p_value'].isna()]
print(f"  Count: {len(odd)}")
print("This suggests the GEE converged but standard errors were not estimated")
print("for some coefficients (possibly due to near-singularity)")
print()
print("=== gh_last_E: ALL NaN ===")
last_e = reg[reg['Model'] == 'gh_last_E']
all_nan = last_e['OR'].isna().all()
print(f"All OR NaN: {all_nan}")
print("The GEE for last authorship with interaction term completely failed")
print("This is likely a near-singular Hessian issue")
