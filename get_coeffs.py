import pandas as pd
r = pd.read_csv('analysis_results/regression_results.csv')
for model in ['gh_first_A','gh_last_A','gh_corr_A']:
    row = r[(r['Model']==model) & (r['Variable']=='bloc_Multi-bloc')]
    if len(row): print(f"{model} multi-bloc OR={row.iloc[0]['OR']:.4f} [{row.iloc[0]['CI_lo']:.4f},{row.iloc[0]['CI_hi']:.4f}]")
for model in ['gh_first_B','gh_last_B','gh_corr_B']:
    row = r[(r['Model']==model) & (r['Variable']=='country_count')]
    if len(row): print(f"{model} country_count OR={row.iloc[0]['OR']:.4f} [{row.iloc[0]['CI_lo']:.4f},{row.iloc[0]['CI_hi']:.4f}]")
for model in ['gh_first_C','gh_last_C','gh_corr_C']:
    for v in ['bloc_Multi-bloc','country_count']:
        row = r[(r['Model']==model) & (r['Variable']==v)]
        if len(row): print(f"{model} {v} OR={row.iloc[0]['OR']:.4f} [{row.iloc[0]['CI_lo']:.4f},{row.iloc[0]['CI_hi']:.4f}]")
