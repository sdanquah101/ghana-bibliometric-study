import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd, numpy as np
from config import *
from statsmodels.api import Logit, add_constant

works = pd.read_parquet(INTERMEDIATE_DIR / 'works.parquet')
authorships = pd.read_parquet(INTERMEDIATE_DIR / 'authorships.parquet')

print("=== BIVARIATE TREND (year only) ===")
for pos_name, pos_filter in [('First', 'first'), ('Last', 'last'), ('Corresponding', None)]:
    if pos_filter:
        pos_data = authorships[authorships['author_position'] == pos_filter].copy()
    else:
        pos_data = authorships[authorships['is_corresponding_combined'] == True].copy()
    pos_data['gh'] = pos_data['affiliation_category'].isin(['Ghanaian', 'Dual-affiliated']).astype(int)
    X = add_constant(pos_data['publication_year'] - 2000)
    m = Logit(pos_data['gh'], X).fit(disp=0)
    c = m.params.iloc[1]
    o = np.exp(c)
    ci = np.exp(m.conf_int().iloc[1])
    d = "INCREASING" if c > 0 else "DECREASING"
    print(f"  {pos_name}: coeff={c:.5f}, OR={o:.4f} ({ci[0]:.4f}-{ci[1]:.4f}), {d}")

print("\n=== MULTIVARIATE (year_centered from model) ===")
print("  First: OR=1.019 -> INCREASING after controls")
print("  Last: year not significant (p=0.685)")
print("  Corresponding: OR=1.018 -> INCREASING after controls")

print(f"\n=== MULTI-BLOC CONTEXT ===")
n_mb = (works['partner_bloc'] == 'Multi-bloc').sum()
print(f"  N={n_mb:,} ({n_mb/len(works)*100:.1f}% of study set)")

print("\n=== INSTITUTIONAL LEADERSHIP RATES ===")
def to_bool(val):
    if pd.isna(val): return False
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() == 'true'
    return bool(val)

gh_auths = authorships[authorships['affiliation_category'].isin(['Ghanaian', 'Dual-affiliated'])].copy()
inst_records = []
for _, row in gh_auths.iterrows():
    if pd.notna(row.get('gh_institution_names')):
        for inst in str(row['gh_institution_names']).split('|'):
            if inst.strip():
                inst_records.append({
                    'work_id': row['work_id'],
                    'institution': inst.strip(),
                    'author_position': row['author_position'],
                    'is_corresponding': row.get('is_corresponding_combined', False),
                })
inst_df = pd.DataFrame(inst_records)
top5 = inst_df['institution'].value_counts().head(5).index.tolist()
for inst_name in top5:
    sub = inst_df[inst_df['institution'] == inst_name]
    n_works = sub['work_id'].nunique()
    inst_work_ids = set(sub['work_id'])
    inst_auths = authorships[authorships['work_id'].isin(inst_work_ids)]
    
    first = inst_auths[inst_auths['author_position'] == 'first']
    f_gh = first['affiliation_category'].isin(['Ghanaian', 'Dual-affiliated']).sum()
    f_pct = f_gh / len(first) * 100
    
    last = inst_auths[inst_auths['author_position'] == 'last']
    l_gh = last['affiliation_category'].isin(['Ghanaian', 'Dual-affiliated']).sum()
    l_pct = l_gh / len(last) * 100
    
    corr = inst_auths[inst_auths['is_corresponding_combined'] == True]
    c_gh = corr['affiliation_category'].isin(['Ghanaian', 'Dual-affiliated']).sum()
    c_pct = c_gh / len(corr) * 100
    
    print(f"  {inst_name}: N={n_works:,}, First={f_pct:.1f}%, Last={l_pct:.1f}%, Corr={c_pct:.1f}%")
