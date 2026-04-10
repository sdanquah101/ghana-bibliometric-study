import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd, numpy as np

# Write output to file to avoid Windows encoding issues
_outfile = open('analysis_results/ci_results.txt', 'w', encoding='utf-8')
_orig_print = print
def print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    _outfile.write(' '.join(str(a) for a in args) + '\n')
    _outfile.flush()
from config import *
from statsmodels.stats.proportion import proportion_confint

works = pd.read_parquet(INTERMEDIATE_DIR / 'works.parquet')
authorships = pd.read_parquet(INTERMEDIATE_DIR / 'authorships.parquet')

def wilson_ci(count, total):
    if total == 0: return 0, 0
    lo, hi = proportion_confint(count, total, alpha=0.05, method='wilson')
    return lo*100, hi*100

# 1. PARTNER BLOC CIs (Ghanaian+Dual first author)
print("=== PARTNER BLOC CIs ===")
blocs = ["Western", "East Asian", "South Asian", "African", "MENA", "Latin American", "Multi-bloc"]
for bloc in blocs:
    bw = works[works["partner_bloc"] == bloc]
    ba = authorships[(authorships["work_id"].isin(bw["work_id"])) & (authorships["author_position"] == "first")]
    gh = ba["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).sum()
    total = len(ba)
    pct = gh/total*100 if total > 0 else 0
    lo, hi = wilson_ci(gh, total)
    print(f"  {bloc}: N={len(bw):,}, GH+Dual first={pct:.1f}% (95% CI: {lo:.1f}-{hi:.1f}%)")

# 2. COUNTRY DEEP DIVE CIs
print("\n=== COUNTRY DEEP DIVE CIs ===")
for code, name in [("CN", "China"), ("IN", "India"), ("ZA", "South Africa"), ("BR", "Brazil")]:
    mask = works["partner_countries"].fillna("").str.contains(f"(^|\\|){code}(\\||$)", regex=True)
    cw = works[mask]
    ca = authorships[(authorships["work_id"].isin(cw["work_id"])) & (authorships["author_position"] == "first")]
    gh = ca["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).sum()
    total = len(ca)
    pct = gh/total*100 if total > 0 else 0
    lo, hi = wilson_ci(gh, total)
    print(f"  {name}: N={len(cw):,}, GH+Dual first={pct:.1f}% (95% CI: {lo:.1f}-{hi:.1f}%)")

# 3. PRE/POST COVID COMBINED GH+DUAL
print("\n=== PRE/POST COVID COMBINED ===")
for era_name, era_val in [("Pre-COVID", 0), ("Post-COVID", 1)]:
    ew = works[works["covid_era"] == era_val]
    ea = authorships[authorships["work_id"].isin(ew["work_id"])]
    for pos_name, pos_filter in [("First", "first"), ("Last", "last"), ("Corresponding", None)]:
        if pos_filter:
            pos = ea[ea["author_position"] == pos_filter]
        else:
            pos = ea[ea["is_corresponding_combined"] == True]
        gh = (pos["affiliation_category"] == "Ghanaian").sum()
        dual = (pos["affiliation_category"] == "Dual-affiliated").sum()
        total = len(pos)
        print(f"  {era_name} {pos_name}: GH={gh/total*100:.1f}%, Dual={dual/total*100:.1f}%, Combined={((gh+dual)/total*100):.1f}%")

# 4. BILATERAL VS CONSORTIUM decomposition
print("\n=== BILATERAL VS CONSORTIUM ===")
bilateral = works[works["partner_bloc"] != "Multi-bloc"]
consortium = works[works["partner_bloc"] == "Multi-bloc"]
for subset_name, subset_works in [("Bilateral", bilateral), ("Multi-bloc", consortium)]:
    sw_ids = set(subset_works["work_id"])
    sa = authorships[authorships["work_id"].isin(sw_ids)]
    for pos_name, pos_filter in [("First", "first"), ("Last", "last"), ("Corresponding", None)]:
        if pos_filter:
            pos = sa[sa["author_position"] == pos_filter]
        else:
            pos = sa[sa["is_corresponding_combined"] == True]
        gh_dual = pos["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).sum()
        pct = gh_dual/len(pos)*100 if len(pos) > 0 else 0
        print(f"  {subset_name} (N={len(subset_works):,}) {pos_name}: GH+Dual={pct:.1f}%")

# 5. SDG analysis - quick check
print("\n=== SDG ANALYSIS ===")
try:
    sdgs = pd.read_parquet(INTERMEDIATE_DIR / 'sdgs.parquet')
    print(f"  SDG records: {len(sdgs):,}")
    top_sdgs = sdgs.groupby("display_name")["work_id"].nunique().sort_values(ascending=False).head(5)
    for sdg, n in top_sdgs.items():
        print(f"  {sdg}: {n:,} works")
    # Check leadership for top 2 SDGs
    for sdg_name in top_sdgs.index[:3]:
        sdg_works = set(sdgs[sdgs["display_name"] == sdg_name]["work_id"])
        sa = authorships[(authorships["work_id"].isin(sdg_works)) & (authorships["author_position"] == "first")]
        gh_pct = sa["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).mean() * 100
        print(f"    {sdg_name}: GH+Dual first author = {gh_pct:.1f}%")
except Exception as e:
    print(f"  SDG analysis failed: {e}")
