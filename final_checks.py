import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd, numpy as np
from config import *

_outfile = open('analysis_results/final_checks.txt', 'w', encoding='utf-8')
_orig_print = print
def print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    _outfile.write(' '.join(str(a) for a in args) + '\n')
    _outfile.flush()

works = pd.read_parquet(INTERMEDIATE_DIR / 'works.parquet')
authorships = pd.read_parquet(INTERMEDIATE_DIR / 'authorships.parquet')

def to_bool(val):
    if pd.isna(val): return False
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() == 'true'
    return bool(val)

# ── CHECK 1: Citation impact denominators ──
print("=== CHECK 1: CITATION DENOMINATORS ===")
first_auths = authorships[authorships["author_position"] == "first"][["work_id", "affiliation_category"]].drop_duplicates("work_id")
first_auths.rename(columns={"affiliation_category": "first_cat"}, inplace=True)
wc = works.merge(first_auths, on="work_id", how="left")

# FWCI median analysis: how many have FWCI?
for cat in ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]:
    subset_fwci = wc[(wc["first_cat"] == cat) & wc["fwci"].notna()]
    subset_all = wc[wc["first_cat"] == cat]
    print(f"  {cat}: FWCI available={len(subset_fwci):,}, total={len(subset_all):,}")

# Top 10% analysis
if "is_in_top_10_percent" in wc.columns:
    print("\n  Top 10% column exists. Checking denominators:")
    for cat in ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]:
        # What denominator was used? Check if NaN top10 are excluded
        subset = wc[wc["first_cat"] == cat]
        subset_notna = subset[subset["is_in_top_10_percent"].notna()]
        top10_true = subset_notna["is_in_top_10_percent"].apply(to_bool).sum()
        print(f"  {cat}: total={len(subset):,}, with top10 data={len(subset_notna):,}, in top10={top10_true:,}")
else:
    print("  No is_in_top_10_percent column. Checking fwci-based top10:")
    fwci_threshold = wc["fwci"].quantile(0.90)
    print(f"  FWCI 90th percentile threshold: {fwci_threshold:.2f}")
    for cat in ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]:
        subset = wc[wc["first_cat"] == cat]
        subset_fwci = subset[subset["fwci"].notna()]
        top10 = (subset_fwci["fwci"] >= fwci_threshold).sum()
        print(f"  {cat}: FWCI available={len(subset_fwci):,}, in top10={top10:,} ({top10/len(subset_fwci)*100:.1f}%)")

# ── CHECK 2: COVID sensitivity with COMBINED rates ──
print("\n=== CHECK 2: COVID SENSITIVITY COMBINED ===")
# Remove COVID papers, then compute combined GH+Dual
if "abstract" in works.columns:
    covid_mask = works["abstract"].fillna("").str.contains("COVID|SARS-CoV|coronavirus|pandemic", case=False, regex=True)
elif "title" in works.columns:
    covid_mask = works["title"].fillna("").str.contains("COVID|SARS-CoV|coronavirus|pandemic", case=False, regex=True)
else:
    covid_mask = pd.Series(False, index=works.index)
    print("  WARNING: No abstract or title column for COVID detection")

non_covid = works[~covid_mask]
print(f"  COVID papers removed: {covid_mask.sum():,}")
print(f"  Non-COVID papers: {len(non_covid):,}")

for era_name, era_val in [("Pre-COVID", 0), ("Post-COVID", 1)]:
    ew = non_covid[non_covid["covid_era"] == era_val]
    ea = authorships[(authorships["work_id"].isin(ew["work_id"])) & (authorships["author_position"] == "first")]
    gh = (ea["affiliation_category"] == "Ghanaian").sum()
    dual = (ea["affiliation_category"] == "Dual-affiliated").sum()
    total = len(ea)
    gh_pct = gh/total*100
    dual_pct = dual/total*100
    combined = (gh+dual)/total*100
    print(f"  {era_name}: GH={gh_pct:.1f}%, Dual={dual_pct:.1f}%, Combined={combined:.1f}%, N={len(ew):,}")

# ── CHECK 8: Corresponding author imputation ──
print("\n=== CHECK 8: CORRESPONDING AUTHOR IMPUTATION ===")
# How many papers have first author = corresponding author?
corr_auths = authorships[authorships["is_corresponding_combined"] == True]
first_auths_all = authorships[authorships["author_position"] == "first"]

# For each work, check if first author is also corresponding
works_with_corr = corr_auths["work_id"].unique()
works_with_first = first_auths_all["work_id"].unique()
print(f"  Works with any corresponding author: {len(works_with_corr):,}")
print(f"  Works with first author: {len(works_with_first):,}")

# Merge to check overlap
first_corr = authorships[(authorships["author_position"] == "first") & (authorships["is_corresponding_combined"] == True)]
print(f"  Works where first author IS corresponding: {first_corr['work_id'].nunique():,}")
first_is_corr_pct = first_corr["work_id"].nunique() / len(works) * 100
print(f"  Rate: {first_is_corr_pct:.1f}%")

# How many corresponding authors per work on average?
corr_per_work = corr_auths.groupby("work_id").size()
print(f"  Mean corresponding authors per work: {corr_per_work.mean():.2f}")
print(f"  Median corresponding authors per work: {corr_per_work.median():.0f}")
print(f"  Works with exactly 1 corresponding author: {(corr_per_work == 1).sum():,} ({(corr_per_work == 1).mean()*100:.1f}%)")
print(f"  Works with >1 corresponding author: {(corr_per_work > 1).sum():,} ({(corr_per_work > 1).mean()*100:.1f}%)")

# Check the source of corresponding author data
if "is_corresponding" in authorships.columns:
    native_corr = authorships["is_corresponding"].apply(to_bool).sum()
    print(f"\n  Native is_corresponding=True: {native_corr:,}")
if "is_corresponding_combined" in authorships.columns:
    combined_corr = (authorships["is_corresponding_combined"] == True).sum()
    print(f"  Combined is_corresponding_combined=True: {combined_corr:,}")

_outfile.close()
