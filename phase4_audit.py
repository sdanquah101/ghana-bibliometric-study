"""
PHASE 4: DATA AUDIT
Ghana Bibliometric Study
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

works = pd.read_parquet("analysis_results/intermediate/works.parquet")
authorships = pd.read_parquet("analysis_results/intermediate/authorships.parquet")

print("=" * 80)
print("PHASE 4: DATA AUDIT")
print("=" * 80)

# 4.1 Team size distribution
print("\n--- 4.1 Team Size Distribution ---")
total = len(works)
for label, cond in [("Exactly 2 authors", works["author_count"] == 2),
                     ("Exactly 3 authors", works["author_count"] == 3),
                     ("4+ authors", works["author_count"] >= 4)]:
    n = cond.sum()
    print(f"  {label}: {n:,} ({100*n/total:.1f}%)")

print(f"\n  Team size - Mean: {works['author_count'].mean():.1f}, Median: {works['author_count'].median():.0f}, "
      f"IQR: {works['author_count'].quantile(0.25):.0f}-{works['author_count'].quantile(0.75):.0f}")

# 4.2 Corresponding author overlap (already computed in Phase 3, verify)
print("\n--- 4.2 Corresponding Author Overlap ---")
n_first_is_corr = works["first_is_corr"].sum()
n_total_with_corr = works["first_is_corr"].notna().sum()
print(f"  First author = Corresponding author: {int(n_first_is_corr):,} ({100*n_first_is_corr/n_total_with_corr:.1f}%)")
print(f"  First author != Corresponding author: {int(n_total_with_corr - n_first_is_corr):,} ({100*(n_total_with_corr-n_first_is_corr)/n_total_with_corr:.1f}%)")

# 4.3 Funding distribution
print("\n--- 4.3 Funding Distribution ---")
def to_bool(val):
    if pd.isna(val): return False
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() == "true"
    return bool(val)

has_fund = works["has_funding"].apply(to_bool)
print(f"  has_funding = True: {has_fund.sum():,} ({100*has_fund.sum()/total:.1f}%)")
print(f"  has_funding = False: {(~has_fund).sum():,} ({100*(~has_fund).sum()/total:.1f}%)")
print(f"\n  Funder category breakdown:")
print(works["funder_category"].value_counts().to_string())

# 4.4 Field distribution verification
print("\n--- 4.4 Field Distribution Verification ---")
print("  Full field_name value counts:")
print(works["field_name"].value_counts().to_string())
bad_fields = {"Chemical Engineering", "Materials Science", "Environmental Science",
              "Civil and Structural Engineering", "Computer Science", "Mathematics"}
found_bad = works["field_name"].isin(bad_fields).sum()
print(f"\n  Non-biomedical field contamination check: {found_bad} papers (should be 0)")

# 4.5 Country count
print("\n--- 4.5 Country Count ---")
all_countries = set()
for c in works["countries"].dropna():
    for code in str(c).split("|"):
        code = code.strip()
        if code:
            all_countries.add(code)
print(f"  Total unique country/territory codes: {len(all_countries)}")
if len(all_countries) > 193:
    print(f"  Note: >193 = includes territories (HK, TW, PR, etc.)")

# 4.6 Overall study characteristics
print("\n--- 4.6 Overall Study Characteristics ---")
print(f"  Total works: {len(works):,}")
print(f"  Total authorships: {len(authorships):,}")
print(f"  Unique authors: {authorships['author_id'].nunique():,}")

print(f"\n  Affiliation category distribution:")
print(authorships["affiliation_category"].value_counts().to_string())

print(f"\n  Median team size: {works['author_count'].median():.0f} (IQR: {works['author_count'].quantile(0.25):.0f}-{works['author_count'].quantile(0.75):.0f})")
print(f"  Median country count: {works['country_count'].median():.0f} (IQR: {works['country_count'].quantile(0.25):.0f}-{works['country_count'].quantile(0.75):.0f})")

pre = (works["covid_era"] == 0).sum()
post = (works["covid_era"] == 1).sum()
print(f"\n  Pre-COVID (2000-2019): {pre:,} ({100*pre/total:.1f}%)")
print(f"  Post-COVID (2020-2025): {post:,} ({100*post/total:.1f}%)")

print(f"\n  Partner bloc distribution:")
print(works["partner_bloc"].value_counts().to_string())

print(f"\n  Time period distribution:")
print(works["time_period"].value_counts().sort_index().to_string())

print(f"\n  Publication type distribution:")
print(works["type"].value_counts().head(10).to_string())

# Save audit results
audit = {
    "total_works": int(len(works)),
    "total_authorships": int(len(authorships)),
    "unique_authors": int(authorships['author_id'].nunique()),
    "team_size_2": int((works["author_count"] == 2).sum()),
    "team_size_3": int((works["author_count"] == 3).sum()),
    "team_size_4plus": int((works["author_count"] >= 4).sum()),
    "median_team_size": float(works['author_count'].median()),
    "median_country_count": float(works['country_count'].median()),
    "first_is_corr_pct": float(100 * n_first_is_corr / n_total_with_corr),
    "has_funding_pct": float(100 * has_fund.sum() / total),
    "unique_countries": len(all_countries),
    "pre_covid": int(pre),
    "post_covid": int(post),
    "field_contamination": int(found_bad),
}
RESULTS_DIR = Path("analysis_results")
with open(RESULTS_DIR / "audit_results.json", "w") as f:
    json.dump(audit, f, indent=2)

print("\nPhase 4 complete. Audit results saved.")
