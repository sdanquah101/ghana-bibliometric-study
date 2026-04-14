"""
Phase 2-3 (v2): Data Preparation
=================================
Applies inclusion criteria to the biomedical-filtered dataset and constructs
all derived variables for analysis.

Changes from original phase23_inclusion_prep.py:
  1. Excludes retracted papers (n=43) and paratext (n=5)
  2. Fixes corresponding author dedup (sort by position_index first)
  3. Adds log_author_count for regression
  4. Adds year_centered_sq for quadratic temporal modelling
  5. Extracts primary Ghanaian institution for GEE clustering
  6. Removes covid_era from default predictors (sensitivity only)

Outputs:
  analysis_results/intermediate/v2_works.parquet
  analysis_results/intermediate/v2_authorships.parquet
  analysis_results/v2_prisma_numbers.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from utils import to_bool, BASE, INTERMEDIATE, RESULTS

print("=" * 70)
print("PHASE 2-3 (v2): DATA PREPARATION")
print("=" * 70)

# ── Load raw biomedical-filtered data ────────────────────────────────────
print("\n1. Loading raw data...")
works_raw = pd.read_csv(BASE / "filtered_biomedical" / "works_metadata_filtered.csv",
                        low_memory=False)
authorships_raw = pd.read_csv(BASE / "filtered_biomedical" / "authorships_filtered.csv",
                              low_memory=False)
print(f"   Biomedical works: {len(works_raw):,}")
print(f"   Biomedical authorships: {len(authorships_raw):,}")

total_biomedical = len(works_raw)

# ── Step 1: Study period filter (2000–2025) ──────────────────────────────
print("\n2. Applying study period filter (2000-2025)...")
excluded_years = works_raw[
    (works_raw["publication_year"] < 2000) | (works_raw["publication_year"] > 2025)
]
works = works_raw[
    (works_raw["publication_year"] >= 2000) & (works_raw["publication_year"] <= 2025)
].copy()
n_excluded_years = len(excluded_years)
print(f"   Excluded outside 2000-2025: {n_excluded_years:,}")
print(f"   Remaining: {len(works):,}")

# ── Step 2: International collaboration filter ───────────────────────────
print("\n3. Applying international collaboration filter...")
works["is_international_collab"] = to_bool(works["is_international_collab"])
excluded_domestic = works[~works["is_international_collab"]]
works = works[works["is_international_collab"]].copy()
n_excluded_domestic = len(excluded_domestic)
print(f"   Excluded domestic-only: {n_excluded_domestic:,}")
print(f"   Remaining: {len(works):,}")

n_after_intl = len(works)

# ── Step 3: Multi-authored filter (≥2 authors) ──────────────────────────
print("\n4. Applying multi-authored filter (>=2 authors)...")
excluded_single = works[works["author_count"] < 2]
works = works[works["author_count"] >= 2].copy()
n_excluded_single = len(excluded_single)
print(f"   Excluded single-author: {n_excluded_single:,}")
print(f"   Remaining: {len(works):,}")

n_after_multi = len(works)

# ── Step 4: Exclude retracted papers (NEW) ───────────────────────────────
print("\n5. Excluding retracted papers...")
works["is_retracted"] = to_bool(works["is_retracted"])
excluded_retracted = works[works["is_retracted"]]
works = works[~works["is_retracted"]].copy()
n_excluded_retracted = len(excluded_retracted)
print(f"   Excluded retracted: {n_excluded_retracted:,}")
print(f"   Remaining: {len(works):,}")

# ── Step 5: Exclude paratext (NEW) ──────────────────────────────────────
print("\n6. Excluding paratext records...")
works["is_paratext"] = to_bool(works["is_paratext"])
excluded_paratext = works[works["is_paratext"]]
works = works[~works["is_paratext"]].copy()
n_excluded_paratext = len(excluded_paratext)
print(f"   Excluded paratext: {n_excluded_paratext:,}")
print(f"   FINAL STUDY SET: {len(works):,}")

final_n = len(works)

# ── Filter authorships to final works ────────────────────────────────────
print("\n7. Filtering authorships to study set...")
authorships = authorships_raw[
    authorships_raw["work_id"].isin(works["work_id"])
].copy()
print(f"   Authorships in study set: {len(authorships):,}")
n_unique_authors = authorships["author_id"].nunique()
print(f"   Unique authors: {n_unique_authors:,}")

# ── Affiliation classification ───────────────────────────────────────────
print("\n8. Classifying author affiliations...")

def get_affiliation_category(row):
    gh = row.get("has_gh_affiliation", False)
    non_gh = row.get("has_non_gh_affiliation", False)
    if isinstance(gh, str):
        gh = gh.strip().lower() == "true"
    if isinstance(non_gh, str):
        non_gh = non_gh.strip().lower() == "true"
    if gh and non_gh:
        return "Dual-affiliated"
    elif gh:
        return "Ghanaian"
    else:
        return "Non-Ghanaian"

authorships["affiliation_category"] = authorships.apply(get_affiliation_category, axis=1)

cat_counts = authorships["affiliation_category"].value_counts()
for cat in ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]:
    n = cat_counts.get(cat, 0)
    pct = n / len(authorships) * 100
    print(f"   {cat}: {n:,} ({pct:.1f}%)")

# ── Corresponding author: is_corresponding_combined ─────────────────────
print("\n9. Creating combined corresponding author flag...")
authorships["is_corresponding"] = to_bool(authorships["is_corresponding"])
authorships["is_corresponding_from_work"] = to_bool(
    authorships["is_corresponding_from_work"])
authorships["is_corresponding_combined"] = (
    authorships["is_corresponding"] | authorships["is_corresponding_from_work"]
)

# ── FIXED: Sort by position index before deduplicating ──────────────────
# This ensures "first-listed corresponding author" is truly the one
# with the lowest position index, not an arbitrary row.
print("   Fixing corresponding author dedup (sort by position_index)...")
corr_authors = (authorships[authorships["is_corresponding_combined"]]
                .sort_values(["work_id", "author_position_index"])
                .drop_duplicates("work_id"))
corr_map = corr_authors.set_index("work_id")["affiliation_category"]

# First-is-corresponding flag
first_authors_for_corr = (authorships[authorships["author_position"] == "first"]
                          .drop_duplicates("work_id"))
first_corr_merge = first_authors_for_corr.merge(
    corr_authors[["work_id", "author_id"]].rename(
        columns={"author_id": "corr_author_id"}),
    on="work_id", how="inner")
first_is_corr = (first_corr_merge["author_id"] == first_corr_merge["corr_author_id"])
first_is_corr_pct = first_is_corr.sum() / len(first_is_corr) * 100
print(f"   First-is-corresponding: {first_is_corr_pct:.1f}%")
works["first_is_corr"] = works["work_id"].map(
    first_corr_merge.set_index("work_id").apply(
        lambda r: r["author_id"] == r["corr_author_id"], axis=1)
).fillna(False)

# ── Partner bloc classification ──────────────────────────────────────────
print("\n10. Classifying partner blocs...")

WESTERN = {
    "US", "CA", "GB", "AU", "NZ",
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    "IS", "LI", "NO", "CH",
}
AFRICAN = {
    "DZ", "AO", "BJ", "BW", "BF", "BI", "CV", "CM", "CF", "TD",
    "KM", "CD", "CG", "CI", "DJ", "EG", "GQ", "ER", "SZ", "ET",
    "GA", "GM", "GN", "GW", "KE", "LS", "LR", "LY", "MG", "MW",
    "ML", "MR", "MU", "MA", "MZ", "NA", "NE", "NG", "RW", "ST",
    "SN", "SC", "SL", "SO", "ZA", "SS", "SD", "TZ", "TG", "TN",
    "UG", "ZM", "ZW",
}
EAST_ASIAN = {
    "CN", "JP", "KR", "TW", "SG", "MY", "TH", "VN", "PH", "ID",
    "MM", "KH", "LA", "BN", "MN",
}
SOUTH_ASIAN = {"IN", "BD", "PK", "LK", "NP", "BT", "MV"}
MENA = {
    "TR", "IL", "IR", "IQ", "SA", "AE", "QA", "KW", "BH", "OM",
    "YE", "JO", "LB", "SY", "PS", "EG", "LY", "TN", "DZ", "MA",
}
LATIN_AMERICAN = {
    "BR", "MX", "AR", "CL", "CO", "PE", "VE", "EC", "BO", "PY",
    "UY", "GY", "SR", "CU", "DO", "HT", "JM", "TT", "CR", "PA",
    "GT", "HN",
}

def assign_partner_bloc(countries_str):
    """Assign partner bloc based on non-Ghanaian co-author countries."""
    if pd.isna(countries_str):
        return "Unknown", [], False
    # Split on pipe separator — safe for 2-letter ISO codes
    codes = [c.strip() for c in str(countries_str).split("|")]
    partner_codes = [c for c in codes if c != "GH"]
    if not partner_codes:
        return "Unknown", partner_codes, False

    blocs_present = set()
    for code in partner_codes:
        if code in WESTERN:
            blocs_present.add("Western")
        elif code in AFRICAN:
            blocs_present.add("African")
        elif code in EAST_ASIAN:
            blocs_present.add("East Asian")
        elif code in SOUTH_ASIAN:
            blocs_present.add("South Asian")
        elif code in MENA:
            blocs_present.add("MENA")
        elif code in LATIN_AMERICAN:
            blocs_present.add("Latin American")
        else:
            blocs_present.add("Other")

    if len(blocs_present) == 1:
        return blocs_present.pop(), partner_codes, True
    else:
        return "Multi-bloc", partner_codes, False

bloc_results = works["countries"].apply(assign_partner_bloc)
works["partner_bloc"] = bloc_results.apply(lambda x: x[0])
works["partner_countries"] = bloc_results.apply(lambda x: x[1])
works["is_bilateral"] = bloc_results.apply(lambda x: x[2])

bloc_dist = works["partner_bloc"].value_counts()
for bloc, cnt in bloc_dist.items():
    print(f"   {bloc}: {cnt:,} ({cnt/len(works)*100:.1f}%)")

# ── Emerging Powers / BRICS+ classification ──────────────────────────────
print("\n10b. Classifying Emerging Powers (BRICS+) vs Traditional Western...")

EMERGING_POWERS = {"CN", "IN", "BR", "ZA"}  # Core BRICS (minus Russia)
TRADITIONAL_WESTERN_TOP = {"US", "GB", "DE", "FR", "NL", "CA", "AU", "CH", "SE", "BE", "DK", "NO"}

def classify_partner_type(countries_str, partner_bloc):
    """Classify into Traditional Western / Emerging Power / African / Other."""
    if pd.isna(countries_str):
        return "Unknown"
    codes = [c.strip() for c in str(countries_str).split("|")]
    partner_codes = [c for c in codes if c != "GH"]
    if not partner_codes:
        return "Unknown"

    has_western = any(c in WESTERN for c in partner_codes)
    has_emerging = any(c in EMERGING_POWERS for c in partner_codes)
    has_african = any(c in AFRICAN for c in partner_codes)

    # Count how many broad types are present
    types_present = sum([has_western, has_emerging, has_african])
    has_other = any(c not in WESTERN and c not in EMERGING_POWERS and c not in AFRICAN
                    for c in partner_codes)
    if has_other:
        types_present += 1

    if types_present > 1:
        return "Mixed"
    elif has_western:
        return "Traditional Western"
    elif has_emerging:
        return "Emerging Power"
    elif has_african:
        return "African"
    else:
        return "Other"

works["partner_type"] = works.apply(
    lambda r: classify_partner_type(r["countries"], r["partner_bloc"]), axis=1)

# Individual top-partner country flags (for country-level analysis)
TOP_PARTNERS = ["US", "GB", "CN", "IN", "ZA", "NG", "KE", "DE", "NL", "FR", "CA", "AU", "BR"]
for cc in TOP_PARTNERS:
    works[f"has_{cc}"] = works["partner_countries"].apply(
        lambda x: int(cc in x) if isinstance(x, list) else 0)

pt_dist = works["partner_type"].value_counts()
for pt, cnt in pt_dist.items():
    print(f"   {pt}: {cnt:,} ({cnt/len(works)*100:.1f}%)")

# ── Derived variables ────────────────────────────────────────────────────
print("\n11. Constructing derived variables...")

# Temporal
works["year_centered"] = works["publication_year"] - 2000
works["year_centered_sq"] = works["year_centered"] ** 2
# covid_era computed but NOT used in primary model
works["covid_era"] = (works["publication_year"] >= 2020).astype(int)

# Team size
works["log_author_count"] = np.log(works["author_count"])
works["team_size_cat"] = pd.cut(
    works["author_count"],
    bins=[0, 2, 3, 5, 10, 50, 99999],
    labels=["2", "3", "4-5", "6-10", "11-50", ">50"],
)

# Time periods
works["time_period"] = pd.cut(
    works["publication_year"],
    bins=[1999, 2005, 2010, 2015, 2019, 2025],
    labels=["2000-2005", "2006-2010", "2011-2015", "2016-2019", "2020-2025"],
)

# Funding
works["has_funding_bool"] = to_bool(works["has_funding"])
works["has_funding_int"] = works["has_funding_bool"].astype(int)

def classify_funder(names_str):
    if pd.isna(names_str):
        return "No data"
    names = str(names_str).lower()
    northern_kw = ["nih", "wellcome", "gates", "usaid", "who", "world health",
                  "dfid", "bmgf", "pepfar", "global fund", "unitaid",
                  "european commission", "eu ", "nsf", "cdc"]
    gh_kw = ["getfund", "university of ghana", "ghana health", "knust",
            "council for scientific", "csir"]
    for kw in northern_kw:
        if kw in names:
            return "International (Northern)"
    for kw in gh_kw:
        if kw in names:
            return "Ghanaian"
    return "Other"

works["funder_category"] = works["funder_names"].apply(classify_funder)

# OA
works["is_oa_bool"] = to_bool(works["is_oa"])
works["is_oa_int"] = works["is_oa_bool"].astype(int)

# Field for regression
works["field_name"] = works["field_name"].fillna(works["domain_name"])
works["subfield_name"] = works["subfield_name"].fillna("")

def assign_field_reg(row):
    field = str(row.get("field_name", ""))
    if field in ("Medicine", "Nursing", "Health Professions", "Dentistry",
                 "Veterinary"):
        return "Health Sciences"
    elif field == "Biochemistry, Genetics and Molecular Biology":
        return "Biochemistry, Genetics and Molecular Biology"
    elif field == "Immunology and Microbiology":
        return "Immunology and Microbiology"
    elif field == "Neuroscience":
        return "Neuroscience"
    elif field == "Pharmacology, Toxicology and Pharmaceutics":
        return "Pharmacology, Toxicology and Pharmaceutics"
    elif field == "Engineering":
        return "Engineering"
    else:
        return "Other"

works["field_reg"] = works.apply(assign_field_reg, axis=1)

# Collapse "Other" if < 50 obs
field_counts = works["field_reg"].value_counts()
small_fields = field_counts[field_counts < 50].index.tolist()
if small_fields:
    works.loc[works["field_reg"].isin(small_fields), "field_reg"] = "Other"
    print(f"   Collapsed small fields into Other: {small_fields}")

# First author ID for alternative clustering
first_auth = (authorships[authorships["author_position"] == "first"]
              .drop_duplicates("work_id")[["work_id", "author_id"]])
works = works.merge(first_auth.rename(columns={"author_id": "first_author_id"}),
                    on="work_id", how="left")

# ── Primary Ghanaian institution (NEW — for GEE clustering) ─────────────
print("\n12. Extracting primary Ghanaian institution for clustering...")

gh_auths = authorships[to_bool(authorships["has_gh_affiliation"])].copy()
gh_auths["primary_gh_inst"] = gh_auths["gh_institution_names"].apply(
    lambda s: str(s).split("|")[0].strip() if pd.notna(s) else None
)

# For each paper, take the most common GH institution among its GH authors
paper_inst = gh_auths.groupby("work_id")["primary_gh_inst"].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
)
works = works.merge(paper_inst.rename("primary_gh_institution"),
                    left_on="work_id", right_index=True, how="left")

# Fill any remaining NaN with a unique placeholder
missing_inst = works["primary_gh_institution"].isna()
if missing_inst.any():
    works.loc[missing_inst, "primary_gh_institution"] = (
        "_unknown_" + works.loc[missing_inst].index.astype(str))
    print(f"   Papers with no GH institution: {missing_inst.sum()} (assigned unique IDs)")

inst_counts = works["primary_gh_institution"].value_counts()
print(f"   Unique GH institutions: {len(inst_counts)}")
print(f"   Median papers per institution: {inst_counts.median():.0f}")
print(f"   Top 5:")
for inst, cnt in inst_counts.head(5).items():
    print(f"     {cnt:5,d} | {inst}")

# ── Unique countries ─────────────────────────────────────────────────────
all_countries = set()
for c in works["countries"].dropna():
    all_countries.update(str(c).split("|"))
all_countries.discard("GH")
n_countries = len(all_countries) + 1  # +1 for Ghana

# ── Save ─────────────────────────────────────────────────────────────────
print("\n13. Saving prepared data...")
INTERMEDIATE.mkdir(parents=True, exist_ok=True)
works.to_parquet(INTERMEDIATE / "v2_works.parquet", index=False)
authorships.to_parquet(INTERMEDIATE / "v2_authorships.parquet", index=False)

# PRISMA numbers — all from actual counts, not subtraction
n_within_period = len(works_raw[
    (works_raw["publication_year"] >= 2000) & (works_raw["publication_year"] <= 2025)
])
prisma = {
    "total_openalex": 127332,
    "total_biomedical": total_biomedical,
    "excluded_outside_years": total_biomedical - n_within_period,
    "within_study_period": n_within_period,
    "excluded_domestic_only": n_excluded_domestic,
    "international_collabs": n_after_intl,
    "excluded_single_author": n_excluded_single,
    "after_multi_author": n_after_multi,
    "excluded_retracted": n_excluded_retracted,
    "excluded_paratext": n_excluded_paratext,
    "final_study_set": final_n,
    "total_authorships": len(authorships),
    "unique_authors": n_unique_authors,
    "unique_countries": n_countries,
    "first_is_corr_pct": round(first_is_corr_pct, 1),
    "n_multi_corr_authors": len(
        authorships[authorships["is_corresponding_combined"]]
        .groupby("work_id").size().pipe(lambda s: s[s >= 2])
    ),
}
# Verify arithmetic
assert prisma["within_study_period"] - prisma["excluded_domestic_only"] == prisma["international_collabs"], \
    f"PRISMA arithmetic failed: {prisma['within_study_period']} - {prisma['excluded_domestic_only']} != {prisma['international_collabs']}"
assert prisma["after_multi_author"] - prisma["excluded_retracted"] - prisma["excluded_paratext"] == prisma["final_study_set"], \
    "PRISMA arithmetic failed at quality exclusions"
with open(RESULTS / "v2_prisma_numbers.json", "w") as f:
    json.dump(prisma, f, indent=2)

print(f"\n{'='*70}")
print(f"DONE. Final study set: {final_n:,} works")
print(f"  Authorships: {len(authorships):,}")
print(f"  Unique authors: {n_unique_authors:,}")
print(f"  Countries: {n_countries}")
print(f"{'='*70}")
