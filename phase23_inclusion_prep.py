"""
PHASE 2 + 3: Study Inclusion Criteria & Data Preparation
Ghana Bibliometric Study
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("PHASE 2: STUDY INCLUSION CRITERIA")
print("=" * 80)

works = pd.read_csv("filtered_biomedical/works_metadata_filtered.csv", low_memory=False)
authorships = pd.read_csv("filtered_biomedical/authorships_filtered.csv", low_memory=False)

total_biomed = len(works)
print(f"Biomedical works (all years, all types): {total_biomed:,}")

# Criterion 1: Year range
works = works[(works["publication_year"] >= 2000) & (works["publication_year"] <= 2025)]
after_year = len(works)
print(f"After year filter (2000-2025): {after_year:,}  [excluded: {total_biomed - after_year:,}]")

# Criterion 2: International collaboration
works = works[works["is_international_collab"] == True]
after_intl = len(works)
print(f"After international collab filter: {after_intl:,}  [excluded: {after_year - after_intl:,}]")

# Criterion 3: >=2 authors
works = works[works["author_count"] >= 2]
final_n = len(works)
print(f"After >=2 authors filter: {final_n:,}  [excluded: {after_intl - final_n:,}]  <- FINAL STUDY SET")

# Filter authorships to study set
study_ids = set(works["work_id"])
authorships = authorships[authorships["work_id"].isin(study_ids)]
print(f"Authorships in study set: {len(authorships):,}")
print(f"Unique authors: {authorships['author_id'].nunique():,}")

# Save PRISMA numbers
prisma = {
    "total_openalex": 127332,
    "excluded_non_biomedical": 127332 - total_biomed,
    "total_biomedical": total_biomed,
    "excluded_outside_years": total_biomed - after_year,
    "within_study_period": after_year,
    "excluded_domestic_only": after_year - after_intl,
    "international_collabs": after_intl,
    "excluded_single_author": after_intl - final_n,
    "final_study_set": final_n,
    "total_authorships": len(authorships),
    "unique_authors": int(authorships['author_id'].nunique()),
}
RESULTS_DIR = Path("analysis_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / "prisma_numbers.json", "w") as f:
    json.dump(prisma, f, indent=2)
print(f"\nPRISMA numbers saved to analysis_results/prisma_numbers.json")

print("\n" + "=" * 80)
print("PHASE 3: DATA PREPARATION")
print("=" * 80)

# 3.1 Affiliation categories
print("\n--- 3.1 Affiliation Categories ---")
def get_affiliation_category(row):
    gh = row.get("has_gh_affiliation", False)
    non_gh = row.get("has_non_gh_affiliation", False)
    if pd.isna(gh): gh = False
    if pd.isna(non_gh): non_gh = False
    if isinstance(gh, str): gh = gh.strip().lower() == "true"
    if isinstance(non_gh, str): non_gh = non_gh.strip().lower() == "true"
    if gh and non_gh:
        return "Dual-affiliated"
    elif gh:
        return "Ghanaian"
    else:
        return "Non-Ghanaian"

authorships["affiliation_category"] = authorships.apply(get_affiliation_category, axis=1)
print(authorships["affiliation_category"].value_counts().to_string())

# 3.2 Corresponding author flag
print("\n--- 3.2 Corresponding Author Flag ---")
def to_bool(val):
    if pd.isna(val): return False
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() == "true"
    return bool(val)

authorships["is_corresponding_combined"] = (
    authorships["is_corresponding"].apply(to_bool) |
    authorships["is_corresponding_from_work"].apply(to_bool)
)
n_corr = authorships["is_corresponding_combined"].sum()
print(f"Total corresponding author flags: {n_corr:,}")
n_works_with_corr = authorships[authorships["is_corresponding_combined"]]["work_id"].nunique()
print(f"Works with at least one corresponding author: {n_works_with_corr:,} / {final_n:,} ({100*n_works_with_corr/final_n:.1f}%)")

# 3.3 first_is_corr flag
print("\n--- 3.3 First = Corresponding Overlap ---")
first_authors = authorships[authorships["author_position"] == "first"][["work_id", "author_id"]].drop_duplicates("work_id")
first_authors.rename(columns={"author_id": "first_author_id"}, inplace=True)

corr_authors = authorships[authorships["is_corresponding_combined"] == True].drop_duplicates("work_id")[["work_id", "author_id"]]
corr_authors.rename(columns={"author_id": "corr_author_id"}, inplace=True)

overlap = first_authors.merge(corr_authors, on="work_id", how="inner")
overlap["first_is_corr"] = overlap["first_author_id"] == overlap["corr_author_id"]

works = works.merge(overlap[["work_id", "first_is_corr"]], on="work_id", how="left")
works["first_is_corr"] = works["first_is_corr"].fillna(False)

n_overlap = works["first_is_corr"].sum()
n_with_both = len(overlap)
print(f"Works with both first and corresponding author identified: {n_with_both:,}")
print(f"First author IS corresponding author: {n_overlap:,} ({100*n_overlap/n_with_both:.1f}%)")
print(f"First author IS NOT corresponding author: {n_with_both - n_overlap:,} ({100*(n_with_both-n_overlap)/n_with_both:.1f}%)")

# 3.4 Partner blocs
print("\n--- 3.4 Partner Blocs ---")
PARTNER_BLOCS = {
    "Western": {
        "US", "GB", "CA", "AU", "NZ",
        "DE", "FR", "NL", "SE", "DK", "NO", "CH", "BE", "IT", "ES",
        "AT", "FI", "IE", "PT", "LU", "GR", "CZ", "PL", "HU", "RO",
        "BG", "HR", "SK", "SI", "LT", "LV", "EE", "MT", "CY", "IS", "LI",
    },
    "East Asian": {
        "CN", "JP", "KR", "TW", "SG", "HK", "MY", "TH", "VN",
        "ID", "PH", "MM", "KH", "LA", "BN",
    },
    "South Asian": {"IN", "BD", "PK", "LK", "NP", "BT", "MV"},
    "Latin American": {
        "BR", "MX", "AR", "CO", "CL", "PE", "CU", "EC", "VE", "BO",
        "UY", "PY", "CR", "PA", "DO", "GT", "HN", "NI", "SV", "JM", "TT", "HT",
    },
    "African": {
        "ZA", "NG", "KE", "TZ", "ET", "UG", "CM", "SN", "BF", "MW",
        "RW", "BJ", "ML", "NE", "GN", "CI", "CD", "MZ", "ZW", "ZM",
        "MG", "AO", "GA", "SD", "SS", "SO", "TG", "SL", "LR", "ER",
        "DJ", "MR", "GM", "GW", "CV", "ST", "KM", "MU", "SC", "SZ",
        "LS", "BW", "NA", "TD", "CF", "CG", "GQ", "BI",
    },
    "MENA": {
        "EG", "SA", "IR", "IL", "TR", "MA", "TN", "AE", "QA",
        "JO", "LB", "IQ", "SY", "YE", "OM", "KW", "BH", "LY", "DZ", "PS",
    },
}

COUNTRY_TO_BLOC = {}
for bloc_name, codes in PARTNER_BLOCS.items():
    for code in codes:
        COUNTRY_TO_BLOC[code] = bloc_name

def assign_partner_bloc(countries_str):
    if pd.isna(countries_str):
        return "Unknown", ""
    codes = [c.strip() for c in str(countries_str).split("|") if c.strip()]
    non_gh = [c for c in codes if c != "GH"]
    if not non_gh:
        return "Unknown", ""
    blocs_present = set()
    for c in non_gh:
        blocs_present.add(COUNTRY_TO_BLOC.get(c, "Other"))
    partner_countries = "|".join(non_gh)
    if len(blocs_present) == 1:
        return blocs_present.pop(), partner_countries
    else:
        return "Multi-bloc", partner_countries

bloc_results = works["countries"].apply(assign_partner_bloc)
works["partner_bloc"] = [r[0] for r in bloc_results]
works["partner_countries"] = [r[1] for r in bloc_results]
works["is_bilateral"] = (works["partner_bloc"] != "Multi-bloc").astype(int)

print(works["partner_bloc"].value_counts().to_string())
print(f"\nBilateral: {works['is_bilateral'].sum():,}")
print(f"Multi-bloc: {(~works['is_bilateral'].astype(bool)).sum():,}")

# 3.5 Other derived variables
print("\n--- 3.5 Derived Variables ---")
works["covid_era"] = (works["publication_year"] >= 2020).astype(int)
works["year_centered"] = works["publication_year"] - 2000

works["team_size_cat"] = pd.cut(
    works["author_count"],
    bins=[0, 2, 3, 5, 10, 50, 99999],
    labels=["2", "3", "4-5", "6-10", "11-50", ">50"]
)

TIME_PERIODS = {
    "2000-2005": (2000, 2005), "2006-2010": (2006, 2010),
    "2011-2015": (2011, 2015), "2016-2019": (2016, 2019),
    "2020-2025": (2020, 2025),
}
def assign_time_period(year):
    for name, (s, e) in TIME_PERIODS.items():
        if s <= year <= e: return name
    return "Unknown"
works["time_period"] = works["publication_year"].apply(assign_time_period)

# Funding classification
NORTHERN_FUNDERS_KW = [
    "NIH", "National Institutes of Health", "Wellcome", "Gates", "USAID",
    "World Health Organization", "WHO", "DFID", "Medical Research Council",
    "European Commission", "EU", "Global Fund", "EDCTP", "World Bank",
    "CDC", "National Science Foundation", "UKRI", "BMGF", "Fogarty",
    "PEPFAR", "UNICEF",
]
GHANAIAN_FUNDERS_KW = [
    "Ghana", "GETFund", "KNUST", "University of Ghana", "Noguchi",
    "Ghana Health Service", "CSIR-Ghana",
]

try:
    funding = pd.read_csv("filtered_biomedical/works_funding_filtered.csv", low_memory=False)
    def classify_funder(name):
        if pd.isna(name): return "Other/Unclassified"
        upper = str(name).upper()
        for kw in NORTHERN_FUNDERS_KW:
            if kw.upper() in upper: return "International (Northern)"
        for kw in GHANAIAN_FUNDERS_KW:
            if kw.upper() in upper: return "Ghanaian"
        return "Other/Unclassified"

    if len(funding) > 0:
        funding["funder_class"] = funding["funder_name"].apply(classify_funder)
        work_funder = funding.groupby("work_id")["funder_class"].apply(
            lambda x: "International (Northern)" if "International (Northern)" in x.values
            else ("Ghanaian" if "Ghanaian" in x.values else "Other/Unclassified")
        ).reset_index(name="funder_category")
        works = works.merge(work_funder, on="work_id", how="left")
        works["funder_category"] = works["funder_category"].fillna("No funding data")
    else:
        works["funder_category"] = "No funding data"
except Exception as e:
    print(f"Funding classification error: {e}")
    works["funder_category"] = "No funding data"

print("Funder category distribution:")
print(works["funder_category"].value_counts().to_string())

# Save first_author_id on works
first_auth_ids = authorships[authorships["author_position"] == "first"][["work_id", "author_id"]].drop_duplicates("work_id")
first_auth_ids.rename(columns={"author_id": "first_author_id"}, inplace=True)
works = works.merge(first_auth_ids, on="work_id", how="left")

# 3.6 Field category for regression
print("\n--- 3.6 Field Category for Regression ---")
topics = pd.read_csv("filtered_biomedical/works_topics_filtered.csv", low_memory=False)
if "is_primary" in topics.columns:
    primary_topics = topics[topics["is_primary"] == True][["work_id", "field_name", "subfield_name"]].drop_duplicates("work_id")
elif "topic_rank" in topics.columns:
    primary_topics = topics[topics["topic_rank"] == 0][["work_id", "field_name", "subfield_name"]].drop_duplicates("work_id")
    if len(primary_topics) == 0:
        primary_topics = topics[topics["topic_rank"] == 1][["work_id", "field_name", "subfield_name"]].drop_duplicates("work_id")
else:
    primary_topics = topics[["work_id", "field_name", "subfield_name"]].drop_duplicates("work_id")

# Avoid duplicate columns if field_name already exists
if "field_name" in works.columns:
    works = works.drop(columns=["field_name"])
if "subfield_name" in works.columns:
    works = works.drop(columns=["subfield_name"])

works = works.merge(primary_topics, on="work_id", how="left")

health_fields = {"Medicine", "Nursing", "Health Professions", "Dentistry"}
works["field_reg"] = works["field_name"].apply(
    lambda x: "Health Sciences" if x in health_fields else (x if pd.notna(x) else "Other")
)

top_fields = works["field_reg"].value_counts().head(6).index.tolist()
works["field_reg"] = works["field_reg"].apply(lambda x: x if x in top_fields else "Other")

print("Field distribution for regression:")
print(works["field_reg"].value_counts().to_string())

# 3.7 Ensure has_funding is clean boolean
works["has_funding_bool"] = works["has_funding"].apply(to_bool).astype(int)
works["is_oa_bool"] = works["is_oa"].apply(to_bool).astype(int)

# 3.7 Save prepared data
print("\n--- 3.7 Saving Prepared Data ---")
INTERMEDIATE_DIR = Path("analysis_results/intermediate")
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

works.to_parquet(INTERMEDIATE_DIR / "works.parquet", index=False)
authorships.to_parquet(INTERMEDIATE_DIR / "authorships.parquet", index=False)

print(f"Works saved: {len(works):,} rows, {len(works.columns)} columns")
print(f"Authorships saved: {len(authorships):,} rows, {len(authorships.columns)} columns")
print(f"Works columns: {works.columns.tolist()}")

print("\nPhase 2+3 complete.")
