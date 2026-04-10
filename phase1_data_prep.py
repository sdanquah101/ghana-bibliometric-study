"""
Phase 1: Data Preparation & Validation
=======================================
Loads all CSVs, filters to study set, creates analytical variables,
saves processed data, and prints validation summary.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Import shared config
from config import *

print("=" * 60)
print("PHASE 1: DATA PREPARATION & VALIDATION")
print("=" * 60)

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================
# 1. LOAD DATA
# =============================================================
print("\n[1/8] Loading CSV files...")

try:
    works = pd.read_csv(DATA_DIR / "works_metadata_filtered.csv", low_memory=False)
    print(f"  works_metadata_filtered.csv: {len(works):,} rows")
except FileNotFoundError:
    print("ERROR: works_metadata_filtered.csv not found. Stopping.")
    sys.exit(1)

try:
    authorships = pd.read_csv(DATA_DIR / "authorships_filtered.csv", low_memory=False)
    print(f"  authorships_filtered.csv: {len(authorships):,} rows")
except FileNotFoundError:
    print("ERROR: authorships_filtered.csv not found. Stopping.")
    sys.exit(1)

try:
    affiliations = pd.read_csv(DATA_DIR / "author_affiliations_filtered.csv", low_memory=False)
    print(f"  author_affiliations_filtered.csv: {len(affiliations):,} rows")
except FileNotFoundError:
    print("WARNING: author_affiliations_filtered.csv not found.")
    affiliations = pd.DataFrame()

try:
    topics = pd.read_csv(DATA_DIR / "works_topics_filtered.csv", low_memory=False)
    print(f"  works_topics_filtered.csv: {len(topics):,} rows")
except FileNotFoundError:
    print("WARNING: works_topics_filtered.csv not found.")
    topics = pd.DataFrame()

try:
    funding = pd.read_csv(DATA_DIR / "works_funding_filtered.csv", low_memory=False)
    print(f"  works_funding_filtered.csv: {len(funding):,} rows")
except FileNotFoundError:
    print("WARNING: works_funding_filtered.csv not found.")
    funding = pd.DataFrame()

try:
    keywords = pd.read_csv(DATA_DIR / "works_keywords_filtered.csv", low_memory=False)
    print(f"  works_keywords_filtered.csv: {len(keywords):,} rows")
except FileNotFoundError:
    print("WARNING: works_keywords_filtered.csv not found.")
    keywords = pd.DataFrame()

try:
    sdgs = pd.read_csv(DATA_DIR / "works_sdgs_filtered.csv", low_memory=False)
    print(f"  works_sdgs_filtered.csv: {len(sdgs):,} rows")
except FileNotFoundError:
    print("WARNING: works_sdgs_filtered.csv not found.")
    sdgs = pd.DataFrame()

# =============================================================
# 2. FILTER TO STUDY SET
# =============================================================
print("\n[2/8] Filtering to study set...")

total_before = len(works)
print(f"  Total works in filtered set: {total_before:,}")

works = works[(works["publication_year"] >= STUDY_START) & (works["publication_year"] <= STUDY_END)]
after_year = len(works)
print(f"  After year filter ({STUDY_START}-{STUDY_END}): {after_year:,}")

works = works[works["is_international_collab"] == True]
after_intl = len(works)
print(f"  After intl collab filter: {after_intl:,}")

works = works[works["author_count"] >= 2]
print(f"  After >=2 authors filter: {len(works):,}  <- FINAL STUDY SET")

study_work_ids = set(works["work_id"].unique())

# Filter other tables
authorships = authorships[authorships["work_id"].isin(study_work_ids)]
print(f"  Authorships in study set: {len(authorships):,}")

if len(affiliations) > 0:
    affiliations = affiliations[affiliations["work_id"].isin(study_work_ids)]
    print(f"  Affiliations in study set: {len(affiliations):,}")

if len(topics) > 0:
    topics = topics[topics["work_id"].isin(study_work_ids)]
    print(f"  Topics in study set: {len(topics):,}")

if len(funding) > 0:
    funding = funding[funding["work_id"].isin(study_work_ids)]
    print(f"  Funding rows in study set: {len(funding):,}")

if len(keywords) > 0:
    keywords = keywords[keywords["work_id"].isin(study_work_ids)]
    print(f"  Keywords in study set: {len(keywords):,}")

if len(sdgs) > 0:
    sdgs = sdgs[sdgs["work_id"].isin(study_work_ids)]
    print(f"  SDGs in study set: {len(sdgs):,}")

# =============================================================
# 3. CREATE AFFILIATION CATEGORY
# =============================================================
print("\n[3/8] Creating affiliation categories...")

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
print("  Affiliation categories assigned:")
for cat, count in authorships["affiliation_category"].value_counts().items():
    pct = count / len(authorships) * 100
    print(f"    {cat}: {count:,} ({pct:.1f}%)")

# =============================================================
# 4. CREATE CORRESPONDING AUTHOR FLAG
# =============================================================
print("\n[4/8] Creating corresponding author flag...")

def to_bool(val):
    if pd.isna(val): return False
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.strip().lower() == "true"
    return bool(val)

authorships["is_corresponding_combined"] = (
    authorships["is_corresponding"].apply(to_bool) |
    authorships["is_corresponding_from_work"].apply(to_bool)
)

corr_works = authorships[authorships["is_corresponding_combined"] == True]["work_id"].nunique()
total_works_n = len(works)
print(f"  Works with >=1 corresponding author: {corr_works:,} ({corr_works/total_works_n*100:.1f}%)")

# =============================================================
# 5. PARTNER BLOC + COVID ERA + TIME PERIOD
# =============================================================
print("\n[5/8] Assigning partner blocs, COVID era, and time periods...")

bloc_results = works["countries"].apply(assign_partner_bloc)
works["partner_bloc"] = [r[0] for r in bloc_results]
works["partner_countries"] = ["|".join(r[1]) if r[1] else "" for r in bloc_results]
works["is_western_collab"] = [r[2] for r in bloc_results]
works["western_vs_nonwestern"] = [r[3] for r in bloc_results]

print("  Partner bloc distribution:")
for bloc, count in works["partner_bloc"].value_counts().items():
    pct = count / len(works) * 100
    print(f"    {bloc}: {count:,} ({pct:.1f}%)")

# COVID era
works["covid_era"] = (works["publication_year"] >= COVID_YEAR).astype(int)
# Time period
works["time_period"] = works["publication_year"].apply(assign_time_period)

pre = (works["covid_era"] == 0).sum()
post = (works["covid_era"] == 1).sum()
print(f"\n  COVID era:")
print(f"    Pre-COVID (2000-2019): {pre:,} ({pre/len(works)*100:.1f}%)")
print(f"    Post-COVID (2020-2025): {post:,} ({post/len(works)*100:.1f}%)")

# =============================================================
# 6. CLASSIFY FUNDERS
# =============================================================
print("\n[6/8] Classifying funders...")

if len(funding) > 0:
    funding["funder_category"] = funding["funder_name"].apply(classify_funder)
    print("  Funder category distribution:")
    for cat, count in funding["funder_category"].value_counts().items():
        print(f"    {cat}: {count:,}")

    # Work-level: assign highest-priority funder category
    priority = {"International (Northern)": 0, "Ghanaian": 1, "Other/Unclassified": 2}
    funding["_priority"] = funding["funder_category"].map(priority).fillna(3)
    work_funder_cat = (
        funding.sort_values("_priority")
        .drop_duplicates(subset=["work_id"], keep="first")[["work_id", "funder_category"]]
    )
    works = works.merge(work_funder_cat, on="work_id", how="left")
    works["funder_category"] = works["funder_category"].fillna("No funding data")
    funding.drop(columns=["_priority"], inplace=True)
else:
    works["funder_category"] = "No funding data"

print("  Work-level funder categories:")
for cat, count in works["funder_category"].value_counts().items():
    pct = count / len(works) * 100
    print(f"    {cat}: {count:,} ({pct:.1f}%)")

# =============================================================
# 7. COVID TOPIC IDENTIFICATION
# =============================================================
print("\n[7/8] Identifying COVID-related papers...")

covid_keywords = ["COVID", "SARS-COV-2", "CORONAVIRUS", "COVID-19", "NCOV", "PANDEMIC"]

title_match = works["title"].fillna("").str.upper().apply(
    lambda t: any(kw in t for kw in covid_keywords)
)
abstract_match = works["abstract"].fillna("").str.upper().apply(
    lambda t: any(kw in t for kw in covid_keywords)
)

if len(keywords) > 0:
    kw_covid_works = keywords[
        keywords["keyword"].fillna("").str.upper().apply(
            lambda k: any(kw in k for kw in covid_keywords)
        )
    ]["work_id"].unique()
    kw_match = works["work_id"].isin(kw_covid_works)
else:
    kw_match = pd.Series(False, index=works.index)

topic_covid_keywords = ["COVID", "CORONAVIRUS", "PANDEMIC", "SARS"]
if len(topics) > 0 and "topic_name" in topics.columns:
    if "is_primary" in topics.columns:
        primary_topics = topics[topics["is_primary"] == True]
    elif "topic_rank" in topics.columns:
        primary_topics = topics[topics["topic_rank"] == 1]
    else:
        primary_topics = topics
    topic_covid_works = primary_topics[
        primary_topics["topic_name"].fillna("").str.upper().apply(
            lambda t: any(kw in t for kw in topic_covid_keywords)
        )
    ]["work_id"].unique()
    topic_match = works["work_id"].isin(topic_covid_works)
else:
    topic_match = pd.Series(False, index=works.index)

works["is_covid_related"] = title_match | abstract_match | kw_match | topic_match
covid_count = works["is_covid_related"].sum()
print(f"  COVID-related papers: {covid_count:,} ({covid_count/len(works)*100:.1f}%)")

# =============================================================
# 8. SAVE & VALIDATE
# =============================================================
print("\n[8/8] Saving processed data...")

works.to_parquet(INTERMEDIATE_DIR / "works.parquet", index=False)
authorships.to_parquet(INTERMEDIATE_DIR / "authorships.parquet", index=False)
if len(funding) > 0:
    funding.to_parquet(INTERMEDIATE_DIR / "funding.parquet", index=False)
if len(topics) > 0:
    topics.to_parquet(INTERMEDIATE_DIR / "topics.parquet", index=False)
if len(keywords) > 0:
    keywords.to_parquet(INTERMEDIATE_DIR / "keywords.parquet", index=False)
if len(sdgs) > 0:
    sdgs.to_parquet(INTERMEDIATE_DIR / "sdgs.parquet", index=False)
if len(affiliations) > 0:
    affiliations.to_parquet(INTERMEDIATE_DIR / "affiliations.parquet", index=False)

works.to_csv(OUTPUT_DIR / "study_works.csv", index=False)
print(f"  Saved study_works.csv ({len(works):,} rows)")

# =============================================================
# VALIDATION SUMMARY
# =============================================================
print("\n")
print("=" * 60)
print("STUDY DATASET SUMMARY")
print("-" * 60)

print(f"Total works in filtered set:           {total_before:,}")
print(f"After year filter ({STUDY_START}-{STUDY_END}):         {after_year:,}")
print(f"After intl collab filter:              {after_intl:,}")
print(f"After >=2 authors filter:              {len(works):,}  <- FINAL STUDY SET")
print(f"Total authorships:                     {len(authorships):,}")

n_unique_authors = authorships["author_id"].nunique()
print(f"Unique authors:                        {n_unique_authors:,}")

if len(affiliations) > 0 and "institution_id" in affiliations.columns:
    n_unique_inst = affiliations["institution_id"].nunique()
elif "gh_institution_names" in authorships.columns:
    all_inst = set()
    for col in ["gh_institution_names", "non_gh_institution_names"]:
        if col in authorships.columns:
            for val in authorships[col].dropna():
                for inst in str(val).split("|"):
                    if inst.strip():
                        all_inst.add(inst.strip())
    n_unique_inst = len(all_inst)
else:
    n_unique_inst = 0
print(f"Unique institutions:                   {n_unique_inst:,}")

n_unique_countries = len(set(
    c.strip() for val in works["countries"].dropna()
    for c in str(val).split("|") if c.strip()
))
print(f"Unique countries:                      {n_unique_countries:,}")

med_auth = works["author_count"].median()
q1_auth = works["author_count"].quantile(0.25)
q3_auth = works["author_count"].quantile(0.75)
print(f"\nMedian authors per paper (IQR):        {med_auth:.0f} ({q1_auth:.0f}-{q3_auth:.0f})")

med_ctry = works["country_count"].median()
q1_ctry = works["country_count"].quantile(0.25)
q3_ctry = works["country_count"].quantile(0.75)
print(f"Median countries per paper (IQR):      {med_ctry:.0f} ({q1_ctry:.0f}-{q3_ctry:.0f})")

print(f"\nAFFILIATION CATEGORY DISTRIBUTION (all authorships):")
for cat in ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]:
    n = (authorships["affiliation_category"] == cat).sum()
    pct = n / len(authorships) * 100
    print(f"  {cat + ':':40s} {n:>8,} ({pct:.1f}%)")

print(f"\nAUTHOR POSITION DISTRIBUTION:")
for pos in ["first", "middle", "last"]:
    n = (authorships["author_position"] == pos).sum()
    print(f"  {pos.capitalize() + ':':40s} {n:>8,}")

print(f"\nDATA COVERAGE:")
corr_count = authorships[authorships["is_corresponding_combined"]]["work_id"].nunique()
print(f"  Works with corresponding author:     {corr_count:,} ({corr_count/len(works)*100:.1f}%)")

if "has_funding" in works.columns:
    fund_count = works["has_funding"].apply(to_bool).sum()
else:
    fund_count = 0
print(f"  Works with funding data:             {fund_count:,} ({fund_count/len(works)*100:.1f}%)")

fwci_count = works["fwci"].notna().sum() if "fwci" in works.columns else 0
print(f"  Works with FWCI:                     {fwci_count:,} ({fwci_count/len(works)*100:.1f}%)")

abs_count = works["abstract"].notna().sum() if "abstract" in works.columns else 0
print(f"  Works with abstract:                 {abs_count:,} ({abs_count/len(works)*100:.1f}%)")

if "is_author_truncated" in works.columns:
    trunc_count = works["is_author_truncated"].apply(to_bool).sum()
else:
    trunc_count = 0
print(f"  Works with author truncation:        {trunc_count:,} ({trunc_count/len(works)*100:.1f}%)")

print(f"\nTOP 10 FUNDERS:")
if len(funding) > 0:
    top_funders = funding.groupby("funder_name")["work_id"].nunique().sort_values(ascending=False).head(10)
    for i, (name, cnt) in enumerate(top_funders.items(), 1):
        disp = name[:50] if isinstance(name, str) else str(name)[:50]
        print(f"  {i:>2}. {disp:50s} {cnt:>5,} works")

    print(f"\nFUNDING SOURCE DISTRIBUTION:")
    if "source" in funding.columns:
        for src, cnt in funding["source"].value_counts().items():
            print(f"  {src}: {cnt:,}")
else:
    print("  No funding data available.")

print(f"\nDOMAIN DISTRIBUTION:")
if "domain_name" in works.columns:
    for dom, cnt in works["domain_name"].value_counts().items():
        pct = cnt / len(works) * 100
        print(f"  {str(dom):40s} {cnt:>8,} ({pct:.1f}%)")

print(f"\nPARTNER BLOC DISTRIBUTION:")
for bloc in ["Western", "African", "East Asian", "South Asian", "MENA",
             "Latin American", "Multi-bloc", "Other", "Unknown"]:
    n = (works["partner_bloc"] == bloc).sum()
    if n > 0:
        pct = n / len(works) * 100
        print(f"  {bloc + ':':40s} {n:>8,} ({pct:.1f}%)")

print(f"\nCOVID ERA:")
pre_n = (works["covid_era"] == 0).sum()
post_n = (works["covid_era"] == 1).sum()
print(f"  Pre-COVID (2000-2019):               {pre_n:,} ({pre_n/len(works)*100:.1f}%)")
print(f"  Post-COVID (2020-2025):              {post_n:,} ({post_n/len(works)*100:.1f}%)")

print(f"\nCOVID-RELATED PAPERS:")
print(f"  Identified:                          {covid_count:,} ({covid_count/len(works)*100:.1f}%)")

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE -- Review numbers above before proceeding.")
print("=" * 60)

# Save validation to file
with open(OUTPUT_DIR / "validation_output.txt", "w", encoding="utf-8") as f:
    f.write(f"Study works: {len(works):,}\n")
    f.write(f"Study authorships: {len(authorships):,}\n")
    f.write(f"Unique authors: {n_unique_authors:,}\n")
    f.write(f"Unique institutions: {n_unique_inst:,}\n")
    f.write(f"Unique countries: {n_unique_countries:,}\n")
    f.write(f"Corresponding author coverage: {corr_count/len(works)*100:.1f}%\n")
    f.write(f"Funding coverage: {fund_count/len(works)*100:.1f}%\n")
    f.write(f"FWCI coverage: {fwci_count/len(works)*100:.1f}%\n")
