"""
PHASE 0 + 1: Data Inspection and Biomedical Filtering
Ghana Bibliometric Study
"""
import json
import pandas as pd
from pathlib import Path

print("=" * 80)
print("PHASE 0: DATA INSPECTION")
print("=" * 80)

# 0.1 Extraction summary
print("\n--- 0.1 Extraction Summary ---")
with open("extraction_summary.json") as f:
    summary = json.load(f)
print(json.dumps(summary, indent=2))

# 0.2 Works metadata
print("\n--- 0.2 Works Metadata ---")
works = pd.read_csv("works_metadata.csv", low_memory=False)
print(f"Total works: {len(works):,}")
print(f"Columns: {works.columns.tolist()}")
print(f"Year range: {works['publication_year'].min()} - {works['publication_year'].max()}")
print(f"Types: {works['type'].value_counts().head(10).to_string()}")

# 0.3 Authorships
print("\n--- 0.3 Authorships ---")
auths = pd.read_csv("authorships.csv", low_memory=False)
print(f"Total authorship records: {len(auths):,}")
print(f"Columns: {auths.columns.tolist()}")
print(f"Author positions: {auths['author_position'].value_counts().to_string()}")

# 0.4 Topic taxonomy
print("\n--- 0.4 Topic Taxonomy ---")
topics = pd.read_csv("works_topics.csv", low_memory=False)
print(f"Total topic records: {len(topics):,}")
print(f"Columns: {topics.columns.tolist()}")

# Get primary topic per work
if "is_primary" in topics.columns:
    primary = topics[topics["is_primary"] == True].copy()
elif "topic_rank" in topics.columns:
    primary = topics[topics["topic_rank"] == 0].copy()  # Check if 0-indexed
    if len(primary) == 0:
        primary = topics[topics["topic_rank"] == 1].copy()
else:
    primary = topics.drop_duplicates("work_id").copy()

print(f"Works with primary topic: {len(primary):,}")

print("\n=== ALL DOMAINS ===")
print(primary["domain_name"].value_counts().to_string())

print("\n=== ALL FIELDS ===")
print(primary["field_name"].value_counts().to_string())

print("\n=== ALL SUBFIELDS (top 80) ===")
print(primary["subfield_name"].value_counts().head(80).to_string())

# Specifically show Engineering subfields
print("\n=== SUBFIELDS UNDER 'Engineering' FIELD ===")
eng = primary[primary["field_name"].str.contains("Engineering", na=False)]
print(eng["subfield_name"].value_counts().to_string())

# Show Life Sciences subfields  
print("\n=== SUBFIELDS UNDER 'Life Sciences' DOMAIN ===")
life = primary[primary["domain_name"] == "Life Sciences"]
print(life["subfield_name"].value_counts().head(40).to_string())

# Show anything biomedical-sounding outside Health Sciences
print("\n=== POTENTIALLY BIOMEDICAL SUBFIELDS IN NON-HEALTH-SCIENCE DOMAINS ===")
bio_kw = ["Biomedical", "Biomaterial", "Health Informatics", "Parasitol", "Virol", 
           "Medicinal Chem", "Toxicol", "Pharm", "Pathol", "Epidemiol", "Anatomy",
           "Physiol", "Clinical", "Medical"]
non_health = primary[primary["domain_name"] != "Health Sciences"]
for kw in bio_kw:
    matches = non_health[non_health["subfield_name"].str.contains(kw, case=False, na=False)]
    if len(matches) > 0:
        print(f"  '{kw}' matches in non-Health domains:")
        for _, row in matches[["domain_name", "field_name", "subfield_name"]].drop_duplicates().iterrows():
            count = len(matches[matches["subfield_name"] == row["subfield_name"]])
            print(f"    {row['domain_name']} > {row['field_name']} > {row['subfield_name']} ({count} works)")

# 0.5 Other files
print("\n--- 0.5 Other Files ---")
for fname in ["works_funding.csv", "works_keywords.csv", "works_sdgs.csv", "author_affiliations.csv"]:
    try:
        df = pd.read_csv(fname, low_memory=False, nrows=3)
        print(f"\n{fname}: {df.columns.tolist()}")
    except Exception as e:
        print(f"\n{fname}: NOT FOUND or EMPTY ({e})")


print("\n" + "=" * 80)
print("PHASE 1: BIOMEDICAL FILTER")
print("=" * 80)

# 1.1 Define inclusion criteria
domain_include = {"Health Sciences"}

field_include = {
    "Biochemistry, Genetics and Molecular Biology",
    "Immunology and Microbiology",
    "Neuroscience",
    "Pharmacology, Toxicology and Pharmaceutics",
}

# Biomedical engineering subfields only
subfield_include = {
    "Biomedical Engineering",
}

# 1.2 Apply the filter
primary["include"] = (
    primary["domain_name"].isin(domain_include) |
    primary["field_name"].isin(field_include) |
    primary["subfield_name"].isin(subfield_include)
)

included_work_ids = set(primary[primary["include"]]["work_id"])

# Print included
included = primary[primary["include"]]
print(f"\nWorks matching biomedical filter: {len(included_work_ids):,}")
print(f"Works excluded: {len(primary) - len(included):,}")

print("\n=== INCLUDED: Domains ===")
print(included["domain_name"].value_counts().to_string())
print("\n=== INCLUDED: Fields ===")
print(included["field_name"].value_counts().to_string())
print("\n=== INCLUDED: Top 30 Subfields ===")
print(included["subfield_name"].value_counts().head(30).to_string())

# Print excluded (spot-check)
excluded = primary[~primary["include"]]
print("\n=== EXCLUDED: Fields ===")
print(excluded["field_name"].value_counts().to_string())
print("\n=== EXCLUDED: Top 20 Subfields ===")
print(excluded["subfield_name"].value_counts().head(20).to_string())

# 1.3 Save filtered data
FILTERED_DIR = Path("filtered_biomedical")
FILTERED_DIR.mkdir(exist_ok=True)

works_filtered = works[works["work_id"].isin(included_work_ids)]
works_filtered.to_csv(FILTERED_DIR / "works_metadata_filtered.csv", index=False)
print(f"\nSaved works_metadata_filtered.csv: {len(works_filtered):,} rows")

for fname, csv_name in [
    ("authorships.csv", "authorships_filtered.csv"),
    ("author_affiliations.csv", "author_affiliations_filtered.csv"),
    ("works_funding.csv", "works_funding_filtered.csv"),
    ("works_keywords.csv", "works_keywords_filtered.csv"),
    ("works_sdgs.csv", "works_sdgs_filtered.csv"),
    ("works_topics.csv", "works_topics_filtered.csv"),
]:
    try:
        df = pd.read_csv(fname, low_memory=False)
        df_filtered = df[df["work_id"].isin(included_work_ids)]
        df_filtered.to_csv(FILTERED_DIR / csv_name, index=False)
        print(f"  {csv_name}: {len(df_filtered):,} rows")
    except Exception as e:
        print(f"  {csv_name}: SKIPPED ({e})")

# 1.4 Filter log
filter_log = {
    "study_universe": "Biomedical science and biomedical engineering",
    "filter_applied_to": "Primary OpenAlex topic classification per work",
    "domains_included_entirely": sorted(domain_include),
    "fields_included_entirely": sorted(field_include),
    "subfields_included_selectively": sorted(subfield_include),
    "total_works_before_filter": int(len(works)),
    "total_works_after_filter": int(len(works_filtered)),
    "works_excluded_by_filter": int(len(works) - len(works_filtered)),
}
with open(FILTERED_DIR / "filter_log.json", "w") as f:
    json.dump(filter_log, f, indent=2)
print("\nFilter log saved.")

print("\n✓ Phase 0+1 complete.")
