"""
Re-run biomedical filter with Bioengineering added to subfield_include.
"""
import json
import pandas as pd
from pathlib import Path

works = pd.read_csv("works_metadata.csv", low_memory=False)
topics = pd.read_csv("works_topics.csv", low_memory=False)

# Get primary topic per work
primary = topics[topics["is_primary"] == True].copy()

domain_include = {"Health Sciences"}
field_include = {
    "Biochemistry, Genetics and Molecular Biology",
    "Immunology and Microbiology",
    "Neuroscience",
    "Pharmacology, Toxicology and Pharmaceutics",
}
subfield_include = {
    "Biomedical Engineering",
    "Bioengineering",  # Added per user request (9 works)
}

primary["include"] = (
    primary["domain_name"].isin(domain_include) |
    primary["field_name"].isin(field_include) |
    primary["subfield_name"].isin(subfield_include)
)

included_work_ids = set(primary[primary["include"]]["work_id"])
print(f"Works matching biomedical filter: {len(included_work_ids):,}")

# Save filtered data
FILTERED_DIR = Path("filtered_biomedical")
FILTERED_DIR.mkdir(exist_ok=True)

works_filtered = works[works["work_id"].isin(included_work_ids)]
works_filtered.to_csv(FILTERED_DIR / "works_metadata_filtered.csv", index=False)
print(f"Saved works_metadata_filtered.csv: {len(works_filtered):,} rows")

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

# Update filter log
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

print("Filter log updated. Done.")
