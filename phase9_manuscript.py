"""
PHASE 9: MANUSCRIPT TEXT GENERATION
Ghana Bibliometric Study
"""
import json
from pathlib import Path

OUTPUT_DIR = Path("analysis_results")

# Load key numbers
with open(OUTPUT_DIR / "prisma_numbers.json") as f:
    prisma = json.load(f)
with open(OUTPUT_DIR / "audit_results.json") as f:
    audit = json.load(f)

# Read key CSV results
import pandas as pd
table1 = pd.read_csv(OUTPUT_DIR / "table1_leadership_proportions.csv")
covid_df = pd.read_csv(OUTPUT_DIR / "pre_post_covid.csv")
bloc_df = pd.read_csv(OUTPUT_DIR / "leadership_by_bloc.csv")
country_df = pd.read_csv(OUTPUT_DIR / "country_level_analysis.csv")
reg_results = pd.read_csv(OUTPUT_DIR / "regression_results.csv")
rob_df = pd.read_csv(OUTPUT_DIR / "robustness_summary.csv")

print("=" * 80)
print("PHASE 9: MANUSCRIPT TEXT")
print("=" * 80)

manuscript = f"""
# MANUSCRIPT TEXT SNIPPETS
## Ghana Bibliometric Study — Authorship Equity in International Biomedical Research

---

## Methods — Study Universe

Works were classified as biomedical using their primary OpenAlex topic assignment. 
The study universe included all works whose primary topic fell within the Health Sciences 
domain (encompassing Medicine, Nursing, Health Professions, and Dentistry); the Life Sciences 
fields of Biochemistry, Genetics and Molecular Biology, Immunology and Microbiology, 
Neuroscience, and Pharmacology, Toxicology and Pharmaceutics; and the Biomedical Engineering 
and Bioengineering subfields within Engineering. General engineering fields (Chemical, Civil, 
Electrical, Environmental, Materials, and Mechanical Engineering) were excluded because 
authorship conventions in these disciplines do not reliably follow the first-last-corresponding 
framework that underlies our positional analysis.

## Methods — Affiliation Classification

"Ghanaian" is determined by institutional affiliation, not nationality: a British researcher 
at the University of Ghana is classified as "Ghanaian"; a Ghanaian national at Harvard with 
no Ghanaian affiliation is classified as "Non-Ghanaian." Researchers who listed both Ghanaian 
and non-Ghanaian institutional affiliations on a given paper were classified as "Dual-affiliated."

## Methods — Authorship Positions

We use the term "research leadership" to refer specifically to occupying these conventionally 
recognized authorship positions (first, last, and corresponding). This operationalization 
captures positional credit within established disciplinary norms, which may not reflect the 
full breadth of intellectual and logistical contributions to collaborative research.

## Methods — Bilateral/Consortium

Papers were classified as bilateral if all non-Ghanaian co-authors belonged to a single 
geographic bloc, and as multi-bloc otherwise. This classification is derived from the partner 
bloc variable used in the regression models.

## Methods — Country-Level

Country-level analyses include all papers with at least one co-author from that country, 
regardless of bloc classification. For comparison, bilateral-only rates are also reported.

---

## Results — Study Characteristics

From {prisma['total_openalex']:,} Ghanaian-affiliated works in OpenAlex, {prisma['total_biomedical']:,} 
met our biomedical field criteria. After restricting to the 2000-2025 study period 
({prisma['within_study_period']:,} works), requiring international collaboration 
({prisma['international_collabs']:,} works), and excluding single-author papers, our final 
study set comprised {prisma['final_study_set']:,} works with {audit['total_authorships']:,} 
authorships from {audit['unique_authors']:,} unique authors across {audit['unique_countries']} 
countries and territories.

The median team size was {audit['median_team_size']:.0f} (IQR: ...), and the median number 
of countries per paper was {audit['median_country_count']:.0f}. {audit['pre_covid']:,} works 
({100*audit['pre_covid']/(audit['pre_covid']+audit['post_covid']):.1f}%) were published before 
the COVID-19 pandemic (2000-2019), and {audit['post_covid']:,} 
({100*audit['post_covid']/(audit['pre_covid']+audit['post_covid']):.1f}%) during or after 
(2020-2025).

## Results — Overall Leadership Proportions

Ghanaian-affiliated researchers held first authorship on {table1[(table1['Position']=='First') & (table1['Category']=='Ghanaian')]['Pct'].values[0]}% 
of international collaborative publications, with an additional 
{table1[(table1['Position']=='First') & (table1['Category']=='Dual-affiliated')]['Pct'].values[0]}% 
authored by dual-affiliated researchers (combined: 
{table1[(table1['Position']=='First') & (table1['Category']=='Ghanaian')]['Pct'].values[0] + table1[(table1['Position']=='First') & (table1['Category']=='Dual-affiliated')]['Pct'].values[0]:.1f}%).

The leadership deficit was most pronounced for last authorship, where only 
{table1[(table1['Position']=='Last') & (table1['Category']=='Ghanaian')]['Pct'].values[0]}% 
of papers had a Ghanaian last author (combined with Dual: 
{table1[(table1['Position']=='Last') & (table1['Category']=='Ghanaian')]['Pct'].values[0] + table1[(table1['Position']=='Last') & (table1['Category']=='Dual-affiliated')]['Pct'].values[0]:.1f}%).

Corresponding authorship showed similar patterns to first authorship 
({table1[(table1['Position']=='Corresponding') & (table1['Category']=='Ghanaian')]['Pct'].values[0]}% 
Ghanaian, {table1[(table1['Position']=='Corresponding') & (table1['Category']=='Dual-affiliated')]['Pct'].values[0]}% 
Dual-affiliated). However, {audit['first_is_corr_pct']:.1f}% of corresponding authors were 
also the first author, limiting the independent analytical contribution of corresponding 
authorship as a distinct measure of epistemic ownership.

## Results — Simpson's Paradox in Temporal Trends

The bivariate trend in Ghanaian first authorship showed no significant change over time 
(Mann-Kendall tau=-0.077, p=0.597). However, within-stratum analysis revealed a significant 
increasing trend in bilateral partnerships (tau=0.440, p=0.002), masked by the growing share 
of multi-bloc consortium publications, which rose from 20.8% of output in 2000-2005 to 41.5% 
in 2020-2025. Multi-bloc papers had substantially lower Ghanaian leadership rates across all 
positions, producing a confounding adjustment reversal (Simpson's paradox).

---

## Discussion — Theoretical Framing

Our finding that emerging-power partnerships replicate authorship inequities suggests the 
mechanism is structural-economic -- driven by resource concentration and institutional power -- 
rather than specifically postcolonial. This does not invalidate the decolonization framework 
but demands its expansion beyond the North-South binary.

---

## Limitations

### Selection Bias
This study examines only internationally collaborative publications. Domestic publications -- 
where Ghanaian researchers hold all authorship positions by definition -- are excluded. Our 
findings characterize leadership dynamics within international collaborations specifically.

### Affiliation-Based Classification
Although expatriate academics comprise less than 2% of Ghana's university communities 
(Agyeman, 2023), they are likely disproportionately represented in internationally 
collaborative publications. Our affiliation-based classification would count such researchers 
as "Ghanaian," potentially inflating local leadership estimates.

### Journal Prestige
Journal prestige is an unmeasured confounder. Papers in high-impact international journals 
likely have different authorship patterns than those in regional journals, and we were unable 
to control for this.

---

## Text Fixes Checklist
- [x] ICMJE citation: 2024, not 2026
- [x] Fisher citation: Fisher's exact test ~1934, not Fisher (1922)
- [x] "Felipe et.al, 2023" -> "Felipe et al., 2023"
- [x] Remove dangling "and" at end of partner bloc list
- [x] Remove double "However" in corresponding author methods paragraph
- [x] Fill empty citation "( )" in bilateral-consortium section
- [x] "exponential growth" -> "rapid growth" (unless exponential model fitted)
- [x] "203 countries" -> "{audit['unique_countries']} countries and territories"
- [x] Table 5 must state it reports combined Ghanaian + Dual-affiliated rates
- [x] Simpson's paradox: bilateral shows increasing trend, aggregate flat -> confounding adjustment reversal confirmed
- [x] Temper corresponding authorship claims: {audit['first_is_corr_pct']:.1f}% overlap limits independent contribution
"""

# Save manuscript text
with open(OUTPUT_DIR / "manuscript_text.md", "w", encoding="utf-8") as f:
    f.write(manuscript)

print(manuscript)
print(f"\nManuscript text saved to {OUTPUT_DIR / 'manuscript_text.md'}")
print("\nPhase 9 complete.")
print("\n" + "=" * 80)
print("ALL PHASES COMPLETE.")
print("=" * 80)
