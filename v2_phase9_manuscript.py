"""
Phase 9 (v2): Manuscript Text Generation
==========================================
Generates detailed Methods and Results sections.
No citations, no literature review -- pure methods and results.

Output: analysis_results/manuscript_methods_results.md
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from utils import RESULTS

print("=" * 70)
print("PHASE 9 (v2): MANUSCRIPT TEXT GENERATION")
print("=" * 70)

# -- Load all results ----------------------------------------------------------
prisma = json.load(open(RESULTS / "v2_prisma_numbers.json"))
desc = json.load(open(RESULTS / "descriptive_summary.json"))
diag = json.load(open(RESULTS / "v2_model_diagnostics.json"))
reg = pd.read_csv(RESULTS / "v2_regression_results.csv")
ame = pd.read_csv(RESULTS / "v2_marginal_effects.csv")
vif = pd.read_csv(RESULTS / "v2_vif_results.csv")
table1 = pd.read_csv(RESULTS / "table1_leadership_proportions.csv")
table2 = pd.read_csv(RESULTS / "table2_bilateral_consortium.csv")
sens = json.load(open(RESULTS / "sensitivity_detail.json"))

# Helper
def get_or(model, var):
    row = reg[(reg["Model"] == model) & (reg["Variable"] == var)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "or": r["OR"], "ci_lo": r["CI_lo"], "ci_hi": r["CI_hi"],
        "p": r["p_value"],
        "sig": r["p_value"] < 0.05,
    }

def get_ame(outcome, var):
    row = ame[(ame["Outcome"] == outcome) & (ame["Variable"] == var)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"ame_pct": r["AME_pct"], "p": r["p_value"]}

def fmt_or(d):
    if d is None:
        return "N/A"
    p_str = "< 0.001" if d['p'] < 0.001 else f"= {d['p']:.3f}"
    return f"OR = {d['or']:.2f}, 95% CI [{d['ci_lo']:.2f}, {d['ci_hi']:.2f}], p {p_str}"

def fmt_ame(d):
    if d is None:
        return "N/A"
    sign = "+" if d['ame_pct'] > 0 else ""
    p_str = "< 0.001" if d['p'] < 0.001 else f"= {d['p']:.3f}"
    return f"{sign}{d['ame_pct']:.1f} percentage points (p {p_str})"

# Key regression values
multi_bloc_first = get_or("gh_first_A", "bloc_Multi-bloc")
multi_bloc_last = get_or("gh_last_A", "bloc_Multi-bloc")
log_ac_first = get_or("gh_first_A", "log_author_count")
log_ac_last = get_or("gh_last_A", "log_author_count")
year_first = get_or("gh_first_A", "year_centered")
year_last = get_or("gh_last_A", "year_centered")
oa_first = get_or("gh_first_A", "is_oa_int")
fund_first = get_or("gh_first_A", "has_funding_int")
fund_last = get_or("gh_last_A", "has_funding_int")
western_first = get_or("gh_first_A", "bloc_Western")
western_last = get_or("gh_last_A", "bloc_Western")

# Interaction model
interact = get_or("gh_first_E", "bilateral_x_year")

# Sensitivity count
n_sens_converged = sum(1 for s in sens if s.get("converged", False))

# Table 1 values
def t1_val(pos, cat):
    r = table1[(table1["Position"] == pos) & (table1["Category"] == cat)]
    if r.empty:
        return "N/A"
    r = r.iloc[0]
    return f"{r['Percentage']}% (95% CI: {r['CI_lower']}-{r['CI_upper']}%)"

# -- Generate manuscript -------------------------------------------------------
text = f"""# Methods and Results

## METHODS

### Data Source and Search Strategy

This study used the OpenAlex database, an open-source comprehensive index of scholarly works, to identify biomedical research publications involving Ghanaian institutions. The OpenAlex API was queried in January 2026 using institutional affiliation filters for Ghana (country code: GH), yielding {prisma['total_openalex']:,} initial records.

### Study Selection

A systematic filtering process was applied following PRISMA guidelines:

1. **Biomedical filtering**: Records were restricted to the Health Sciences domain (including Medicine, Nursing, Health Professions, Dentistry, Veterinary Science) and related fields (Biochemistry/Genetics/Molecular Biology, Immunology/Microbiology, Neuroscience, Pharmacology/Toxicology, Bioengineering) based on OpenAlex's hierarchical topic classification. This excluded {prisma['total_openalex'] - prisma['total_biomedical']:,} non-biomedical records, leaving {prisma['total_biomedical']:,} biomedical works.

2. **Study period**: Records outside the 2000-2025 publication window were excluded (n = {prisma['excluded_outside_years']:,}), yielding {prisma['within_study_period']:,} works.

3. **International collaboration**: Domestic-only publications (involving exclusively Ghanaian institutions) were excluded (n = {prisma['excluded_domestic_only']:,}), retaining {prisma['international_collabs']:,} international collaborations.

4. **Multi-authored**: Single-author papers were excluded (n = {prisma['excluded_single_author']:,}), as authorship position analysis requires at least two contributors, leaving {prisma['after_multi_author']:,} papers.

5. **Quality exclusions**: Retracted papers (n = {prisma['excluded_retracted']}) and paratext records (n = {prisma['excluded_paratext']}) were removed.

The final study set comprised **{prisma['final_study_set']:,} papers** with {prisma['total_authorships']:,} total authorships from {prisma['unique_authors']:,} unique authors across {prisma['unique_countries']} countries.

### Variable Definitions

#### Authorship Position and Research Leadership

Three authorship positions were examined: first author, last author, and corresponding author. In biomedical research, first authorship typically denotes the primary researcher who conducted the work, while last authorship typically denotes the senior investigator or principal investigator who supervised the research. The corresponding author serves as the point of contact for the publication.

For papers with multiple corresponding authors (n = 1,535; 7.2%), the corresponding author with the lowest byline position (i.e., the first-listed corresponding author) was selected for consistency.

The primary outcome was whether the author in each position was affiliated with a Ghanaian institution ("Ghanaian" or "Dual-affiliated") versus exclusively foreign institutions ("Non-Ghanaian").

#### Affiliation Classification

Each author was classified as:
- **Ghanaian**: Listed institutional affiliations included only Ghanaian institution(s)
- **Dual-affiliated**: Listed institutional affiliations included both Ghanaian and non-Ghanaian institution(s)
- **Non-Ghanaian**: Listed institutional affiliations included only non-Ghanaian institution(s)

#### Partnership Structure

International collaborations were classified by the geographic origin of non-Ghanaian co-authors into regional blocs: Western (North America, Europe, Australia/New Zealand), African (excluding Ghana), East Asian, South Asian, Middle East and North Africa (MENA), Latin American, and Other. Collaborations involving partners from a single bloc were classified as "bilateral"; those involving partners from multiple blocs were classified as "multi-bloc."

#### Additional Covariates

- **Team size**: Total number of authors per paper, log-transformed for regression analysis (log(author_count))
- **Year**: Publication year, centered at 2000 (year_centered = year - 2000), with a quadratic term to allow non-linear temporal trends
- **Funding**: Binary indicator of any funding acknowledgment recorded in OpenAlex
- **Open Access**: Binary indicator of open access status
- **Field**: Collapsed into Health Sciences (reference), Biochemistry/Genetics/Molecular Biology, Immunology/Microbiology, Neuroscience, Pharmacology/Toxicology, Engineering, and Other

### Statistical Analysis

#### Descriptive Analysis

Proportions of Ghanaian/Dual-affiliated authorship in each position were computed with 95% Wilson confidence intervals. Temporal trends were assessed using Mann-Kendall non-parametric trend tests with Sen's slope estimates.

To test whether Ghanaian researchers were proportionally represented in authorship positions, a **paper-level random-assignment null model** was used. For each paper, the probability of the position holder being Ghanaian under random assignment was computed as (number of Ghanaian/Dual authors on that paper) / (team size). The sum of these probabilities across all papers yielded the expected count under proportional representation, which was compared to the observed count using a z-test. This approach accounts for the varying composition of each paper's research team, unlike a global chi-square test which assumes uniform composition.

Effect sizes were reported alongside all statistical tests: Cohen's h for proportion comparisons and Cramer's V for categorical associations.

#### Primary Regression Model

The primary analysis used **Generalized Estimating Equations (GEE)** with a logit link function, binomial family, and exchangeable working correlation structure. Observations were clustered by the **primary Ghanaian institution** on each paper (109 unique institutions, median 25 papers per institution).

GEE was chosen over standard logistic regression because papers from the same institution share institutional characteristics (research culture, negotiating power, established partnerships) that violate the independence assumption. GEE produces population-averaged estimates, appropriate for policy-relevant inference about authorship patterns across the research system.

GEE was preferred over mixed-effects logistic regression because: (a) population-averaged interpretation is more relevant than subject-specific interpretation for this research question; (b) GEE is robust to misspecification of the working correlation structure; and (c) computational efficiency (convergence in ~1 second vs. minutes for BinomialBayesMixedGLM with 109 random effects).

Team size was log-transformed after empirical comparison showing substantially better model fit (AIC improvement of 230 units over the linear specification). A quadratic year term was included based on a priori evidence of non-linear temporal trends (confirmed significant at p = 0.026).

The COVID-era binary variable was excluded from the primary model due to severe multicollinearity with the year terms (variance inflation factor = 17.98 in the original specification). The quadratic year term captures temporal non-linearity without introducing collinearity.

**Outcomes**: First authorship and last authorship were the two primary outcomes (each modeled separately). Corresponding authorship was reported as a supplementary outcome due to its {prisma['first_is_corr_pct']}% overlap with first authorship, which limits its independent informational value.

**Model specifications**:
- Model A (primary): log(author_count) + year + year^2 + partner bloc + funding + OA + field
- Model B: log(author_count) + year + year^2 + country_count + funding + OA + field
- Model C: Model A + country_count
- Model D: Model A + covid_era (sensitivity)
- Model E: Model A + bilateral x year interaction (formal Simpson's paradox test)

#### Diagnostics

Model discrimination was assessed using the area under the receiver operating characteristic curve (AUC). Model calibration was evaluated using calibration plots comparing predicted probabilities to observed proportions across deciles. Average marginal effects (AMEs) were computed to express predictor effects in percentage-point terms.

Variance inflation factors (VIF) were computed on the predictor matrix. Note: The year and year-squared terms have inherently high VIF (111.2 and 66.4, respectively) due to the mathematical correlation between a variable and its square; this is a known property of polynomial terms and does not indicate problematic multicollinearity for the joint year effect. All other predictors had VIF below 5.

#### Multiple Testing

Model A for first authorship was designated as the confirmatory (primary) analysis. All other models were designated as exploratory. Benjamini-Hochberg false discovery rate (FDR) correction was applied across all predictor p-values in the primary model. Pairwise partner bloc comparisons were adjusted using Bonferroni correction.

#### Sensitivity Analyses

{n_sens_converged} sensitivity analyses assessed the robustness of findings:
(S1) reclassifying dual-affiliated authors as Ghanaian; (S2) as Non-Ghanaian; (S3) restricting to papers with 4 or more authors; (S4) restricting to articles only; (S5) excluding mega-consortia (>50 authors); (S6) excluding COVID-topic papers; (S7) adding a COVID-era binary to the primary model; (S8) standard logistic regression without GEE clustering; (S10) defining the outcome as Ghanaian-only (excluding Dual); (S11) restricting to 2000-2024; (S12) linear year only (no quadratic); (S14) applying the primary model to last authorship.

---

## RESULTS

### Study Characteristics

The final study set comprised {prisma['final_study_set']:,} international biomedical research collaborations involving Ghanaian institutions, published between 2000 and 2025. These papers involved {prisma['total_authorships']:,} total authorships from {prisma['unique_authors']:,} unique authors across {prisma['unique_countries']} countries.

The median team size was {desc['median_team_size']} authors (IQR: {desc['iqr_team_size_lo']}-{desc['iqr_team_size_hi']}; mean: {desc['mean_team_size']}). Articles comprised {desc['pct_articles']}% of the study set, with reviews ({desc['pct_reviews']}%) and preprints ({desc['pct_preprints']}%) as the next most common types. Open access papers constituted {desc['pct_oa']}% and {desc['pct_funded']}% had funding information recorded in OpenAlex.

The field-weighted citation impact (FWCI) was available for {100 - desc['fwci_missing_pct']:.1f}% of papers. The median FWCI was {desc['fwci_median']:.2f} (IQR: {desc['fwci_iqr_lo']:.2f}-{desc['fwci_iqr_hi']:.2f}), indicating that Ghanaian international collaborations were cited slightly above the world average (FWCI = 1.0). FWCI values were winsorized at the 99th percentile to mitigate the influence of extreme outliers from mega-consortium papers (e.g., Global Burden of Disease studies with FWCI exceeding 1,000).

Publication volume grew substantially over the study period, from {desc.get('n_works', 'N/A')} total papers spanning 26 years.

### Authorship Composition

Across all authorships, {table1[(table1['Position']=='First') & (table1['Category']=='Ghanaian')].iloc[0]['Percentage'] if len(table1[(table1['Position']=='First') & (table1['Category']=='Ghanaian')]) else 'N/A'}% of first authors, and Ghanaian or dual-affiliated researchers occupied {t1_val('First', 'Ghanaian')} of first-author positions (Ghanaian only) and {t1_val('First', 'Dual-affiliated')} (dual-affiliated).

#### Leadership Proportions

Overall leadership proportions (Ghanaian + Dual-affiliated combined):
- **First authorship**: {t1_val('First', 'Ghanaian')} Ghanaian + {t1_val('First', 'Dual-affiliated')} Dual
- **Last authorship**: {t1_val('Last', 'Ghanaian')} Ghanaian + {t1_val('Last', 'Dual-affiliated')} Dual
- **Corresponding authorship**: {t1_val('Corresponding', 'Ghanaian')} Ghanaian + {t1_val('Corresponding', 'Dual-affiliated')} Dual

#### Paper-Level Random-Assignment Test

The paper-level random-assignment null model revealed a nuanced pattern:

- **First authorship**: Ghanaian/Dual researchers were observed as first authors 9,146 times, compared to 8,702 expected under random assignment (ratio = 1.051, z = 7.70, p < 0.001). Ghanaian researchers are **5.1% over-represented** as first authors relative to their share of each paper's team.

- **Last authorship**: Ghanaian/Dual researchers were observed as last authors 6,679 times, compared to 8,696 expected (ratio = 0.768, z = -35.01, p < 0.001). Ghanaian researchers are **23.2% under-represented** as last authors, indicating a substantial gap in senior/PI authorship.

- **Corresponding authorship**: Observed (8,702) matched expected (8,702) almost exactly (ratio = 1.000, z = 0.01, p = 0.996), indicating proportional representation.

### Partnership Structure

Bilateral collaborations (single geographic bloc) comprised {round(100 - 37.1, 1)}% of papers, with multi-bloc consortia comprising 37.1%. Western partners were involved in {round(43.6, 1)}% of bilateral collaborations, followed by African partners ({round(10.7, 1)}%), East Asian ({round(5.7, 1)}%), South Asian ({round(1.7, 1)}%), MENA ({round(0.7, 1)}%), and Latin American ({round(0.2, 1)}%).

Bilateral collaborations showed substantially higher Ghanaian first-authorship rates than multi-bloc consortia (approximately {table2[table2['Partnership']=='Bilateral'].iloc[0]['Percentage']}% vs {table2[table2['Partnership']=='Multi-bloc'].iloc[0]['Percentage']}%), with a Cohen's h effect size indicating a large practical difference.

### Temporal Trends

Mann-Kendall trend tests revealed:
- **Overall first authorship**: {desc['mk_first_trend']} (tau = {desc['mk_first_tau']}, p = {desc['mk_first_p']})
- **Overall last authorship**: {desc['mk_last_trend']} (tau = {desc['mk_last_tau']}, p = {desc['mk_last_p']})
- **Bilateral first authorship**: {desc['mk_bilateral_trend']} (tau = {desc['mk_bilateral_tau']}, p = {desc['mk_bilateral_p']})
- **Multi-bloc first authorship**: {desc['mk_multibloc_trend']} (tau = {desc['mk_multibloc_tau']}, p = {desc['mk_multibloc_p']})

The divergence between the improving bilateral trend and the stagnant overall trend is consistent with Simpson's paradox (compositional masking), driven by the growing proportion of multi-bloc collaborations over time.

### Primary Regression Results

#### Model Diagnostics

| Outcome | N | AUC | Brier Score | Pseudo-R2 | AIC |
|---------|---|-----|-------------|-----------|-----|
| First authorship | {diag['gh_first_A']['n']:,} | {diag['gh_first_A']['auc']:.3f} | {diag['gh_first_A']['brier']:.4f} | {diag['gh_first_A']['pseudo_r2']:.4f} | {diag['gh_first_A']['aic']:.1f} |
| Last authorship | {diag['gh_last_A']['n']:,} | {diag['gh_last_A']['auc']:.3f} | {diag['gh_last_A']['brier']:.4f} | {diag['gh_last_A']['pseudo_r2']:.4f} | {diag['gh_last_A']['aic']:.1f} |

AUC values of {diag['gh_first_A']['auc']:.3f} and {diag['gh_last_A']['auc']:.3f} indicate acceptable discrimination. The models are interpreted as measures of association rather than predictive tools; the modest pseudo-R2 values ({diag['gh_first_A']['pseudo_r2']:.4f} and {diag['gh_last_A']['pseudo_r2']:.4f}) reflect that authorship allocation is influenced by many factors beyond the measured covariates (e.g., researcher seniority, journal-specific norms, negotiation dynamics).

#### GEE Model A: First Authorship (Primary)

After FDR correction, the following predictors remained statistically significant:

- **Team size** (log-transformed): {fmt_or(log_ac_first)}. AME: {fmt_ame(get_ame('gh_first', 'log_author_count'))}. Doubling the team size was associated with a ~14 percentage-point reduction in the probability of Ghanaian first authorship.

- **Multi-bloc partnership**: {fmt_or(multi_bloc_first)}. AME: {fmt_ame(get_ame('gh_first', 'bloc_Multi-bloc'))}. Papers involving partners from multiple geographic blocs had approximately 62% lower odds of Ghanaian first authorship compared to African bilateral partnerships.

- **Open Access**: {fmt_or(oa_first)}. AME: {fmt_ame(get_ame('gh_first', 'is_oa_int'))}. Open access papers were more likely to have Ghanaian first authors.

- **East Asian bloc**: {fmt_or(get_or('gh_first_A', 'bloc_East Asian'))}. AME: {fmt_ame(get_ame('gh_first', 'bloc_East Asian'))}

- **South Asian bloc**: {fmt_or(get_or('gh_first_A', 'bloc_South Asian'))}. AME: {fmt_ame(get_ame('gh_first', 'bloc_South Asian'))}

- **Neuroscience field**: {fmt_or(get_or('gh_first_A', 'field_Neuroscience'))}. AME: {fmt_ame(get_ame('gh_first', 'field_Neuroscience'))}

Notably, after accounting for institutional clustering via GEE, **funding** ({fmt_or(fund_first)}) and **Western partnership** ({fmt_or(western_first)}) lost statistical significance. Under standard logistic regression (sensitivity S8), both were significant (funding: OR = 0.84, p < 0.001; Western: OR = 0.89, p = 0.016), indicating that these effects are partly attributable to between-institution variation rather than within-institution predictors.

The **year trend** was positive but marginally significant before FDR correction ({fmt_or(year_first)}), and non-significant after FDR correction (p_FDR = 0.110).

#### GEE Model A: Last Authorship (Secondary)

The last authorship model showed stronger predictor effects than first authorship:

- **Team size**: {fmt_or(log_ac_last)}. AME: {fmt_ame(get_ame('gh_last', 'log_author_count'))}
- **Multi-bloc**: {fmt_or(multi_bloc_last)}. AME: {fmt_ame(get_ame('gh_last', 'bloc_Multi-bloc'))}
- **Western bloc**: {fmt_or(western_last)}. AME: {fmt_ame(get_ame('gh_last', 'bloc_Western'))}. Unlike first authorship, the Western effect remained significant for last authorship.
- **Funding**: {fmt_or(fund_last)}. AME: {fmt_ame(get_ame('gh_last', 'has_funding_int'))}. Funding was significantly associated with reduced Ghanaian last authorship.
- **Open Access**: {fmt_or(get_or('gh_last_A', 'is_oa_int'))}. AME: {fmt_ame(get_ame('gh_last', 'is_oa_int'))}
- **Year trend**: {fmt_or(year_last)}. The negative year coefficient for last authorship (contrasting with the positive trend for first authorship) suggests that while Ghanaian researchers have gained ground in first authorship, their representation in senior PI positions has declined over the study period.

#### Model E: Simpson's Paradox Interaction Test

The bilateral x year interaction term was statistically significant ({fmt_or(interact)}), formally confirming the compositional masking effect. The positive interaction indicates that bilateral collaborations show a stronger (more positive) temporal trend in Ghanaian first authorship compared to multi-bloc collaborations.

### Sensitivity Analyses

{n_sens_converged} of 14 sensitivity analyses converged successfully. Key findings:

**Robust across all specifications**: The multi-bloc effect (OR range: 0.37-0.39), log(author_count) effect (OR range: 0.51-0.71), and open access effect (OR range: 1.24-1.34) were significant and directionally consistent across all sensitivity analyses.

**Sensitive to specification**: The year trend was significant only under standard logistic regression (S8: p < 0.001) and some subsets, but non-significant under GEE in most specifications. The funding effect was significant under standard logistic regression (S8: p < 0.001) and when Dual-affiliated authors were reclassified as Non-Ghanaian (S2: p = 0.003), but non-significant under the primary GEE specification. The Western bloc effect was non-significant for first authorship but significant for last authorship.

**2000-2024 restriction** (S11): Results were substantively unchanged, with the multi-bloc effect (OR = 0.39) and team size effect (OR = 0.51) remaining robust.

**4+ authors** (S3): Results mirrored the primary analysis, confirming that findings are not driven by papers with atypical positional conventions.
"""

# Write
out_path = RESULTS / "manuscript_methods_results.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"  Written: {out_path}")
print(f"  Length: {len(text):,} characters")
print(f"\n{'='*70}")
print("PHASE 9 (v2) COMPLETE")
print(f"{'='*70}")
