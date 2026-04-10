# Research Leadership in International Collaborations Involving Ghanaian Researchers in Biomedical Science and Engineering: A Bibliometric Analysis, 2000–2025

## 1. Introduction

International research collaboration has become the dominant mode of knowledge production in the biomedical sciences and engineering. For researchers in low- and middle-income countries (LMICs), these collaborations offer access to funding, advanced infrastructure, methodological expertise, and global scholarly networks. Yet a persistent concern in the global health and science policy literature is whether local researchers from LMICs genuinely lead these partnerships—conceptualising the research questions, directing the analysis, and stewarding the manuscript through publication—or whether they serve primarily as data collectors and field implementers while intellectual credit accrues to partners from high-income countries.

A growing body of work has examined authorship equity in international health research. Hedt-Gauthier et al. (2019, *BMJ Global Health*) demonstrated that researchers from LMICs were significantly underrepresented in first and last authorship positions in global health research. Mbaye et al. (2019) documented author inequity in African health research specifically, while Iyer et al. (2018) analysed North-South partnerships in public health. Rees et al. (2021) examined trends in authorship equity across multiple global health journals. However, these studies share several limitations: most did not analyse corresponding authorship at scale, none disaggregated findings by emerging-power partnerships (e.g., China, India, Brazil), and none examined the COVID-19 period or applied multivariate adjustment to separate structural confounders from genuine trend change.

Authorship position remains the primary currency of academic credit and career advancement. In the biomedical sciences, three positions carry distinct leadership signals. First authorship typically denotes the researcher who performed the most substantive work and led the writing. Last authorship signals senior intellectual leadership—the principal investigator, laboratory director, or senior mentor who conceived and supervised the research. Corresponding authorship designates the researcher who manages the manuscript throughout peer review, handles editorial correspondence, and serves as the long-term point of contact for the work. Together, these three positions constitute the measurable surface area of research leadership.

Ghana provides an informative case for studying LMIC research leadership. As the second-largest economy in West Africa, Ghana possesses an established university system anchored by the University of Ghana (founded 1948) and Kwame Nkrumah University of Science and Technology (founded 1952), strong international research partnerships, and one of the highest rates of international co-publication among sub-Saharan African countries. This combination of institutional maturity and extensive collaboration makes Ghana an ideal site to examine whether partnership leads to parity.

This study examined 24,768 international collaborative publications involving Ghanaian researchers in biomedical science and engineering between 2000 and 2025. Using bibliometric data from OpenAlex, we quantified leadership across first, last, and corresponding authorship positions; identified structural predictors through multivariate logistic regression; assessed temporal trends with both parametric and non-parametric methods; and disaggregated findings by partner region, country, field, and funding source. The study contributes four novel findings to the literature: (a) a corresponding author analysis at scale with 99.9% data coverage, including validation of potential imputation bias; (b) the first analysis of authorship equity in emerging-power partnerships (China, India, Brazil); (c) a pre-/post-COVID decomposition with topic-based sensitivity testing; and (d) the identification of a Simpson's paradox in temporal trends via multivariate adjustment.

## 2. Study Characteristics and Data Quality

The analytical dataset was derived from a wider corpus of 46,945 filtered publications. After applying inclusion criteria—international collaboration (at least one non-Ghanaian co-author), publication between 2000 and 2025, and at least two authors—the final study set comprised 24,768 works. These works contained 223,453 individual authorship records representing 93,203 unique authors affiliated with institutions across 203 countries. An additional 320,537 affiliation records, 69,961 topic classifications, 28,541 funding records, and 292,364 keywords were linked to the study set.

The median number of authors per paper was 6 (IQR: 4–10), and the median number of countries per paper was 2 (IQR: 2–3). Health Sciences dominated the field composition at 69.4%, followed by Physical Sciences (16.6%) and Life Sciences (14.0%). The temporal distribution was heavily skewed toward recent years: the pre-COVID period (2000–2019) accounted for 9,675 works (39.1%), while the post-COVID period (2020–2025) contributed 15,093 works (60.9%).

Data quality was excellent. Field-Weighted Citation Impact (FWCI) was available for 91.6% of works. No works suffered from author list truncation. Corresponding author information was available for 24,752 works (99.9%). However, this unusually high coverage warrants scrutiny. Analysis revealed that in 80.1% of works, the first author was also flagged as the corresponding author. This rate is substantially higher than the 50–60% observed in typical biomedical publishing, suggesting that OpenAlex may impute the first author as corresponding when explicit corresponding author metadata is unavailable. This means the corresponding author analysis partially collapses into the first author analysis—a caveat that should be borne in mind when interpreting corresponding authorship findings. We retain the analysis because the 80.1% overlap rate, while high, still leaves roughly 5,000 works where corresponding and first authorship diverge, and because the corresponding author results show meaningfully different patterns from first authorship (e.g., a steeper temporal decline, larger funding chi-square), suggesting the measure captures distinct signal even with imputation.

Within the authorship records, affiliation categories were distributed as follows: Non-Ghanaian authors comprised 153,305 records (68.6%), Ghanaian authors 51,647 records (23.1%), and dual-affiliated authors—those holding simultaneous affiliations in Ghana and at least one other country—18,501 records (8.3%). An important methodological note: "Ghanaian" is defined by institutional affiliation, not nationality. A British researcher employed at the University of Ghana counts as "Ghanaian" in this analysis, while a Ghanaian national working at Harvard with no Ghanaian institutional affiliation counts as "Non-Ghanaian." This classification likely biases toward higher reported Ghanaian leadership, as some researchers categorised as "Ghanaian" may be expatriate investigators at Ghanaian institutions.

Partner bloc classification assigned each work to one of eight categories: Western (10,836 works; 43.8%), Multi-bloc (8,691; 35.1%), African (2,671; 10.8%), East Asian (1,788; 7.2%), South Asian (441; 1.8%), MENA (222; 0.9%), Other (73; 0.3%), and Latin American (46; 0.2%).

## 3. Overall Leadership Proportions

The central finding is that non-Ghanaian researchers hold a disproportionate share of leadership positions across all three measures. Non-Ghanaian researchers occupied 55.1% (95% CI: 54.5–55.8%) of first authorships, 66.2% (95% CI: 65.6–66.8%) of last authorships, and 59.9% (95% CI: 59.4–60.5%) of corresponding authorships. Ghanaian researchers held 25.8% (95% CI: 25.3–26.3%) of first authorships, 22.0% (95% CI: 21.5–22.5%) of last authorships, and 24.5% (95% CI: 24.0–25.0%) of corresponding authorships. Dual-affiliated researchers accounted for 19.1% (95% CI: 18.6–19.6%) of first authorships, 11.8% (95% CI: 11.4–12.2%) of last authorships, and 15.6% (95% CI: 15.2–16.0%) of corresponding authorships.

Chi-square goodness-of-fit tests determined whether the distribution of affiliation categories in leadership positions deviated from the overall authorship composition. All three revealed highly significant deviations: first author χ²(2) = 4,212.26, p < 0.001; last author χ²(2) = 407.07, p < 0.001; corresponding author χ²(2) = 2,372.25, p < 0.001. The large chi-square for first authorship was driven by the substantial overrepresentation of dual-affiliated and Ghanaian researchers in first authorship relative to their overall share of authorships (44.9% combined first authorship vs. 31.4% of all authorships). This indicates that first authorship is the position where Ghanaian-affiliated researchers are best represented. Conversely, the last authorship result reflected the most consequential inequity: Ghanaian and dual-affiliated researchers held only 33.8% of last authorships, while non-Ghanaian researchers held 66.2%—a position associated with the intellectual architect of the study, the senior investigator who secures funding, designs the research programme, and mentors junior researchers.

## 4. Temporal Trends

### 4.1 Bivariate Trend Analysis

Bivariate logistic regression with publication year (centred at 2000) as the sole predictor revealed that the proportion of Ghanaian or dual-affiliated researchers in leadership positions is not improving over time when examined without adjustment. First authorship showed a statistically significant decline (OR = 0.994, 95% CI: 0.990–0.999, p = 0.008), meaning the odds of Ghanaian leadership decreased by approximately 0.6% per year. Corresponding authorship showed a steeper decline (OR = 0.993, 95% CI: 0.989–0.997, p < 0.001), representing a 0.7% annual reduction in odds. Last authorship was statistically stable (OR = 1.001, 95% CI: 0.996–1.006, p = 0.685).

### 4.2 Non-Parametric Trend Tests

Mann-Kendall trend tests on annual Ghanaian leadership proportions corroborated the bivariate findings. Corresponding authorship was the only position showing a statistically significant monotonic decline (τ = –0.3200, p = 0.023, Sen's slope = –0.15 percentage points per year)—a cumulative erosion of nearly four percentage points over the study period. First authorship showed a non-significant declining trend (τ = –0.1692, p = 0.234, Sen's slope = –0.07 pp/year) and last authorship a non-significant decline (τ = –0.2185, p = 0.123, Sen's slope = –0.18 pp/year).

### 4.3 Multivariate Trend Analysis: A Simpson's Paradox

The multivariate logistic regression models—which control for team size, number of partner countries, partner bloc, field, funding, and open access status—revealed a striking reversal. After adjustment, the year coefficient became positive for both first authorship (OR = 1.019, 95% CI: 1.011–1.027, p < 0.001) and corresponding authorship (OR = 1.018, 95% CI: 1.010–1.026, p < 0.001). This constitutes a classic Simpson's paradox: the raw trend is negative, but the adjusted trend is positive once confounders are controlled.

The mechanism is identifiable. The rapid growth of multi-bloc consortia and larger team sizes over the study period acts as a structural confounder. Multi-bloc collaborations—which show Ghanaian first authorship of only 27.1%—grew substantially as a proportion of total output. Because the composition of the collaboration landscape shifted toward configurations that inherently suppress local leadership, the aggregate proportion declined even as Ghanaian researchers' conditional probability of leadership in comparable publications increased. To illustrate concretely: a Ghanaian researcher leading a bilateral study with a UK partner in 2005 and a comparable bilateral study in 2020 would have higher odds of first authorship in 2020. But the 2020 research landscape also includes far more consortium papers—large multi-site clinical trials, pandemic-response studies, genomic consortia—where no single country leads. The aggregate proportion, which mixes bilateral and consortium papers, declines even though the within-type probability improves.

## 5. The Bilateral–Consortium Divide

This decomposition is arguably the most important structural finding in the study. When the dataset is split into bilateral partnerships (16,077 works, 64.9%) and multi-bloc collaborations (8,691 works, 35.1%), the leadership picture diverges dramatically:

| Position | Bilateral (N=16,077) | Multi-bloc (N=8,691) |
|---|---|---|
| First author (GH+Dual) | 54.5% | 27.1% |
| Last author (GH+Dual) | 40.7% | 20.9% |
| Corresponding author (GH+Dual) | 51.2% | 23.6% |

In bilateral partnerships, Ghanaian and dual-affiliated researchers hold a **majority** of first authorships (54.5%) and a majority of corresponding authorships (51.2%). The authorship equity problem is not fundamentally about bilateral partnerships—it is almost entirely a consortium phenomenon. In multi-bloc collaborations, Ghanaian leadership drops to roughly one in four for first authorship and one in five for last authorship.

This finding reframes the equity narrative. It is not that international collaboration per se disadvantages Ghanaian researchers. Rather, the structure of large consortia—with their complex governance hierarchies, centralised data analysis units, multiple writing committees, and funder-driven authorship norms—systematically reduces local leadership. Given that multi-bloc papers comprise 35.1% of the dataset and have grown rapidly, this structural dilution is the primary driver of the aggregate leadership deficit.

## 6. Leadership by Partner Region and Country

### 6.1 Partner Bloc Analysis

The geographic composition of partnerships profoundly shapes Ghanaian leadership. African collaborations yielded the highest rates: 61.2% (95% CI: 59.4–63.0%) of first authors were Ghanaian or dual-affiliated, along with 48.8% of last authors and 60.6% of corresponding authors. MENA partnerships showed similarly high rates (61.7%, 95% CI: 55.2–67.9% first author), though the sample was small (N = 222). Western partnerships produced intermediate rates at 54.4% (95% CI: 53.5–55.4%) first authorship. East Asian partnerships showed notably lower rates at 45.1% (95% CI: 42.8–47.4%), while South Asian partnerships yielded 48.3% (95% CI: 43.7–53.0%). Multi-bloc collaborations showed the lowest rates at 27.1% (95% CI: 26.2–28.0%).

Chi-square tests confirmed highly significant differences between Western and African partnerships across all three positions: first author χ² = 39.48 (p < 0.001), last author χ² = 82.04 (p < 0.001), and corresponding author χ² = 80.31 (p < 0.001). The largest chi-square for last authorship indicates that the senior leadership gap between North-South and South-South collaborations is most pronounced at the supervisory level. Western versus non-Western binary comparisons were also significant (first χ² = 9.55, p = 0.002; last χ² = 22.98, p < 0.001; corresponding χ² = 13.13, p < 0.001). Bonferroni-corrected pairwise comparisons confirmed significant differences for first authorship between Western and African (χ² = 39.48, p < 0.001), Western and East Asian (χ² = 53.09, p < 0.001), and Western and South Asian partnerships (χ² = 6.21, p = 0.013), all surviving the corrected α = 0.0167.

### 6.2 Country-Level Deep Dives

Country-level analyses with Wilson confidence intervals revealed substantial variation. China, the largest East Asian partner (2,040 collaborations), showed 30.7% Ghanaian first authorship (95% CI: 28.8–32.8%). India (1,414 collaborations) showed a lower rate of 23.4% (95% CI: 21.3–25.7%). These are novel findings: most authorship equity studies focus on traditional North-South (US/UK) partnerships, and demonstrating that emerging-power partnerships replicate—and in some cases worsen—the same inequities has not been previously documented at this scale.

South Africa, the largest African partner (3,063 collaborations), yielded 37.1% Ghanaian first authorship (95% CI: 35.4–38.8%)—higher than Asian partners but significantly below the African bloc average, reflecting South Africa's position as a regional research power whose institutional hierarchies may mirror North-South dynamics within Africa. Brazil (514 collaborations) showed the lowest rate at 11.5% (95% CI: 9.0–14.5%). While the confidence interval is wide, even the upper bound (14.5%) falls far below the study average, confirming that Brazil-Ghana collaborations are characterised by unusually low Ghanaian leadership.

## 7. Leadership by Disciplinary Field

Field-level analysis revealed significant variation in Ghanaian leadership. Engineering showed consistently higher rates across all three positions, confirmed in the regression models: first authorship OR = 1.259 (95% CI: 1.119–1.417, p < 0.001), last authorship OR = 2.035 (95% CI: 1.788–2.317, p < 0.001), and corresponding authorship OR = 1.588 (95% CI: 1.410–1.788, p < 0.001). The strong last authorship effect (OR = 2.035) indicates that Ghanaian engineers are approximately twice as likely to occupy senior leadership positions compared to the reference field, likely reflecting the locally-driven nature of engineering research—infrastructure, water treatment, agricultural technology—where problems are defined by local context.

Medicine also showed significantly positive effects for last authorship (OR = 1.486, 95% CI: 1.321–1.672, p < 0.001) and corresponding authorship (OR = 1.207, 95% CI: 1.087–1.340, p < 0.001). Health Professions (OR = 1.696, 95% CI: 1.468–1.960) and Nursing (OR = 1.272, 95% CI: 1.072–1.509) were significant positive predictors for last authorship. These health-related fields, involving community-based or clinical research requiring local expertise and ethical oversight, provide structural reasons for Ghanaian investigators to occupy senior positions.

## 8. Funding and Leadership

Funding was one of the strongest and most consistent predictors. External funding was a significant negative predictor of Ghanaian first authorship (OR = 0.783, 95% CI: 0.739–0.830, p < 0.001), last authorship (OR = 0.667, 95% CI: 0.627–0.710, p < 0.001), and corresponding authorship (OR = 0.724, 95% CI: 0.683–0.767, p < 0.001). The effect was strongest for last authorship, where funded papers showed 33.3% lower odds—consistent with the expectation that the principal investigator role, often tied to the grant, is retained by the funder's institutional nominee.

Chi-square tests comparing Northern-funded, Ghanaian-funded, and unfunded papers confirmed highly significant effects: first author χ²(2) = 634.57, last author χ²(2) = 488.43, and corresponding author χ²(2) = 1,206.93 (all p < 0.001). The corresponding authorship chi-square was nearly double that for first authorship, indicating that funding source exerts the strongest control over who serves as the intellectual custodian of the published work. These findings provide robust quantitative support for the "he who pays the piper" hypothesis: when external funders support research conducted in Ghana, Ghanaian investigators are substantially less likely to occupy the positions that confer credit, career advancement, and ongoing control of the research narrative.

## 9. COVID-19 Effects

### 9.1 Pre-/Post-COVID Compositional Shifts

Comparing the pre-COVID (2000–2019) and post-COVID (2020–2025) periods revealed significant compositional shifts whose net effects differ by position. For first authorship, Ghanaian-only researchers rose from 24.5% to 26.6%, but dual-affiliated first authors declined from 21.6% to 17.4%. The net combined position (Ghanaian + dual-affiliated) therefore **declined** from 46.1% to 44.0%—a 2.1 percentage point reduction. The rise in Ghanaian-only first authorship was more than offset by the dual-affiliated decline.

For last authorship, the picture was genuinely positive. Combined Ghanaian + dual-affiliated last authorship improved from 32.7% to 34.5%—a net gain of 1.8 percentage points, making this the only position showing real combined improvement.

For corresponding authorship, the pattern mirrored first authorship. Combined corresponding authorship declined from 41.6% to 39.3%—a 2.3 percentage point reduction, driven by a sharp 5.2 percentage point decline in dual-affiliated corresponding authorship (19.0% to 13.8%), the single largest category shift in the dataset.

Chi-square tests confirmed significance at all positions: first χ²(2) = 69.20, last χ²(2) = 46.89, corresponding χ²(2) = 149.06 (all p < 0.001).

### 9.2 COVID Topic Sensitivity

To test whether these shifts simply reflected pandemic-response research, we removed 781 COVID-topic papers and re-examined the remaining 23,987 non-COVID works. Using the combined Ghanaian + dual-affiliated metric for consistency with Section 9.1, first authorship was 46.2% pre-COVID versus 44.6% post-COVID—confirming that the combined decline persists even after removing COVID-topic papers. Using Ghanaian-only first authorship, the rate rose from 24.5% to 27.0%, confirming that the Ghanaian-only improvement is also structural rather than COVID-driven. The net interpretation is that domestic capacity is genuinely growing (Ghanaian-only first authorship increasing), but diasporic leadership is declining faster (dual-affiliated share falling), and the combined position has slightly worsened.

## 10. Multivariate Logistic Regression Models

### 10.1 Model Specifications and Diagnostics

Three models were fit, one per leadership position, with the outcome defined as Ghanaian or dual-affiliated (1) versus non-Ghanaian (0). Variance Inflation Factor analysis for year_centered and covid_era yielded values of 2.80—below the threshold of 5.0—permitting simultaneous inclusion. The pseudo-R² values (0.089–0.100) are modest but characteristic of bibliometric logistic regressions, where authorship decisions depend on unmeasurable factors: personal relationships, laboratory hierarchies, career stage, and negotiation dynamics. The models identify structural factors that systematically shift the probability of Ghanaian leadership, recognising substantial unexplained variance.

### 10.2 Model 1: First Authorship (AIC = 30,693.3, pseudo-R² = 0.100)

Each additional partner country reduced odds by 31.0% (OR = 0.690, 95% CI: 0.667–0.712)—the single strongest predictor. Each additional co-author reduced odds by 2.2% (OR = 0.978, 95% CI: 0.972–0.983). Funded papers showed 21.7% lower odds (OR = 0.783). Open access was a positive predictor (OR = 1.213, 95% CI: 1.138–1.293). After controlling for all other factors, year showed a significant positive effect (OR = 1.019, p < 0.001).

The COVID era coefficient requires careful interpretation. The model shows COVID as a small negative predictor (OR = 0.899, 95% CI: 0.820–0.984, p = 0.022). This does not contradict the pre-/post-COVID comparison: the positive year trend (OR = 1.019 per year) accumulates over 20 years, so by 2020 the cumulative year effect is strongly positive. The covid_era binary captures the additional discrete shift beyond the year trend—and that residual is slightly negative for first authorship, meaning the COVID period slightly underperformed what the 20-year trend alone would have predicted. The overall post-COVID level remains higher than baseline because the accumulated year-effect dominates the small negative COVID residual.

Partner bloc effects were pronounced relative to the African reference group: East Asian OR = 0.531, South Asian OR = 0.522, Multi-bloc OR = 0.646, Western OR = 0.853 (all p < 0.001). Engineering was the sole significant positive field (OR = 1.261).

### 10.3 Model 2: Last Authorship (AIC = 28,723.1, pseudo-R² = 0.089)

Similar structural patterns emerged with larger effect sizes. Country count reduced odds by 29.3% per country (OR = 0.707). Author count reduced odds by 3.9% per author (OR = 0.961)—substantially larger than the first author effect of 2.2%, reflecting how larger teams dilute the probability of any single country occupying the last position. Funding showed the strongest negative effect across all three models (OR = 0.667, 95% CI: 0.627–0.710).

The COVID era was a significant **positive** predictor for last authorship (OR = 1.196, 95% CI: 1.087–1.316, p < 0.001)—contrasting with the negative COVID coefficient for first authorship and consistent with the genuine combined improvement from 32.7% to 34.5%.

Field effects were broadly positive: Engineering OR = 2.032, Health Sciences OR = 1.496. This broad field effect, absent from the first author model, suggests Ghanaian senior researchers have a structural advantage in applied and clinical disciplines where local knowledge is indispensable.

### 10.4 Model 3: Corresponding Authorship (AIC = 31,003.5, pseudo-R² = 0.090)

Country count (OR = 0.749), author count (OR = 0.980), and funding (OR = 0.724) were significant negative predictors. Open access showed the strongest positive effect of all models (OR = 1.368, 95% CI: 1.283–1.459). East Asian partnerships showed the strongest negative association across all models (OR = 0.435, 95% CI: 0.383–0.494)—56.5% lower odds. Engineering (OR = 1.590) and Health Sciences (OR = 1.188) were significant positive fields. Year showed a positive trend (OR = 1.018, p < 0.001). Noting the caveat from Section 2.4 regarding potential imputation of first authors as corresponding, the similarity between this model and Model 1 should be interpreted conservatively.

## 11. Citation Impact Analysis

Citation analysis using FWCI (coverage: 91.6%) revealed a significant gap. Among the 22,684 works with FWCI data, the median FWCI for papers with a non-Ghanaian first author was 1.27 (IQR: 0.20–3.42, N = 12,585), compared to 0.86 (IQR: 0.00–2.39, N = 5,866) for Ghanaian-led papers and 0.80 (IQR: 0.00–2.35, N = 4,233) for dual-affiliated-led papers. Mann-Whitney U tests confirmed significance: Ghanaian versus non-Ghanaian (U = 32,400,962, p < 0.001) and dual-affiliated versus non-Ghanaian (U = 23,255,833, p < 0.001).

The finding that dual-affiliated-led papers show a lower median FWCI (0.80) than Ghanaian-only-led papers (0.86) is counterintuitive. Several explanations are plausible: dual-affiliated authors may take first authorship primarily on smaller bilateral studies while their higher-impact work is credited to their non-Ghanaian affiliation; the dual-affiliation category may include administrative or courtesy appointments with limited research engagement; and selection effects may operate whereby the bilateral studies that naturally place dual-affiliated researchers in first author role (e.g., capacity-building, training publications) inherently have lower citation profiles.

Among the same 22,684 FWCI-available works, 31.3% of non-Ghanaian-led papers (3,938 of 12,585) fell in the top 10% of cited publications, compared to 22.6% of Ghanaian-led papers (1,328 of 5,866) and 22.6% of dual-affiliated-led papers (957 of 4,233). This gap likely reflects structural advantages—journal prestige networks, topic selection, language support, institutional reputation—rather than inherent quality differences. Nevertheless, it compounds the career disadvantage: Ghanaian researchers not only lead fewer papers but see their led papers accumulate fewer citations.

## 12. Dual-Affiliation Dynamics

Dual-affiliated researchers showed a notable temporal decline as a proportion of first authors, falling from 25.1% in 2000–2010 to 17.7% in 2020–2025. This decline was partially compensated by a rise in Ghanaian-only first authors, suggesting growing domestic capacity. However, as the pre-/post-COVID arithmetic in Section 9.1 demonstrates, the compensation is incomplete: dual-affiliated losses more than offset Ghanaian-only gains for first and corresponding authorship.

The geographic distribution of dual affiliates concentrated in a few countries: the United States (8,708 instances), Zambia (2,690), the United Kingdom (2,241), South Africa (2,224), China (1,397), Australia (1,058), Germany (1,046), Nigeria (981), the Netherlands (783), and India (738). The US/UK prominence reflects historical patterns of Ghanaian academic migration and training relationships. Zambia, South Africa, and Nigeria likely reflect multi-site clinical trial networks across sub-Saharan Africa.

## 13. Institutional Analysis

Among the top five Ghanaian institutions, meaningful differences in leadership rates emerged. The University of Cape Coast and the University of Health and Allied Sciences (UHAS) showed the highest rates, both achieving 55.4% first authorship, with corresponding authorship of 49.2% and 50.7% respectively. KNUST followed at 52.3% first, 38.2% last, and 46.4% corresponding. The Noguchi Memorial Institute showed 49.6% first but lower last (34.4%) and corresponding (38.7%), suggesting international partners retain senior roles even at a research-focused institution.

The University of Ghana, despite the highest volume (7,709 works), showed the lowest leadership rates: 44.5% first, 33.0% last, 38.7% corresponding. This is consistent with UG's deeper involvement in major international consortia, mirroring the bilateral-consortium divide: universities embedded in consortium networks show lower leadership rates despite (or because of) their higher international visibility.

## 14. Sensitivity Analyses

Six sensitivity analyses confirmed robustness.

**Sensitivity 1: Dual-affiliation reclassification.** Generous (dual = Ghanaian): first authorship rose to 44.9%. Conservative (dual = non-Ghanaian): remained at 25.8%. Core findings unchanged in either scenario.

**Sensitivity 2: Article-only restriction.** Journal articles only (19,788 works, 79.9%): Ghanaian first authorship 26.5%, dual 19.1%, non-Ghanaian 54.5%—nearly identical to full dataset.

**Sensitivity 3: Corresponding author coverage.** All 26 study years met ≥50% corresponding author coverage, ensuring temporal trends are not artefacts of differential data availability.

**Sensitivity 4: Author truncation.** No works had truncated author lists. Excluding 337 works with >50 authors: Ghanaian first authorship 26.1%.

**Sensitivity 5: COVID-topic exclusion.** After removing 781 COVID-topic papers (23,987 remaining), combined (Ghanaian + dual-affiliated) first authorship was 46.2% pre-COVID versus 44.6% post-COVID—confirming the combined decline persists. Ghanaian-only first authorship rose from 24.5% to 27.0%, confirming domestic capacity growth is structural.

**Sensitivity 6: Western vs. non-Western pairwise.** Bonferroni-corrected comparisons: Western vs. African (54.4% vs. 61.2%, χ² = 39.48, p < 0.001), Western vs. East Asian (54.4% vs. 45.1%, χ² = 53.09, p < 0.001), Western vs. South Asian (54.4% vs. 48.3%, χ² = 6.21, p = 0.013). All survived corrected α = 0.0167.

## 15. Implications and Policy Recommendations

**First, funders should mandate local corresponding authorship for locally-conducted research.** The corresponding author role carries the strongest funding chi-square (1,206.93), shows a significant monotonic decline (Mann-Kendall τ = –0.32, p = 0.023), and experienced the largest compositional shift in the COVID transition (dual-affiliated corresponding authorship fell 5.2 percentage points). Funders such as the NIH, Wellcome Trust, and Gates Foundation should require that corresponding authors for research conducted primarily in Ghana be based at a Ghanaian institution.

**Second, authorship equity should become a reportable grant metric.** Making first, last, and corresponding authorship distributions by affiliation country a reporting requirement would create transparency and accountability.

**Third, intra-African collaboration funding deserves targeted expansion.** African partnerships show the highest Ghanaian leadership (61.2% first, 48.8% last, 60.6% corresponding) and the smallest equity gaps. South-South funding mechanisms should be scaled.

**Fourth, consortium governance should include authorship equity provisions.** Multi-bloc collaborations (35.1% of the dataset) show the lowest Ghanaian leadership (27.1% first author vs. 54.5% in bilateral partnerships). This is the single largest structural driver of the aggregate deficit. Large consortia should adopt authorship rotation policies, minimum local leadership quotas, or mentoring mandates that prepare LMIC investigators for senior authorship.

## 16. Conclusions

This bibliometric analysis of 24,768 international collaborative publications reveals a persistent leadership deficit for Ghanaian researchers—but one whose nature is more nuanced than aggregate proportions suggest. In bilateral partnerships (64.9% of collaborations), Ghanaian and dual-affiliated researchers hold a majority of first authorships (54.5%) and corresponding authorships (51.2%). The aggregate deficit is driven primarily by multi-bloc consortia, where leadership drops to 27.1% for first authorship. External funding reduces odds of Ghanaian leadership by 21.7–33.3%. The corresponding author role—the only position showing a statistically significant monotonic decline—is eroding at 0.15 percentage points per year, though this finding should be interpreted cautiously given that 80.1% of corresponding authors are also first authors, suggesting possible imputation in the source data. A Simpson's paradox in temporal trends reveals that after controlling for structural confounders, Ghanaian leadership is genuinely improving (OR = 1.019 per year)—but this gain is being absorbed by the shift toward larger consortia. The post-COVID period brought real improvement only in last authorship (combined 32.7% → 34.5%); first authorship (46.1% → 44.0%) and corresponding authorship (41.6% → 39.3%) both declined in combined terms. Novel findings on emerging-power partnerships—China (30.7%, 95% CI: 28.8–32.8%), India (23.4%, 95% CI: 21.3–25.7%), and Brazil (11.5%, 95% CI: 9.0–14.5%)—demonstrate that authorship inequity is not exclusively a North-South phenomenon.
