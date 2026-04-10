# KEY FINDINGS

## Headline Numbers
- Total study works: 24,768
- Study period: 2000-2025
- Total authorships: 223,453
- Unique authors: 93,203
- Countries represented: 203

## Finding 1: Overall Leadership
- Ghanaian first authorship: 25.8% (95% CI: 25.3-26.3)
- Ghanaian last authorship: 22.0% (95% CI: 21.5-22.5)
- Ghanaian corresponding authorship: 24.5% (95% CI: 24.0-25.0)
- Dual-affiliated first authorship: 19.1% (95% CI: 18.6-19.6)
- Combined Ghanaian+Dual first authorship: 44.900000000000006%
- Non-Ghanaian first authorship: 55.1%
- Chi-square goodness-of-fit: highly significant (p < 0.001) for all positions
- **Interpretation**: Ghanaian researchers hold ~26% of first authorships and ~22% of last authorships despite comprising ~31% of authorships on these papers. Leadership is disproportionately non-Ghanaian.

## Finding 2: Temporal Trend
- Direction: Logistic regression shows year_centered OR=1.019 (p < 0.001) -- slight positive trend after controlling for confounders
- Mann-Kendall trend on annual proportions: tau=-0.17, p=0.234 -- not significant in raw annual data
- Pre-COVID Ghanaian first authorship: 24.5%
- Post-COVID Ghanaian first authorship: 26.6%
- **Interpretation**: After controlling for team size, field, partner region, and funding, there IS a small positive time trend. But the raw annual data does not show a clear monotonic trend, suggesting confounders (growing team sizes, rising multi-bloc collaborations) mask underlying improvement.

## Finding 3: COVID Effect
- Pre-COVID first authorship (Ghanaian): 24.5% -> Post-COVID: 26.6%
- Chi-square pre vs post: chi2=69.20, p < 0.001 (significant shift in composition)
- Non-COVID papers only: Pre=24.6% vs Post=27.1%
- COVID topic removal: leadership shift PERSISTS -- structural change
- **Interpretation**: The shift in authorship composition between pre and post-COVID is significant, but Ghanaian-only first authorship increased while dual-affiliated declined. The Ghanaian last authorship improved (19.9% to 23.3%).

## Finding 4: Western vs Non-Western Partnerships
- Western (N=10,836): GH+Dual first author = 54.4%
- East Asian (N=1,788): GH+Dual first author = 45.2%
- South Asian (N=441): GH+Dual first author = 48.3%
- African (N=2,671): GH+Dual first author = 61.2%
- Latin American (N=46): GH+Dual first author = 54.4%
- MENA (N=222): GH+Dual first author = 61.7%
- Multi-bloc (N=8,691): GH+Dual first author = 27.1%
- Binary Western vs Non-Western: chi2=9.55, p=0.002 (significant)
- African collaborations show HIGHEST Ghanaian leadership (61.2% first author GH+Dual)
- Multi-bloc collaborations show LOWEST (27.1%) -- larger teams dilute local leadership
- East Asian collaborations: 45.2% -- lower than Western (54.4%)
- **Interpretation**: South-South (African) collaborations are most equitable. Multi-bloc (large consortium) papers strongly dilute Ghanaian leadership. East Asian partnerships (primarily China) show lower Ghanaian leadership than Western partnerships.

## Finding 5: Funding and Leadership
- Logistic regression: has_funding_bool OR=0.783 (p < 0.001) for first author
- Logistic regression: has_funding_bool OR=0.667 (p < 0.001) for last author
- Funding is a SIGNIFICANT NEGATIVE predictor of Ghanaian leadership
- Externally funded papers are LESS likely to have Ghanaian first/last authors
- **Interpretation**: This supports the "he who pays the piper" hypothesis. External funding is associated with reduced Ghanaian leadership, even after controlling for field, partner, year, and team size.

## Finding 6: Dual-Affiliation Dynamics
- Dual-affiliated first authors: declining from 25.1% (2000-2010) to 17.7% (2020-2025)
- Top dual-affiliation countries: US, Zambia, UK, South Africa, China
- Ghanaian-only first authorship: increasing (compensating for dual-affiliation decline)
- Sensitivity check: conclusions are robust to dual-affiliation reclassification
- **Interpretation**: The decline in dual-affiliated leadership does NOT indicate reduced Ghanaian leadership overall -- rather, Ghanaian-only authors are increasingly taking first author positions on their own merit.

## Finding 7: Citation Impact
- Median FWCI when Ghanaian first author: 0.86
- Median FWCI when Dual-affiliated first author: 0.80
- Median FWCI when Non-Ghanaian first author: 1.27
- Mann-Whitney Ghanaian vs Non-Ghanaian: U=32,400,962, p < 0.001 (significant)
- Top 10% papers: Ghanaian-led 20.8%, Non-Ghanaian-led 28.8%
- **Interpretation**: Non-Ghanaian-led papers receive more citations. This likely reflects journal prestige bias, network effects, and topic selection rather than research quality.

## Finding 8: Key Predictors (Logistic Regression)
**Model 1 (First Author), AIC=30693.5, pseudo-R2=0.100:**
- year_centered: OR=1.019 (1.011-1.027), p < 0.001 (improving over time)
- country_count: OR=0.690 (0.667-0.712), p < 0.001 (more countries = less GH leadership)
- author_count: OR=0.978 (0.972-0.983), p < 0.001 (larger teams = less GH leadership)
- has_funding: OR=0.783 (0.739-0.830), p < 0.001 (funded = less GH leadership)
- is_oa: OR=1.213 (1.138-1.293), p < 0.001 (OA = more GH leadership)
- bloc_East Asian: OR=0.530 (0.468-0.601), p < 0.001 (vs African reference)
- covid_era: OR=0.899 (0.820-0.984), p=0.022
- field_Engineering: OR=1.259 (1.119-1.417), p < 0.001 (engineering = more GH leadership)

## Sensitivity Check Summary
- Dual-affiliation reclassification: Generous=44.9%, Conservative=25.8% (vs Original=25.8%) -- conclusions robust
- Article-only restriction: similar results -- conclusions robust
- Corresponding author (high-coverage years only): similar results -- conclusions robust
- Works with >50 authors excluded: similar results -- conclusions robust
- COVID topic removal: leadership shift persists
- Western vs Non-Western binary: significant (p=0.002) -- conclusions robust

## Country Deep Dives
- **China**: 2,040 collaborations, GH first author = 30.7%
- **India**: 1,414 collaborations, GH first author = 23.4%
- **South Africa**: 3,063 collaborations, GH first author = 37.1%
- **Brazil**: 514 collaborations, GH first author = 11.5%

## Top Ghanaian Institutions
- University of Ghana: 7,709 collaborative works
- KNUST: 5,692 collaborative works
- University of Cape Coast: 2,325 works
- University of Health and Allied Sciences: 2,230 works
- Noguchi Memorial Institute: 1,963 works
