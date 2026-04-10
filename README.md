# Ghana Bibliometric Study: Research Leadership in International Collaborations

A comprehensive bibliometric analysis (2000–2025) examining research leadership patterns in 24,768 international collaborative publications involving Ghanaian researchers in biomedical science and engineering.

## Research Question

**In international collaborations involving Ghanaian researchers, who leads?** This study quantifies Ghanaian research leadership through first, last, and corresponding authorship positions across 25 years of publication data.

## Key Findings

- **Corresponding author decline**: While first and last authorship show modest improvements, Ghanaian corresponding authorship is declining (Mann-Kendall τ = –0.32, p = 0.023)
- **Funding effect**: Funded papers show 22–34% lower odds of Ghanaian leadership across all positions
- **Bilateral–Consortium divide**: Bilateral partnerships show 54.5% Ghanaian first authorship vs. 27.1% in multi-bloc consortia
- **Simpson's Paradox**: Raw trends show declining leadership, but multivariate models controlling for structural changes reveal improving odds

## Pipeline Architecture

The analysis runs as a 4-script sequential pipeline:

| Script | Purpose |
|--------|---------|
| `phase1_data_prep.py` | Data ingestion, filtering, validation |
| `phase2_descriptive_core.py` | Descriptive statistics, leadership proportions, citation impact |
| `phase3_advanced.py` | Logistic regression, COVID analysis, partner bloc dynamics |
| `phase4_sensitivity.py` | Six sensitivity analyses, findings summary |

### Supporting Scripts
- `config.py` — Shared configuration (paths, palettes, bloc mappings)
- `compute_cis.py` — Wilson confidence intervals for key metrics
- `generate_prisma.py` — PRISMA-style flow diagram
- `run_collapsed_reg.py` — Health Sciences collapsed regression models

## Data Source

Publication metadata retrieved from [OpenAlex](https://openalex.org/), an open-access scholarly database indexing 250M+ works with structured affiliation, authorship, and citation data.

**Note**: Raw data files are excluded from this repository due to size (~3GB). The analysis scripts and all generated outputs (charts, tables, findings) are included.

## Outputs

All outputs are saved to `analysis_results/`:
- **22 charts** (PNG format) covering temporal trends, leadership distributions, forest plots, and more
- **21+ CSV tables** with detailed statistical results
- **LaTeX manuscript** (`methodology_results.tex`) — full methodology and results section
- **Findings document** (`findings_document.md`) — 5,000-word forensically verified narrative

## Dependencies

```
pandas
numpy
scipy
statsmodels
matplotlib
seaborn
pyarrow
```

## Usage

```bash
python phase1_data_prep.py
python phase2_descriptive_core.py
python phase3_advanced.py
python phase4_sensitivity.py
```

## License

This research is conducted for academic purposes.
