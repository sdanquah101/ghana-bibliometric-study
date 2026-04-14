"""
Shared utility functions for the Ghana Bibliometric Study.
All analysis scripts import from this module instead of duplicating code.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from pathlib import Path
import json, warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
INTERMEDIATE = BASE / "analysis_results" / "intermediate"
RESULTS = BASE / "analysis_results"

# ── Plot style ───────────────────────────────────────────────────────────
def setup_plot_style():
    """Configure consistent matplotlib defaults for all charts."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# ── Boolean conversion ───────────────────────────────────────────────────
def to_bool(series: pd.Series) -> pd.Series:
    """Safely convert a mixed-type series (str/bool/int/NaN) to boolean."""
    def _convert(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val) if not pd.isna(val) else False
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes")
        return False
    return series.apply(_convert)

# ── Wilson confidence interval ───────────────────────────────────────────
def wilson_ci(count: int, total: int, alpha: float = 0.05) -> tuple:
    """
    Wilson score confidence interval for a binomial proportion.
    Returns (lower, upper) as proportions [0, 1].
    """
    if total == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = count / total
    denom = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return (max(0, centre - margin), min(1, centre + margin))

# ── Effect size: Cohen's h ───────────────────────────────────────────────
def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for comparing two proportions.
    |h| < 0.2 = small, 0.2-0.5 = medium, > 0.8 = large.
    """
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

# ── Effect size: Cramér's V ──────────────────────────────────────────────
def cramers_v(contingency_table: np.ndarray) -> float:
    """Cramer's V from a contingency table (numpy 2D array)."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum()
    r, c = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))

# ── Leadership extraction ────────────────────────────────────────────────
def get_leadership(works: pd.DataFrame, authorships: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the affiliation category of first, last, and corresponding
    authors for each work. Returns a DataFrame indexed by work_id with
    columns: first_cat, last_cat, corr_cat, plus derived boolean columns
    gh_first, gh_last, gh_corr (True if Ghanaian or Dual-affiliated).
    """
    # First author
    first = (authorships[authorships["author_position"] == "first"]
             .drop_duplicates("work_id")
             [["work_id", "affiliation_category"]]
             .rename(columns={"affiliation_category": "first_cat"}))

    # Last author
    last = (authorships[authorships["author_position"] == "last"]
            .drop_duplicates("work_id")
            [["work_id", "affiliation_category"]]
            .rename(columns={"affiliation_category": "last_cat"}))

    # Corresponding author — sort by position index first to get earliest
    corr = authorships[to_bool(authorships["is_corresponding_combined"])].copy()
    corr = corr.sort_values(["work_id", "author_position_index"])
    corr = (corr.drop_duplicates("work_id")
            [["work_id", "affiliation_category"]]
            .rename(columns={"affiliation_category": "corr_cat"}))

    # Merge
    leadership = works[["work_id"]].merge(first, on="work_id", how="left")
    leadership = leadership.merge(last, on="work_id", how="left")
    leadership = leadership.merge(corr, on="work_id", how="left")

    # Boolean indicators
    gh_cats = {"Ghanaian", "Dual-affiliated"}
    leadership["gh_first"] = leadership["first_cat"].isin(gh_cats)
    leadership["gh_last"] = leadership["last_cat"].isin(gh_cats)
    leadership["gh_corr"] = leadership["corr_cat"].isin(gh_cats)

    return leadership

# ── Chart saving ─────────────────────────────────────────────────────────
def save_chart(fig, name: str, dpi: int = 300):
    """Save figure as both PNG and SVG in the results directory."""
    png_path = RESULTS / f"{name}.png"
    svg_path = RESULTS / f"{name}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {png_path.name}, {svg_path.name}")

# ── Load prepared data ───────────────────────────────────────────────────
def load_study_data():
    """Load the prepared works + authorships parquet files."""
    works = pd.read_parquet(INTERMEDIATE / "v2_works.parquet")
    authorships = pd.read_parquet(INTERMEDIATE / "v2_authorships.parquet")
    return works, authorships

# ── Paper-level random-assignment null ───────────────────────────────────
def random_assignment_test(works: pd.DataFrame, authorships: pd.DataFrame,
                           position: str = "first"):
    """
    Test whether GH/Dual authors are over/under-represented in a given
    authorship position relative to their share of each paper's team.
    
    Returns dict with expected_count, observed_count, ratio, z_stat, p_value.
    """
    # Count GH+Dual authors per paper
    paper_comp = authorships.groupby("work_id").agg(
        total=("author_id", "size"),
        gh_dual=("affiliation_category",
                 lambda x: x.isin(["Ghanaian", "Dual-affiliated"]).sum())
    ).reset_index()

    # Get the actual author in the position
    if position == "first":
        pos_data = authorships[authorships["author_position"] == "first"]
    elif position == "last":
        pos_data = authorships[authorships["author_position"] == "last"]
    else:  # corresponding
        pos_data = authorships[to_bool(authorships["is_corresponding_combined"])]
        pos_data = pos_data.sort_values(["work_id", "author_position_index"])

    pos_data = pos_data.drop_duplicates("work_id")
    pos_data["is_gh"] = pos_data["affiliation_category"].isin(
        ["Ghanaian", "Dual-affiliated"])

    merged = paper_comp.merge(pos_data[["work_id", "is_gh"]], on="work_id", how="inner")
    merged["p_random"] = merged["gh_dual"] / merged["total"]

    expected = merged["p_random"].sum()
    observed = merged["is_gh"].sum()
    
    # Variance under the null (sum of Bernoulli variances)
    var_null = (merged["p_random"] * (1 - merged["p_random"])).sum()
    se = np.sqrt(var_null)
    z = (observed - expected) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "position": position,
        "n_papers": len(merged),
        "expected": expected,
        "observed": int(observed),
        "ratio": observed / expected if expected > 0 else np.nan,
        "difference": int(observed - expected),
        "z_stat": z,
        "p_value": p,
        "direction": "over" if observed > expected else "under"
    }

# ── FWCI cleaning ────────────────────────────────────────────────────────
def clean_fwci(series: pd.Series, winsorize_pct: float = 99.0) -> pd.Series:
    """
    Clean FWCI values: convert to numeric, winsorize at the given percentile.
    """
    fwci = pd.to_numeric(series, errors="coerce")
    cap = fwci.quantile(winsorize_pct / 100)
    return fwci.clip(upper=cap)
