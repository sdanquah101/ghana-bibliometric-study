"""
Sensitivity analysis: Ghanaian-only vs Ghanaian+Dual-affiliated
Computes O/E ratios for both groupings and key descriptive stats.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json

# Load data
INTERMEDIATE = Path("analysis_results/intermediate")
works = pd.read_parquet(INTERMEDIATE / "v2_works.parquet")
authorships = pd.read_parquet(INTERMEDIATE / "v2_authorships.parquet")

N = len(works)
print(f"Study set: {N:,} papers")

# ── Author composition ───────────────────────────────────────────────
print("\n" + "="*70)
print("AUTHOR COMPOSITION")
print("="*70)
print(authorships["affiliation_category"].value_counts().to_string())

# ── Helper: Random-assignment test for a given set of categories ─────
def null_test(authorships_df, works_df, gh_cats, position="first"):
    """Test over/under-representation for given affiliation categories."""
    paper_comp = authorships_df.groupby("work_id").agg(
        total=("author_id", "size"),
        gh_count=("affiliation_category", lambda x: x.isin(gh_cats).sum())
    ).reset_index()

    if position == "first":
        pos = authorships_df[authorships_df["author_position"] == "first"]
    elif position == "last":
        pos = authorships_df[authorships_df["author_position"] == "last"]
    pos = pos.drop_duplicates("work_id")
    pos["is_gh"] = pos["affiliation_category"].isin(gh_cats)

    merged = paper_comp.merge(pos[["work_id", "is_gh"]], on="work_id", how="inner")
    merged["p_random"] = merged["gh_count"] / merged["total"]

    expected = merged["p_random"].sum()
    observed = int(merged["is_gh"].sum())
    var_null = (merged["p_random"] * (1 - merged["p_random"])).sum()
    se = np.sqrt(var_null)
    z = (observed - expected) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    ratio = observed / expected if expected > 0 else np.nan

    return {
        "observed": observed,
        "expected": round(expected, 1),
        "ratio": round(ratio, 3),
        "z": round(z, 2),
        "p": p,
        "pct": round(observed / len(merged) * 100, 1),
    }


# ── Run for both groupings ───────────────────────────────────────────
print("\n" + "="*70)
print("NULL MODEL: GHANAIAN + DUAL-AFFILIATED (as in primary analysis)")
print("="*70)
gh_cats_combined = {"Ghanaian", "Dual-affiliated"}
for pos in ["first", "last"]:
    r = null_test(authorships, works, gh_cats_combined, pos)
    print(f"  {pos:6s}: observed={r['observed']:,}, expected={r['expected']:,.1f}, "
          f"O/E={r['ratio']:.3f}, z={r['z']:.2f}, p={r['p']:.4g}, rate={r['pct']}%")

print("\n" + "="*70)
print("NULL MODEL: GHANAIAN-ONLY (excluding Dual-affiliated)")
print("="*70)
gh_cats_only = {"Ghanaian"}
for pos in ["first", "last"]:
    r = null_test(authorships, works, gh_cats_only, pos)
    print(f"  {pos:6s}: observed={r['observed']:,}, expected={r['expected']:,.1f}, "
          f"O/E={r['ratio']:.3f}, z={r['z']:.2f}, p={r['p']:.4g}, rate={r['pct']}%")

# ── Detailed breakdown ───────────────────────────────────────────────
print("\n" + "="*70)
print("DETAILED RATES BY AFFILIATION CATEGORY")
print("="*70)

# First author affiliation category
first_authors = authorships[authorships["author_position"] == "first"].drop_duplicates("work_id")
last_authors = authorships[authorships["author_position"] == "last"].drop_duplicates("work_id")

print(f"\nFirst author affiliation breakdown (N={len(first_authors):,}):")
fc = first_authors["affiliation_category"].value_counts()
for cat in ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]:
    if cat in fc.index:
        print(f"  {cat:18s}: {fc[cat]:,} ({100*fc[cat]/len(first_authors):.1f}%)")

print(f"\nLast author affiliation breakdown (N={len(last_authors):,}):")
lc = last_authors["affiliation_category"].value_counts()
for cat in ["Ghanaian", "Dual-affiliated", "Non-Ghanaian"]:
    if cat in lc.index:
        print(f"  {cat:18s}: {lc[cat]:,} ({100*lc[cat]/len(last_authors):.1f}%)")

# ── Bilateral vs multi-bloc rates for both groupings ─────────────────
print("\n" + "="*70)
print("BILATERAL vs MULTI-BLOC: Ghanaian-only vs Combined")
print("="*70)

# Need leadership info with detailed categories
from utils import get_leadership
leadership = get_leadership(works, authorships)
df = works.merge(leadership, on="work_id", how="left")

# Ghanaian-only indicators
df["gh_only_first"] = df["first_cat"] == "Ghanaian"
df["gh_only_last"] = df["last_cat"] == "Ghanaian"
df["dual_first"] = df["first_cat"] == "Dual-affiliated"
df["dual_last"] = df["last_cat"] == "Dual-affiliated"

bilateral = df[df["is_bilateral"] == 1]
multibloc = df[df["is_bilateral"] == 0]

print(f"\n{'':20s} {'Combined':>12s} {'GH-only':>12s} {'Dual-only':>12s}")
print("-" * 60)

for label, subset in [("All papers", df), ("Bilateral", bilateral), ("Multi-bloc", multibloc)]:
    n = len(subset)
    comb_first = subset["gh_first"].mean() * 100
    only_first = subset["gh_only_first"].mean() * 100
    dual_first = subset["dual_first"].mean() * 100
    print(f"{label+' (first)':20s} {comb_first:11.1f}% {only_first:11.1f}% {dual_first:11.1f}%")
    
    comb_last = subset["gh_last"].mean() * 100
    only_last = subset["gh_only_last"].mean() * 100
    dual_last = subset["dual_last"].mean() * 100
    print(f"{label+' (last)':20s} {comb_last:11.1f}% {only_last:11.1f}% {dual_last:11.1f}%")
    print()

# ── Partner bloc rates for Ghanaian-only ─────────────────────────────
print("="*70)
print("PARTNER BLOC RATES: Ghanaian-only")
print("="*70)
for bloc in ["Western", "African", "East Asian", "South Asian", "Multi-bloc"]:
    sub = df[df["partner_bloc"] == bloc]
    if len(sub) < 30:
        continue
    print(f"  {bloc:15s}: N={len(sub):,}, "
          f"GH-only first={sub['gh_only_first'].mean()*100:.1f}%, "
          f"Combined first={sub['gh_first'].mean()*100:.1f}%, "
          f"GH-only last={sub['gh_only_last'].mean()*100:.1f}%, "
          f"Combined last={sub['gh_last'].mean()*100:.1f}%")

# ── Save results as JSON ────────────────────────────────────────────
results = {
    "combined_first": null_test(authorships, works, gh_cats_combined, "first"),
    "combined_last": null_test(authorships, works, gh_cats_combined, "last"),
    "ghonly_first": null_test(authorships, works, gh_cats_only, "first"),
    "ghonly_last": null_test(authorships, works, gh_cats_only, "last"),
    "rates": {
        "all_combined_first": round(df["gh_first"].mean() * 100, 1),
        "all_ghonly_first": round(df["gh_only_first"].mean() * 100, 1),
        "all_dual_first": round(df["dual_first"].mean() * 100, 1),
        "all_combined_last": round(df["gh_last"].mean() * 100, 1),
        "all_ghonly_last": round(df["gh_only_last"].mean() * 100, 1),
        "all_dual_last": round(df["dual_last"].mean() * 100, 1),
        "bilateral_combined_first": round(bilateral["gh_first"].mean() * 100, 1),
        "bilateral_ghonly_first": round(bilateral["gh_only_first"].mean() * 100, 1),
        "bilateral_combined_last": round(bilateral["gh_last"].mean() * 100, 1),
        "bilateral_ghonly_last": round(bilateral["gh_only_last"].mean() * 100, 1),
        "multibloc_combined_first": round(multibloc["gh_first"].mean() * 100, 1),
        "multibloc_ghonly_first": round(multibloc["gh_only_first"].mean() * 100, 1),
        "multibloc_combined_last": round(multibloc["gh_last"].mean() * 100, 1),
        "multibloc_ghonly_last": round(multibloc["gh_only_last"].mean() * 100, 1),
    }
}

with open("analysis_results/sensitivity_ghonly.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n\nResults saved to analysis_results/sensitivity_ghonly.json")
print("\nDone.")
