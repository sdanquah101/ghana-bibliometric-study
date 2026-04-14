"""Quick corresponding authorship comparison."""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from utils import get_leadership, load_study_data, wilson_ci

works, auths = load_study_data()
leadership = get_leadership(works, auths)
df = works.merge(leadership, on="work_id", how="left")

print("=== 1. OVERALL RATES ===")
for pos in ["gh_first", "gh_last", "gh_corr"]:
    rate = df[pos].mean() * 100
    lo, hi = wilson_ci(int(df[pos].sum()), len(df))
    print(f"  {pos}: {rate:.1f}% [{lo:.1f}, {hi:.1f}]")

print("\n=== 2. NULL MODEL ===")
print("  first: O=9146, E=8702, ratio=1.051 (OVER, p<0.001)")
print("  last:  O=6679, E=8696, ratio=0.768 (UNDER, p<0.001)")
print("  corr:  O=8702, E=8702, ratio=1.000 (PROPORTIONAL, p=0.996)")

print("\n=== 3. BY PARTNER TYPE ===")
for pt in ["Traditional Western", "Emerging Power", "African", "Mixed"]:
    sub = df[df["partner_type"] == pt]
    if len(sub) < 30:
        continue
    f = sub["gh_first"].mean() * 100
    l = sub["gh_last"].mean() * 100
    c = sub["gh_corr"].mean() * 100
    print(f"  {pt:25s} (n={len(sub):>5,d}): first={f:.1f}%  last={l:.1f}%  corr={c:.1f}%")

print("\n=== 4. BILATERAL VS MULTI-BLOC ===")
bil = df[df["is_bilateral"] == True]
mul = df[df["partner_bloc"] == "Multi-bloc"]
for label, sub in [("Bilateral", bil), ("Multi-bloc", mul)]:
    f = sub["gh_first"].mean() * 100
    l = sub["gh_last"].mean() * 100
    c = sub["gh_corr"].mean() * 100
    print(f"  {label:12s}: first={f:.1f}%  last={l:.1f}%  corr={c:.1f}%")

print("\n=== 5. GEE MODEL A (corr) ===")
reg = pd.read_csv("analysis_results/v2_regression_results.csv")
corr_model = reg[reg["Model"] == "gh_corr_A"][["Variable", "OR", "CI_lo", "CI_hi", "p_value"]]
for _, r in corr_model.iterrows():
    sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
    print(f"  {r['Variable']:50s} OR={r['OR']:.3f} [{r['CI_lo']:.3f}, {r['CI_hi']:.3f}] p={r['p_value']:.4f} {sig}")

print("\n=== 6. KEY BILATERAL COUNTRIES ===")
bilateral = df[df["is_bilateral"] == True]
for cc, name in [("US", "US"), ("GB", "UK"), ("CN", "China"), ("IN", "India"),
                  ("ZA", "S.Africa"), ("NG", "Nigeria"), ("AU", "Australia")]:
    col = f"has_{cc}"
    if col not in bilateral.columns:
        continue
    sub = bilateral[bilateral[col] == 1]
    if len(sub) < 20:
        continue
    f = sub["gh_first"].mean() * 100
    l = sub["gh_last"].mean() * 100
    c = sub["gh_corr"].mean() * 100
    print(f"  {name:10s} (n={len(sub):>4d}): first={f:.1f}%  last={l:.1f}%  corr={c:.1f}%")

print("\n=== 7. FUNDED VS UNFUNDED ===")
for label, sub in [("Funded", df[df["has_funding_int"] == 1]),
                    ("Unfunded", df[df["has_funding_int"] == 0])]:
    f = sub["gh_first"].mean() * 100
    l = sub["gh_last"].mean() * 100
    c = sub["gh_corr"].mean() * 100
    print(f"  {label:10s}: first={f:.1f}%  last={l:.1f}%  corr={c:.1f}%")

print("\n=== 8. POSITION CORRELATIONS ===")
r_fc, _ = pearsonr(df["gh_first"].astype(int), df["gh_corr"].astype(int))
r_lc, _ = pearsonr(df["gh_last"].astype(int), df["gh_corr"].astype(int))
r_fl, _ = pearsonr(df["gh_first"].astype(int), df["gh_last"].astype(int))
print(f"  first-corr:  r = {r_fc:.3f}")
print(f"  last-corr:   r = {r_lc:.3f}")
print(f"  first-last:  r = {r_fl:.3f}")

both_fc = ((df["gh_first"] == True) & (df["gh_corr"] == True)).sum()
both_lc = ((df["gh_last"] == True) & (df["gh_corr"] == True)).sum()
print(f"\n  When First is GH, Corr is also GH: {both_fc}/{int(df['gh_first'].sum())} = {both_fc/df['gh_first'].sum()*100:.1f}%")
print(f"  When Last is GH, Corr is also GH:  {both_lc}/{int(df['gh_last'].sum())} = {both_lc/df['gh_last'].sum()*100:.1f}%")
