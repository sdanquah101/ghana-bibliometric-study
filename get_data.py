import pandas as pd, json
R = "analysis_results/"

t1 = pd.read_csv(R+"table1_leadership_proportions.csv")
print("=== TABLE 1 ===")
print(t1.to_string())

reg = pd.read_csv(R+"v2_regression_results.csv")
first_A = reg[reg["Model"]=="gh_first_A"][["Variable","OR","CI_lo","CI_hi","p_value"]]
last_A = reg[reg["Model"]=="gh_last_A"][["Variable","OR","CI_lo","CI_hi","p_value"]]
print("\n=== FIRST A ===")
print(first_A.to_string())
print("\n=== LAST A ===")
print(last_A.to_string())

d = json.load(open(R+"v2_model_diagnostics.json"))
print(f"\nFirst AUC: {d['gh_first_A']['auc']}, N: {d['gh_first_A']['n_obs']}")
print(f"Last AUC: {d['gh_last_A']['auc']}, N: {d['gh_last_A']['n_obs']}")

pt = pd.read_csv(R+"partner_type_equity.csv")
print("\n=== PARTNER TYPE ===")
print(pt.to_string())

bc = pd.read_csv(R+"bilateral_country_equity.csv")
print("\n=== BILATERAL COUNTRY ===")
print(bc.to_string())

ml = json.load(open(R+"ml_results.json"))
print(f"\nXGB First AUC: {ml['xgb_first']['cv_auc']}")
print(f"XGB Last AUC: {ml['xgb_last']['cv_auc']}")

ptr = pd.read_csv(R+"partner_type_regression.csv")
print("\n=== PARTNER TYPE REGRESSION ===")
print(ptr.to_string())

btr = pd.read_csv(R+"bilateral_type_regression.csv")
print("\n=== BILATERAL TYPE REGRESSION ===")
print(btr.to_string())

sens = pd.read_csv(R+"sensitivity_summary.csv")
print("\n=== SENSITIVITY ===")
print(sens[["Analysis","N","multibloc_OR","multibloc_p"]].to_string())

# Advanced results
adv = json.load(open(R+"advanced_results.json"))
print("\n=== FAIRLIE ===")
for v in adv.get("fairlie_decomposition",{}).get("variable_contributions",[]):
    print(f"  {v['variable']}: {v['contribution']:.4f} ({v['pct_of_explained']:.1f}%)")
print(f"\n=== MEDIATION ===")
med = adv.get("mediation",{})
print(f"  Path a: {med.get('path_a')}")
print(f"  Path b: {med.get('path_b')}")
print(f"  Indirect: {med.get('indirect_effect')}")
print(f"  Pct mediated: {med.get('pct_mediated')}")
print(f"  Sobel z: {med.get('sobel_z')}, p: {med.get('sobel_p')}")

em = json.load(open(R+"emerging_powers_results.json"))
print("\n=== EMERGING DETAIL ===")
for e in em.get("emerging_detail",[]):
    print(f"  {e['Country']}: total={e['N_total']}, bil={e['N_bilateral']}, first={e['GH_first_all']}%, last={e['GH_last_all']}%")
