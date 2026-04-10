import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.api import Logit, add_constant
from pathlib import Path

from config import INTERMEDIATE_DIR, OUTPUT_DIR

# Load data
works = pd.read_parquet(INTERMEDIATE_DIR / "works.parquet")
authorships = pd.read_parquet(INTERMEDIATE_DIR / "authorships.parquet")

def to_bool(v):
    if pd.isna(v): return False
    if isinstance(v, str): return v.lower() == 'true'
    return bool(v)

# Prepare regression dataset at work level
first_auths = authorships[authorships["author_position"] == "first"][["work_id", "affiliation_category"]].copy()
first_auths["gh_first"] = first_auths["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).astype(int)
first_auths = first_auths.drop_duplicates("work_id")

last_auths = authorships[authorships["author_position"] == "last"][["work_id", "affiliation_category"]].copy()
last_auths["gh_last"] = last_auths["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).astype(int)
last_auths = last_auths.drop_duplicates("work_id")

corr_auths = authorships[authorships["is_corresponding_combined"] == True][["work_id", "affiliation_category"]].copy()
corr_work = corr_auths.groupby("work_id").apply(
    lambda x: int(x["affiliation_category"].isin(["Ghanaian", "Dual-affiliated"]).any())
).reset_index(name="gh_corr")

reg_df = works[["work_id", "publication_year", "covid_era", "field_name",
                "partner_bloc", "country_count", "author_count",
                "has_funding", "is_oa"]].copy()
reg_df["has_funding_bool"] = reg_df["has_funding"].apply(to_bool).astype(int)
reg_df["is_oa_bool"] = reg_df["is_oa"].apply(to_bool).astype(int)
reg_df["year_centered"] = reg_df["publication_year"] - 2000

reg_df = reg_df.merge(first_auths[["work_id", "gh_first"]], on="work_id", how="left")
reg_df = reg_df.merge(last_auths[["work_id", "gh_last"]], on="work_id", how="left")
reg_df = reg_df.merge(corr_work, on="work_id", how="left")

# Collapse small blocs
bloc_counts = reg_df["partner_bloc"].value_counts()
small_blocs = bloc_counts[bloc_counts < 30].index.tolist()
reg_df["partner_bloc_reg"] = reg_df["partner_bloc"].apply(lambda x: "Other" if x in small_blocs else x)

# ----------------------------------------------------
# Collapse health sciences
# ----------------------------------------------------
reg_df["field_reg"] = reg_df["field_name"].fillna("Other")
health_fields = ["Medicine", "Nursing", "Health Professions"]
reg_df["field_reg"] = reg_df["field_reg"].apply(lambda x: "Health Sciences" if x in health_fields else x)

top_fields = reg_df["field_reg"].value_counts().head(5).index.tolist()
# Check if Biochemistry, Genetics and Molecular Biology is still in top 5, otherwise add it to act as reference 
# (assuming it was the reference before)
ref_field = "Biochemistry, Genetics and Molecular Biology"
if ref_field not in top_fields:
    top_fields.append(ref_field)

reg_df["field_reg"] = reg_df["field_reg"].apply(lambda x: x if x in top_fields else "Other")

# Create output text
out_text = []
out_text.append("LOGISTIC REGRESSION RESULTS (COLLAPSED HEALTH SCIENCES FIELD)\n")
out_text.append("="*80 + "\n")

def run_logistic_model(dep_var, model_name):
    model_df = reg_df.dropna(subset=[dep_var]).copy()
    numeric_cols = ["year_centered", "covid_era", "country_count", "author_count", "has_funding_bool", "is_oa_bool"]

    bloc_dummies = pd.get_dummies(model_df["partner_bloc_reg"], prefix="bloc", drop_first=False)
    # drop African as reference
    if "bloc_African" in bloc_dummies.columns:
        bloc_dummies = bloc_dummies.drop(columns=["bloc_African"])
    else:
        bloc_dummies = bloc_dummies.iloc[:, 1:]

    field_dummies = pd.get_dummies(model_df["field_reg"], prefix="field", drop_first=False)
    # drop reference field
    if f"field_{ref_field}" in field_dummies.columns:
        field_dummies = field_dummies.drop(columns=[f"field_{ref_field}"])
    else:
        field_dummies = field_dummies.iloc[:, 1:] # drop first alphabetically if ref not present somehow

    X = pd.concat([model_df[numeric_cols].reset_index(drop=True),
                    bloc_dummies.reset_index(drop=True),
                    field_dummies.reset_index(drop=True)], axis=1)
    X = add_constant(X)
    y = model_df[dep_var].reset_index(drop=True)

    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    model = Logit(y, X.astype(float)).fit(disp=0, maxiter=100)
    
    out_text.append(f"--- {model_name} ---")
    out_text.append(f"AIC: {model.aic:.1f}, Pseudo R-squared: {model.prsquared:.4f}\n")
    
    results = []
    for var in model.params.index:
        if var == "const": continue
        or_val = np.exp(model.params[var])
        ci = np.exp(model.conf_int().loc[var])
        p_val = model.pvalues[var]
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
        res_str = f"{var:30s} OR={or_val:.3f} (95% CI: {ci[0]:.3f} - {ci[1]:.3f}), p={p_str}"
        out_text.append(res_str)
    
    out_text.append("\n" + "-"*40 + "\n")


run_logistic_model("gh_first", "Model 1: First Authorship")
run_logistic_model("gh_last", "Model 2: Last Authorship")
run_logistic_model("gh_corr", "Model 3: Corresponding Authorship")

with open(OUTPUT_DIR / "collapsed_regression_results.txt", "w") as f:
    f.write("\n".join(out_text))
print("Done! Saved to analysis_results/collapsed_regression_results.txt")
