"""
Phase 11 (v2): Machine Learning Analyses
==========================================
5 ML approaches to complement GEE findings:
  1. XGBoost + SHAP (feature importance, interactions, non-linearities)
  2. K-Means Clustering (collaboration archetypes)
  3. Partial Dependence / ICE plots (non-linear dose-response)
  4. Anomaly Detection (inequity outlier papers)
  5. Change-Point Detection (temporal regime shifts)
"""

import pandas as pd
import numpy as np
import json, warnings, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, classification_report,
                              accuracy_score, f1_score)
from sklearn.cluster import KMeans
from sklearn.inspection import PartialDependenceDisplay

from utils import (
    setup_plot_style, save_chart, get_leadership, load_study_data,
    to_bool, RESULTS
)

warnings.filterwarnings("ignore")
setup_plot_style()

print("=" * 70)
print("PHASE 11 (v2): MACHINE LEARNING ANALYSES")
print("=" * 70)

# -- Load data ----------------------------------------------------------------
works, authorships = load_study_data()
leadership = get_leadership(works, authorships)
df = works.merge(leadership, on="work_id", how="left")
N = len(df)
print(f"Study set: {N:,}")

# -- Build ML feature matrix --------------------------------------------------
# Use same predictors as GEE but keep them numeric/encoded
feature_cols = [
    "log_author_count", "year_centered", "year_centered_sq",
    "has_funding_int", "is_oa_int", "country_count", "author_count",
]

# Encode partner_bloc
le_bloc = LabelEncoder()
df["bloc_encoded"] = le_bloc.fit_transform(df["partner_bloc"].fillna("Unknown"))

# Encode field
le_field = LabelEncoder()
df["field_encoded"] = le_field.fit_transform(df["field_reg"].fillna("Other"))

# Binary features from bloc
df["is_multibloc"] = (df["partner_bloc"] == "Multi-bloc").astype(int)
df["is_western"] = (df["partner_bloc"] == "Western").astype(int)
df["is_african"] = (df["partner_bloc"] == "African").astype(int)
df["is_bilateral"] = df["is_bilateral"].astype(int)

ml_features = feature_cols + [
    "is_multibloc", "is_western", "is_african", "is_bilateral",
    "bloc_encoded", "field_encoded",
]

X = df[ml_features].copy()
y_first = df["gh_first"].astype(int).values
y_last = df["gh_last"].astype(int).values

print(f"Features: {len(ml_features)}")
print(f"First auth positive rate: {y_first.mean():.3f}")
print(f"Last auth positive rate:  {y_last.mean():.3f}")

# ==============================================================================
# ANALYSIS 1: XGBoost + SHAP
# ==============================================================================
print(f"\n{'='*70}")
print("ANALYSIS 1: XGBoost + SHAP")
print(f"{'='*70}")

import xgboost as xgb
import shap

# -- Train XGBoost with cross-validation --
xgb_params = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "auc",
}

results_ml = {}

for outcome_name, y in [("first", y_first), ("last", y_last)]:
    print(f"\n  --- {outcome_name.upper()} AUTHORSHIP ---")
    
    # 5-fold CV for honest AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(
        xgb.XGBClassifier(**xgb_params),
        X, y, cv=cv, method="predict_proba"
    )[:, 1]
    cv_auc = roc_auc_score(y, cv_probs)
    cv_acc = accuracy_score(y, (cv_probs > 0.5).astype(int))
    cv_f1 = f1_score(y, (cv_probs > 0.5).astype(int))
    print(f"  5-fold CV AUC: {cv_auc:.4f}")
    print(f"  5-fold CV Accuracy: {cv_acc:.4f}")
    print(f"  5-fold CV F1: {cv_f1:.4f}")
    
    # Train final model on full data for SHAP
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X, y, verbose=False)
    
    # Feature importance (native)
    imp = pd.DataFrame({
        "Feature": ml_features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    print(f"\n  XGBoost Feature Importance:")
    for _, r in imp.head(10).iterrows():
        print(f"    {r['Feature']:25s} {r['Importance']:.4f}")
    imp.to_csv(RESULTS / f"ml_xgb_importance_{outcome_name}.csv", index=False)
    
    # SHAP values
    print(f"  Computing SHAP values...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    elapsed = time.time() - t0
    print(f"  SHAP computed in {elapsed:.1f}s")
    
    # SHAP summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X, feature_names=ml_features,
                      show=False, max_display=13)
    plt.title(f"SHAP Summary: GH {outcome_name.title()} Authorship", fontsize=13)
    plt.tight_layout()
    save_chart(plt.gcf(), f"ml_shap_summary_{outcome_name}")
    plt.close("all")
    
    # SHAP bar plot (mean absolute)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=ml_features,
                      plot_type="bar", show=False, max_display=13)
    plt.title(f"SHAP Mean |Impact|: GH {outcome_name.title()} Authorship", fontsize=13)
    plt.tight_layout()
    save_chart(plt.gcf(), f"ml_shap_bar_{outcome_name}")
    plt.close("all")
    
    # SHAP dependence plots for top 3 features
    top3 = imp.head(3)["Feature"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, feat in enumerate(top3):
        feat_idx = ml_features.index(feat)
        shap.dependence_plot(feat_idx, shap_values, X,
                             feature_names=ml_features,
                             ax=axes[i], show=False)
        axes[i].set_title(f"SHAP: {feat}")
    plt.suptitle(f"SHAP Dependence: GH {outcome_name.title()} Authorship",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_chart(fig, f"ml_shap_dependence_{outcome_name}")
    plt.close("all")
    
    # Mean absolute SHAP for comparison with GEE
    mean_abs_shap = pd.DataFrame({
        "Feature": ml_features,
        "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0),
        "Mean_SHAP": shap_values.mean(axis=0),
    }).sort_values("Mean_abs_SHAP", ascending=False)
    mean_abs_shap.to_csv(RESULTS / f"ml_shap_values_{outcome_name}.csv", index=False)
    
    results_ml[f"xgb_{outcome_name}"] = {
        "cv_auc": round(cv_auc, 4),
        "cv_accuracy": round(cv_acc, 4),
        "cv_f1": round(cv_f1, 4),
        "top_feature": imp.iloc[0]["Feature"],
        "top_shap": mean_abs_shap.iloc[0]["Feature"],
    }

# GEE vs XGBoost AUC comparison
print(f"\n  --- GEE vs XGBoost AUC Comparison ---")
diag = json.load(open(RESULTS / "v2_model_diagnostics.json"))
print(f"  First auth - GEE AUC: {diag['gh_first_A']['auc']:.4f}, "
      f"XGBoost CV AUC: {results_ml['xgb_first']['cv_auc']:.4f}")
print(f"  Last auth  - GEE AUC: {diag['gh_last_A']['auc']:.4f}, "
      f"XGBoost CV AUC: {results_ml['xgb_last']['cv_auc']:.4f}")

# ==============================================================================
# ANALYSIS 2: K-MEANS CLUSTERING (Collaboration Archetypes)
# ==============================================================================
print(f"\n{'='*70}")
print("ANALYSIS 2: K-MEANS CLUSTERING — Collaboration Archetypes")
print(f"{'='*70}")

cluster_features = ["log_author_count", "country_count", "year_centered",
                     "has_funding_int", "is_oa_int", "is_multibloc",
                     "is_western", "is_african"]

X_clust = df[cluster_features].copy()
scaler = StandardScaler()
X_clust_scaled = scaler.fit_transform(X_clust)

# Elbow method
inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clust_scaled)
    inertias.append(km.inertia_)

# Fit with k=4 (reasonable for interpretability)
k_best = 4
km_final = KMeans(n_clusters=k_best, random_state=42, n_init=20)
df["cluster"] = km_final.fit_predict(X_clust_scaled)

# Profile each cluster
print(f"\n  {k_best} Cluster Profiles:")
cluster_profiles = []
for c in range(k_best):
    sub = df[df["cluster"] == c]
    profile = {
        "Cluster": c + 1,
        "N": len(sub),
        "Pct": round(len(sub) / N * 100, 1),
        "Median_team": int(sub["author_count"].median()),
        "Mean_countries": round(sub["country_count"].mean(), 1),
        "Pct_multibloc": round(sub["is_multibloc"].mean() * 100, 1),
        "Pct_western": round(sub["is_western"].mean() * 100, 1),
        "Pct_funded": round(sub["has_funding_int"].mean() * 100, 1),
        "Pct_OA": round(sub["is_oa_int"].mean() * 100, 1),
        "GH_first_pct": round(sub["gh_first"].mean() * 100, 1),
        "GH_last_pct": round(sub["gh_last"].mean() * 100, 1),
        "Mean_year": round(sub["publication_year"].mean(), 0),
    }
    cluster_profiles.append(profile)
    
    print(f"\n  Cluster {c+1} (n={len(sub):,}, {len(sub)/N*100:.0f}%):")
    print(f"    Team: median {profile['Median_team']}, countries: {profile['Mean_countries']}")
    print(f"    Multi-bloc: {profile['Pct_multibloc']}%, Western: {profile['Pct_western']}%")
    print(f"    Funded: {profile['Pct_funded']}%, OA: {profile['Pct_OA']}%")
    print(f"    -> GH first: {profile['GH_first_pct']}%, GH last: {profile['GH_last_pct']}%")

cluster_df = pd.DataFrame(cluster_profiles)
cluster_df.to_csv(RESULTS / "ml_cluster_profiles.csv", index=False)

# Name clusters based on dominant characteristics
cluster_names = []
for _, row in cluster_df.iterrows():
    if row["Pct_multibloc"] > 60:
        name = "Multi-bloc Consortium"
    elif row["Pct_western"] > 50:
        name = "Western Bilateral"
    elif row["Median_team"] <= 4:
        name = "Small Team"
    else:
        name = "Non-Western Bilateral"
    
    if row["Pct_funded"] > 55:
        name += " (Funded)"
    cluster_names.append(name)

cluster_df["Name"] = cluster_names

# Chart: cluster comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Cluster equity comparison
x_pos = np.arange(k_best)
width = 0.35
axes[0].bar(x_pos - width/2, cluster_df["GH_first_pct"],
            width, label="First Author", color="#2563eb", alpha=0.8)
axes[0].bar(x_pos + width/2, cluster_df["GH_last_pct"],
            width, label="Last Author", color="#dc2626", alpha=0.8)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(cluster_names, fontsize=8, rotation=15, ha="right")
axes[0].set_ylabel("GH/Dual Leadership (%)")
axes[0].set_title("Authorship Equity by Collaboration Archetype")
axes[0].legend()
for i, row in cluster_df.iterrows():
    axes[0].text(i - width/2, row["GH_first_pct"] + 1,
                 f"{row['GH_first_pct']:.0f}%", ha="center", fontsize=8)
    axes[0].text(i + width/2, row["GH_last_pct"] + 1,
                 f"{row['GH_last_pct']:.0f}%", ha="center", fontsize=8)

# Elbow plot
axes[1].plot(list(K_range), inertias, "bo-")
axes[1].axvline(k_best, color="red", ls="--", label=f"k={k_best}")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Inertia")
axes[1].set_title("Elbow Plot")
axes[1].legend()

plt.tight_layout()
save_chart(fig, "ml_cluster_archetypes")

# Cluster size chart (radar/spider or stacked bar)
fig, ax = plt.subplots(figsize=(10, 6))
props = ["Pct_multibloc", "Pct_western", "Pct_funded", "Pct_OA"]
prop_labels = ["Multi-bloc", "Western", "Funded", "Open Access"]
x = np.arange(len(prop_labels))
bar_width = 0.2
for i, (_, row) in enumerate(cluster_df.iterrows()):
    vals = [row[p] for p in props]
    ax.bar(x + i * bar_width, vals, bar_width, label=cluster_names[i], alpha=0.8)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(prop_labels)
ax.set_ylabel("Percentage (%)")
ax.set_title("Cluster Characteristic Profiles")
ax.legend(fontsize=8)
plt.tight_layout()
save_chart(fig, "ml_cluster_profiles")

# ==============================================================================
# ANALYSIS 3: PARTIAL DEPENDENCE PLOTS
# ==============================================================================
print(f"\n{'='*70}")
print("ANALYSIS 3: PARTIAL DEPENDENCE / ICE PLOTS")
print(f"{'='*70}")

# Use the first authorship XGBoost model
model_first = xgb.XGBClassifier(**xgb_params)
model_first.fit(X, y_first, verbose=False)

# Features to plot
pdp_features = ["log_author_count", "year_centered", "country_count",
                 "author_count", "is_multibloc"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat = axes.flatten()

for i, feat in enumerate(pdp_features):
    if i >= 6:
        break
    feat_idx = ml_features.index(feat)
    
    # Manual PDP computation for more control
    feat_values = np.linspace(X[feat].quantile(0.01),
                               X[feat].quantile(0.99), 50)
    pdp_vals = []
    for val in feat_values:
        X_mod = X.copy()
        X_mod[feat] = val
        pred = model_first.predict_proba(X_mod)[:, 1].mean()
        pdp_vals.append(pred)
    
    ax = axes_flat[i]
    ax.plot(feat_values, pdp_vals, color="#2563eb", lw=2)
    ax.set_xlabel(feat)
    ax.set_ylabel("P(GH First Author)")
    ax.set_title(f"PDP: {feat}")
    ax.grid(alpha=0.3)
    
    # Add rug plot
    sample = X[feat].sample(min(500, len(X)), random_state=42)
    ax.plot(sample, [min(pdp_vals)] * len(sample), "|", color="gray",
            alpha=0.3, markersize=10)

# Hide unused subplot
if len(pdp_features) < 6:
    axes_flat[5].axis("off")

plt.suptitle("Partial Dependence Plots: GH First Authorship", fontsize=14, y=1.01)
plt.tight_layout()
save_chart(fig, "ml_pdp_first")

# ICE plot for team size (most important feature)
print("  Computing ICE for log_author_count...")
fig, ax = plt.subplots(figsize=(10, 6))
feat = "log_author_count"
feat_values = np.linspace(X[feat].quantile(0.01), X[feat].quantile(0.99), 50)

# Sample 200 papers for ICE
np.random.seed(42)
ice_sample_idx = np.random.choice(len(X), size=200, replace=False)
X_ice_sample = X.iloc[ice_sample_idx]

ice_curves = []
for val in feat_values:
    X_mod = X_ice_sample.copy()
    X_mod[feat] = val
    preds = model_first.predict_proba(X_mod)[:, 1]
    ice_curves.append(preds)

ice_array = np.array(ice_curves)  # shape: (50, 200)

# Plot individual ICE curves
for j in range(ice_array.shape[1]):
    ax.plot(feat_values, ice_array[:, j], color="#2563eb", alpha=0.05, lw=0.5)

# PDP (mean)
ax.plot(feat_values, ice_array.mean(axis=1), color="#dc2626", lw=3,
        label="PDP (mean)")
ax.set_xlabel("log(Author Count)")
ax.set_ylabel("P(GH First Author)")
ax.set_title("ICE Plot: Effect of Team Size on GH First Authorship")
ax.legend()

# Add team size scale on top
ax_top = ax.twiny()
team_sizes = np.exp(feat_values)
ax_top.set_xlim(team_sizes[0], team_sizes[-1])
ax_top.set_xlabel("Team Size (actual)")
plt.tight_layout()
save_chart(fig, "ml_ice_teamsize")

print("  PDPs and ICE plots saved.")

# ==============================================================================
# ANALYSIS 4: ANOMALY DETECTION — Inequity Outlier Papers
# ==============================================================================
print(f"\n{'='*70}")
print("ANALYSIS 4: ANOMALY DETECTION (Inequity Outliers)")
print(f"{'='*70}")

# Predict P(GH first author) for each paper
df["pred_gh_first"] = model_first.predict_proba(X)[:, 1]

# Inequity outliers: papers where model predicts HIGH probability of GH first
# author (>0.7) but actual outcome is non-GH
high_pred_nongh = df[(df["pred_gh_first"] > 0.7) & (df["gh_first"] == False)]
low_pred_gh = df[(df["pred_gh_first"] < 0.3) & (df["gh_first"] == True)]

print(f"  Papers with P(GH first) > 0.7 but actual non-GH: {len(high_pred_nongh):,}")
print(f"  Papers with P(GH first) < 0.3 but actual GH: {len(low_pred_gh):,}")

# Profile inequity outliers vs normal papers
normal = df[(df["pred_gh_first"] > 0.5) & (df["gh_first"] == True)]

anomaly_profile = pd.DataFrame({
    "Metric": ["N", "Median team size", "Mean countries", "% Multi-bloc",
               "% Western", "% Funded", "% OA", "Mean year"],
    "Inequity Outliers": [
        len(high_pred_nongh),
        int(high_pred_nongh["author_count"].median()),
        round(high_pred_nongh["country_count"].mean(), 1),
        round(high_pred_nongh["is_multibloc"].mean() * 100, 1),
        round(high_pred_nongh["is_western"].mean() * 100, 1),
        round(high_pred_nongh["has_funding_int"].mean() * 100, 1),
        round(high_pred_nongh["is_oa_int"].mean() * 100, 1),
        round(high_pred_nongh["publication_year"].mean(), 0),
    ],
    "Expected GH (normal)": [
        len(normal),
        int(normal["author_count"].median()),
        round(normal["country_count"].mean(), 1),
        round(normal["is_multibloc"].mean() * 100, 1),
        round(normal["is_western"].mean() * 100, 1),
        round(normal["has_funding_int"].mean() * 100, 1),
        round(normal["is_oa_int"].mean() * 100, 1),
        round(normal["publication_year"].mean(), 0),
    ],
})
print(f"\n{anomaly_profile.to_string(index=False)}")
anomaly_profile.to_csv(RESULTS / "ml_anomaly_profile.csv", index=False)

results_ml["anomaly"] = {
    "n_inequity_outliers": len(high_pred_nongh),
    "n_unexpected_gh": len(low_pred_gh),
    "outlier_pct": round(len(high_pred_nongh) / N * 100, 1),
}

# Chart: prediction distribution
fig, ax = plt.subplots(figsize=(10, 6))
gh_preds = df[df["gh_first"] == True]["pred_gh_first"]
nongh_preds = df[df["gh_first"] == False]["pred_gh_first"]

ax.hist(nongh_preds, bins=50, alpha=0.6, color="#dc2626",
        label="Actual: Non-GH First", density=True)
ax.hist(gh_preds, bins=50, alpha=0.6, color="#059669",
        label="Actual: GH First", density=True)
ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Decision boundary")
ax.axvline(0.7, color="#f59e0b", ls=":", lw=1.5, label="Inequity threshold (0.7)")
ax.set_xlabel("Predicted P(GH First Author)")
ax.set_ylabel("Density")
ax.set_title("XGBoost Prediction Distribution by Actual Outcome")
ax.legend(fontsize=9)
plt.tight_layout()
save_chart(fig, "ml_prediction_distribution")

# Scatter: predicted vs actual (aggregated by year)
fig, ax = plt.subplots(figsize=(10, 5))
annual = df.groupby("publication_year").agg(
    pred_mean=("pred_gh_first", "mean"),
    actual_mean=("gh_first", "mean"),
).reset_index()
ax.plot(annual["publication_year"], annual["actual_mean"] * 100, "o-",
        color="#059669", label="Actual GH First %", lw=2)
ax.plot(annual["publication_year"], annual["pred_mean"] * 100, "s--",
        color="#2563eb", label="XGBoost Predicted %", lw=2)
ax.set_xlabel("Year")
ax.set_ylabel("GH First Authorship Rate (%)")
ax.set_title("Model Calibration Over Time")
ax.legend()
plt.tight_layout()
save_chart(fig, "ml_calibration_temporal")

# ==============================================================================
# ANALYSIS 5: CHANGE-POINT DETECTION
# ==============================================================================
print(f"\n{'='*70}")
print("ANALYSIS 5: CHANGE-POINT DETECTION")
print(f"{'='*70}")

import ruptures as rpt

# Annual GH first authorship rates
annual_rates = df.groupby("publication_year").agg(
    first_rate=("gh_first", "mean"),
    last_rate=("gh_last", "mean"),
    n=("work_id", "size"),
).reset_index()

# Only use years with sufficient data
annual_rates = annual_rates[annual_rates["n"] >= 30].sort_values("publication_year")

cp_results = {}
for col, label in [("first_rate", "First"), ("last_rate", "Last")]:
    signal = annual_rates[col].values
    years = annual_rates["publication_year"].values
    
    # PELT (Pruned Exact Linear Time) algorithm
    algo = rpt.Pelt(model="rbf", min_size=3).fit(signal)
    try:
        cps = algo.predict(pen=0.1)
        # Remove the last point (always returned by ruptures)
        cp_indices = [c for c in cps if c < len(signal)]
        cp_years = [years[i] for i in cp_indices if i < len(years)]
    except Exception:
        cp_years = []
    
    # Also try Binseg for comparison
    algo2 = rpt.Binseg(model="l2", min_size=3).fit(signal)
    try:
        cps2 = algo2.predict(n_bkps=2)
        cp_indices2 = [c for c in cps2 if c < len(signal)]
        cp_years2 = [years[i] for i in cp_indices2 if i < len(years)]
    except Exception:
        cp_years2 = []
    
    cp_results[col] = {
        "pelt_changepoints": [int(y) for y in cp_years],
        "binseg_changepoints": [int(y) for y in cp_years2],
    }
    
    print(f"  {label} authorship:")
    print(f"    PELT changepoints: {cp_years}")
    print(f"    BinSeg changepoints: {cp_years2}")

# Chart: change-point visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for i, (col, label) in enumerate([("first_rate", "First Authorship"),
                                    ("last_rate", "Last Authorship")]):
    ax = axes[i]
    ax.plot(annual_rates["publication_year"], annual_rates[col] * 100,
            "o-", color="#2563eb", lw=2, markersize=5)
    
    # Mark changepoints
    for cp_year in cp_results[col]["pelt_changepoints"]:
        ax.axvline(cp_year, color="#dc2626", ls="--", lw=2, alpha=0.7,
                   label=f"PELT: {int(cp_year)}")
    for cp_year in cp_results[col]["binseg_changepoints"]:
        ax.axvline(cp_year, color="#f59e0b", ls=":", lw=2, alpha=0.7,
                   label=f"BinSeg: {int(cp_year)}")
    
    ax.set_ylabel(f"GH {label} Rate (%)")
    ax.set_title(f"Change-Point Detection: {label}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

axes[1].set_xlabel("Publication Year")
plt.tight_layout()
save_chart(fig, "ml_changepoints")

# Also bilateral vs multi-bloc change points
annual_bil = df[df["is_bilateral"] == 1].groupby("publication_year")["gh_first"].mean()
annual_mul = df[df["is_multibloc"] == 1].groupby("publication_year")["gh_first"].mean()

# ==============================================================================
# SAVE ALL RESULTS
# ==============================================================================
results_ml["clustering"] = {
    "k": k_best,
    "profiles": cluster_profiles,
}
results_ml["changepoints"] = cp_results

with open(RESULTS / "ml_results.json", "w") as f:
    json.dump(results_ml, f, indent=2, default=str)

print(f"\n{'='*70}")
print("PHASE 11 (v2) COMPLETE -- ALL 5 ML ANALYSES")
print(f"{'='*70}")
print(f"Results: ml_results.json")
print(f"Charts: ~12 new charts")
print(f"CSVs: xgb_importance, shap_values, cluster_profiles, anomaly_profile")
