import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, roc_curve, recall_score, precision_score, confusion_matrix
from sklearn.calibration import calibration_curve
import warnings
import random
import shap
warnings.filterwarnings("ignore", category=FutureWarning)

# Random setup
SEED = 7 
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
sklearn_random_state = SEED

FEATURE_CSV   = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\radiomics.csv'
TOPO_CSV      = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\finaltopo.csv'
FRACTAL_CSV   = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\fractal.csv'
AGE_CSV       = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\AgeSex.xlsx'
OUTCOME_CSV   = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\outcomes.csv'
features_include = r'Topology-Fractal-Radiomics-Age-Sex-20250613_074011/features_final.csv'

# Assume best_params.json is in the current working directory
STUDY_DIR = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Topology-Fractal-Radiomics-Age-Sex-20250613_074011"
#PARAMS_PATH = os.path.join(STUDY_DIR, 'best_params.json')

# Standardize matplotlib font sizes
plt.rcParams.update({
    'font.size': 12,            # Base font size
    'axes.titlesize': 14,       # Title size of axes
    'axes.labelsize': 12,       # X and Y label size
    'xtick.labelsize': 10,      # X tick label size
    'ytick.labelsize': 10,      # Y tick label size
    'legend.fontsize': 11,      # Legend font size
    'figure.titlesize': 14      # Figure title size
})


#Function to drop correlated features
def drop_correlated_features(df, threshold):
    if threshold is None: #if none is given as threshold, return full DF
        return df
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def bootstrap_auc(y_true, y_prob, n_bootstraps=1000, ci=0.95):
    rng = np.random.RandomState(SEED)
    bootstraps = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        bootstraps.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lower = np.percentile(bootstraps, (1-ci)/2*100)
    upper = np.percentile(bootstraps, (ci + (1-ci)/2)*100)
    return lower, upper

results_df = pd.read_csv(r"C:\Users\tjz5\Downloads\predictions.csv")

y_true = results_df['y_true'].values
y_prob = results_df['y_prob'].values
y_pred = results_df['y_pred'].values

# Metrics and outputs
auc = roc_auc_score(y_true, y_prob)
print(f"AUC: {auc:.3f}")
lo, hi = bootstrap_auc(y_true, y_prob)

fpr, tpr, _ = roc_curve(y_true, y_prob)

n_bootstraps = 1000
ci = 0.95
rng = np.random.RandomState(SEED)
base_fpr = np.linspace(0, 1, 101)
tprs_boot = []

for _ in range(n_bootstraps):
    idx = rng.randint(0, len(y_true), len(y_true))
    if len(np.unique(y_true[idx])) < 2:
        continue
    fpr_bs, tpr_bs, _ = roc_curve(y_true[idx], y_prob[idx])
    # interpolate this bootstrap's TPR onto base_fpr
    tprs_boot.append(np.interp(base_fpr, fpr_bs, tpr_bs))

tprs_boot = np.vstack(tprs_boot)
lower_perc = ((1 - ci) / 2) * 100
upper_perc = (ci + (1 - ci) / 2) * 100
tpr_lower = np.percentile(tprs_boot, lower_perc, axis=0)
tpr_upper = np.percentile(tprs_boot, upper_perc, axis=0)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {auc:.3f}")
# shade between lower & upper CI
plt.fill_between(
    base_fpr, tpr_lower, tpr_upper,
    color='lightblue', alpha=0.3,
    label=f"{int(ci*100)}% CI ({lo:.3f}–{hi:.3f})"
)

plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.title("ROC curve with Shaded 95% CI")
plt.tight_layout()
plt.savefig(os.path.join(STUDY_DIR, "TEST-SET-roc_curve_with_ci.png"), dpi=1200)
plt.close()

# Calibration curve
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
plt.figure()
plt.plot(mean_pred, frac_pos, marker='o', color='blue',lw=2)
plt.plot([0, 1], [0, 1], '--',color='gray')
plt.xlabel('Mean Predicted'); plt.ylabel('Fraction Positives')
plt.tight_layout()
plt.savefig(os.path.join(STUDY_DIR, 'TEST-SET-calibration_curve.png'), dpi=1200)
plt.close()

def bootstrap_auprc(y_true, y_prob, n_bootstraps=1000, ci=0.95):
    rng = np.random.RandomState(SEED)
    bootstraps = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        bootstraps.append(average_precision_score(y_true[idx], y_prob[idx]))
    lower = np.percentile(bootstraps, (1-ci)/2*100)
    upper = np.percentile(bootstraps, (ci + (1-ci)/2)*100)
    return lower, upper

auprc = average_precision_score(y_true, y_prob)
auprc_lo, auprc_hi = bootstrap_auprc(y_true, y_prob)
# Classification metrics
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
ppv = precision_score(y_true, y_pred)
npv = tn / (tn + fn)
brier = brier_score_loss(y_true, y_prob)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = 2 * (precision * recall) / (precision + recall)
accuracy = (tp + tn) / (tp + tn + fp + fn)
# write to metrics.txt
with open(os.path.join(STUDY_DIR, 'metrics.txt'), 'a') as mf:
    mf.write(f"AUPRC: {auprc:.3f}\n")
    mf.write(f"AUPRC 95% CI: {auprc_lo:.3f}-{auprc_hi:.3f}\n")
    mf.write(f"Brier Score: {brier:.3f}\n")
    mf.write(f"Accuracy: {accuracy:.3f}\n")
    mf.write(f"F1 Score: {f1:.3f}\n")
    mf.write(f"Recall: {recall:.3f}\n")
    mf.write(f"PPV: {ppv:.3f}\n")
    mf.write(f"NPV: {npv:.3f}\n")
    mf.write(f"Specificity: {spec:.3f}\n")
    mf.write(f"TP: {tp}\n")
    mf.write(f"TN: {tn}\n")
    mf.write(f"FP: {fp}\n")
    mf.write(f"FN: {fn}\n")
    
# Compute ROC
fpr, tpr, thresholds = roc_curve(y_true, y_prob)

# Metrics at Youden's threshold
youden_j    = tpr - fpr
idx_youden  = np.argmax(youden_j)
th_youden   = thresholds[idx_youden]
sens_youden = tpr[idx_youden]
spec_youden = 1 - fpr[idx_youden]
ppv_youden  = precision_score(y_true, (y_prob >= th_youden).astype(int))
y_pred_youden = (y_prob >= th_youden).astype(int)
tn_y, fp_y, fn_y, tp_y = confusion_matrix(y_true, y_pred_youden).ravel()

# Compute metrics at Youden's threshold
accuracy_youden = (tp_y + tn_y) / (tp_y + tn_y + fp_y + fn_y)
npv_youden = tn_y / (tn_y + fn_y) if (tn_y + fn_y) > 0 else 0
f1_youden = 2 * tp_y / (2 * tp_y + fp_y + fn_y) if (2 * tp_y + fp_y + fn_y) > 0 else 0
print(f"[Youden's Threshold = {th_youden:.3f}]")
print(f"Accuracy: {accuracy_youden:.3f}")
print(f"NPV: {npv_youden:.3f}")
print(f"F1 Score: {f1_youden:.3f}")

# Metrics at 85% sensitivity threshold
target_sens = 0.85
idx85 = np.where(tpr >= target_sens)[0]
idx85 = idx85[0] if len(idx85) else np.argmax(tpr)
th_85    = thresholds[idx85]
sens_85  = tpr[idx85]
spec_85  = 1 - fpr[idx85]
ppv_85   = precision_score(y_true, (y_prob >= th_85).astype(int))

with open(os.path.join(STUDY_DIR, 'TEST-metrics.txt'), 'a') as mf:
    mf.write("\nYouden's threshold: {:.3f}\n".format(th_youden))
    mf.write("Sensitivity@Youden: {:.3f}, Specificity@Youden: {:.3f}, PPV@Youden: {:.3f}\n"
             .format(sens_youden, spec_youden, ppv_youden))
    mf.write(f"Accuracy@Youden: {accuracy_youden:.3f}\n")
    mf.write(f"NPV@Youden: {npv_youden:.3f}\n")
    mf.write(f"F1 Score@Youden: {f1_youden:.3f}\n")
    mf.write("\n85% Sensitivity threshold: {:.3f}\n".format(th_85))
    mf.write("Sensitivity@85%: {:.3f}, Specificity@85%: {:.3f}, PPV@85%: {:.3f}\n"
             .format(sens_85, spec_85, ppv_85))
    
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

#ROC Curve
axes[0].plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {auc:.3f}")
axes[0].fill_between(base_fpr, tpr_lower, tpr_upper, color='lightblue', alpha=0.3,
                     label=f"{int(ci*100)}% CI ({lo:.3f}–{hi:.3f})")
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")

#Calibration Curve
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
axes[1].plot(mean_pred, frac_pos, marker='o', color='blue', lw=2, label='Calibration')
axes[1].plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
axes[1].set_xlabel("Mean Predicted Probability")
axes[1].set_ylabel("Fraction of Positives")
axes[1].legend(loc="lower right")

# Layout and save
plt.tight_layout()
plt.savefig(os.path.join(STUDY_DIR, "TEST-SET-panel.png"), dpi=1200)
plt.close()

cv_results_df = pd.read_csv(r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Topology-Fractal-Radiomics-Age-Sex-20250613_074011\cv_predictions.csv")
y_true_cv = cv_results_df['y_true'].values
y_prob_cv = cv_results_df['y_prob'].values
y_pred_cv = cv_results_df['y_pred'].values
# --- Bootstrap AUC for CV ---
auc_cv = roc_auc_score(y_true_cv, y_prob_cv)
lo_cv, hi_cv = bootstrap_auc(y_true_cv, y_prob_cv)

# --- Bootstrap ROC CI for CV ---
base_fpr_cv = np.linspace(0, 1, 101)
tprs_cv_boot = []
for _ in range(n_bootstraps):
    idx = rng.randint(0, len(y_true_cv), len(y_true_cv))
    if len(np.unique(y_true_cv[idx])) < 2:
        continue
    fpr_bs, tpr_bs, _ = roc_curve(y_true_cv[idx], y_prob_cv[idx])
    tprs_cv_boot.append(np.interp(base_fpr_cv, fpr_bs, tpr_bs))

tprs_cv_boot = np.vstack(tprs_cv_boot)
tpr_cv_lower = np.percentile(tprs_cv_boot, lower_perc, axis=0)
tpr_cv_upper = np.percentile(tprs_cv_boot, upper_perc, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust size or layout as needed

# --- Panel A: ROC Curve with shaded CI and CV in background ---
# Plot CV ROC curve (light gray background)
axes[0].plot(base_fpr_cv, np.mean(tprs_cv_boot, axis=0), color='darkgray', lw=2,
             label=f"CV ROC: AUC = {auc_cv:.3f} (95% CI: {lo_cv:.3f}-{hi_cv:.3f})")
axes[0].fill_between(base_fpr_cv, tpr_cv_lower, tpr_cv_upper,
                     color='darkgray', alpha=0.4)

# Plot external/test ROC curve (blue foreground)
axes[0].plot(base_fpr, np.mean(tprs_boot, axis=0), color='blue', lw=2,
             label=f"Test ROC: AUC = {auc:.3f} (95% CI: {lo:.3f}-{hi:.3f})")
axes[0].fill_between(base_fpr, tpr_lower, tpr_upper,
                     color='royalblue', alpha=0.3)

# Diagonal line
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")

# --- Panel B: Calibration Curve (Test Set Only) ---
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
axes[1].plot(mean_pred, frac_pos, marker='o', color='blue', lw=2, label="SVM Calibration")
axes[1].plot([0, 1], [0, 1], '--', color='gray', label="Perfect Calibration")
axes[1].set_xlabel("Mean Predicted Probability")
axes[1].set_ylabel("Fraction of Positives")
axes[1].legend(loc="lower right")

# Layout and save
plt.tight_layout()
plt.savefig(os.path.join(STUDY_DIR, "Combo-panel.png"), dpi=1200)
plt.savefig(os.path.join(STUDY_DIR, "Combo-panel.svg"))
plt.close()

shap_df = pd.read_csv(r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Topology-Fractal-Radiomics-Age-Sex-20250613_074011\shap_values_svm.csv")
shap_values = shap_df.values
feature_names = shap_df.columns.tolist()
X_sample = pd.read_csv(r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Xsample.csv')  
X_sample.drop('Unnamed: 0', axis=1, inplace=True)

name_mapping = {
    "original_firstorder_Skewness_T2W_R3":    "First Order Skewness (T2W, R3)",
    "original_glcm_Correlation_T1N_R3":       "GLCM Correlation (T1N, R3)",
    "original_shape_MinorAxisLength_T1C_R3":  "Shape Minor Axis Length (T1C, R3)",
    "original_firstorder_90Percentile_T2F_R3":"First Order 90th Percentile (T2F, R3)",
    "original_gldm_LargeDependenceHighGrayLevelEmphasis_T1N_R3": "GLDM Large Dependence High Gray-Level Emphasis (T1N, R3)",
    "original_firstorder_Minimum_T1C_R2":     "First Order Minimum (T1C, R2)",
    "original_firstorder_Skewness_T1N_R3":    "First Order Skewness (T1N, R3)",
    "original_firstorder_Median_T1C_R3":      "First Order Median (T1C, R3)",
    "original_shape_Maximum2DDiameterRow_T1C_R3": "Shape Maximum 2D Diameter Row (T1C, R3)",
    "original_firstorder_90Percentile_T1C_R3":"First Order 90th Percentile (T1C, R3)",
    "T2W_379":                                "T2W Feature 379",
    "Sex":                                    "Patient Sex",
    "T2W_399":                                "T2W Feature 399",
    "original_firstorder_Maximum_T2W_R3":     "First Order Maximum (T2W, R3)",
    "original_firstorder_Skewness_T2F_R1":    "First Order Skewness (T2F, R1)",
    "original_firstorder_Variance_T2F_R2":    "First Order Variance (T2F, R2)",
    "original_gldm_LargeDependenceHighGrayLevelEmphasis_T1C_R3": "GLDM Large Dependence High Gray-Level Emphasis (T1C, R3)",
    "original_glszm_SizeZoneNonUniformity_T1N_R3": "GLSZM Size Zone Non-Uniformity (T1N, R3)",
    "T2F_399":                                "T2F Feature 399",
    "T2F_379":                                "T2F Feature 379",
}

X_plot = X_sample.rename(columns=name_mapping)

fig = plt.figure(figsize=(20, 8))
shap.summary_plot(
    shap_values,
    X_plot,            # uses the renamed columns
    plot_type="dot",
    plot_size=(15, 8), # controls the SHAP axes size if supported
    show=False
)

fig.subplots_adjust(left=0.30, right=0.95)

plt.tight_layout()
fig.savefig(
    "shap_summary.png",
    dpi=1200,                # high resolution
    bbox_inches="tight"     # crop whitespace
)
fig.savefig(
    "shap_summary.svg",
    bbox_inches="tight"     # crop whitespace
)
plt.show()

# Rename columns for display
X_plot = X_sample.rename(columns=name_mapping)

# Compute mean absolute SHAP values per feature
shap_importance = np.abs(shap_values).mean(axis=0)
top5_indices = np.argsort(shap_importance)[-5:][::-1]  # top 5, descending
top5_features = X_plot.columns[top5_indices]

# Subset to top 5 features
X_top5 = X_plot[top5_features]
shap_top5 = shap_values[:, top5_indices]

# Plot
fig = plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_top5,
    X_top5,
    plot_type="dot",
    plot_size=(10, 8),
    show=False, 
)
# Hide y-axis labels (feature names)
plt.gca().set_yticklabels([])  # removes labels
plt.gca().tick_params(axis='y', length=0)  # optionally remove tick marks

plt.tight_layout()

# Save figures
fig.savefig("shap_summary_top5.png", dpi=1200, bbox_inches="tight")
fig.savefig("shap_summary_top5.svg", bbox_inches="tight")
plt.show()



# Filter SHAP and X_sample to only features in name_mapping
valid_features = list(name_mapping.keys())

X_filtered = X_sample[valid_features]
X_plot = X_filtered.rename(columns=name_mapping)

# Compute SHAP importance for filtered features
shap_values_filtered = shap_values[:, [X_sample.columns.get_loc(f) for f in valid_features]]
shap_importance = np.abs(shap_values_filtered).mean(axis=0)

# Build series with renamed column names
importance_series = pd.Series(shap_importance, index=X_plot.columns)

# Drop features with 0 importance
importance_series = importance_series[importance_series != 0]
importance_sorted = importance_series.sort_values(ascending=False)

# Plot configs
max_features = 100
height_per_feature = 0.3
max_height = 20
fig_height = min(max_height, len(importance_sorted[:max_features]) * height_per_feature + 2)

# -------- Bar Plot: ALL (Filtered) --------
fig, ax = plt.subplots(figsize=(12, 8))
importance_sorted[:max_features].plot(kind='barh', ax=ax)
ax.set_title(f"SHAP Feature Importance (Top {min(max_features, len(importance_sorted))})")
ax.set_xlabel("Mean |SHAP value|")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig("shap_bar_all.png", dpi=300, bbox_inches="tight")
fig.savefig("shap_bar_all.svg", bbox_inches="tight")
plt.show()

# -------- Bar Plot: TOP 5 --------
top5 = importance_sorted.head(5)
fig, ax = plt.subplots(figsize=(6, 3))
top5.plot(kind='barh', color='tab:blue', ax=ax)
ax.set_title("SHAP Feature Importance (Top 5)")
ax.set_xlabel("Mean |SHAP value|")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig("shap_bar_top5.png", dpi=300, bbox_inches="tight")
fig.savefig("shap_bar_top5.svg", bbox_inches="tight")
plt.show()