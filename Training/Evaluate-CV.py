# Evaluate-CV.py
# Evaluate best performing non-ANN model

#Import libraries
import os
import json
import time
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle, resample
from xgboost import XGBClassifier
from optuna.integration import BoTorchSampler
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from optuna.exceptions import TrialPruned
import shap
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
import warnings
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, recall_score
import random
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.metrics import roc_curve

#RANDOM setup
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

# Assume best_params.json is in the current working directory
STUDY_DIR = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Topology-Fractal-Radiomics-Age-Sex-20250613_074011"
#PARAMS_PATH = os.path.join(STUDY_DIR, 'best_params.json')

# Utility functions
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

# Load and merge data
feats = pd.read_csv(FEATURE_CSV)
outcomes = pd.read_csv(OUTCOME_CSV)
fractal = pd.read_csv(FRACTAL_CSV)
topo = pd.read_csv(TOPO_CSV)
age = pd.read_excel(AGE_CSV)
age = age[['BraTS ID', 'Age', 'Sex']].rename(columns={'BraTS ID': 'PatientID'})
age['Sex'] = age['Sex'].map({'Male': 1, 'Female': 0})
age['Age'] = age['Age'].fillna(age['Age'].mean())

df = age.merge(topo, on='PatientID', how='inner').merge(fractal,on="PatientID",how='inner').merge(feats,on="PatientID",how="inner").merge(outcomes, on='PatientID', how='inner')
df = df[df['PatientID'] != 'BraTS-MEN-01103-000']

demodf = df[['PatientID', 'Age', 'Sex', 'Outcome']]
demodf.to_csv(os.path.join(STUDY_DIR, 'demographics.csv'), index=False)

df.to_csv(os.path.join(STUDY_DIR, 'merged_features.csv'), index=False)

# Merge features
X = df.drop(columns=['PatientID', 'Outcome'])
y = df['Outcome']

# Drop diagnostic columns
diag_cols = [c for c in X.columns if 'diagnostic' in c.lower()]
if diag_cols:
    X = X.drop(columns=diag_cols)

'''
selector = SelectKBest(score_func=f_classif, k=400)
scaler = RobustScaler()

clf = CatBoostClassifier(
    random_seed=SEED,
    iterations=982,
    depth=3,
    learning_rate=0.11205103917362272,
    l2_leaf_reg=0.008592748795770015,
    leaf_estimation_iterations=2,
    colsample_bylevel=0.7564983565750154,
    verbose=0,
    task_type='CPU',
    allow_writing_files=False,
    save_snapshot=False
)


corr_threshold = 0.7

'''

scaler = StandardScaler()
selector = SelectFromModel(LassoCV(cv=3, max_iter=100000, tol=1e-3, random_state=SEED))
corr_threshold = None

clf = SVC(
    random_state=SEED,
    C=1.1455279871277018,
    gamma=0.011675340804279044,
    kernel='rbf',
    probability=True
)


'''
scaler = StandardScaler()
selector = SelectKBest(score_func=mutual_info_classif, k="all")
corr_threshold = None
clf = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    max_iter=10000,
    random_state=SEED,
    C=0.005717258510850064,
    l1_ratio=0.0017054992034081908
)
'''
'''
scaler = RobustScaler()
selector = SelectKBest(score_func=mutual_info_classif, k=250)
corr_threshold = 0.9
clf = XGBClassifier(
    n_estimators=440,
    max_depth = 190,
    learning_rate = 0.2909225067221857,
    reg_alpha = 2.1321407724998007,
    reg_lambda = 3.1279883286525587,
    min_child_weight = 5,
    subsample = 0.7515775597770649,
    colsample_bytree = 0.5520855214697309,
    eval_metric='logloss',
    random_state=SEED,
    tree_method='hist',
    device='cpu',
)


scaler = StandardScaler()
k=10
selector = SelectFromModel(LassoCV(cv=3, max_iter=100000, tol=1e-3, random_state=SEED))
corr_threshold = None
clf = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    max_iter=10000,
    random_state=SEED,
    C=1.145527987127701,
    l1_ratio= 0.011675340804279044
)
'''
pipeline = ImbPipeline([
    ('scaler', scaler),
    ('selector', selector),
    ('smote', SMOTE(sampling_strategy='auto',random_state=SEED)),
    ('clf', clf)
])

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
y_true_all, y_prob_all, y_pred_all = [], [], []
auroc_scores = []
i=1
for train_idx, val_idx in skf.split(X, y):
    X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
    # Apply correlation filtering
    X_train = drop_correlated_features(X_train_raw, corr_threshold)
    # Align columns in X_val to match filtered X_train
    X_val = X_val_raw[X_train.columns]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    mask = pipeline.named_steps['selector'].get_support()
    feats = X_train.columns[mask]
    # Print actual columns used in this fold
    print(f"Fold {i} selected features ({len(feats)}): {list(feats)}")
    # Save to CSV
    pd.DataFrame({'feature': feats}).to_csv(
        os.path.join(STUDY_DIR, f'features_fold_{i}.csv'), index=False)

        # compute and log this fold’s AUC
    fold_auc = roc_auc_score(y_val, y_prob)
    auroc_scores.append(fold_auc)
    print(f"Fold {i} AUC: {fold_auc:.3f}")
    with open(os.path.join(STUDY_DIR, 'metrics.txt'), 'a') as mf:
        mf.write(f"Fold {i} AUROC: {fold_auc:.3f}\n")
    # then extend your lists as before
    y_true_all.extend(y_val)
    y_prob_all.extend(y_prob)
    y_pred_all.extend(y_pred)

    i+=1


y_true = np.array(y_true_all)
y_prob = np.array(y_prob_all)
y_pred = np.array(y_pred_all)


results_df = pd.DataFrame({
    'y_true': y_true,
    'y_prob': y_prob,
    'y_pred': y_pred
})
results_df.to_csv(os.path.join(STUDY_DIR, 'cv_predictions.csv'), index=False)

# Metrics and outputs
auc = roc_auc_score(y_true, y_prob)
lo, hi = bootstrap_auc(y_true, y_prob)

with open(os.path.join(STUDY_DIR, 'metrics.txt'), 'w') as mf:
    mf.write(f"AUROC: {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})\n")
    mf.write(f"Fold AUCs: {', '.join(f'{a:.3f}' for a in auroc_scores)}\n")
    mf.write(f"Mean Fold AUC: {np.mean(auroc_scores):.3f}\n")
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
    # interpolate this bootstrap’s TPR onto base_fpr
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
plt.savefig(os.path.join(STUDY_DIR, "roc_curve_with_ci.png"), dpi=1200)
plt.close()

# Calibration curve
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
plt.figure()
plt.plot(mean_pred, frac_pos, marker='o')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('Mean Predicted'); plt.ylabel('Fraction Positives')
plt.savefig(os.path.join(STUDY_DIR, 'calibration_curve.png'), dpi=1200)
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
# Metrics at Youden’s threshold
youden_j    = tpr - fpr
idx_youden  = np.argmax(youden_j)
th_youden   = thresholds[idx_youden]
sens_youden = tpr[idx_youden]
spec_youden = 1 - fpr[idx_youden]
ppv_youden  = precision_score(y_true, (y_prob >= th_youden).astype(int))

# Metrics at 85% sensitivity threshold
target_sens = 0.85
idx85 = np.where(tpr >= target_sens)[0]
idx85 = idx85[0] if len(idx85) else np.argmax(tpr)
th_85    = thresholds[idx85]
sens_85  = tpr[idx85]
spec_85  = 1 - fpr[idx85]
ppv_85   = precision_score(y_true, (y_prob >= th_85).astype(int))

with open(os.path.join(STUDY_DIR, 'metrics.txt'), 'a') as mf:
    mf.write("\nYouden's threshold: {:.3f}\n".format(th_youden))
    mf.write("Sensitivity@Youden: {:.3f}, Specificity@Youden: {:.3f}, PPV@Youden: {:.3f}\n"
             .format(sens_youden, spec_youden, ppv_youden))
    mf.write("\n85% Sensitivity threshold: {:.3f}\n".format(th_85))
    mf.write("Sensitivity@85%: {:.3f}, Specificity@85%: {:.3f}, PPV@85%: {:.3f}\n"
             .format(sens_85, spec_85, ppv_85))


# re-fit on the globally filtered set
X_filtered = drop_correlated_features(X, corr_threshold)
pipeline.fit(X_filtered, y)
#num final features
mask = pipeline.named_steps['selector'].get_support()
feats = X_filtered.columns[mask]
print(f"Final selected features ({len(feats)}): {list(feats)}")
pd.DataFrame({'feature': feats}).to_csv(
    os.path.join(STUDY_DIR, f'features_final.csv'), index=False)

perm = permutation_importance(
    pipeline,
    X_filtered,
    y,
    n_repeats=30,
    random_state=SEED
)
imps = perm.importances_mean
idxs = np.argsort(imps)[::-1][:10]            # top 10 only
top_feats = X_filtered.columns[idxs]

plt.figure(figsize=(8, 4))
plt.barh(range(10), imps[idxs][::-1])
plt.yticks(range(10), top_feats[::-1])
plt.xlabel("Mean Permutation Importance")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(STUDY_DIR, 'permutation_importance_top10.png'), dpi=1200)
plt.close()


svm_model = pipeline.named_steps['clf']

try:
 
    background_size = min(100, len(X_filtered))
    background = shap.sample(X_filtered, background_size, random_state=SEED)

    explainer = shap.Explainer(pipeline.predict_proba, background)

    sample_size = min(200, len(X_filtered))
    X_sample = shap.sample(X_filtered, sample_size, random_state=SEED)
    X_sample.to_csv("Xsample.csv")
    
    print("Computing SHAP values for SVM (this may take a few minutes)...")
    shap_values = explainer(X_sample)
    
    if hasattr(shap_values, 'values') and shap_values.values.ndim == 3:
        shap_values_positive = shap_values[:, :, 1]  # positive class
    else:
        shap_values_positive = shap_values
    
    plt.figure(figsize=(20, 8))
    shap.summary_plot(
        shap_values_positive, 
        X_sample, 
        show=False, 
        plot_type="dot"
    )
    plt.title("SHAP Summary (beeswarm) - SVM")
    plt.tight_layout()
    plt.savefig(os.path.join(STUDY_DIR, "shap_beeswarm_svm.png"), dpi=1200, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(20, 10))
    shap.summary_plot(
        shap_values_positive, 
        X_sample, 
        show=False, 
        plot_type="bar"
    )
    plt.title("SHAP Feature Importance (bar) - SVM")
    plt.tight_layout()
    plt.savefig(os.path.join(STUDY_DIR, "shap_bar_svm.png"), dpi=1200, bbox_inches='tight')
    plt.close()
    
    for i in range(min(3, len(X_sample))):
        plt.figure(figsize=(20, 8))
        shap.waterfall_plot(shap_values_positive[i], show=False)
        plt.title(f"SHAP Waterfall Plot - Sample {i+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(STUDY_DIR, f"shap_waterfall_sample_{i+1}.png"), 
                   dpi=1200, bbox_inches='tight')
        plt.close()
    
    print("SHAP analysis completed successfully!")
    
except Exception as e:
    print(f"Error with unified Explainer, trying KernelExplainer: {e}")
    
    try:
        background_size = min(50, len(X_filtered))
        background = shap.sample(X_filtered, background_size, random_state=SEED)
        
        explainer = shap.KernelExplainer(
            lambda x: pipeline.predict_proba(x)[:, 1],  
            background
        )
        
        sample_size = min(50, len(X_filtered))
        X_sample = shap.sample(X_filtered, sample_size, random_state=SEED)
        X_sample.to_csv("XSample.csv")
        print("Computing SHAP values with KernelExplainer (this will take longer)...")
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(20, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            show=False, 
            plot_type="dot"
        )
        plt.title("SHAP Summary (beeswarm) - SVM (Kernel)")
        plt.tight_layout()
        plt.savefig(os.path.join(STUDY_DIR, "shap_beeswarm_svm_kernel.png"), 
                   dpi=1200, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(20, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            show=False, 
            plot_type="bar"
        )
        plt.title("SHAP Feature Importance (bar) - SVM (Kernel)")
        plt.tight_layout()
        plt.savefig(os.path.join(STUDY_DIR, "shap_bar_svm_kernel.png"), 
                   dpi=1200, bbox_inches='tight')
        plt.close()
        
        print("SHAP analysis with KernelExplainer completed!")
        
    except Exception as e2:
        print(f"Both SHAP methods failed: {e2}")
        print("Skipping SHAP analysis for SVM.")

try:
    if 'shap_values' in locals():
        if hasattr(shap_values, 'values'):
            shap_df = pd.DataFrame(shap_values.values, columns=X_sample.columns)
        else:
            shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
        shap_df.to_csv(os.path.join(STUDY_DIR, 'shap_values_svm.csv'), index=False)
        print("SHAP values saved to CSV.")
except Exception as e:
    print(f"Could not save SHAP values: {e}")