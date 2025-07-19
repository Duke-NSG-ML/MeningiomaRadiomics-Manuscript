# FinalModel-Calibration.py
# Train, calibrate, and save a radiomics model pipeline that internally selects only the final features
# Uses SVC with RBF kernel as the base classifier and Platt scaling for calibration

#Import libraries
import os
import json
import datetime
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

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
FEATURES_INCLUDE = r'Topology-Fractal-Radiomics-Age-Sex-20250613_074011/features_final.csv'

# Assume best_params.json is in the current working directory
STUDY_DIR = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Topology-Fractal-Radiomics-Age-Sex-20250613_074011"
#PARAMS_PATH = os.path.join(STUDY_DIR, 'best_params.json')
OUTPUT_DIR = STUDY_DIR  # saving inside study dir

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
features_include = pd.read_csv(FEATURES_INCLUDE, header=None).iloc[:, 0].tolist()
print(features_include)
age = age[['BraTS ID', 'Age', 'Sex']].rename(columns={'BraTS ID': 'PatientID'})
age['Sex'] = age['Sex'].map({'Male': 1, 'Female': 0})
age['Age'] = age['Age'].fillna(age['Age'].mean())

df = age.merge(topo, on='PatientID', how='inner').merge(fractal,on="PatientID",how='inner').merge(feats,on="PatientID",how="inner").merge(outcomes, on='PatientID', how='inner')
df = df[df['PatientID'] != 'BraTS-MEN-01103-000']  #This patient had solitary fibrous tumor, exclude

# Merge features
X = df.drop(columns=['PatientID', 'Outcome'])
X = X[features_include]
print(X)

y = df['Outcome']

# Drop diagnostic columns
diag_cols = [c for c in X.columns if 'diagnostic' in c.lower()]
if diag_cols:
    X = X.drop(columns=diag_cols)


# Feature selection settings
corr_threshold = None
scaler   = StandardScaler()
smote    = SMOTE(sampling_strategy='auto', random_state=SEED)
clf      = SVC(random_state=SEED, C=1.1455279871277018, gamma=0.011675340804279044, kernel='rbf', probability=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
auc_scores = []
for train_idx, val_idx in skf.split(X, y):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    X_tr = drop_correlated_features(X_tr, corr_threshold)
    X_va = X_va[X_tr.columns]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    pipe = ImbPipeline([
        ('scaler', scaler),
        ('smote', smote),
        ('clf', clf)
    ])
    pipe.fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_va)[:,1]
    auc_scores.append(roc_auc_score(y_va, probs))

print(f"CV AUCs: {np.round(auc_scores,3)} | Mean: {np.mean(auc_scores):.3f}")

# Final feature selection
X_filtered = drop_correlated_features(X, corr_threshold)
final_features = X_filtered.columns.tolist()



full_pipe = ImbPipeline([
    ('scaler', scaler),
    ('smote', smote),
    ('clf', clf)
])

# Fit full pipeline
print("\nFitting full pipeline on all data...")
full_pipe.fit(X_filtered, y)

# Calibrate using Platt scaling 
CALIBRATION_METHOD = 'sigmoid'  
calibrator = CalibratedClassifierCV(
    estimator=full_pipe,
    method=CALIBRATION_METHOD,
    cv='prefit'
)
print(f"Calibrating pipeline using {CALIBRATION_METHOD}...")
calibrator.fit(X_filtered, y)

# Save calibrated pipeline and feature names
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(calibrator, os.path.join(OUTPUT_DIR, 'calibrated_pipeline.pkl'))
joblib.dump(final_features, os.path.join(OUTPUT_DIR, 'model_feature_names.pkl'))

# Save configuration
config = {
    'features': final_features,
    'seed': SEED,
    'calibration': CALIBRATION_METHOD,
    'cv_splits': 5
}
with open(os.path.join(OUTPUT_DIR, 'model_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print('=== Final calibrated pipeline saved! ===')
