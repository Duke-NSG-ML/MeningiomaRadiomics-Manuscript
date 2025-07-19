#Evaluate-ANN-CV.py
#This script evaluates the ANN model using cross-validation to obtain final performance metrics

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
import random
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

#RANDOM setup
SEED = 7 
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
sklearn_random_state = SEED

FEATURE_CSV   = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\radiomics.csv'
TOPO_CSV      = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\finaltopo.csv'
FRACTAL_CSV   = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\fractal.csv'
AGE_CSV       = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\AgeSex.xlsx'
OUTCOME_CSV   = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\outcomes.csv'

# Study directory - update this to your ANN study directory
STUDY_DIR = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Topo-Fractal-Age-Sex-ANN-20250618_155912"  # Update this path
PARAMS_PATH = os.path.join(STUDY_DIR, 'best_params.json')

#enviornment setup
USE_GPU = True
N_FOLDS = 5
EPOCHS_CV = 25  
EPOCHS_FINAL = 25 

# Load best parameters
with open(PARAMS_PATH, 'r') as f:
    best_params = json.load(f)

def validate_data(df, name="DataFrame"):
    """Validate data for NaN, inf, and other issues"""
    print(f"\n=== Validating {name} ===")
    print(f"Shape: {df.shape}")
    
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        print(f"Columns with NaN: {nan_cols}")
        for col in nan_cols:
            nan_count = df[col].isnull().sum()
            print(f"  {col}: {nan_count} NaN values ({nan_count/len(df)*100:.1f}%)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
    if inf_cols:
        print(f"Columns with inf values: {inf_cols}")
    
    constant_cols = []
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    if constant_cols:
        print(f"Constant columns (will cause issues): {constant_cols}")
    
    return nan_cols, inf_cols, constant_cols

def clean_data(df):
    """Clean data by handling NaN, inf, and constant values - EXACTLY as in optimization"""
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    constant_cols = []
    for col in numeric_cols:
        if df_clean[col].nunique() <= 1:
            constant_cols.append(col)
    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        df_clean = df_clean.drop(columns=constant_cols)
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            if pd.isna(median_val):
                print(f"Warning: Column {col} has all NaN values, filling with 0")
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna(median_val)
    
    return df_clean

def drop_correlated_features(df, threshold):
    """Drop correlated features - EXACTLY as in optimization"""
    if threshold is None or df.shape[1] <= 1:
        return df
    
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("Warning: No numeric columns found for correlation analysis")
            return df
        
        if numeric_df.isnull().any().any() or np.isinf(numeric_df.values).any():
            print("Warning: NaN/inf values found, cleaning before correlation")
            numeric_df = clean_data(numeric_df)
        
        corr_matrix = numeric_df.corr()
        
        if corr_matrix.isnull().any().any():
            print("Warning: NaN in correlation matrix, filling with 0")
            corr_matrix = corr_matrix.fillna(0)
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        remaining_df = df.drop(columns=to_drop)
        print(f"Dropped {len(to_drop)} correlated features (threshold={threshold})")
        return remaining_df
        
    except Exception as e:
        print(f"Error in correlation filtering: {e}")
        return df

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

class ResidualBlock(nn.Module):
    def __init__(self, dim, act_fn, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = act_fn
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = act_fn
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.act1(self.bn1(self.fc1(x)))
        out = self.drop1(out)
        out = self.drop2(self.bn2(self.fc2(out)))
        return self.act2(out + identity)

class PyTorchANN(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_layers=[64], activation="relu", dropout=0.5,
                 lr=1e-3, optimizer_name="adam", momentum=0.9,
                 epochs=30, batch_size=32, use_gpu=True, use_skip=False, use_amp=True):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.use_skip = use_skip
        self.use_amp = use_amp
        if self.use_gpu:
            torch.backends.cudnn.benchmark = True

    def _get_activation(self):
        return {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }.get(self.activation, nn.ReLU())

    def _build_model(self):
        act_fn = self._get_activation()
        layers = []
        in_dim = self.input_dim

        for h in self.hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn)
            layers.append(nn.Dropout(self.dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X_np = X.values if hasattr(X, "values") else X
        y_np = y.values if hasattr(y, "values") else y
        
        if np.isnan(X_np).any() or np.isinf(X_np).any():
            raise ValueError("NaN or inf values found in X during fit")
        if np.isnan(y_np).any() or np.isinf(y_np).any():
            raise ValueError("NaN or inf values found in y during fit")

        device = torch.device("cuda" if self.use_gpu else "cpu")
        self.input_dim = X_np.shape[1]
        
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
        
        if torch.isnan(X_tensor).any():
            raise ValueError("NaN in X_tensor")
        if torch.isnan(y_tensor).any():
            raise ValueError("NaN in y_tensor")
        
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.use_gpu
        )
        
        self.model = self._build_model().to(device)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)
        
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-8)
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=self.momentum
            )
        elif self.optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                self.model.parameters(), lr=self.lr, momentum=self.momentum, eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer_name}")
            
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_gpu and self.use_amp)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_idx, (xb, yb) in enumerate(loader):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                
                if torch.isnan(xb).any():
                    print(f"NaN in input batch {batch_idx}, epoch {epoch}")
                    continue
                if torch.isnan(yb).any():
                    print(f"NaN in target batch {batch_idx}, epoch {epoch}")
                    continue
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.use_gpu and self.use_amp):
                    preds = self.model(xb)
                    
                    if torch.isnan(preds).any():
                        print(f"NaN in predictions at epoch {epoch}, batch {batch_idx}")
                        continue
                    
                    loss = criterion(preds, yb)
                    
                    if torch.isnan(loss):
                        print(f"NaN loss at epoch {epoch}, batch {batch_idx}")
                        continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            
            if epoch_loss > 0:
                scheduler.step(epoch_loss / len(loader))
            
            if self.use_gpu:
                torch.cuda.empty_cache()
        
        return self

    def predict_proba(self, X):
        X_np = X.values if hasattr(X, "values") else X
        
        if np.isnan(X_np).any() or np.isinf(X_np).any():
            raise ValueError("NaN or inf values found in X during predict_proba")
        
        device = torch.device("cuda" if self.use_gpu else "cpu")
        loader = DataLoader(
            TensorDataset(torch.tensor(X_np, dtype=torch.float32)),
            batch_size=self.batch_size,
            pin_memory=self.use_gpu
        )

        self.model.eval()
        logits_list = []
        
        with torch.no_grad():
            for i, (xb,) in enumerate(loader):
                xb = xb.to(device, non_blocking=True)
                
                if torch.isnan(xb).any():
                    raise ValueError(f"NaN in input batch {i} during prediction")
                
                with torch.cuda.amp.autocast(enabled=self.use_gpu and self.use_amp):
                    logits = self.model(xb)
                
                if torch.isnan(logits).any():
                    raise ValueError(f"NaN in logits (batch {i})")
                
                logits_list.append(logits.cpu())

        all_logits = torch.cat(logits_list, dim=0)
        all_logits = torch.clamp(all_logits, min=-50, max=50)
        probs_tensor = torch.sigmoid(all_logits)
        probs = probs_tensor.numpy().flatten()
        
        if np.isnan(probs).any():
            nan_indices = np.where(np.isnan(probs))[0]
            raise ValueError(f"NaN in sigmoid outputs at indices: {nan_indices[:10]}")
        
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

print("=== Loading Data ===")
features = pd.read_csv(FEATURE_CSV)
outcome = pd.read_csv(OUTCOME_CSV)
topo = pd.read_csv(TOPO_CSV)
fractal = pd.read_csv(FRACTAL_CSV)
age = pd.read_excel(AGE_CSV)

age = age[['BraTS ID', 'Age', 'Sex']]
sex_map = {"Male": 1, "Female": 0}
age['Sex'] = age['Sex'].map(sex_map)
age = age.rename(columns={'BraTS ID': 'PatientID'})
age['Age'] = age["Age"].fillna(age["Age"].mean())

print("=== Merging Datasets ===")
df = age.merge(fractal, on="PatientID", how='inner').merge(topo, on='PatientID', how='inner').merge(outcome, on='PatientID', how='inner')
df = df[df["PatientID"] != 'BraTS-MEN-01103-000']

# Initial data validation
print("=== Initial Data Validation ===")
nan_cols, inf_cols, const_cols = validate_data(df, "Merged DataFrame")

# Extract features and target
X = df.drop(columns=["PatientID", 'Outcome'])
diagnostic_cols = [col for col in X.columns if 'diagnostic' in col.lower()]
if diagnostic_cols:
    print(f"Removing diagnostic columns: {diagnostic_cols}")
    X = X.drop(columns=diagnostic_cols)
y = df['Outcome']

print("\n=== Cleaning Feature Matrix ===")
X_clean = clean_data(X)
print(f"Original X shape: {X.shape}, Cleaned X shape: {X_clean.shape}")

# Final validation
print("\n=== Final Validation ===")
validate_data(X_clean, "Final X")
validate_data(pd.DataFrame({'Outcome': y}), "Final y")

if y.isnull().any():
    print("Warning: NaN values found in target variable, removing corresponding rows")
    valid_indices = ~y.isnull()
    X_clean = X_clean[valid_indices]
    y = y[valid_indices]

print(f"\nFinal dataset shape: X={X_clean.shape}, y={len(y)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

print("\n=== Starting Evaluation ===")

# Extract parameters
scaler_map = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}
scaler_name = best_params['scaler']
selector_type = best_params['selector']
select_k = best_params.get('select_k', X_clean.shape[1])
corr_threshold = best_params.get('corr_thresh', None)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
y_true_all, y_prob_all, y_pred_all = [], [], []
auroc_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_clean, y)):
    try:
        X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train = clean_data(X_train)
        X_val = clean_data(X_val)

        X_train_f = drop_correlated_features(X_train, corr_threshold)
        if X_train_f.shape[1] == 0:
            print(f"No features after correlation filter in fold {fold_idx}, skipping")
            continue
        
        common_cols = [col for col in X_train_f.columns if col in X_val.columns]
        if len(common_cols) == 0:
            print(f"No common columns in fold {fold_idx}, skipping")
            continue
        
        X_train_f = X_train_f[common_cols]
        X_val_f = X_val[common_cols]

        k = min(select_k, X_train_f.shape[1])
        
        if selector_type == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif selector_type == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        else:  
            selector = SelectFromModel(
                LassoCV(cv=3, max_iter=10000, tol=1e-4, random_state=SEED), 
                threshold='mean'
            )
        
        X_train_s = selector.fit_transform(X_train_f, y_train)
        X_val_s = selector.transform(X_val_f)
        
        if X_train_s.shape[1] == 0:
            print(f"No features after selection in fold {fold_idx}, skipping")
            continue

        scaler = scaler_map[scaler_name]
        X_train_scaled = scaler.fit_transform(X_train_s)
        X_val_scaled = scaler.transform(X_val_s)
        
        if np.isnan(X_train_scaled).any() or np.isnan(X_val_scaled).any():
            print(f"NaN after scaling in fold {fold_idx}, skipping")
            continue

        try:
            minority_count = min(len(y_train[y_train==0]), len(y_train[y_train==1]))
            k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
            smote = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
            X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
        except Exception as e:
            print(f"SMOTE failed in fold {fold_idx}: {e}")
            X_res, y_res = X_train_scaled, y_train

        hidden_layers = [best_params[f'ann_h_{i}'] for i in range(best_params['ann_layers'])]
        clf = PyTorchANN(
            input_dim=X_res.shape[1],
            hidden_layers=hidden_layers,
            activation=best_params['ann_act'],
            dropout=best_params['ann_dropout'],
            lr=best_params['ann_lr'],
            optimizer_name=best_params['ann_opt'],
            momentum=best_params.get('ann_mom', 0.0),
            epochs=EPOCHS_CV,  # THIS IS THE KEY: Use 25 epochs like in optimization CV
            batch_size=min(best_params['batch_size'], len(X_res)//2),
            use_gpu=USE_GPU
        )
        
        clf.fit(pd.DataFrame(X_res), pd.Series(y_res))

        preds = clf.predict_proba(pd.DataFrame(X_val_scaled))[:, 1]
        
        if np.isnan(preds).any():
            print(f"NaN predictions in fold {fold_idx}, skipping")
            continue

        auc = roc_auc_score(y_val, preds)
        auroc_scores.append(auc)
        print(f"Fold {fold_idx+1} AUC: {auc:.3f}")
        
        mask = selector.get_support()
        feats = X_train_f.columns[mask]
        print(f"Fold {fold_idx+1} selected features ({len(feats)})")
        pd.DataFrame({'feature': feats}).to_csv(
            os.path.join(STUDY_DIR, f'eval_features_fold_{fold_idx+1}.csv'), index=False)
        
        y_true_all.extend(y_val)
        y_prob_all.extend(preds)
        y_pred_all.extend((preds > 0.5).astype(int))
        
    except Exception as e:
        print(f"Error in fold {fold_idx}: {e}")
        continue

if len(auroc_scores) == 0:
    raise ValueError("No successful folds in evaluation!")

# Convert to arrays
y_true = np.array(y_true_all)
y_prob = np.array(y_prob_all)
y_pred = np.array(y_pred_all)

# Save predictions
results_df = pd.DataFrame({
    'y_true': y_true,
    'y_prob': y_prob,
    'y_pred': y_pred
})
results_df.to_csv(os.path.join(STUDY_DIR, 'eval_cv_predictions.csv'), index=False)

# Compute overall metrics
auc = roc_auc_score(y_true, y_prob)
lo, hi = bootstrap_auc(y_true, y_prob)

print(f"\n=== EVALUATION RESULTS ===")
print(f"Mean CV AUC: {np.mean(auroc_scores):.3f}")
print(f"Overall AUC: {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
print(f"Individual fold AUCs: {auroc_scores}")

with open(os.path.join(STUDY_DIR, 'evaluation_metrics.txt'), 'w') as mf:
    mf.write("=== MATCHING OPTIMIZATION EVALUATION ===\n")
    mf.write(f"Used {EPOCHS_CV} epochs (matching optimization CV)\n\n")
    mf.write(f"AUROC: {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})\n")
    mf.write(f"Fold AUCs: {', '.join(f'{a:.3f}' for a in auroc_scores)}\n")
    mf.write(f"Mean Fold AUC: {np.mean(auroc_scores):.3f}\n")

