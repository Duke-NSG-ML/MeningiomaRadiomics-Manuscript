#CV-ANN.py
#This script performs hyperparameter optimization for an ANN model using Optuna with cross-validation 
import os
import numpy as np
import pandas as pd
import json
import pickle
import logging
import optuna
import datetime
from sqlalchemy import text
from optuna.storages import RDBStorage
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import  StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LassoCV
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
import random
from optuna.exceptions import TrialPruned
import csv

#RANDOM Setup
SEED = 7 
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
sklearn_random_state = SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Environment setup
USE_GPU = True
N_TRIALS = 1000
N_FOLDS = 5
EPOCHS = 50

# Load in data
FEATURE_CSV = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\radiomics.csv'
topo_csv = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\finaltopo.csv"
fractal_csv = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\fractal.csv"
age_csv = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\AgeSex.xlsx"
OUTCOME_CSV = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\outcomes.csv'

#Study setup
study_name = "Radiomics-Age-Sex" + "-ANN-" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
study_descriptor = "Included Variables: Radiomics, Age, Sex"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Make dir for study results if it doesn't exist with studyname 
if not os.path.exists(study_name):
    os.makedirs(study_name)
if study_descriptor:
    with open(f"{study_name}/study_descriptor.txt", "w") as f:
        f.write(study_descriptor)

# Optuna storage path in the study directory
storage = f"sqlite:///{study_name}/{study_name}.db"

def validate_data(df, name="DataFrame"):
    """Validate data for NaN, inf, and other issues"""
    print(f"\n=== Validating {name} ===")
    print(f"Shape: {df.shape}")
    
    # Check for NaN values
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        print(f"Columns with NaN: {nan_cols}")
        for col in nan_cols:
            nan_count = df[col].isnull().sum()
            print(f"  {col}: {nan_count} NaN values ({nan_count/len(df)*100:.1f}%)")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
    if inf_cols:
        print(f"Columns with inf values: {inf_cols}")
    
    # Check for constant columns
    constant_cols = []
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    if constant_cols:
        print(f"Constant columns (will cause issues): {constant_cols}")
    
    return nan_cols, inf_cols, constant_cols

def clean_data(df):
    """Clean data by handling NaN, inf, and constant values"""
    df_clean = df.copy()
    
    # Handle infinite values - replace with NaN first
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Remove constant columns
    constant_cols = []
    for col in numeric_cols:
        if df_clean[col].nunique() <= 1:
            constant_cols.append(col)
    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        df_clean = df_clean.drop(columns=constant_cols)
    
    # Handle NaN values - median imputation for numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            if pd.isna(median_val):  # If all values are NaN
                print(f"Warning: Column {col} has all NaN values, filling with 0")
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna(median_val)
    
    return df_clean

def drop_correlated_features(df, threshold):
    """Drop correlated features with better error handling"""
    if threshold is None or df.shape[1] <= 1:
        return df
    
    try:
        # Ensure all columns are numeric
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("Warning: No numeric columns found for correlation analysis")
            return df
        
        # Check for NaN/inf before correlation
        if numeric_df.isnull().any().any() or np.isinf(numeric_df.values).any():
            print("Warning: NaN/inf values found, cleaning before correlation")
            numeric_df = clean_data(numeric_df)
        
        corr_matrix = numeric_df.corr()
        
        # Handle NaN in correlation matrix
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

# Process radiomics df
def preprocess(df, outcomes_df):
    # Identify columns containing 'diagnostic'
    diagnostic_cols = [col for col in df.columns if 'diagnostic' in col.lower()]
    # Drop those columns
    df = df.drop(columns=diagnostic_cols)
    return df.drop(columns=['PatientID']), df['PatientID']

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
        
        # Check for NaN/inf in inputs
        if np.isnan(X_np).any() or np.isinf(X_np).any():
            raise ValueError("NaN or inf values found in X during fit")
        if np.isnan(y_np).any() or np.isinf(y_np).any():
            raise ValueError("NaN or inf values found in y during fit")

        device = torch.device("cuda" if self.use_gpu else "cpu")
        self.input_dim = X_np.shape[1]
        
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
        
        # Check tensors for NaN
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
        
        # Optimizer setup with gradient clipping
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
                
                # Check for NaN in batch
                if torch.isnan(xb).any():
                    print(f"NaN in input batch {batch_idx}, epoch {epoch}")
                    continue
                if torch.isnan(yb).any():
                    print(f"NaN in target batch {batch_idx}, epoch {epoch}")
                    continue
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.use_gpu and self.use_amp):
                    preds = self.model(xb)
                    
                    # Check predictions for NaN
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
        
        # Input validation
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
                
                # Check input batch for NaN
                if torch.isnan(xb).any():
                    raise ValueError(f"NaN in input batch {i} during prediction")
                
                with torch.cuda.amp.autocast(enabled=self.use_gpu and self.use_amp):
                    logits = self.model(xb)
                
                # Check logits for NaN
                if torch.isnan(logits).any():
                    raise ValueError(f"NaN in logits (batch {i})")
                
                logits_list.append(logits.cpu())

        # Concatenate and apply sigmoid
        all_logits = torch.cat(logits_list, dim=0)
        
        # Clamp logits to prevent numerical overflow in sigmoid
        all_logits = torch.clamp(all_logits, min=-50, max=50)
        probs_tensor = torch.sigmoid(all_logits)

        # Convert to numpy and validate
        probs = probs_tensor.numpy().flatten()
        
        if np.isnan(probs).any():
            nan_indices = np.where(np.isnan(probs))[0]
            raise ValueError(f"NaN in sigmoid outputs at indices: {nan_indices[:10]}")
        
        # Ensure probabilities are in valid range
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        
        # Return in two-column format
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def objective(trial, X, y):
    try:
        # Validate inputs
        if X.isnull().any().any() or np.isinf(X.values).any():
            X = clean_data(X)
        
        corr_thresh = trial.suggest_categorical('corr_thresh', [None, 0.8, 0.9, 0.95])
        select_k = trial.suggest_int('select_k', 10, min(50, X.shape[1]))  # Cap max features
        scaler_name = trial.suggest_categorical('scaler', ['standard'])
        
        scaler = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }[scaler_name]

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        aucs = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            try:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Clean data for this fold
                X_train = clean_data(X_train)
                X_val = clean_data(X_val)

                X_train_f = drop_correlated_features(X_train, corr_thresh)
                if X_train_f.shape[1] == 0:
                    trial.set_user_attr('error', f'no features after corr filter in fold {fold_idx}')
                    raise TrialPruned()
                
                common_cols = [col for col in X_train_f.columns if col in X_val.columns]
                if len(common_cols) == 0:
                    trial.set_user_attr('error', f'no common columns in fold {fold_idx}')
                    raise TrialPruned()
                
                X_train_f = X_train_f[common_cols]
                X_val_f = X_val[common_cols]

                k = min(select_k, X_train_f.shape[1])
                selector_type = trial.suggest_categorical('selector', ['f_classif', 'mutual_info'])

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
                    trial.set_user_attr('error', f'no features after selection in fold {fold_idx}')
                    raise TrialPruned()

                X_train_scaled = scaler.fit_transform(X_train_s)
                X_val_scaled = scaler.transform(X_val_s)
                
                if np.isnan(X_train_scaled).any() or np.isnan(X_val_scaled).any():
                    trial.set_user_attr('error', f'NaN after scaling in fold {fold_idx}')
                    raise TrialPruned()

                #SMOTE
                try:
                    minority_count = min(len(y_train[y_train==0]), len(y_train[y_train==1]))
                    k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
                    smote = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
                    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
                except Exception as e:
                    print(f"SMOTE failed in fold {fold_idx}: {e}")
                    # Use original data if SMOTE fails
                    X_res, y_res = X_train_scaled, y_train

                batch_size = trial.suggest_categorical('batch_size', [16,32,64,128])
                n_layers = trial.suggest_int('ann_layers', 1, 8)  # Reduce complexity
                hidden = [trial.suggest_int(f'ann_h_{i}', 16, 512, step=16) for i in range(n_layers)]
                dropout = trial.suggest_float('ann_dropout', 0.1, 0.5)
                lr = trial.suggest_float('ann_lr', 1e-5, 1e-3, log=True)
                act = trial.suggest_categorical('ann_act', ['relu', 'leaky_relu'])  # Stick to stable activations
                optn = trial.suggest_categorical('ann_opt', ['adam'])  # Use most stable optimizer
                
                clf = PyTorchANN(
                    input_dim=X_res.shape[1], 
                    hidden_layers=hidden,
                    activation=act, 
                    dropout=dropout,
                    lr=lr, 
                    optimizer_name=optn, 
                    momentum=0.0,
                    epochs=min(EPOCHS, 25),  # Reduce epochs to prevent overfitting
                    batch_size=min(batch_size, len(X_res)//2),
                    use_gpu=USE_GPU
                )
                
                clf.fit(pd.DataFrame(X_res), pd.Series(y_res))

                preds = clf.predict_proba(pd.DataFrame(X_val_scaled))[:, 1]
                
                if np.isnan(preds).any():
                    trial.set_user_attr('error', f'NaN predictions in fold {fold_idx}')
                    raise TrialPruned()

                auc = roc_auc_score(y_val, preds)
                aucs.append(auc)
                
            except Exception as e:
                print(f"Error in fold {fold_idx}: {e}")
                trial.set_user_attr('error', f'fold {fold_idx} failed: {str(e)}')
                raise TrialPruned()

        if len(aucs) == 0:
            trial.set_user_attr('error', 'no successful folds')
            raise TrialPruned()
        
        trial.set_user_attr('aucs', aucs)
        trial.set_user_attr('n_successful_folds', len(aucs))
        return np.mean(aucs)
        
    except Exception as e:
        trial.set_user_attr('error', str(e))
        raise TrialPruned()

def run_optuna(X, y):
    # Configure RDB storage for parallel trials
    rdb = RDBStorage(
        url=storage,
        engine_kwargs={
            'connect_args': {
                'timeout': 600,
                'check_same_thread': False
            }
        }
    )
    # Enable Write-Ahead Logging
    with rdb.engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))

    # Create or load study using RDB storage
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        storage=rdb,
        sampler=TPESampler(n_startup_trials=75, seed=SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3),
        load_if_exists=True
    )

    # Run optimization in parallel
    study.optimize(lambda t: objective(t, X, y), n_trials=N_TRIALS, n_jobs=25)
    print(f"Run dashboard: optuna-dashboard {storage} {study.study_name}")

    # Save study results
    best = study.best_trial
    results = {
        'best_trial': best.number,
        'value': best.value,
        'params': best.params,
        'user_attrs': best.user_attrs
    }
    with open(f"{study_name}/study_results.json","w") as f:
        json.dump(results,f,indent=4)
    with open(f"{study_name}/trial_auc_summary.csv","w",newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trial","mean_auc","n_successful_folds"])
        for t in study.trials:
            aucs = t.user_attrs.get('aucs', [])
            writer.writerow([
                t.number,
                np.mean(aucs) if len(aucs) > 0 else np.nan,
                len(aucs)
            ])

    # Retrain final model with best params
    scaler_map = {'standard':StandardScaler(),'minmax':MinMaxScaler(),'robust':RobustScaler()}
    scaler = scaler_map[best.params['scaler']]
    sel_name = best.params['selector']
    k = best.params.get('select_k', X.shape[1])
    
    if sel_name == 'f_classif':
        selector = SelectKBest(f_classif, k=k)
    elif sel_name == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=k)
    else:  # lasso
        selector = SelectFromModel(LassoCV(cv=3, random_state=SEED), threshold='mean')

    # Build final model with best hyperparameters
    clf = PyTorchANN(
        input_dim=X.shape[1],
        hidden_layers=[best.params[f'ann_h_{i}'] for i in range(best.params['ann_layers'])],
        activation=best.params['ann_act'],
        dropout=best.params['ann_dropout'],
        lr=best.params['ann_lr'],
        optimizer_name=best.params['ann_opt'],
        momentum=best.params.get('ann_mom',0),
        epochs=EPOCHS,
        batch_size=best.params['batch_size'],
        use_gpu=USE_GPU
    )
    
    final = Pipeline([('scaler',scaler),('select',selector),('clf',clf)])
    X_filt = drop_correlated_features(X, best.params.get('corr_thresh',None))
    final.fit(X_filt, y)

    # Save final artifacts
    with open(f"{study_name}/best_model.pkl","wb") as f:
        pickle.dump(final,f)
    pd.Series(X_filt.columns).to_csv(f"{study_name}/selected_features.csv", index=False)
    with open(f"{study_name}/best_params.json","w") as f:
        json.dump(best.params,f,indent=4)
    print(f"Artifacts saved in {study_name}")

# Load and process data
print("=== Loading Data ===")
features = pd.read_csv(FEATURE_CSV)
outcome = pd.read_csv(OUTCOME_CSV)
topo = pd.read_csv(topo_csv)
fractal = pd.read_csv(fractal_csv)
age = pd.read_excel(age_csv)

assert 'Outcome' in outcome.columns

# Process the age df
age = age[['BraTS ID', 'Age', 'Sex']]
sex_map = {"Male": 1, "Female": 0}
age['Sex'] = age['Sex'].map(sex_map)
age = age.rename(columns={'BraTS ID': 'PatientID'})
age['Age'] = age["Age"].fillna(age["Age"].mean())

# Merge all datasets
print("=== Merging Datasets ===")
df = age.merge(features, on="PatientID", how='inner').merge(outcome, on='PatientID', how='inner')
df = df[df["PatientID"] != 'BraTS-MEN-01103-000'] #patient with hemangiopericytoma 

# Initial data validation
print("=== Initial Data Validation ===")
nan_cols, inf_cols, const_cols = validate_data(df, "Merged DataFrame")

# Make sure there is an "Outcome" column
if 'Outcome' not in df.columns:
    raise ValueError("The DataFrame must contain an 'Outcome' column.")

# Extract features and target
X = df.drop(columns=["PatientID", 'Outcome'])
diagnostic_cols = [col for col in X.columns if 'diagnostic' in col.lower()]
if diagnostic_cols:
    print(f"Removing diagnostic columns: {diagnostic_cols}")
    X = X.drop(columns=diagnostic_cols)
y = df['Outcome']

# Clean the feature matrix
print("\n=== Cleaning Feature Matrix ===")
X_clean = clean_data(X)
print(f"Original X shape: {X.shape}, Cleaned X shape: {X_clean.shape}")

# Final validation
print("\n=== Final Validation ===")
validate_data(X_clean, "Final X")
validate_data(pd.DataFrame({'Outcome': y}), "Final y")

# Ensure y is clean too
if y.isnull().any():
    print("Warning: NaN values found in target variable, removing corresponding rows")
    valid_indices = ~y.isnull()
    X_clean = X_clean[valid_indices]
    y = y[valid_indices]

print(f"\nFinal dataset shape: X={X_clean.shape}, y={len(y)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Run the study
print("\n=== Starting Optuna Optimization ===")
run_optuna(X_clean, y)