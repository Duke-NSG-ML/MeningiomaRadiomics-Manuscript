#CV.py
#This file conducsts hyperparameter tuning and cross validation using optuna on non-ANN models used in the manuscript

#Import libraries
import os
import json
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
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, recall_score
from sklearn.model_selection import  StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from optuna.exceptions import TrialPruned
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, LassoCV
import warnings
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import random
warnings.filterwarnings("ignore", category=FutureWarning)

#RANDOM setup
SEED = 7 
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
sklearn_random_state = SEED
#enviornment setup
USE_GPU = True                  # Flag to indicate whether GPU should be used 
N_TRIALS = 1000              # Total number of Optuna trials
N_FOLDS = 5                     # Number of folds for cross-validation

#Load in data
FEATURE_CSV = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\radiomics.csv' #radiomics extraction csv
topo_csv = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\finaltopo.csv" #topology extraction csv
fractal_csv = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\fractal.csv"
age_csv = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\AgeSex.xlsx" #demographic info csv
OUTCOME_CSV = r'C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\outcomes.csv' #outcomes csv


#Study setup
study_name = "Fractal-Age-Sex" + "-" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
study_descriptor = "Included Variables: Fractal, Age, Sex"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if not os.path.exists(study_name):
    os.makedirs(study_name)
if study_descriptor:
    with open(f"{study_name}/study_descriptor.txt", "w") as f:
        f.write(study_descriptor)

#optuna storage path in the study directory
storage_path = f"sqlite:///{study_name}/{study_name}.db"

#Process radiomics df
def preprocess(df, outcomes_df):
    # Identify columns containing 'diagnostic'
    diagnostic_cols = [col for col in df.columns if 'diagnostic' in col.lower()]

    # Drop those columns
    df = df.drop(columns=diagnostic_cols)

    return df.drop(columns=['PatientID']), df['PatientID']

features = pd.read_csv(FEATURE_CSV) #create radiomics feature df
outcome = pd.read_csv(OUTCOME_CSV) #create outcome df
topo = pd.read_csv(topo_csv) #create topology feature df
fractal = pd.read_csv(fractal_csv)
age = pd.read_excel(age_csv) #create age/sex df
assert 'Outcome' in outcome.columns #ensure outcome list is in outcomes df

#Process the age df
age = age[['BraTS ID', 'Age', 'Sex']] #keep only applicable columns
sex_map = {"Male": 1, "Female":0} #Int Sex encoding
age['Sex'] = age['Sex'].map(sex_map) #Remap to our encoding
age = age.rename(columns={'BraTS ID': 'PatientID'}) #Rename column to match other DFs

age['Age'] = age["Age"].fillna(age["Age"].mean()) #There is an Na for one patients age, impute with mean 

#EDIT THIS LINE BASED ON WHAT FEATURES YOU WANT TO INCLUDE
df =  age.merge(fractal, on="PatientID",how='inner').merge(outcome, on='PatientID', how='inner')
df = df[df["PatientID"]!='BraTS-MEN-01103-000'] #This patient has a solitary fibrous tumor, exclude

#make sure there is an "Outcome" column
if 'Outcome' not in df.columns:
    raise ValueError("The DataFrame must contain an 'Outcome' column.")

X = df.drop(columns=["PatientID",'Outcome'])  # Ensure 'PatientID' is dropped if present in the features
disgnostic_cols = [col for col in X.columns if 'diagnostic' in col.lower()] #these are output by our extraction, drop them
if disgnostic_cols:
    X = X.drop(columns=disgnostic_cols)
y = df['Outcome'] #Target variable

#Function to drop correlated features
def drop_correlated_features(df, threshold):
    if threshold is None: #if none is given as threshold, return full DF
        return df
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

#Define the optuna objective function
def objective(trial, X, y):
    scaler_name = trial.suggest_categorical('scaler', ['standard', 'minmax', 'robust']) #Data scalers to select between
    scaler = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}[scaler_name] #map categorical name to actual function 

        # Feature selection method
    selector_name = trial.suggest_categorical('selector', ['f_classif', 'mutual_info', 'lasso'])
    k_option = trial.suggest_categorical('select__k', [5,10,20,30,100, 200,250, 400, 1000, 'all'])
    k_best = None if k_option == 'all' else k_option
    # Correlation threshold option 
    corr_threshold = trial.suggest_categorical('corr_thresh', [None,.2,.3,.7,.8,.9])

    # Feature selection object
    if selector_name == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k_best or 'all')
    elif selector_name == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k_best or 'all')
    elif selector_name == 'lasso':
        selector = SelectFromModel(LassoCV(cv=3, max_iter=100000, tol=1e-3, random_state=SEED))
        
    model_type = trial.suggest_categorical('model', ['xgboost', 'random_forest', 'knn', 'lightgbm','cat_boost','elastic_net','svm'])
    if model_type == 'xgboost':
        clf_params = {
            'n_estimators': trial.suggest_int('xgb__n_estimators', 20, 500, step=20),
            'max_depth': trial.suggest_int('xgb__max_depth', 10,280, step=20),
            'learning_rate': trial.suggest_float('xgb__learning_rate', 0.01, 0.9, log=True),
            'xgb__reg_alpha': trial.suggest_float('xgb__reg_alpha', 0.0, 5.0),
            'xgb__reg_lambda': trial.suggest_float('xgb__reg_lambda', 0.0, 5.0),
            'xgb__min_child_weight': trial.suggest_int('xgb__min_child_weight', 1, 20),
            'xgb__subsample': trial.suggest_float('xgb__subsample', 0.5, 1.0),
            'xgb__colsample_bytree': trial.suggest_float('xgb__colsample_bytree', 0.5, 1.0),
            'eval_metric': 'logloss',
            'random_state': SEED,
            'tree_method': 'hist',
            'device': 'cpu'
        }
        clf = XGBClassifier(**clf_params)

    elif model_type == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int('rf__n_estimators', 20, 500, step=20),
            max_depth=trial.suggest_categorical('rf__max_depth', [None, 10, 30, 50,100,400]),
            random_state=SEED
        )

    elif model_type == 'svm':
        kernel = trial.suggest_categorical('svm__kernel', ['rbf'])
        svm_params = {
            'C': trial.suggest_float('svm__C', 1e-2, 1e3, log=True),
            'kernel': kernel,
            'probability': True,
            'random_state': SEED
        }
        if kernel == 'rbf':
            svm_params['gamma'] = trial.suggest_float('svm__gamma', 1e-3, 1, log=True)
        clf = SVC(**svm_params)

    elif model_type == 'knn':
        clf = KNeighborsClassifier(
            n_neighbors=trial.suggest_int('knn__n_neighbors', 3, 50),
            metric=trial.suggest_categorical('knn__metric', ['euclidean', 'manhattan'])
        )

    elif model_type == 'elastic_net':
        clf = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            max_iter=10000,
            random_state=SEED,
            C=trial.suggest_float('en__C', 1e-3, 1e3, log=True),
            l1_ratio=trial.suggest_float('en__l1_ratio', 1e-3, 1.0, log=True)
        )

    elif model_type == 'lightgbm':
        clf = LGBMClassifier(
            n_estimators=trial.suggest_int('lgbm__n_estimators', 20,400, step=5),
            max_depth=trial.suggest_categorical('lgbm__max_depth', [3, 5, 7, 13,100,200, -1]),
            learning_rate=trial.suggest_float('lgbm__learning_rate', 0.01, 0.9, log=True),
            num_leaves=trial.suggest_int('lgbm__num_leaves', 15, 64),
            reg_alpha=trial.suggest_float('lgbm__reg_alpha', 0.0, 1.0),
            reg_lambda=trial.suggest_float('lgbm__reg_lambda', 0.0, 1.0),
            random_state=SEED,
            device_type='cpu'
        )

    elif model_type == 'cat_boost':
        clf = CatBoostClassifier(
            iterations=trial.suggest_int('cat__iterations', 100, 1000),
            depth=trial.suggest_int('cat__depth', 3, 10),
            learning_rate=trial.suggest_float('cat__learning_rate', 0.01, 0.9, log=True),
            l2_leaf_reg=trial.suggest_float('cat__l2_leaf_reg', 1e-3, 1e3, log=True),
            leaf_estimation_iterations = trial.suggest_int("leaf_estimation_iterations", 1, 10),
            colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            random_state=SEED,
            verbose=0,
            task_type='CPU',
             allow_writing_files=False,
            save_snapshot=False
        )

    pipeline = ImbPipeline([
    ('scaler', scaler),
    ('select', selector),
    ('smote', SMOTE(sampling_strategy='auto', random_state=SEED)),
    ('clf', clf)
])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs, aprcs, recalls, briers = [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]

        # Apply correlation filtering
        X_train = drop_correlated_features(X_train_raw, corr_threshold)
        

        # Align columns in X_val to match filtered X_train
        X_val = X_val_raw[X_train.columns]

        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        try:
            pipeline.fit(X_train, y_train)
        except ValueError as e:
            trial.set_user_attr("error", str(e))
            raise TrialPruned()
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, y_prob))
        aprcs.append(average_precision_score(y_val, y_prob))
        recalls.append(recall_score(y_val, y_pred))
        briers.append(brier_score_loss(y_val, y_prob))
    


    trial.set_user_attr("aprc", aprcs)
    trial.set_user_attr("recall", recalls)
    trial.set_user_attr("brier_score",briers)
    trial.set_user_attr("selected_features", X_train.columns.tolist())
    trial.set_user_attr('aucs', aucs)
    
    """"
    y_prob_all = cross_val_predict(pipeline, X, y, cv=skf, method='predict_proba')[:,1]
    fpr, tpr, _ = roc_curve(y, y_prob_all)
    trial.set_user_attr("cv_fpr", fpr.tolist())
    trial.set_user_attr("cv_tpr", tpr.tolist())
    """

    return np.mean(aucs)


def run_optuna(X, y):

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        storage=storage_path,
        sampler=TPESampler(n_startup_trials=75, seed=SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3),
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS, n_jobs=1)

    print(f"\nRun the dashboard with:\noptuna-dashboard {storage_path} {study_name}")
    print(f"Best parameters: {study.best_params}")

    roc_dir = os.path.join(study_name, "roc_data")
    os.makedirs(roc_dir, exist_ok=True)

    # also dump a summary CSV of mean AUC per trial
    import csv
    summary_csv = os.path.join(study_name, "trial_auc_summary.csv")
    with open(summary_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["trial_number", "mean_auc"] + [f"fold_{i}" for i in range(N_FOLDS)])
        for t in study.trials:
            aucs = t.user_attrs.get("aucs", [])
            mean_auc = np.mean(aucs) if aucs else None
            writer.writerow([t.number, mean_auc] + aucs)
    
    results = {
        'best_trial_number': study.best_trial.number,
        'best_value':        study.best_value,
        'best_params':       study.best_params,
        'best_user_attrs':   study.best_trial.user_attrs
    }

    with open(os.path.join(study_name, 'study_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    df = study.trials_dataframe()
    df.to_csv(os.path.join(study_name, 'all_trials.csv'), index=False)

    '''
    # save detailed ROC arrays per trial
    for t in study.trials:
        attrs = t.user_attrs
        if "cv_fpr" in attrs and "cv_tpr" in attrs:
            np.save(os.path.join(roc_dir, f"trial_{t.number}_fpr.npy"), np.array(attrs["cv_fpr"]))
            np.save(os.path.join(roc_dir, f"trial_{t.number}_tpr.npy"), np.array(attrs["cv_tpr"]))
    '''
    # Restore transformers from best params
    best_params = study.best_params
    scaler_map = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    selector_name = best_params['selector']
    scaler = scaler_map[best_params['scaler']]
    k_best = best_params['select__k']
    k_best = None if k_best == 'all' else k_best
    corr_threshold = best_params.get('corr_thresh', None)

    # Build selector
    if selector_name == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k_best or 'all')
    elif selector_name == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k_best or 'all')
    elif selector_name == 'lasso':
        selector = SelectFromModel(LassoCV(cv=3, max_iter=1000000, tol=1e-3, random_state=SEED))

    # Classifier mapping
    model_type = best_params['model']
    if model_type == 'xgboost':
        clf = XGBClassifier(
            n_estimators=best_params['xgb__n_estimators'],
            max_depth=best_params['xgb__max_depth'],
            learning_rate=best_params['xgb__learning_rate'],
            xgb__reg_alpha=best_params['xgb__reg_alpha'],
            xgb__reg_lambda=best_params['xgb__reg_lambda'],
            xgb__min_child_weight=best_params['xgb__min_child_weight'],
            xgb__subsample=best_params['xgb__subsample'],
            xgb__colsample_bytree=best_params['xgb__colsample_bytree'],
            eval_metric='logloss',
            random_state=SEED,
            tree_method='hist',
            device='cuda' if USE_GPU else 'cpu',
            use_label_encoder=False
        )
    elif model_type == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=best_params['rf__n_estimators'],
            max_depth=best_params['rf__max_depth'],
            random_state=SEED
        )
    elif model_type == 'svm':
        kernel = best_params['svm__kernel']
        svm_params = {
            'C': best_params['svm__C'],
            'kernel': kernel,
            'probability': True,
            'random_state': SEED
        }
        if kernel == 'rbf' and 'svm__gamma' in best_params:
            svm_params['gamma'] = best_params['svm__gamma']
        clf = SVC(**svm_params)
    elif model_type == 'knn':
        clf = KNeighborsClassifier(
            n_neighbors=best_params['knn__n_neighbors'],
            metric=best_params['knn__metric']
        )
    elif model_type == 'elastic_net':
        clf = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            max_iter=10000,
            random_state=SEED,
            C=best_params['en__C'],
            l1_ratio=best_params['en__l1_ratio']
        )
    elif model_type == 'lightgbm':
        clf = LGBMClassifier(
            n_estimators=best_params['lgbm__n_estimators'],
            max_depth=best_params['lgbm__max_depth'],
            learning_rate=best_params['lgbm__learning_rate'],
            num_leaves=best_params['lgbm__num_leaves'],
            reg_alpha=best_params['lgbm__reg_alpha'],
            reg_lambda=best_params['lgbm__reg_lambda'],
            random_state=SEED,
            device_type='cpu'
        )
    
    else:  # cat_boost
        clf = CatBoostClassifier(
            iterations=best_params['cat__iterations'],
            depth=best_params['cat__depth'],
            learning_rate=best_params['cat__learning_rate'],
            l2_leaf_reg=best_params['cat__l2_leaf_reg'],
            random_state=SEED,
            verbose=0,
            task_type='CPU',
             allow_writing_files=False,
            save_snapshot=False
        )

    # Final model pipeline 
    best_model = Pipeline([
        ('scaler', scaler),
        ('select', selector),
        ('clf', clf)
    ])

    X_filtered = drop_correlated_features(X, corr_threshold)
    best_model.fit(X_filtered, y)
    print(f"Selected features: {X_filtered.columns.tolist()}")
    #save the best model 
    model_path = os.path.join(study_name, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logging.info(f"Best model saved to {model_path}")
    # Save the best parameters
    params_path = os.path.join(study_name, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    logging.info(f"Best parameters saved to {params_path}")
    # Save the study results
    study_results_path = os.path.join(study_name, 'study_results.json')

    # Save the selected features in a csv
    selected_features_path = os.path.join(study_name, 'selected_features.csv')
    with open(selected_features_path, 'w') as f:
        f.write(','.join(X_filtered.columns.tolist()))
    return best_model, X_filtered.columns.tolist(), best_params

best_model, cols, params = run_optuna(X, y)