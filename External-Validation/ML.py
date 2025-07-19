#!/usr/bin/env python3
"""
Test a saved calibrated radiomics pipeline with built-in feature selection.
Uses command-line arguments (or environment variables) for all I/O paths.
"""
import os
import random
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_curve
)

# --- Reproducibility
SEED = int(os.environ.get("SEED", 7))
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def bootstrap_metric(y_true, y_score, metric_fn, n_bootstraps=1000, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    lower = np.percentile(scores, (1-ci)/2*100)
    upper = np.percentile(scores, (ci + (1-ci)/2)*100)
    return lower, upper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved radiomics pipeline"
    )
    parser.add_argument(
        "--pipeline",
        default=os.environ.get("PIPELINE_FILE", "calibrated_pipeline.pkl"),
        help="Path to the saved pipeline pickle"
    )
    parser.add_argument(
        "--features",
        default=os.environ.get("FEATURES_FILE", "model_feature_names.pkl"),
        help="Path to the feature names pickle"
    )
    parser.add_argument(
        "--X",
        required=True,
        help="Path to the input features (pickle or CSV)"
    )
    parser.add_argument(
        "--y",
        required=True,
        help="Path to the input labels (pickle or CSV)"
    )
    parser.add_argument(
        "--probs_csv",
        default=os.environ.get("PRED_OUT", "predictions.csv"),
        help="Output path for predictions CSV"
    )
    parser.add_argument(
        "--metrics",
        default=os.environ.get("METRICS_OUT", "metrics.txt"),
        help="Output path for metrics text file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load pipeline and features
    pipeline = joblib.load(args.pipeline)
    features = joblib.load(args.features)
    # Load input data
    if args.X.lower().endswith('.pkl'):
        X = pd.read_pickle(args.X)
    else:
        X = pd.read_csv(args.X)

    if args.y.lower().endswith('.pkl'):
        y_obj = pd.read_pickle(args.y)
    else:
        y_obj = pd.read_csv(args.y)

    print(features)
    # Normalize to numpy array
    if isinstance(y_obj, pd.Series):
        y = y_obj.values
    elif isinstance(y_obj, pd.DataFrame):
        if 'Outcome' in y_obj.columns:
            y = y_obj['Outcome'].values
        else:
            y = y_obj.iloc[:, 0].values
    else:
        raise ValueError(f"Unsupported label type: {type(y_obj)}")


    missing = set(features) - set(X.columns)
    if missing:
        raise ValueError(f"Missing test columns: {missing}")
    X = X[features]

    # Predict probabilities and classes
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    print("PROBS")
    print(y_prob)
    print("pred")
    print(y_pred)
    # Compute metrics
    auc = roc_auc_score(y, y_prob)
    auc_lo, auc_hi = bootstrap_metric(y, y_prob, roc_auc_score)
    ap, ap_lo, ap_hi = (
        average_precision_score(y, y_prob),
        *bootstrap_metric(y, y_prob, average_precision_score)
    )
    brier = brier_score_loss(y, y_prob)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    rec = recall_score(y, y_pred)
    ppv = precision_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    fpr, tpr, thr = roc_curve(y, y_prob)
    j_idx = np.argmax(tpr - fpr)
    youden_thr = thr[j_idx]
    sens_j = tpr[j_idx]
    spec_j = 1 - fpr[j_idx]
    ppv_j = precision_score(y, (y_prob >= youden_thr).astype(int))

    idx85 = np.where(tpr >= 0.85)[0]
    idx85 = idx85[0] if len(idx85) else np.argmax(tpr)
    thr85 = thr[idx85]
    sens85 = tpr[idx85]
    spec85 = 1 - fpr[idx85]
    ppv85 = precision_score(y, (y_prob >= thr85).astype(int))

    # Write metrics
    with open(args.metrics, 'w') as mf:
        mf.write(f"AUROC: {auc:.3f} (95% CI {auc_lo:.3f}-{auc_hi:.3f})\n")
        mf.write(f"AUPRC: {ap:.3f} (95% CI {ap_lo:.3f}-{ap_hi:.3f})\n")
        mf.write(f"Brier: {brier:.3f}\n")
        mf.write(f"Accuracy: {acc:.3f}\nF1: {f1:.3f}\nRecall: {rec:.3f}\n")
        mf.write(f"PPV: {ppv:.3f}\nNPV: {npv:.3f}\nSpec: {spec:.3f}\n")
        mf.write(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
        mf.write(
            f"Youden thr: {youden_thr:.3f} "
            f"(Sens {sens_j:.3f}, Spec {spec_j:.3f}, PPV {ppv_j:.3f})\n"
        )
        mf.write(
            f"85% Sens thr: {thr85:.3f} "
            f"(Sens {sens85:.3f}, Spec {spec85:.3f}, PPV {ppv85:.3f})\n"
        )
    print(f"Metrics saved to {args.metrics}")

    # Save predictions
    dfp = pd.DataFrame({'y_true': y, 'y_pred': y_pred, 'y_prob': y_prob})
    dfp.to_csv(args.probs_csv, index=False)
    print(f"Predictions saved to {args.probs_csv}")


if __name__ == '__main__':
    main()
