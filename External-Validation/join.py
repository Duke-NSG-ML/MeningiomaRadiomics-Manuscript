#!/usr/bin/env python3
import os, argparse, random, pickle, numpy as np, pandas as pd
seed = 7
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)

def preprocess(features_df: pd.DataFrame):
    # build a unique feature label per sequence+region
    print(features_df)
    features_df['FeatureLabel'] = features_df['Sequence'] + '_R' + features_df['RegionLabel'].astype(str)
    pivot = features_df.pivot(index='PatientID', columns='FeatureLabel', values=[
        col for col in features_df.columns if col not in ['PatientID', 'Sequence', 'RegionLabel']
    ])
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns]
    pivot = pivot.reset_index()
    df = pivot
    print(df)
    # drop any columns containing 'diagnostic'
    diagnostic_cols = [c for c in df.columns if 'diagnostic' in c.lower()]
    df = df.drop(columns=diagnostic_cols)

    # fill missing with zero
    df = df.fillna(0)


    return df

# --- Main script ---
def main():
    parser = argparse.ArgumentParser(
        description="Merge feature, topology, age, and outcome data into X/y datasets"
    )
    parser.add_argument(
        "--feature_csv", required=True,
        help="Path to pivoted radiomics features CSV"
    )
    parser.add_argument(
        "--topo_csv", required=True,
        help="Path to topology CSV"
    )
    parser.add_argument(
        "--age_csv", required=True,
        help="Path to age/sex CSV"
    )
    parser.add_argument(
        "--outcome_csv", required=True,
        help="Path to outcomes CSV"
    )
    parser.add_argument(
        "--x_out", default="X.pkl",
        help="Output pickle path for X"
    )
    parser.add_argument(
        "--y_out", default="y.pkl",
        help="Output pickle path for y"
    )
    parser.add_argument(
        "--x_csv", default="X.csv",
        help="Output CSV path for X"
    )
    parser.add_argument(
        "--y_csv", default="y.csv",
        help="Output CSV path for y"
    )
    args = parser.parse_args()

    # Load data
    features = pd.read_csv(args.feature_csv)
    topo = pd.read_csv(args.topo_csv)
    age = pd.read_csv(args.age_csv)
    #outcome = pd.read_csv(args.outcome_csv)
    age['Grade'] = pd.to_numeric(age['Grade'],errors='coerce')
    age['Outcome'] = age["Grade"].apply(lambda x: 1 if x in [2,3] else 0)
    outcome = age[['BraTS ID', "Outcome"]]
    outcome = outcome.rename(columns={"BraTS ID":"PatientID"})
    
    features = preprocess(features)
    # Check for required columns
    if 'PatientID' not in features.columns:
        raise ValueError("feature_csv must include 'PatientID' column")
    if 'PatientID' not in topo.columns:
        raise ValueError("topo_csv must include 'PatientID' column")
    if 'BraTS ID' in age.columns:
        age = age.rename(columns={'BraTS ID': 'PatientID'})
    if 'PatientID' not in age.columns:
        raise ValueError("age_csv must include 'PatientID' column or 'BraTS ID'")
    if 'PatientID' not in outcome.columns or 'Outcome' not in outcome.columns:
        raise ValueError("outcome_csv must include 'PatientID' and 'Outcome' columns")

    # Prepare age/sex
    df_age = age[['PatientID', 'Age', 'Sex']].copy()
    df_age['Sex'] = df_age['Sex'].map({'Male': 1, 'Female': 0})
    df_age['Age'] = df_age['Age'].fillna(df_age['Age'].median())

    # Drop diagnostic columns if present
    diag_cols = [c for c in features.columns if 'diagnostic' in c.lower()]
    if diag_cols:
        features = features.drop(columns=diag_cols)

    # Merge all on PatientID
    df = (
        df_age
        .merge(topo, on='PatientID', how='inner')
        .merge(features, on='PatientID', how='inner')
        .merge(outcome, on='PatientID', how='inner')
    )

    # Separate X and y
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Save pickles
    with open(args.x_out, 'wb') as fx:
        pickle.dump(X, fx)
    with open(args.y_out, 'wb') as fy:
        pickle.dump(y, fy)

    # also save as CSV
    X.to_csv(args.x_csv, index=False)
    y.to_csv(args.y_csv, index=False)
    print(len(X))
    print(X.head())
    print(y)
    print(f"✔ Joined data; wrote {args.x_out} & {args.y_out}")

if __name__ == "__main__":
    main()
