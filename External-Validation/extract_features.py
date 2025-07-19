# extract_features.py
import argparse
from radiomics_extraction import process_all_patients, save_features_to_csv
import pandas as pd

def preprocess(features_df: pd.DataFrame):
    # build a unique feature label per sequence+region
    features_df['FeatureLabel'] = (
        features_df['Sequence']
        + '_R'
        + features_df['RegionLabel'].astype(str)
    )

    # pivot so each PatientID is one row
    feature_cols = [
        c for c in features_df.columns
        if c not in ['PatientID', 'Sequence', 'RegionLabel']
    ]
    pivot = features_df.pivot(
        index='PatientID',
        columns='FeatureLabel',
        values=feature_cols
    )
    # flatten the MultiIndex columns
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns]
    df = pivot.reset_index()


    # drop any columns containing 'diagnostic'
    diagnostic_cols = [c for c in df.columns if 'diagnostic' in c.lower()]
    df = df.drop(columns=diagnostic_cols)

    # fill missing with zero
    df = df.fillna(0)


    return df


parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help="Folder containing patient folders")
parser.add_argument('--age', required=True, help="Path to outcomes CSV")
parser.add_argument('--outcome', default='outcomes.csv', help="Path to outcomes CSV")
parser.add_argument('--output', default='features.csv', help="Output CSV path")
args = parser.parse_args()

age = pd.read_csv(args.age)
age['Grade'] = pd.to_numeric(age['Grade'],errors='coerce')
age['Outcome'] = age["Grade"].apply(lambda x: 1 if x in [2,3] else 0)
outcome = age[['BraTS ID', "Outcome"]]
outcome = outcome.rename(columns={"BraTS ID":"PatientID"})
outcome.to_csv(args.outcome,index=False)

features_df = process_all_patients(data_dir=args.input, outcome_csv=args.outcome)
save_features_to_csv(features_df, output_csv=args.output)
x = preprocess(features_df)
x.to_csv("X.csv",index=False)