#!/usr/bin/env python3

import os
import argparse
import random
import logging
import numpy as np
import SimpleITK as sitk
import gudhi as gd
from persim import PersImage
import pickle
import pandas as pd

# Constants
SEQUENCES = ['T1C', 'T1N', 'T2F', 'T2W']
PIM = PersImage(pixels=[20, 20], spread=1.0)
seed = 7
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["SITK_SHOW_WARNINGS"] = "0"
sitk.ProcessObject_SetGlobalWarningDisplay(False)
random.seed(seed)
np.random.seed(seed)


def to_float_list(x):
    # If it’s already a sequence, just cast to list
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    # Otherwise assume it’s a string like "[0.1,0.2,...]"
    s = str(x).strip('[]')
    if not s:
        return []
    return list(np.fromstring(s, sep=','))


def get_patient_files(patient_folder):
    """
    Return a dict mapping each MRI sequence and the segmentation mask to their file paths.
    """
    files = os.listdir(patient_folder)
    seq_dict = {
        seq: next(
            (os.path.join(patient_folder, f)
             for f in files if seq.lower() in f.lower() and f.endswith('.nii.gz')),
            None
        )
        for seq in SEQUENCES
    }
    seg_dict = {
        'mask': next(
            (os.path.join(patient_folder, f)
             for f in files if '-seg' in f.lower() and f.endswith('.nii.gz')),
            None
        )
    }
    return {**seq_dict, **seg_dict}

def extract_topology(data_dir: str, outcome_csv: str, output_pkl="outputs/topo.pkl",output_csv='outputs/topo.csv'):
    outcomes_df = pd.read_csv(outcome_csv)
    if 'PatientID' in outcomes_df.columns:
        valid_patients = set(outcomes_df['PatientID'].astype(str).tolist())
    else:
        valid_patients = set(outcomes_df.iloc[:, 0].astype(str).tolist())
    print("Found", len(valid_patients), "valid patient IDs in outcomes.csv.")

    # Extraction
    vectorized_features = {}
    pim = PersImage(pixels=[20, 20], spread=1.0)
    patient_counter = 0

    for subfolder in next(os.walk(data_dir))[1]:
        if subfolder not in valid_patients:
            print(f"Skipping patient {subfolder} as it is not in outcomes.csv.")
            continue

        subfolder_path = os.path.join(data_dir, subfolder)
        patient_files = get_patient_files(subfolder_path)
        patient_images = {}

        # load and normalize
        for modality, file_path in patient_files.items():
            if file_path is None:
                print(f"{modality} file not found in {subfolder}")
                continue
            try:
                img = sitk.ReadImage(file_path)
                img_array = sitk.GetArrayFromImage(img).astype(np.float32)
                min_val, max_val = img_array.min(), img_array.max()
                if max_val > min_val:
                    img_array = (img_array - min_val) / (max_val - min_val)
                patient_images[modality] = img_array
            except Exception as e:
                print(f"Error loading {modality} for {subfolder}: {e}")

        vectorized_features[subfolder] = {}
        if 'mask' in patient_images:
            mask = patient_images['mask']
            binary_mask = (mask > 0).astype(np.float32)

            for seq in SEQUENCES:
                if seq in patient_images:
                    masked_image = patient_images[seq] * binary_mask
                    cc = gd.CubicalComplex(top_dimensional_cells=masked_image)
                    pers_pairs = cc.persistence()
                    diag = np.array([
                        [p[1][0], p[1][1]] if p[1][1] != float('inf') else [p[1][0], -1]
                        for p in pers_pairs
                    ])
                    if diag.size == 0:
                        vec = np.zeros(pim.pixels[0] * pim.pixels[1])
                    else:
                        vec = pim.transform(diag).flatten()
                    vectorized_features[subfolder][seq] = vec
                    print(f"Patient {subfolder}, sequence {seq}: Vectorized feature shape: {vec.shape}")
                else:
                    print(f"{seq} image not found for patient {subfolder}")
        else:
            print(f"Segmentation mask not found for patient {subfolder}")

        patient_counter += 1
        print(f"Processed {patient_counter} valid patient(s) so far.\n")

    print(f"Processed data for {patient_counter} patients. Final vectorized features saved.")
    with open(output_pkl, 'wb') as f:
        pickle.dump(vectorized_features, f)

    rows = []
    for patient_id, modalities in vectorized_features.items():
        row = {"PatientID": patient_id}
        for modality in ['T1C', 'T1N', 'T2F', 'T2W']:
            if modality in modalities:
                row[modality] = modalities[modality].tolist()  # or .tolist() to keep arrays clean
            else:
                row[modality] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    # Convert string representations of arrays into actual lists
    df2 = pd.DataFrame()
    for col in ['T1C', 'T1N', 'T2F', 'T2W']:
        df2[col] = df[col].apply(to_float_list)
    expanded_cols = []
    for col in ['T1C', 'T1N', 'T2F', 'T2W']:
        expanded = pd.DataFrame(df2[col].to_list(), index=df2.index)
        expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
        expanded_cols.append(expanded)

    # Concatenate original patient/outcome data with the expanded features
    df_final = pd.concat([df[['PatientID']], *expanded_cols], axis=1)
    df_final.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract topological features")
    parser.add_argument("--data-dir",    required=True, help="Path to patient folders")
    parser.add_argument("--outcome-csv", required=True, help="CSV listing patient IDs")
    parser.add_argument("--output-pkl",  default="topo.pkl", help="Where to write .pkl")
    parser.add_argument("--output-csv",  default="topo.csv", help="Where to write expanded CSV")
    args = parser.parse_args()

    df = extract_topology(
        data_dir   = args.data_dir,
        outcome_csv= args.outcome_csv,
        output_pkl = args.output_pkl,
        output_csv = args.output_csv
    )
    print(f"Done. Wrote {len(df)} rows to {args.output_csv}")
