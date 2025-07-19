# Extraction-Radiomics.py
# Extract radiomics features from tumor segmentations using PyRadiomics

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import datetime
import time
import pickle
import json
import SimpleITK as sitk
os.environ["SITK_SHOW_WARNINGS"] = "0"
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import scipy.stats as stats
import logging
import multiprocessing
from radiomics import featureextractor
from tqdm import tqdm
import sys
import contextlib

# Logging setup
logging.basicConfig(filename='experiment_log.txt', 
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logger = logging.getLogger("radiomics")
logger.setLevel(logging.INFO)

# RANDOM setup
seed_value = 7
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
sklearn_random_state = seed_value

# Paths
os.chdir(r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data")
DATA_DIR = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\BraTS-MEN-Train"
OUTCOME_CSV = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\outcomes.csv"
SEQUENCES = ['T1C', 'T1N', 'T2F', 'T2W']

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings['normalize'] = True


def extract_patient_features_worker(patient_id, patient_files, return_dict):
    try:
        features = extract_patient_features(patient_id, patient_files)
        return_dict['features'] = features
    except Exception as e:
        return_dict['error'] = str(e)

def get_patient_files(patient_folder):
    files = os.listdir(patient_folder)
    seq_paths = {seq: next((os.path.join(patient_folder, f) for f in files if seq.lower() in f.lower() and f.endswith('.nii.gz')), None) for seq in SEQUENCES}
    mask_path = next((os.path.join(patient_folder, f) for f in files if '-seg' in f.lower() and f.endswith('.nii.gz')), None)

    if mask_path is None:
        logging.warning(f"No mask found in {patient_folder}")
    for seq, path in seq_paths.items():
        if path is None:
            logging.warning(f"Missing {seq} for {os.path.basename(patient_folder)}")

    return seq_paths | {'mask': mask_path}

def extract_patient_features_with_timeout(patient_id, patient_files, timeout=10000, skipped_patients=None):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process = multiprocessing.Process(
        target=extract_patient_features_worker,
        args=(patient_id, patient_files, return_dict)
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        logging.warning(f"Timeout: Skipping patient {patient_id} after {timeout} seconds.")
        if skipped_patients is not None:
            skipped_patients.append((patient_id, "Timeout"))
        return []
    if 'error' in return_dict:
        logging.warning(f"Error extracting {patient_id}: {return_dict['error']}")
        if skipped_patients is not None:
            skipped_patients.append((patient_id, return_dict['error']))
        return []
    return return_dict.get('features', [])

def extract_patient_features(patient_id, patient_files):
    features_list = []
    mask_path = patient_files.get('mask')
    if not mask_path:
        return features_list
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    region_labels = [int(lbl) for lbl in np.unique(mask_array) if lbl != 0]
    if not region_labels:
        logging.warning(f"No nonzero labels in mask for {patient_id}")
        return features_list
    for seq, image_path in patient_files.items():
        if seq == 'mask' or not image_path:
            continue
        for region in region_labels:
            try:
                feat_dict = extractor.execute(image_path, mask_path, label=region)
                features_list.append({'PatientID': patient_id, 'Sequence': seq, 'RegionLabel': region, **feat_dict})
            except Exception as e:
                logging.warning(f"Error in {patient_id}, {seq}, Region {region}: {e}")
    return features_list

def process_all_patients(timeout=5000):
    outcome_df = pd.read_csv(OUTCOME_CSV)
    valid_patient_ids = set(outcome_df['PatientID'].astype(str))

    all_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    missing_from_csv = [d for d in all_folders if d not in valid_patient_ids]
    if missing_from_csv:
        print(f"{len(missing_from_csv)} folders were skipped because they were not in the CSV: {missing_from_csv[:5]}...")

    patient_folders = [os.path.join(DATA_DIR, d) for d in all_folders if d in valid_patient_ids]
    print(f"Processing {len(patient_folders)} patients...")

    all_features, skipped_patients = [], []
    start_all = time.time()

    for folder in tqdm(patient_folders, desc="Extracting features for patients"):
        patient_id = os.path.basename(folder)
        patient_files = get_patient_files(folder)
        start_folder = time.time()
        features = extract_patient_features_with_timeout(patient_id, patient_files, timeout=timeout, skipped_patients=skipped_patients)
        elapsed_folder = time.time() - start_folder
        tqdm.write(f"{patient_id}: {len(features)} features extracted in {elapsed_folder:.2f} seconds ({'skipped' if not features else 'ok'})")
        all_features.extend(features)
        gc.collect()
        total_elapsed = time.time() - start_all
        print(f"Feature extraction completed in {total_elapsed:.2f} seconds")


    if skipped_patients:
        with open("skipped_patients.txt", "w") as f:
            for pid, reason in skipped_patients:
                f.write(f"{pid}: {reason}\n")
        print(f"Skipped {len(skipped_patients)} patients. See 'skipped_patients.txt'.")

    if not all_features:
        print("No features extracted. Please check logs and skipped_patients.txt.")

    return pd.DataFrame(all_features)

def save_features_to_csv(features_df, output_csv='extracted_features-skipped-2.csv'):
    if not features_df.empty:
        features_df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
    else:
        print("Warning: No features to save.")

def main():
    features_df = process_all_patients()
    save_features_to_csv(features_df)
    logging.info("Feature extraction completed.")

if __name__ == '__main__':
    main()
