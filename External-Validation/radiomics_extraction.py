import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import logging
from radiomics import featureextractor

# Disable PyRadiomics warnings
os.environ["SITK_SHOW_WARNINGS"] = "0"
sitk.ProcessObject_SetGlobalWarningDisplay(False)

# Set up logging
logging.basicConfig(filename='radiomics_log.txt', 
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Define imaging sequences used
SEQUENCES = ['T1C', 'T1N', 'T2F', 'T2W']

# Initialize PyRadiomics extractor
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings['normalize'] = True

def get_patient_files(patient_folder):
    files = os.listdir(patient_folder)
    seq_paths = {
        seq: next((os.path.join(patient_folder, f) 
                   for f in files if seq.lower() in f.lower() and f.endswith('.nii.gz')), None)
        for seq in SEQUENCES
    }
    mask_path = next((os.path.join(patient_folder, f) 
                      for f in files if '-seg' in f.lower() and f.endswith('.nii.gz')), None)

    return seq_paths | {'mask': mask_path}

def extract_patient_features(patient_id, patient_files):
    features_list = []
    mask_path = patient_files.get('mask')
    if not mask_path:
        logging.warning(f"No mask found for {patient_id}")
        return features_list

    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    region_labels = [int(lbl) for lbl in np.unique(mask_array) if lbl != 0]
    if not region_labels:
        logging.warning(f"No labeled regions found in mask for {patient_id}")
        return features_list

    for seq, image_path in patient_files.items():
        if seq == 'mask' or not image_path:
            continue
        for region in region_labels:
            try:
                features = extractor.execute(image_path, mask_path, label=region)
                features_list.append({
                    'PatientID': patient_id,
                    'Sequence': seq,
                    'RegionLabel': region,
                    **features
                })
            except Exception as e:
                logging.warning(f"Error extracting {patient_id} | {seq} | Region {region}: {e}")
    return features_list

def process_all_patients(data_dir, outcome_csv):
    outcome_df = pd.read_csv(outcome_csv)
    valid_patient_ids = set(outcome_df['PatientID'].astype(str))

    all_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    patient_folders = [os.path.join(data_dir, d) for d in all_folders if d in valid_patient_ids]

    print(f"Found {len(patient_folders)} valid patient folders.")

    all_features = []

    for folder in patient_folders:
        patient_id = os.path.basename(folder)
        patient_files = get_patient_files(folder)
        features = extract_patient_features(patient_id, patient_files)
        all_features.extend(features)

    return pd.DataFrame(all_features)

def save_features_to_csv(features_df, output_csv='X.csv'):
    if not features_df.empty:
        features_df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
    else:
        print("No features extracted.")
