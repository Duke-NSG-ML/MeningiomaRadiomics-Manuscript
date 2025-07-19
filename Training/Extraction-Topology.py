# Extraction-Topology.py
# Extract persistence diagram features from tumor segmentations using Gudhi and Persim
# Based on ToplogicalTumor repo as written by Chris Tralie

#Import libraries
import os
import numpy as np
import SimpleITK as sitk
os.environ["SITK_SHOW_WARNINGS"] = "0"
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import gudhi as gd
from persim import PersImage
import pickle
import pandas as pd
import random 
import logging

# Logging setup
logging.basicConfig(filename='topology_extraction.txt', 
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logger = logging.getLogger("radiomics")
logger.setLevel(logging.INFO)

# Random setup
os.environ["PYTHONHASHSEED"] = str(7)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
seed_value = 7
random.seed(seed_value)
np.random.seed(seed_value)
sklearn_random_state = seed_value

# Paths
os.chdir(r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data")
DATA_DIR = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\BraTS-MEN-Train"
OUTCOME_CSV = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\outcomes.csv"
SEQUENCES = ['T1C', 'T1N', 'T2F', 'T2W']

outcomes_df = pd.read_csv(OUTCOME_CSV)

if 'PatientID' in outcomes_df.columns:
    valid_patients = set(outcomes_df['PatientID'].astype(str).tolist())
else:
    valid_patients = set(outcomes_df.iloc[:, 0].astype(str).tolist())
print("Found", len(valid_patients), "valid patient IDs in outcomes.csv.")

all_patients = [
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d)) and d in valid_patients
]

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
# dict to store vectorized 
vectorized_features = {}

# set up the persistence image vectorizer
pim = PersImage(pixels=[20, 20], spread=1.0)

# initialize a patient counter.
patient_counter = 0

# iterate over all patient subfolders
for subfolder in next(os.walk(DATA_DIR))[1]:

    if subfolder not in valid_patients:
        print(f"Skipping patient {subfolder} as it is not in outcomes.csv.")
        continue

    subfolder_path = os.path.join(DATA_DIR, subfolder)
    patient_files = get_patient_files(subfolder_path)

    patient_images = {}

    # load T1C, T1N, T2F, T2W, mask
    for modality, file_path in patient_files.items():
      if file_path is None:
          print(f"{modality} file not found in {subfolder}")
          continue

      try:
          img = sitk.ReadImage(file_path)
          img_array = sitk.GetArrayFromImage(img).astype(np.float32)

          # norm
          min_val = np.min(img_array)
          max_val = np.max(img_array)
          if max_val > min_val:
              img_array = (img_array - min_val) / (max_val - min_val)

          patient_images[modality] = img_array
      except Exception as e:
          print(f"Error loading {modality} for {subfolder}: {e}")

    vectorized_features[subfolder] = {}

    if 'mask' in patient_images:
        mask = patient_images['mask']
        # create mask
        binary_mask = (mask > 0).astype(np.float32)

        # apply mask
        for seq in SEQUENCES:
            if seq in patient_images:
                masked_image = patient_images[seq] * binary_mask

                # cubical complex
                cc = gd.CubicalComplex(top_dimensional_cells=masked_image)
                pers_pairs = cc.persistence()

                # birth death pairs array
                diag = np.array([
                    [pair[1][0], pair[1][1]] if pair[1][1] != float('inf')
                    else [pair[1][0], -1]
                    for pair in pers_pairs
                ])

                # vectorize persistence diagram
                if diag.size == 0:
                    vec = np.zeros(pim.pixels[0] * pim.pixels[1])
                else:
                    pim_img = pim.transform(diag)
                    vec = pim_img.flatten()

                vectorized_features[subfolder][seq] = vec
                print(f"Patient {subfolder}, sequence {seq}: Vectorized feature shape: {vec.shape}")
            else:
                print(f"{seq} image not found for patient {subfolder}")
    else:
        print(f"Segmentation mask not found for patient {subfolder}")

    patient_counter += 1
    print(f"Processed {patient_counter} valid patient(s) so far.\n")

    if patient_counter % 10 == 0:
        with open('final-vectorized_features.pkl', 'wb') as f:
            pickle.dump(vectorized_features, f)
        print("Saved vectorized features to file after processing", patient_counter, "patients.")
with open('final-vectorized_features.pkl', 'wb') as f:
    pickle.dump(vectorized_features, f)
print("Processed data for", patient_counter, "patients. Final vectorized features saved.")


rows = []
for patient_id, modalities in vectorized_features.items():
    row = {"PatientID": patient_id}
    for modality in ['T1C', 'T1N', 'T2F', 'T2W']:
        if modality in modalities:
            row[modality] = modalities[modality].tolist()  
        else:
            row[modality] = None
    rows.append(row)

# Create DataFrame and optionally merge with topology.csv for Outcome
df = pd.DataFrame(rows)
df.to_csv('tjz_vectorized_features-final.csv', index=False)
# %%
data = pd.read_pickle('final-vectorized_features.pkl')
# %%
rows = []
for patient_id, modalities in data.items():
    row = {"PatientID": patient_id}
    for modality in ['T1C', 'T1N', 'T2F', 'T2W']:
        if modality in modalities:
            row[modality] = modalities[modality].tolist()  
        else:
            row[modality] = None
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv('tjz_vectorized_features-final.csv', index=False)
# %%
def preprocess2(df):
    # Expand each list into its own set of columns
    
        # Convert string representations of arrays into actual lists
    for col in ['T1C', 'T1N', 'T2F', 'T2W']:
        df[col] = df[col].apply(lambda x: list(np.fromstring(x.strip('[]'), sep=' ')))
    expanded_cols = []
    for col in ['T1C', 'T1N', 'T2F', 'T2W']:
        expanded = pd.DataFrame(df[col].to_list(), index=df.index)
        expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
        expanded_cols.append(expanded)

    # Concatenate original patient/outcome data with the expanded features
    df_final = pd.concat([df[['PatientID']], *expanded_cols], axis=1)
    return df_final

df = pd.read_csv(r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\simple_tda_features.csv")
s = preprocess2(df)
s.to_csv("finaltopo.csv",index=False)
