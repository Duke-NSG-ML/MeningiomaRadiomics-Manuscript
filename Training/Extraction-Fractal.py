# Extraction-Fractal.py
# Extract fractal dimension features from tumor segmentations
# fractal dimension as shown by Chris Tralie in ToplogicalTumors repo.

import os
import random
import numpy as np
import pandas as pd
import logging
from scipy.spatial import KDTree
from scipy.stats import linregress
import SimpleITK as sitk
import pickle
import nibabel as nib
# Reproducibility & threading limits
seed_value = 7
os.environ["PYTHONHASHSEED"] = str(seed_value)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
random.seed(seed_value)
np.random.seed(seed_value)

logging.basicConfig(filename='fractal_extraction.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fractal_extraction")

DATA_DIR = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\BraTS-MEN-Train"
OUTCOME_CSV = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\outcomes.csv"


def binary_volume_2coords(B):
    """
    Extract voxel indices of mask==1 as an (N,3) array of [z,y,x].
    """
    return np.array(np.where(B == 1), dtype=float).T


def get_h0_total(X, alpha):
    """
    Compute the total persistence in H0 by quickly computing the 
    MST as a subset of the Delaunay complex
    
    Parameters
    ----------
    X: ndarray(N, d)
        Point cloud
    alpha: float
        Power to weight each edge length
    
    Returns
    -------
    float: Power weighted total persistence sum
    """
    from scipy import sparse
    from scipy.spatial import Delaunay
    from itertools import combinations

    N = X.shape[0]
    simplices = Delaunay(X).simplices
    I = np.array([])
    J = np.array([])
    V = np.array([])
    counts = np.array([])

    ## Step 1: Compute distances of all edges in the Delaunay complex
    for (i, j) in combinations(range(simplices.shape[1]), 2):
        Ik = simplices[:, i]
        Jk = simplices[:, j]
        Vk = np.sqrt(np.sum((X[Ik, :] - X[Jk, :])**2, axis=1))
        I = np.concatenate((I, Ik))
        J = np.concatenate((J, Jk))
        V = np.concatenate((V, Vk))
        counts = np.concatenate((counts, np.ones(Ik.size)))
    Ds = sparse.coo_matrix((V, (I, J)), shape=(N, N))
    counts = sparse.coo_matrix((counts, (I, J)), shape=(N, N))
    Ds = Ds/counts
    
    ## Step 2: Compute the minimum spanning tree to get H0
    tree = sparse.csgraph.minimum_spanning_tree(Ds)
    ds = np.array(tree[tree.nonzero()]).flatten()
    return np.sum(ds**alpha)
    
def get_h0_fractaldim(X, alpha=0.5, samples=100):
    """
    Compute the H0-based fractal dimension, according to the procedure in [1]
    
    [1] Fractal Dimension Estimation with Persistent Homology: A Comparative Study
    Jonathan Jaquette and Benjamin Schweinhart
    
    Parameters
    ----------
    X: ndarray(N, d)
        Point cloud
    alpha: float
        Power to weight each edge length
    
    Returns
    -------
    float: Fractal dimension estimate
    """
    from scipy.stats import linregress
    N = X.shape[0]
    Ns = 10**np.linspace(np.log10(N/10), np.log10(N), 100)
    Ns = np.array(np.floor(Ns), dtype=int)
    Es = []
    for n in Ns:
        Xn = X[np.random.permutation(N)[0:n], :]
        En = get_h0_total(Xn, alpha)
        Es.append(En)
    beta = linregress(np.log(Ns), np.log(Es)).slope
    return alpha/(1-beta)


def get_patient_files(patient_folder):
    """
    Find mask file in folder; returns mask_path or None.
    """
    for f in os.listdir(patient_folder):
        if '-seg' in f.lower() and f.endswith('.nii.gz'):
            return os.path.join(patient_folder, f)
    return None


def find_dist_thresh(X, n_target, tree=None, eps_initial=1e-3, n_iters=10, verbose=False):
    """
    Find the distance threshold at which n edges exist, using
    Golden sections search
    
    Parameters
    ----------
    X: ndarray(N, d)
        Point cloud
    n_target: int
        Target number of points
    tree: scipy.spatial.KDTree
        KDTree, if precomputed
    eps_initial: float
        Epsilon at which to start searching for interval endpoints
    n_inters: int
        Number of iterations of golden sections
    verbose: bool
        If true, print golden section search iterations
    
    Returns
    -------
    float: Epsilon value that gets close to n_target
    """
    from scipy.spatial import KDTree
    N = X.shape[0]
    if not tree:
        tree = KDTree(X)
    get_n = lambda eps: np.sum(tree.query_ball_point(X, eps, return_length=True))-N
    
    ## Step 1: Figure out interval endpoints
    a = eps_initial
    n_est = get_n(a)
    while n_est > n_target:
        a /= 2
        n_est = get_n(a)
    b = a
    while n_est < n_target:
        b *= 2
        n_est = get_n(b)
    
    ## Step 2: Run golden sections search
    gr = (np.sqrt(5) + 1)/2
    f = lambda eps: np.abs(get_n(eps)-n_target)
    c = b - (b - a)/gr
    d = a + (b - a)/gr
    for i in range(n_iters):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a)/gr
        d = a + (b - a)/gr
        eps = (b + a)/2
        if verbose:
            print(eps, get_n(eps))
        if get_n(eps) == n_target:
            break
    return eps
    
def get_correlation_dimension(X, n_samples=100):
    """
    Compute the correlation dimension for fractal estimation
    
    Parameters
    ----------
    X: ndarray(N, d)
        Point cloud
    n_samples: int
        Number of distance thresholds to take in the estimation
    """
    from scipy.stats import linregress
    from scipy.spatial import KDTree
    N = X.shape[0]
    tree = KDTree(X)
    n1 = N**0.75
    n2 = 50*N

    eps1 = find_dist_thresh(X, n1, tree=tree)
    eps2 = find_dist_thresh(X, n2, tree=tree)
    all_eps = 2**np.linspace(np.log2(eps1), np.log2(eps2), n_samples)
    
    
    get_n = lambda eps: np.sum(tree.query_ball_point(X, eps, return_length=True))-N
    C = [get_n(eps) for eps in all_eps]    

    return linregress(np.log(all_eps), np.log(C)).slope

def process_all_patients(max_points=2000):
    """
    Iterate patients, compute 3D shape correlation dimension.
    Returns DataFrame with PatientID, corr_dim_shape.
    """
    outcomes = pd.read_csv(OUTCOME_CSV)
    valid = set(outcomes['PatientID'].astype(str)) if 'PatientID' in outcomes.columns else set(outcomes.iloc[:,0].astype(str))
    results = []
    for pid in sorted(os.listdir(DATA_DIR)):
        if pid not in valid:
            continue
        folder = os.path.join(DATA_DIR, pid)
        if not os.path.isdir(folder):
            continue
        mask_path = get_patient_files(folder)
        if mask_path is None:
            logger.warning(f"{pid}: no mask, skipping")
            continue
        mask_img = sitk.ReadImage(mask_path)
        mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
        # extract voxel indices [z,y,x]
        coords_zyx = binary_volume_2coords(mask_arr)
        # reorder to [x,y,z] and convert to mm
        coords_vox = coords_zyx[:, [2,1,0]]
        spacing = mask_img.GetSpacing()
        coords_mm = coords_vox * np.array(spacing)
        # subsample
        if coords_mm.shape[0] > max_points:
            idx = np.random.choice(coords_mm.shape[0], max_points, replace=False)
            coords_mm = coords_mm[idx]
        # compute fractal dimension
        print("Starting extraction for {:}".format(pid))
        dim = get_correlation_dimension(coords_mm)
        logger.info(f"{pid} corr_dim_shape={dim:.3f}")
        results.append({'PatientID': pid, 'corr_dim_shape': dim})
    return pd.DataFrame(results)



def load_dictionary(metadata_path):
    """
    Read a CSV whose first column is PatientID (or has a column literally named "PatientID"),
    then return a dict: { patient_id_str : pandas.Series(row_i) }.
    This mirrors PrecomputeAllFeatures.ipynb’s load_dictionary exactly.

    - metadata_path: path to outcomes.csv (or your metadata CSV).
    - Returns: a dict of { patient_id (string) : pd.Series(...) } for each row in the CSV.
    """
    df = pd.read_csv(metadata_path)
    if 'PatientID' in df.columns:
        df['PatientID'] = df['PatientID'].astype(str)
        data = { row['PatientID']: row for _, row in df.iterrows() }
    else:
        first_col = df.columns[0]
        df[first_col] = df[first_col].astype(str)
        data = { row[first_col]: row for _, row in df.iterrows() }
    return data


METADATA_CSV     = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\outcomes.csv"
ALL_DATA_BASE_DIR = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\BraTS-MEN-Train"

# Keep your sequence list the same:
SEQUENCES = ['T1C', 'T1N', 'T2F', 'T2W']

# Where to write per-patient pickles (one file per patient ID):
TOPO_OUTPUT_DIR = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\preprocessed\topology"

# Tumor‐type labels in your segmentation
#   1 → Edema, 2 → Main Tumor, 3 → Necrotic.
tumor_types = ["Edema", "Main Tumor", "Necrotic"]



def get_patient_files(patient_folder):
    """
    Return a dict mapping each MRI sequence and the segmentation mask to their file paths,
    by looking for filenames that contain the lowercase seq string, plus '-seg' for the mask.
    If not found, that key’s value will be None.
    """
    files = os.listdir(patient_folder)
    seq_dict = {
        seq: next(
            (os.path.join(patient_folder, f)
             for f in files if (seq.lower() in f.lower() and f.endswith('.nii.gz'))),
            None
        )
        for seq in SEQUENCES
    }
    seg_dict = {
        'mask': next(
            (os.path.join(patient_folder, f)
             for f in files if ('-seg' in f.lower() and f.endswith('.nii.gz'))),
            None
        )
    }
    # Combine both into one dict
    return { **seq_dict, **seg_dict }


os.makedirs(TOPO_OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Loading metadata CSV → {METADATA_CSV}")
data = load_dictionary(METADATA_CSV)
print(f"[INFO] Found {len(data)} patient IDs in metadata.")



ALL_DATA_BASE_DIR = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\BraTS-MEN-Train"

all_patients = {}

for patient_id, metadata_row in data.items():
    patient_folder = os.path.join(ALL_DATA_BASE_DIR, patient_id)
    if not os.path.isdir(patient_folder):
        continue

    seg_candidates = [
        f for f in os.listdir(patient_folder)
        if (f.lower().endswith(".nii.gz") and "-seg" in f.lower())
    ]
    if len(seg_candidates) == 0:
        continue

    seg_fname = seg_candidates[0]
    seg_path = os.path.join(patient_folder, seg_fname)

    try:
        _ = nib.load(seg_path)
    except Exception as e:
        print(f"[WARNING] Could not nibabel-load {seg_path}: {e}")
        continue

    # Store this under all_patients:
    all_patients[patient_id] = {"tumor_seg_path": seg_path}

print("Built all_patients; found", len(all_patients), "entries.")

def crop_binary_volume(B):
    """
    Crop a binary volume to the axis-aligned bounding box containing
    all of the 1's

    Parameters
    ----------
    B: ndarray(M, N, L)
        A volume of 1s and 0s
    
    Returns:
    ndarray(<=M, <=N, <=L)
    """
    rg = []
    if B.size > 0:
        for i in range(3):
            Bi = B.swapaxes(0, i)
            Bi = np.reshape(Bi, (Bi.shape[0], Bi.shape[1]*Bi.shape[2]))
            idx = np.where(np.sum(Bi, axis=1) > 0)[0]
            if idx.size > 0:
                rg.append((np.min(idx), np.max(idx)))
    ret = np.array([])
    if len(rg) > 0:
        ret = B[rg[0][0]:rg[0][1], rg[1][0]:rg[1][1], rg[2][0]:rg[2][1]]
    return ret

def binary_volume_2coords(B):
    """
    Extract a point cloud representation of the voxels
    that are occupied in a binary volume
    
    Parameters
    ----------
    V: ndarray(M, N, L)
        A volume of 1s and 0s
    """
    return np.array(np.where(B==1), dtype=float).T


iso_names  = {1: "Core", 2: "Flair", 3: "Enchancement"} 
iso_levels = [2, 3, 1]  #[Edema=2, MainTumor=4, Necrotic=1]
tumor_types = ["Flair", "Enchancement", "Core"]

for i, patient_dict in enumerate(all_patients.values()):
    if i % 10 == 0:
        print(".", end="")

    tumor_seg_path = patient_dict["tumor_seg_path"]
    tumor_seg_nifti = nib.load(tumor_seg_path)
    tumor_seg_vol   = tumor_seg_nifti.get_fdata().astype(np.int32)  

    for k, level in enumerate(iso_levels):
        binary = (tumor_seg_vol == level)
        B = crop_binary_volume(binary)
        patient_dict[f"B{ iso_names[level] }"] = B

        X = binary_volume_2coords(binary)
        patient_dict[f"X{ iso_names[level] }"] = X

data = all_patients

to_delete = []
for p in data:
    if not ("XFlair" in data[p].keys()):
        to_delete.append(p)
for p in to_delete:
    del data[p]

print("\nAfter filtering, len(data) =", len(data))



def get_correlation_dimension_grid(B, dmin, dmax, verbose=False):
    """
    Compute the fractal correlation dimension for a point cloud that's
    binned to a regular grid in 3D
    
    Parameters
    ----------
    B: ndarray(M, N, L)
        A volume of 1s and 0s.  1s are locations of points
    dmin: float
        Minimum distance threshold to use in correlation dimension regression
    dmax: float
        Maximum distance threshold to use in correlation dimension regression
    verbose: bool
        Whether to print progress information
    
    Returns
    -------
    float: Fractal dimension estimate
    """
    from scipy.stats import linregress
    
    ## Step 1: Devise grid offsets
    pix = np.arange(0, int(np.ceil(dmax+1)))
    dI, dJ, dK = np.meshgrid(pix, pix, pix, indexing='ij')
    dI, dJ, dK = dI.flatten(), dJ.flatten(), dK.flatten()
    dists = np.sqrt(dI**2+dJ**2+dK**2)
    valid = (dists <= dmax)*(dists > 0)
    dI = dI[valid]
    dJ = dJ[valid]
    dK = dK[valid]
    dists = dists[valid]
    dist_counts = {d:0 for d in np.unique(dists)}
    if verbose:
        print(dI.size, "neighbors")

    ## Step 2: Examine all grid offsets for valid neighbors
    Bi, Bj, Bk = np.meshgrid(*[np.arange(B.shape[k]) for k in range(3)], indexing='ij')
    Bi, Bj, Bk = Bi.flatten(), Bj.flatten(), Bk.flatten()
    all_dists = []
    for count, (di, dj, dk, dist) in enumerate(zip(dI, dJ, dK, dists)):
        if verbose and count%100 == 0:
            print(".", end="")
        idxI = Bi + di
        idxJ = Bj + dj
        idxK = Bk + dk
        valid =  (idxI >= 0)*(idxI < B.shape[0])
        valid *= (idxJ >= 0)*(idxJ < B.shape[1])
        valid *= (idxK >= 0)*(idxK < B.shape[2])
        idxI = idxI[valid]
        idxJ = idxJ[valid]
        idxK = idxK[valid]
        dist_counts[dist] += np.sum(B[idxI, idxJ, idxK])
    
    x = np.array(list(dist_counts.keys()))
    y = np.array([dist_counts[d] for d in x])
    y = np.cumsum(y)
    ## TODO: Does having more distances logarithmically spaced higher up bias the result?
    return linregress(np.log(x[x >= dmin]), np.log(y[x >= dmin])).slope


dmin = 6
dmax = 12

out_dir = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Outputs\fractal"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
for p in data:
    fileout = "{}/{}.pkl".format(out_dir, p)
    if os.path.exists(fileout):
        continue
    dims = np.zeros(len(tumor_types))
    for i, tumor_type in enumerate(tumor_types):
        B = data[p]["B{}".format(tumor_type)]
        if np.sum(B) > 1: # There is at least one edge
            dims[i] = get_correlation_dimension_grid(B, dmin, dmax)
            print(i,dims[i])
    pickle.dump({"x":dims}, open(fileout, "wb"))



metaloc = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Data\outcomes.csv"
fractal_dir = r"C:\Users\tjz5\Desktop\1-Projects\MeningiomaRadiomics\Outputs\fractal"

if os.path.exists(metaloc):
    metadata_df = pd.read_csv(metaloc)
    if 'PatientID' not in metadata_df.columns:
        metadata_df.columns = ['PatientID'] + list(metadata_df.columns[1:])
    metadata_df['PatientID'] = metadata_df['PatientID'].astype(str)
else:
    metadata_df = pd.DataFrame(columns=['PatientID'])
    print(f"[WARNING] Metadata CSV not found at: {metaloc}")

fractal_rows = []
if os.path.isdir(fractal_dir):
    for fname in os.listdir(fractal_dir):
        if fname.endswith(".pkl"):
            pid = fname.replace(".pkl", "")
            fpath = os.path.join(fractal_dir, fname)
            try:
                data = pickle.load(open(fpath, "rb"))
                dims = data.get("x", None)
                if dims is not None and len(dims) == 3:
                    fractal_rows.append({
                        "PatientID": pid,
                        "FD_Edema": dims[0],
                        "FD_MainTumor": dims[1],
                        "FD_Necrotic": dims[2]
                    })
            except Exception as e:
                print(f"[ERROR] Loading {fpath}: {e}")
else:
    print(f"[WARNING] Fractal directory not found at: {fractal_dir}")

fractal_df = pd.DataFrame(fractal_rows)

if not metadata_df.empty and not fractal_df.empty:
    merged_df = metadata_df.merge(fractal_df, on="PatientID", how="left")
else:
    merged_df = metadata_df.copy()

with open('fractal.pkl', 'wb') as f:
    pickle.dump(merged_df, f)