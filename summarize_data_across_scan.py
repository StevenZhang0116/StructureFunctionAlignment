import numpy as np
import pandas as pd 
import os 
import scipy 
import time

import matplotlib.pyplot as plt
import seaborn as sns

import sys 
sys.path.append("../")
sys.path.append("../../")
import helper 

def select_random_columns(A, c, seed=None, return_idx=False):
    """
    """
    if not (0 <= c <= 1):
        raise ValueError("c must be between 0 and 1")

    N, M = A.shape
    K = int(round((1 - c) * M))
    if K == 0:
        raise ValueError("K computed as 0 — decrease c.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(M, size=K, replace=False)
    A_sub = A[:, chosen]

    return (A_sub, chosen) if return_idx else A_sub

def perturb_binary(mat: np.ndarray, k: float, seed=None, return_mask=False):
    """
    Randomly flips each entry of a binary matrix with probability k.

    Parameters
    ----------
    mat : np.ndarray
        2-D numpy array containing only 0s and 1s.
    k   : float
        Flip probability (0 < k < 1).
    seed : int or None
        Optional RNG seed for reproducibility.
    return_mask : bool
        If True, also return the boolean mask of flipped positions.

    Returns
    -------
    dict
        {
            "perturbed": np.ndarray,  # matrix after flips
            "strength": float,        # fraction of elements flipped
            ["mask": np.ndarray]      # (optional) True where flips occurred
        }
    """
    if not (0.0 < k < 1.0):
        raise ValueError("k must be between 0 and 1 (exclusive).")

    rng = np.random.default_rng(seed)
    flip_mask = rng.random(mat.shape) < k        # where to flip
    perturbed = mat.copy()
    perturbed[flip_mask] ^= 1                    # XOR with 1 flips 0↔1

    result = {"perturbed": perturbed,
              "strength": flip_mask.mean()}      # empirical fraction flipped
    if return_mask:
        result["mask"] = flip_mask
    return result

def perturb_preserve_mean_signed(mat: np.ndarray,
                                 k: float,
                                 mode: str = "out",
                                 seed=None,
                                 return_masks=False):
    """
    Balanced perturbation for matrices whose rows/cols contain {0, ±1}.

    Parameters
    ----------
    mat  : np.ndarray (2-D)             – entries are −1, 0, or 1
    k    : float  (0 < k < 1)           – fraction of non-zeros in each row/col to flip
    mode : "out" or "in"                – flip row-wise ("out") or column-wise ("in")
    seed : int | None                   – RNG seed for reproducibility
    return_masks : bool                 – optionally return the flip masks

    Returns
    -------
    dict with keys
        "perturbed" : np.ndarray  – matrix after perturbation
        "strength"  : float       – fraction of total entries changed
        ("mask_sign", "mask_zero") : list[ np.ndarray ]  (optional)
    """
    if not (0 < k < 1):
        raise ValueError("k must lie strictly between 0 and 1")
    if mode not in {"out", "in"}:
        raise ValueError("mode must be 'out' (rows) or 'in' (columns)")

    rng        = np.random.default_rng(seed)
    perturbed  = mat.copy()
    total_flip = 0
    flip_masks = [] if return_masks else None

    axis_iter = range(mat.shape[0]) if mode == "out" else range(mat.shape[1])
    for idx in axis_iter:
        vec = perturbed[idx, :] if mode == "out" else perturbed[:, idx]

        nz_mask = vec != 0
        if not nz_mask.any():        # nothing to flip in this row/col
            continue

        sign         = int(np.sign(vec[nz_mask][0]))    # either −1 or +1
        sign_mask    = vec == sign
        zero_mask    = vec == 0
        n_sign, n_zero = sign_mask.sum(), zero_mask.sum()
        n_flip       = int(np.round(k * n_sign))
        n_flip       = min(n_flip, n_zero)              # avoid running out of zeros
        if n_flip == 0:
            continue

        # choose indices to swap
        sign_idx = rng.choice(np.where(sign_mask)[0],  n_flip, replace=False)
        zero_idx = rng.choice(np.where(zero_mask)[0],  n_flip, replace=False)

        # apply flips
        if mode == "out":
            perturbed[idx, sign_idx] = 0
            perturbed[idx, zero_idx] = sign
        else:
            perturbed[sign_idx, idx] = 0
            perturbed[zero_idx, idx] = sign

        total_flip += 2 * n_flip

        if return_masks:
            mask_sign = np.zeros_like(vec, dtype=bool)
            mask_zero = np.zeros_like(vec, dtype=bool)
            mask_sign[sign_idx] = True
            mask_zero[zero_idx] = True
            flip_masks.append((idx, mask_sign, mask_zero))

    strength = total_flip / mat.size
    out = {"perturbed": perturbed, "strength": strength}
    if return_masks:
        out.update({"mask_sign_zero": flip_masks})
    return out

def sparsify_sign_matrix(M: np.ndarray, k: float, *, inplace: bool = False, seed: int = 0) -> np.ndarray:
    """
    Randomly set k % of the ±1 entries in a {-1, 0, 1} matrix to 0.
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix containing only -1, 0, or 1.
    k : float
        Fraction *or* percentage of non-zero entries to zero-out.  
        • If 0 ≤ k ≤ 1, it is treated as a fraction (e.g. 0.15 = 15 %).  
        • If 1 < k ≤ 100, it is treated as a percentage (e.g. 15 = 15 %).
    inplace : bool, default False
        If True, modify `M` in place; otherwise work on a copy and
        return the modified copy.
    seed : int or None, default None
        Optional seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Matrix with the chosen entries set to 0.
    """
    if not (-1 <= M).all() or not (M <= 1).all():
        raise ValueError("Matrix must contain only -1, 0, or 1.")
    
    # Normalise k to a fraction in [0, 1]
    if k > 1:
        k /= 100.0
    if not (0 <= k <= 1):
        raise ValueError("k must be between 0 and 1 (or 0-100 if given as %).")
    
    rng = np.random.default_rng(seed)
    A = M if inplace else M.copy()
    
    # Indices of non-zero (±1) entries
    nz_rows, nz_cols = np.nonzero(A)
    n_nonzero = nz_rows.size
    
    # Nothing to sparsify?
    if n_nonzero == 0 or k == 0:
        return A
    
    n_to_zero = int(round(k * n_nonzero))
    if n_to_zero == 0:  # k was small but non-zero; choose at least one
        n_to_zero = 1
    
    # Randomly pick the positions to zero-out
    chosen = rng.choice(n_nonzero, size=n_to_zero, replace=False)
    A[nz_rows[chosen], nz_cols[chosen]] = 0
    return A

def check_dale(matrix, mode="in"):
    """
    Check if the matrix follows Dale's principle.
    """
    uniques = np.unique(matrix)     
    assert set(uniques) == {-1, 0, 1} or set(uniques) == {-1, 0} or set(uniques) == {1, 0} or set(uniques) == {0}
    
    if mode == "in":
        for i in range(matrix.shape[1]):
            assert set(matrix[:,i]) == {-1, 0} or set(matrix[:,i]) == {1, 0} or set(matrix[:,i]) == {0}
    elif mode == "out":
        for i in range(matrix.shape[0]):
            assert set(matrix[i,:]) == {-1, 0} or set(matrix[i,:]) == {1, 0} or set(matrix[i,:]) == {0}

def summarize_data(ww, cc, ss, index, scan_specific, perturb=False, percent=0.1):
    """
    """
    assert index in ["in", "out", "activity"]

    output_directory = "./zz_data"

    if index in ["in", "out"]:
        search_string = f"noise_{ww}_cc_{cc}_ss_{ss}"
        dataname = "connectome"
    else:
        search_string = "activity"
        dataname = "activity"
        index = ""

    print(search_string)
    print(index)

    data_files = [
        f for f in os.listdir(output_directory)
        if all(cond in f for cond in [search_string, f"_{index}.mat", "microns"]) and f.endswith(".mat")
    ]

    print(data_files)
    print(len(data_files))

    assert len(data_files) == 12 or len(data_files) == 11

    connectome_lst, tag_lst = [], []

    for mat_file in data_files:
        data = scipy.io.loadmat(f"{output_directory}/{mat_file}")
        conn = data[dataname]
        if index == "in":
            conn = conn.T
        connectome_lst.append(conn)
        tag_lst.extend(list(data['tag'].reshape(-1,1)))

    connectome_array = np.concatenate(connectome_lst, axis=0)
    tag_lst = np.array(tag_lst).reshape(-1,1)

    def find_duplicate_indices(arr):
        # Flatten the array in case it's a column vector
        arr = arr.flatten()
        seen = {}
        duplicate_indices = []

        for i, value in enumerate(arr):
            if value in seen:
                duplicate_indices.append(i)
            else:
                seen[value] = True

        return duplicate_indices

    duplicates_index = find_duplicate_indices(tag_lst)
    tag_lst = np.delete(tag_lst, duplicates_index, axis=0)
    nonduplicate_connectome = np.delete(connectome_array, duplicates_index, axis=0)

    print(tag_lst[0:10])
    print(tag_lst.shape)
    print(nonduplicate_connectome.shape)

    if not perturb:
        scipy.io.savemat(f"{output_directory}/{search_string}_connectome_{index}.mat", {"connectome": nonduplicate_connectome, "tag": tag_lst})
    else:
        print(f"Sparsity: {np.count_nonzero(nonduplicate_connectome)/(nonduplicate_connectome.shape[0]*nonduplicate_connectome.shape[1])}")
        cnt, allcnt = 0, 0
        while cnt < 10:
            # subsample_connectome = select_random_columns(nonduplicate_connectome, percent)
            check_dale(nonduplicate_connectome, mode=index)
            
            method = 2 # which method to be used 
            if method == 1:
                subsample_connectome = perturb_preserve_mean_signed(nonduplicate_connectome, percent, index, seed=np.random.randint(0, 1000000))["perturbed"]
                perturb_index = ""
                
                diff = subsample_connectome - nonduplicate_connectome
                assert set(np.unique(diff)) == {0, 1, -1}
                print(np.sum(np.abs(diff))/(np.sum(nonduplicate_connectome == 1) + np.sum(nonduplicate_connectome == -1)))
                assert np.mean(subsample_connectome) == np.mean(nonduplicate_connectome)
                assert np.sum(subsample_connectome == 1) == np.sum(nonduplicate_connectome == 1)
                assert np.sum(subsample_connectome == -1) == np.sum(nonduplicate_connectome == -1)
                assert np.sum(subsample_connectome == 0) == np.sum(nonduplicate_connectome == 0)
                
            elif method == 2: 
                subsample_connectome = sparsify_sign_matrix(nonduplicate_connectome, percent, inplace=False, seed=np.random.randint(0, 1000000))
                perturb_index = "2"
                
                diff = subsample_connectome - nonduplicate_connectome
                assert set(np.unique(diff)) == {0, 1, -1} if index == "in" else {0, -1}
                print(np.sum(np.abs(diff))/(np.sum(nonduplicate_connectome == 1) + np.sum(nonduplicate_connectome == -1)))
            
            base_corr = np.corrcoef(nonduplicate_connectome, rowvar=True)
            sanity_check = np.corrcoef(subsample_connectome, rowvar=True)
            allcnt += 1
            if not np.isnan(sanity_check).any(): # make sure no isolated node -- perturbation is valid
                figr, axsr = plt.subplots(1, 2, figsize=(4*2, 4))
                sns.heatmap(base_corr, ax=axsr[0], cmap="coolwarm", cbar=True, square=True)
                sns.heatmap(sanity_check, ax=axsr[1], cmap="coolwarm", cbar=True, square=True)
                figr.tight_layout()
                figr.savefig(f"zz_perturb{perturb_index}_{percent}_{index}.png", dpi=1000)
                scipy.io.savemat(f"{output_directory}_perturb/{search_string}_perturb{perturb_index}_{percent}_{cnt}_connectome_{index}.mat", {"connectome": subsample_connectome, "tag": tag_lst})
                cnt += 1
                
        print(f"{cnt}/{allcnt} perturbations were successful.")
            
    
if __name__ == "__main__":
    data = scipy.io.loadmat("./zz_data/noise_normal_cc_count_ss_all_connectome_in.mat")
    cell_table_new = pd.read_feather("../microns_cell_tables/sven/microns_cell_annos_CV_240827.feather")
    synapse_table = pd.read_feather("../microns_cell_tables/sven/synapses_minnie65_phase3_v1_943_combined_incl_trafo_240522.feather")
    
    matching_axon = cell_table_new[(cell_table_new["status_axon"].isin(["extended", "clean"])) & 
        (cell_table_new["classification_system"].isin(["excitatory_neuron", "inhibitory_neuron"]))
    ]

    matching_dendrite = cell_table_new[(cell_table_new["full_dendrite"] == True) & 
        (cell_table_new["classification_system"].isin(["excitatory_neuron", "inhibitory_neuron"]))
    ]
    
    print(matching_axon["layer"].value_counts())
    print(matching_axon["region"].value_counts())
    print(matching_dendrite["layer"].value_counts())
    print(matching_dendrite["region"].value_counts())
    
    tags = data["tag"].flatten() 
    all_rows_df = cell_table_new[cell_table_new["pt_root_id"].isin(tags)].copy()
    print(all_rows_df["layer"].value_counts())
    print(all_rows_df["region"].value_counts())

            
    # print(cnt)
    
    # cell_table_trunc = cell_table_new[cell_table_new["pt_root_id"].isin(data["tag"].flatten())]
    
    # conn, tot_psd_sizes, _, _ = helper.create_connectivity_as_whole(cell_table_trunc, synapse_table)
    # conn_full, tot_psd_sizes, _, _ = helper.create_connectivity_as_whole(cell_table_new, synapse_table)
    
    # print(np.mean(conn))
    # print(np.mean(conn_full))
    
    # summarize_data("normal", "count", "all", "in", False, perturb=False, percent=0.2)
    # summarize_data("normal", "count", "all", "out", False, perturb=False, percent=0.2)
    