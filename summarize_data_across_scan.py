import numpy as np
import pandas as pd 
import os 
import scipy 
import time
import xarray as xr
import sys

import matplotlib.pyplot as plt 
import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

import seaborn as sns
from sklearn.manifold import MDS   

import sys 
sys.path.append("../")
sys.path.append("../../")
import helper 

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']

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
            
def valid_stats(seq):
    """Return count and proportion of finite values (not NaN/Inf)."""
    a = np.asarray(seq, dtype=float)
    mask = np.isfinite(a)
    count_valid = int(mask.sum())
    proportion_valid = count_valid / a.size if a.size else float("nan")
    return count_valid, proportion_valid

def summarize_data(ww, cc, ss, index, scan_specific, perturb=False, percent=0.1):
    """
    """
    assert index in ["in", "out", "activity"]
    
    cell_table = pd.read_feather("../microns_cell_tables/sven/microns_cell_annos_CV_240827.feather")

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
    
    # analyze the spatial bias for these neurons
    # 497 neurons in total for activity recontruction (bottleneck)
    tags = np.asarray(tag_lst).ravel()
    bottleneck = cell_table[cell_table["pt_root_id"].isin(tags)]
    
    xxx = bottleneck["pt_position_x_trafo"].tolist()
    yyy = bottleneck["pt_position_z_trafo"].tolist()
    zzz = bottleneck["pt_position_y_trafo"].tolist()
    xxx = [xx / 1000 for xx in xxx]
    yyy = [yy / 1000 for yy in yyy]
    zzz = [zz / 1000 for zz in zzz]
    
    # spatial distribution
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        xxx, yyy, zzz, 
        c=zzz,             # color by depth (z) for nice contrast
        cmap="viridis",    # smooth colormap
        s=40,              # point size
        alpha=0.8,         # transparency
        edgecolors="k",    # black edges for clarity
        linewidth=0.3
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Cortical Depth (z, microns)")
    ax.set_xlabel("x (microns)")
    ax.set_ylabel("y (microns)")
    ax.set_zlabel("z (microns)")
    ax.view_init(elev=25, azim=35)
    fig.tight_layout()
    fig.savefig("./figures/zz_bottleneck_spatial.png", dpi=300)
    
    # histogram of cortical depth
    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        zzz,
        bins=30,         
        color="skyblue",
        edgecolor="k",
        alpha=0.8
    )

    ax.set_xlabel("Cortical Depth (z, microns)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Cortical Depth (z)")
    fig.tight_layout()
    fig.savefig("./figures/zz_histogram_cortical_depth.png", dpi=300)
    
    # register oracle score
    tag_oracle = {}
    tag_all_oracle = {}
    session_scan = [[5,3],[5,6],[5,7],[6,2],[7,3],[7,5],[9,3],[9,4],[6,4],[8,5],[4,7],[6,6],]
    for (session_info, scan_info) in session_scan:
        prf_coreg = pd.read_csv("./microns/prf_coreg.csv")
        prf_coreg = prf_coreg[(prf_coreg["session"] == session_info) & (prf_coreg["scan_idx"] == scan_info)]

        save_filename = f"./microns/functional_xr/functional_session_{session_info}_scan_{scan_info}.nc"
        session_ds = xr.open_dataset(save_filename)
        for ni in range(len(session_ds["unit_id"].values)):
            if not session_ds["unit_id"].values[ni] in tag_all_oracle.keys():
                tag_all_oracle[session_ds["unit_id"].values[ni]] = [session_ds["oracle_score"].values[ni]]
            else:
                tag_all_oracle[session_ds["unit_id"].values[ni]].append(session_ds["oracle_score"].values[ni])
        for tag in tag_lst:
            tag = tag[0]
            matching_row = prf_coreg[prf_coreg["pt_root_id"] == tag]
            if len(matching_row) > 0: # if there is a match
                if len(matching_row) > 1:
                    matching_row = matching_row.iloc[0]
                unit_id, field = matching_row["unit_id"].item(), matching_row["field"].item()
                # unit_id ordered sequentially
                check_unitid = session_ds["unit_id"].values[unit_id-1]
                check_field = session_ds["field"].values[unit_id-1]
                assert unit_id == check_unitid
                assert field == check_field
                orc_score = session_ds["oracle_score"].values[unit_id-1]
                if tag in tag_oracle.keys():
                    tag_oracle[tag].append(orc_score)
                else:
                    tag_oracle[tag] = [orc_score]

    tag_oracle = {float(k): float(np.nanmean(v)) for k, v in tag_oracle.items()}
    tag_all_oracle = {float(k): float(np.nanmean(v)) for k, v in tag_all_oracle.items()}
    select_oracle = list(tag_oracle.values())
    tag_all_oracle = list(tag_all_oracle.values())

    select_valid = valid_stats(select_oracle)[0]
    select_all = len(select_oracle)
    all_valid = valid_stats(tag_all_oracle)[0]
    all_all = len(tag_all_oracle)
    
    figorchist, axsorchist = plt.subplots(1,1,figsize=(5,5))
    axsorchist.hist(select_oracle, bins=50, color=c_vals[0], alpha=0.5, density=True, label=f"Targeted Neurons: {select_valid}/{select_all}: {select_valid/select_all:.1%} valid")
    axsorchist.hist(tag_all_oracle, bins=50, color=c_vals[1], alpha=0.5, density=True, label=f"All Neurons: {all_valid}/{all_all}: {all_valid/all_all:.1%} valid")
    axsorchist.axvline(np.nanmedian(select_oracle), color=c_vals[0], linestyle='dashed', linewidth=1)
    axsorchist.axvline(np.nanmedian(tag_all_oracle), color=c_vals[1], linestyle='dashed', linewidth=1)
    print(np.nanmedian(select_oracle), np.nanmedian(tag_all_oracle))
    axsorchist.set_xlabel("Oracle Score")
    axsorchist.set_ylabel("Probability")
    axsorchist.set_title("Distribution of Oracle Scores")
    axsorchist.legend(fontsize=8)
    figorchist.tight_layout()
    figorchist.savefig("./figures/zz_histogram_oracle_scores.png", dpi=300)
    sys.exit()

    # plot for Euclidean MDS
    nonduplicate_connectome_corr = np.corrcoef(nonduplicate_connectome, rowvar=True)
    mds = MDS(n_components=2, dissimilarity='precomputed', metric=False)     
    nonduplicate_connectome_coordinate = mds.fit_transform(1 - nonduplicate_connectome_corr)
    
    plt.rcParams.update({
        "axes.spines.right": False,
        "axes.spines.top":   False,
        "axes.labelsize":   13,
        "xtick.labelsize":  11,
        "ytick.labelsize":  11,
    })
    
    figmds, axmds = plt.subplots(figsize=(4.0, 4.0))
    
    hb = axmds.hexbin(nonduplicate_connectome_coordinate[:, 0],
                    nonduplicate_connectome_coordinate[:, 1],
                    gridsize=70,          # resolution of the bins
                    mincnt=1,             # ignore empty hexes
                    # bins='log',           # log‑scaled counts
                    cmap='coolwarm',         # perceptually uniform colormap
                    linewidths=0)         # no hex borders
    
    axmds.set_xlabel("MDS 1")
    axmds.set_ylabel("MDS 2")
    axmds.set_aspect('equal', adjustable='datalim')
    # axmds.set_title("MDS embedding of connectome", pad=10)

    cb = figmds.colorbar(hb, ax=axmds)
    cb.ax.tick_params(labelsize=10)

    figmds.tight_layout(pad=0.5)
    figmds.savefig(f"./figures/zz_mds_{search_string}_{index}.png", dpi=300)

    if not perturb:
        scipy.io.savemat(f"{output_directory}/{search_string}_connectome_{index}.mat", {"connectome": nonduplicate_connectome, "tag": tag_lst})
    else:
        print(f"Sparsity: {np.count_nonzero(nonduplicate_connectome)/(nonduplicate_connectome.shape[0]*nonduplicate_connectome.shape[1])}")
        time.sleep(1000)
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
                figr.savefig(f"./figures/zz_perturb{perturb_index}_{percent}_{index}.png", dpi=1000)
                scipy.io.savemat(f"{output_directory}_perturb/{search_string}_perturb{perturb_index}_{percent}_{cnt}_connectome_{index}.mat", {"connectome": subsample_connectome, "tag": tag_lst})
                cnt += 1
                
        print(f"{cnt}/{allcnt} perturbations were successful.")
            
    
if __name__ == "__main__":
    # data = scipy.io.loadmat("./zz_data/noise_normal_cc_count_ss_all_connectome_in.mat")
    # cell_table_new = pd.read_feather("../microns_cell_tables/sven/microns_cell_annos_CV_240827.feather")
    # synapse_table = pd.read_feather("../microns_cell_tables/sven/synapses_minnie65_phase3_v1_943_combined_incl_trafo_240522.feather")
    
    # matching_axon = cell_table_new[(cell_table_new["status_axon"].isin(["extended", "clean"])) & 
    #     (cell_table_new["classification_system"].isin(["excitatory_neuron", "inhibitory_neuron"]))
    # ]

    # matching_dendrite = cell_table_new[(cell_table_new["full_dendrite"] == True) & 
    #     (cell_table_new["classification_system"].isin(["excitatory_neuron", "inhibitory_neuron"]))
    # ]
    
    # print(matching_axon["layer"].value_counts())
    # print(matching_axon["region"].value_counts())
    # print(matching_dendrite["layer"].value_counts())
    # print(matching_dendrite["region"].value_counts())
    
    # tags = data["tag"].flatten() 
    # all_rows_df = cell_table_new[cell_table_new["pt_root_id"].isin(tags)].copy()
    # print(all_rows_df["layer"].value_counts())
    # print(all_rows_df["region"].value_counts())

            
    # print(cnt)
    
    # cell_table_trunc = cell_table_new[cell_table_new["pt_root_id"].isin(data["tag"].flatten())]
    
    # conn, tot_psd_sizes, _, _ = helper.create_connectivity_as_whole(cell_table_trunc, synapse_table)
    # conn_full, tot_psd_sizes, _, _ = helper.create_connectivity_as_whole(cell_table_new, synapse_table)
    
    # print(np.mean(conn))
    # print(np.mean(conn_full))
    
    # summarize_data("normal", "count", "all", "in", False, perturb=True, percent=0.2)
    summarize_data("normal", "count", "all", "out", False, perturb=True, percent=0.2)
    