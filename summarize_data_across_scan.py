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

    if not perturb:
        scipy.io.savemat(f"{output_directory}/{search_string}_connectome_{index}.mat", {"connectome": nonduplicate_connectome, "tag": tag_lst})
    else:
        cnt, allcnt = 0, 0
        while cnt < 10:
            subsample_connectome = select_random_columns(nonduplicate_connectome, percent)
            base_corr = np.corrcoef(nonduplicate_connectome, rowvar=True)
            sanity_check = np.corrcoef(subsample_connectome, rowvar=True)
            allcnt += 1
            if not np.isnan(sanity_check).any():
                figr, axsr = plt.subplots(1, 2, figsize=(4*2, 4))
                sns.heatmap(base_corr, ax=axsr[0], cmap="coolwarm", cbar=True, square=True)
                sns.heatmap(sanity_check, ax=axsr[1], cmap="coolwarm", cbar=True, square=True)
                figr.tight_layout()
                figr.savefig(f"zz_perturb_{percent}_{index}.png")
                scipy.io.savemat(f"{output_directory}_perturb/{search_string}_perturb_{percent}_{cnt}_connectome_{index}.mat", {"connectome": subsample_connectome, "tag": tag_lst})
                cnt += 1
        print(f"{cnt}/{allcnt} perturbations were successful.")
            
    
if __name__ == "__main__":
    # data = scipy.io.loadmat("./zz_data/noise_normal_cc_count_ss_all_connectome_in.mat")
    # cell_table_new = pd.read_feather("../microns_cell_tables/sven/microns_cell_annos_CV_240827.feather")
    # synapse_table = pd.read_feather("../microns_cell_tables/sven/synapses_minnie65_phase3_v1_943_combined_incl_trafo_240522.feather")
    
    # cnt = 0
    # for tag in data["tag"].flatten():
    #     row = cell_table_new[cell_table_new["pt_root_id"] == tag]
    #     # if row["status_axon"].item() in ("clean", "extended") and row["full_dendrite"].item() == True:
    #     cnt += 1
    
    # cell_table_trunc = cell_table_new[cell_table_new["pt_root_id"].isin(data["tag"].flatten())]
    
    # conn, tot_psd_sizes, _, _ = helper.create_connectivity_as_whole(cell_table_trunc, synapse_table)
    # conn_full, tot_psd_sizes, _, _ = helper.create_connectivity_as_whole(cell_table_new, synapse_table)
    
    # print(np.mean(conn))
    # print(np.mean(conn_full))
    
    summarize_data("normal", "count", "all", "in", False, perturb=True, percent=0.3)
    summarize_data("normal", "count", "all", "out", False, perturb=True, percent=0.3)
    
    # summarize_data("noise", "count", "all", "out", False)