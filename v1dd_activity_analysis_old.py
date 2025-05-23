import pandas as pd 
import numpy as np
import re 
import time
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr

import activity_helper

import sys
sys.path.append("../")
sys.path.append("../../")


import basic
import helper

if __name__ == "__main__":
    data_folder = "./data/"
    output_folder = "./output/"
    use_data = "binary" # binary / count
    symmetry_index = ""
    correlation_index = "row" # column / row
    sub_correlation_index = "nan_fill" # zero_out / nan_fill

    coregisteration = pd.read_feather(f"{data_folder}scan13_coregistration_dataframe_with_assembly_membership.feather")
    roi_id = np.load(f"{data_folder}sessionM409828_13_roi_id_by_index.npy")
    v1dd_cell_table = pd.read_feather("../microns_cell_tables/extra/pre_cell_table_v1dd_proofread_True.feather") 
    v1dd_synapse_table = pd.read_feather("../microns_cell_tables/extra/synapse_table_v1dd_targeting_proofread_True_only_from_proofread_True.feather")   
    activity_covariance_given = np.load(f"{data_folder}sessionM409828_13_covariance.npy")

    dff = np.load(f"{data_folder}sessionM409828_13_dff.npy")
    activity_correlation = np.corrcoef(dff, rowvar=False)

    assert len(roi_id) == activity_covariance_given.shape[0]

    search_root_id_lst = []
    search_root_index_lst = []
    id_iter_lst = []

    for id_iter in range(len(roi_id)):
        theid = roi_id[id_iter]
        match = re.findall(r'(\d+)', theid)
        field = int(match[0])
        unit_id = int(match[1])
        filtered_df = coregisteration[(coregisteration['session'] == 1) & (coregisteration['scan_idx'] == 3) & (coregisteration['field'] == field) & \
                        (coregisteration['unit_id'] == unit_id)]
        if len(filtered_df) > 0: # should have maximum 1 returned result
            search_root_id = filtered_df['pt_root_id'].tolist()[0] # matched row in coregisteration
            # print("====")
            # print(search_root_id)
            # print(coregisteration[coregisteration["pt_root_id"] == search_root_id]["pt_position"].values)
            
            if len(v1dd_cell_table[v1dd_cell_table['pt_root_id'] == search_root_id]) > 0: # matched neuron in V1DD
                id_iter_lst.append(id_iter)
                search_root_id_lst.append(search_root_id) 
                # print(v1dd_cell_table[v1dd_cell_table['pt_root_id'] == search_root_id]["pt_position"].values)
                search_root_index_lst.append(v1dd_cell_table.index[v1dd_cell_table['pt_root_id'] == search_root_id][0]) # index of matched neuron 

    # search_root_index_lst = np.sort(search_root_index_lst) # sort the matched neuron index
    neurons_labels = v1dd_cell_table.iloc[search_root_index_lst]["cell_type"].tolist()

    matched_activity_correlation = activity_correlation[np.ix_(id_iter_lst, id_iter_lst)]
    
    # load connectome data
    ground_truth, totalSyn, synCount, _ = helper.create_connectivity_as_whole(v1dd_cell_table, v1dd_synapse_table)

    # consider undirected graph
    if symmetry_index == "symmetry":
        if use_data == "binary":
            ground_truth = (ground_truth.astype(bool) | ground_truth.T.astype(bool)).astype(int)
        elif use_data == "count":
            ground_truth = ground_truth + ground_truth.T

    # how to calculate the correlation
    ground_truth_correlation = np.corrcoef(ground_truth, rowvar=False if correlation_index=="column" else True) 

    matched_ground_truth = ground_truth[np.ix_(search_root_index_lst, search_root_index_lst)]
    matched_ground_truth_correlation = ground_truth_correlation[np.ix_(search_root_index_lst, search_root_index_lst)]

    assert matched_activity_correlation.shape == matched_ground_truth_correlation.shape

    shape_dimension = matched_activity_correlation.shape[0]
    if sub_correlation_index == "zero_out":
        np.fill_diagonal(matched_activity_correlation, 0)
        np.fill_diagonal(matched_ground_truth_correlation, 0)
    elif sub_correlation_index == "nan_fill":
        np.fill_diagonal(matched_activity_correlation, np.nan)
        np.fill_diagonal(matched_ground_truth_correlation, np.nan)
    
    # comparison
    metric_compare = "correlation" # correlation only
    
    relation_matrix = np.zeros((shape_dimension, shape_dimension))
    for i in range(shape_dimension):
        for j in range(shape_dimension):
            if metric_compare == "correlation":
                if sub_correlation_index == "zero_out":
                    correlation, _ = pearsonr(matched_activity_correlation[i], matched_ground_truth_correlation[j])
                elif sub_correlation_index == "nan_fill":
                    correlation = activity_helper.pearson_correlation_with_nans(matched_activity_correlation[i], matched_ground_truth_correlation[j])

                relation_matrix[i,j] = correlation
            elif metric_compare == "cosine":
                # (?) take out the effect of ith element (because of purposefull setting to diagonal)
                compare1 = matched_activity_correlation[i]
                compare2 = matched_ground_truth_correlation[j]
                compare1 = np.concatenate([compare1[:i], compare1[i+1:]])
                compare2 = np.concatenate([compare2[:j], compare2[j+1:]])

                cosine_relation = cosine_similarity(compare1.reshape(1,-1), compare2.reshape(1,-1))
                relation_matrix[i,j] = cosine_relation[0,0]

    # normal_relation = cosine_similarity(matched_activity_correlation, matched_ground_truth)

    # original diagonal plotting
    subplots_num = 3
    fig, axs = plt.subplots(1,subplots_num,figsize=(6*subplots_num,6))
    sns.heatmap(relation_matrix, ax=axs[0], square=True, cmap="coolwarm", vmin=np.min(relation_matrix), vmax=np.max(relation_matrix))

    # only plotting the diagonal part
    diag_matrix = relation_matrix.copy()
    diag_part = np.diag(diag_matrix).copy()
    diag_matrix.fill(0)
    np.fill_diagonal(diag_matrix, diag_part)
    # sns.heatmap(diag_matrix, ax=axs[1], square=True, cmap="coolwarm", vmin=np.min(diag_matrix), vmax=np.max(diag_matrix))

    # comparing with other ground truth part
    sns.heatmap(matched_ground_truth_correlation, ax=axs[1], square=True)
    sns.heatmap(matched_activity_correlation,ax=axs[2],square=True)

    axs[0].set_title(f"Activity Corr - {use_data} Connectome Corr: {metric_compare}: {sub_correlation_index}")
    # axs[1].set_title(f"Activity Corr - {use_data} Connectome Corr: {metric_compare}: Diagonal")
    axs[1].set_title("Connectome Correlation (Remove Diagonal)")
    axs[2].set_title("Activity Correlation (Remove Diagonal)")
    fig.tight_layout()
    fig.savefig(f"{output_folder}v1dd_{use_data}{symmetry_index}_{metric_compare}_{correlation_index}_test_{sub_correlation_index}.png")

    # significance test
    mean_diagonal, mean_off_diagonal, t_stat, p_value = activity_helper.test_diagonal_significance(relation_matrix)
    print(f"Mean Diagonal: {mean_diagonal}, Mean Off-Diagonal: {mean_off_diagonal}")
    print(f"T-statistic: {t_stat}, P-value: {p_value}")

    # plot (maybe)
    diagonal = np.diag(relation_matrix)
    i, j = np.indices(relation_matrix.shape)
    off_diagonal = relation_matrix[i != j]   

    fig_dist, ax_dist = plt.subplots(1,1,figsize=(10, 5))
    ax_dist.hist(diagonal, bins=20, alpha=0.7, label='Diagonal Elements', color='blue', density=True)
    ax_dist.hist(off_diagonal, bins=20, alpha=0.7, label='Off-Diagonal Elements', color='red', density=True)
    ax_dist.set_xlabel('Value')
    ax_dist.set_ylabel('Frequency')
    ax_dist.set_title(f"p-value: {p_value}")
    fig_dist.legend()
    fig_dist.savefig(f"{output_folder}v1dd_{use_data}{symmetry_index}_{metric_compare}_{correlation_index}_stat_{sub_correlation_index}.png")

    # activity angle analysis 
    ground_truth_consideration_lst = [matched_ground_truth, matched_ground_truth_correlation]
    ground_truth_label_lst = ["Connectome vs. Activity Correlation", "Connectome Correlation vs. Activity Correlation"]

    fig_angle, ax_angle = plt.subplots(1,len(ground_truth_consideration_lst),figsize=(6*len(ground_truth_consideration_lst),6))

    for ground_truth_iter in range(len(ground_truth_consideration_lst)):
        ground_truth_consideration = ground_truth_consideration_lst[ground_truth_iter]

        U_connectome, S_connectome, Vh_connectome = np.linalg.svd(ground_truth_consideration)
        U_activity, S_activity, Vh_activity = np.linalg.svd(matched_activity_correlation)
        
        # 1D case
        U_activity_1, U_connectome_1 = U_activity[:,0], U_connectome[:,0]
        cos_theta = np.dot(U_activity_1, U_connectome_1) / (np.linalg.norm(U_activity_1) * np.linalg.norm(U_connectome_1))
        angle_1 = np.degrees(np.arccos(cos_theta))

        dim_loader, angle_loader = [], []
        for num_dimension in range(1,20):
            U_comps_activity = [U_activity[:,i] for i in range(num_dimension)]
            U_comps_connectome = [U_connectome[:,i] for i in range(num_dimension)]
            angle_in_bewteen = activity_helper.angles_between_flats(U_comps_activity, U_comps_connectome)
            dim_loader.append(num_dimension)
            angle_loader.append(angle_in_bewteen)

        ax_angle[ground_truth_iter].plot(dim_loader, angle_loader, "-o")
        ax_angle[ground_truth_iter].set_title(ground_truth_label_lst[ground_truth_iter])

    for ax in ax_angle:
        ax.set_xlabel("Dimensions")
        ax.set_ylabel("Angle between flats")

    fig_angle.tight_layout()
    fig_angle.savefig(f"{output_folder}v1dd_{use_data}{symmetry_index}_{metric_compare}_{correlation_index}_angle_{sub_correlation_index}.png")