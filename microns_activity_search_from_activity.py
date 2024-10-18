import numpy as np
import pandas as pd
import time
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
import xarray as xr
from scipy.stats import pearsonr
import pickle
import gc

import scipy
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.linalg import subspace_angles
from scipy.spatial.distance import pdist, squareform
from scipy.signal import welch, coherence
from scipy.stats import rankdata, linregress

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import helper
import activity_helper

import microns_across_scans

from netrep.metrics import LinearMetric

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]

def float_to_scientific(value, n=4):
    return f"{value:.{n}e}"

def run(session_info, scan_info, for_construction, R_max, embedding_dimension, raw_data):

    metadata = {}

    # index
    metric_compare = "correlation"
    sub_correlation_index = "nan_fill"

    # proofread & coregisteration overlap information
    prf_coreg = pd.read_csv("./microns/prf_coreg.csv")
    # take the information belong to that specific session
    prf_coreg = prf_coreg[(prf_coreg["session"] == session_info) & (prf_coreg["scan_idx"] == scan_info)]
    print(len(prf_coreg))

    # read in microns cell/synapse information
    cell_table = pd.read_feather("../microns_cell_tables/sven/microns_cell_annos_240524.feather")

    matching_axon = cell_table.index[cell_table["status_axon"].isin(["extended", "clean"])].tolist()
    matching_dendrite = cell_table.index[cell_table["full_dendrite"] == True].tolist()

    # cell_table = cell_table.sort_values(by='cell_type')
    synapse_table = pd.read_feather("../microns_cell_tables/sven/synapses_minnie65_phase3_v1_943_combined_incl_trafo_240522.feather")

    save_filename = f"./microns/functional_xr/functional_session_{session_info}_scan_{scan_info}.nc"
    session_ds = xr.open_dataset(save_filename)

    prf_coreg_ids = set(prf_coreg["pt_root_id"])

    fps_value = session_ds.attrs.get("fps", None)

    selected_neurons = []
    neuron_types = []
    activity_extraction = []
    activity_extraction_extra = []

    for index, pr_root_id in enumerate(cell_table["pt_root_id"]):
        if pr_root_id in prf_coreg_ids:
            cell_row = cell_table[cell_table["pt_root_id"] == pr_root_id]

            selected_neurons.append(index)
            # count += 1
            # Print the corresponding row in prf_coreg
            matching_row = prf_coreg[prf_coreg["pt_root_id"] == pr_root_id]

            if len(matching_row) > 1:
                matching_row = matching_row.iloc[0]
            
            unit_id, field = matching_row["unit_id"].item(), matching_row["field"].item()
            # unit_id ordered sequentially
            check_unitid = session_ds["unit_id"].values[unit_id-1]
            check_field = session_ds["field"].values[unit_id-1]
            # to confirm unitid-1 is the correct searching index
            assert unit_id == check_unitid
            assert field == check_field

            activity_extraction.append(session_ds["fluorescence"].values[unit_id-1,:])
            activity_extraction_extra.append(session_ds["activity"].values[unit_id-1,:])

    assert len(selected_neurons) == len(activity_extraction)

    activity_extraction = np.array(activity_extraction)
    activity_extraction_extra = np.array(activity_extraction_extra)

    # whether to use deconvolved data or raw data
    if raw_data:
        activity_extraction_extra = activity_extraction

    selectss_df = cell_table.loc[selected_neurons]
    soma_locations = selectss_df[['pt_position_x', 'pt_position_y', 'pt_position_z']].to_numpy()
    soma_distances = squareform(pdist(soma_locations, metric='euclidean'))

    selects = [1,12,18,25,32,36]
    ttt = activity_extraction_extra.shape[1]
    timecut = int(ttt/100)
    fignewact, axsnewact = plt.subplots(len(selects),1,figsize=(10,len(selects)*1))
    for ni in selects:
        axsnewact[selects.index(ni)].plot(activity_extraction_extra[ni,:timecut], c=c_vals[selects.index(ni)])
        axsnewact[selects.index(ni)].set_title(f"Neuron {ni}")
        axsnewact[selects.index(ni)].set_xlim([-2, timecut+2])
    fignewact.tight_layout()
    fignewact.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_activity_new.png")

    # calculate activity correlation
    activity_correlation_all = np.corrcoef(activity_extraction_extra, rowvar=True)
    metadata["num_neurons"] = activity_extraction_extra.shape[0]

    fig, ax = plt.subplots(1,1,figsize=(4,4))
    # only take the upper triangular part (ignore the diagonal)
    # in case of plotting duplicated results
    dd = np.triu_indices_from(activity_correlation_all, k=1)
    upper_tri_values = activity_correlation_all[dd].flatten()
    gt_median_corr = np.median(upper_tri_values)
    ax.hist(upper_tri_values, bins=100)
    ax.set_xlabel("Activity Correlation")
    ax.set_ylabel("Frequency/Count")
    ax.set_title(f"Median: {np.round(gt_median_corr,3)}")
    fig.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_acthist.png")

    metadata["gt_median_corr"] = gt_median_corr

    W_all, totalSyn, synCount, _ = helper.create_connectivity_as_whole(cell_table, synapse_table)

    print(f"Original selected_neurons: {len(selected_neurons)}")

    correlation_index_lst = ["column", "row"]
    W_corrs_all, diags, offdiags, W_samples, indices_delete_lst = [], [], [], [], []
    p_values_all = []

    fig, axs = plt.subplots(2,4,figsize=(4*4,4*2))
    fig_dist, ax_dist = plt.subplots(1,3,figsize=(4*3,4*1))

    for correlation_index in correlation_index_lst:
        corrind = correlation_index_lst.index(correlation_index)

        external_data = True
        if not external_data:
            W_corr = np.corrcoef(W, rowvar=False if correlation_index=="column" else True) 
        else:
            # calculated correlation using weighted connectome
            if correlation_index == "row":
                W_trc = synCount[selected_neurons,:]
                W_trc = W_trc[:,matching_dendrite]
                W_corr = np.corrcoef(W_trc, rowvar=True)

            else:
                W_trc = synCount[:,selected_neurons]
                W_trc = W_trc[matching_axon,:]
                W_corr = np.corrcoef(W_trc, rowvar=False)
            
            W_backupcheck = synCount[np.ix_(selected_neurons, selected_neurons)]
            W_samples.append(W_trc)

        W_corr_all = W_corr
        W_corr, activity_correlation, indices_to_delete = activity_helper.sanity_check_W(W_corr_all, activity_correlation_all)
        indices_delete_lst.append(indices_to_delete)
        shape_check = W_corr.shape[0]
        print(f"Shape after sanity check (nan/inf): {shape_check}")

        if sub_correlation_index == "zero_out":
            np.fill_diagonal(activity_correlation, 0)
            np.fill_diagonal(W_corr, 0)

        elif sub_correlation_index == "nan_fill":
            np.fill_diagonal(activity_correlation, np.nan)
            np.fill_diagonal(W_corr, np.nan)

        relation_matrix = np.zeros((shape_check, shape_check))
        relation_matrix_compare = np.zeros((shape_check, shape_check))
        for i in range(shape_check):
            for j in range(shape_check):
                if metric_compare == "correlation":
                    if sub_correlation_index == "zero_out":
                        correlation, _ = pearsonr(activity_correlation[i], W_corr[j])
                    elif sub_correlation_index == "nan_fill":
                        correlation = activity_helper.pearson_correlation_with_nans(activity_correlation[i], W_corr[j])

                    try:
                        correlation_compare, _ = pearsonr(activity_correlation[i], W_backupcheck[j])
                    except:
                        correlation_compare = np.nan

                    relation_matrix[i,j] = correlation
                    relation_matrix_compare[i,j] = correlation_compare


        sns.heatmap(relation_matrix, ax=axs[corrind,0], cbar=True, square=True, center=0, cmap="coolwarm", vmin=np.min(relation_matrix), vmax=np.max(relation_matrix))
        sns.heatmap(relation_matrix_compare, ax=axs[corrind,1], cbar=True, square=True, cmap="coolwarm", vmin=np.min(relation_matrix_compare), vmax=np.max(relation_matrix_compare))
        sns.heatmap(activity_correlation, ax=axs[corrind,2], cbar=True, square=True)
        sns.heatmap(W_corr, ax=axs[corrind,3], cbar=True, square=True)
        axs[corrind,0].set_title(f"Microns Corr(Corr(W), Corr(A)) - {correlation_index}")
        axs[corrind,1].set_title("Microns Corr(W, Corr(A))")
        axs[corrind,2].set_title("Corr(A)")
        axs[corrind,3].set_title("Corr(W)")

        relation_matrix = activity_helper.remove_nan_inf_union(relation_matrix)
        relation_matrix = np.where(np.isnan(relation_matrix) | np.isinf(relation_matrix), 0, relation_matrix)

        mean_diagonal, mean_off_diagonal, t_stat, p_value = activity_helper.test_diagonal_significance(relation_matrix)
        print(f"Mean Diagonal: {mean_diagonal}, Mean Off-Diagonal: {mean_off_diagonal}")
        print(f"T-statistic: {t_stat}, P-value: {p_value}")

        p_values_all.append(p_value)

        diagonal = np.diag(relation_matrix)
        
        i, j = np.indices(relation_matrix.shape)
        off_diagonal = relation_matrix[i != j]   

        diags.append(diagonal)
        offdiags.append(off_diagonal)

        # subspace angle analysis
        # here we refill the diagonal to be 0 (instead of nan/inf as previous)
        np.fill_diagonal(W_corr, 0)
        np.fill_diagonal(W_corr_all, 0)
        W_corrs_all.append(W_corr_all)
        np.fill_diagonal(activity_correlation,0)

        dim_loader, angle_loader = activity_helper.angles_between_flats_wrap(W_corr, activity_correlation)

        metadata[f"{correlation_index}_angle"] = angle_loader

    fig.tight_layout()
    fig.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_heatmap.png")

    def merge_arrays(arrays):
        union_result = arrays[0]    
        for arr in arrays[1:]:
            union_result = np.union1d(union_result, arr)
        return union_result

    # make sure neurons are aligned/matched correctly
    # delete all neurons if it is either nan or inf in any of the correlation matrices (row or column)
    neurons_tobe_deleted = merge_arrays(indices_delete_lst)
    print(f"Neurons union: {neurons_tobe_deleted}")

    activity_extraction_extra_trc = np.delete(activity_extraction_extra, neurons_tobe_deleted, axis=0)

    activity_correlation_all_trc = np.delete(activity_correlation_all, neurons_tobe_deleted, axis=0)  # Delete rows
    activity_correlation_all_trc = np.delete(activity_correlation_all_trc, neurons_tobe_deleted, axis=1)
    np.fill_diagonal(activity_correlation_all_trc, 0)

    in_sample_trc = np.delete(W_samples[0], neurons_tobe_deleted, axis=1)
    out_sample_trc = np.delete(W_samples[1], neurons_tobe_deleted, axis=0)

    in_sample_corr_trc = np.delete(W_corrs_all[0], neurons_tobe_deleted, axis=0)  
    in_sample_corr_trc = np.delete(in_sample_corr_trc, neurons_tobe_deleted, axis=1)

    out_sample_corr_trc = np.delete(W_corrs_all[1], neurons_tobe_deleted, axis=0)  
    out_sample_corr_trc = np.delete(out_sample_corr_trc, neurons_tobe_deleted, axis=1)

    W_corrs_all_trc = [in_sample_corr_trc, out_sample_corr_trc]

    print(activity_extraction_extra_trc.shape)
    print(activity_correlation_all_trc.shape)
    print(out_sample_trc.shape)
    print(in_sample_trc.shape)
    print(out_sample_corr_trc.shape)
    print(in_sample_corr_trc.shape)

    scipy.io.savemat(f"zz_data/microns_{session_info}_{scan_info}_connectome_out.mat", {'connectome': out_sample_trc})
    scipy.io.savemat(f"zz_data/microns_{session_info}_{scan_info}_connectome_in.mat", {'connectome': in_sample_trc})
    scipy.io.savemat(f"zz_data/microns_{session_info}_{scan_info}_activity.mat", {'activity': activity_extraction_extra_trc})


    soma_distances_trc = np.delete(soma_distances, neurons_tobe_deleted, axis=0)  
    soma_distances_trc = np.delete(soma_distances_trc, neurons_tobe_deleted, axis=1)

    print(soma_distances_trc.shape)

    if for_construction:

        def reconstruction(W, K="all", permute=False):
            """
            add option to only take top K components in reconstruction
            if K=None, then using all neurons (without any low-pass filter)
            """
            if K == "all":
                K = W.shape[0]

            filtered_matrix = np.zeros_like(W)
            for i in range(W.shape[0]):
                top_k_indices = np.argsort(W[i])[-K:]        
                filtered_matrix[i, top_k_indices] = W[i, top_k_indices]
            
            row_sums_a = np.sum(np.abs(filtered_matrix), axis=1)

            filtered_matrix_normalized = filtered_matrix / row_sums_a[:, np.newaxis]
            assert np.sum(np.diag(filtered_matrix_normalized)) == 0

            # permute the matrix but still keep it symmetric
            if permute == "rowwise":
                filtered_matrix_normalized = activity_helper.permute_symmetric_matrix(filtered_matrix_normalized)
            elif permute == "cellwise":
                filtered_matrix_normalized = activity_helper.permute_symmetric_matrix_cellwise(filtered_matrix_normalized)

            activity_reconstruct_a = filtered_matrix_normalized @ activity_extraction_extra_trc
            return activity_reconstruct_a, filtered_matrix_normalized

        # diagonal compare
        if len(diags[0]) != len(diags[1]):
            diag1, diag2 = activity_helper.match_list_lengths(diags[0], diags[1])
        else:
            diag1, diag2 = diags[0], diags[1]
        
        t_stat, p_value = stats.ttest_rel(diag1, diag2)
        p_value_one_sided = p_value / 2
        p_value_one_sided_final = p_value_one_sided if t_stat > 0 else 1 - p_value_one_sided

        figallcompare, axsallcompare = plt.subplots(1,2,figsize=(4*2,4))
        axsallcompare[0].hist(diags[0], bins=50, alpha=0.5, label='Column On-Diagonal', color=c_vals[1], density=True)
        axsallcompare[0].hist(diags[1], bins=50, alpha=0.5, label='Row On-Diagonal', color=c_vals_l[1], density=True)
        axsallcompare[0].hist(offdiags[0], bins=50, alpha=0.5, label='Column Off-Diagonal', color=c_vals[0], density=True)
        axsallcompare[0].hist(offdiags[1], bins=50, alpha=0.5, label='Row Off-Diagonal', color=c_vals_l[0], density=True)
        axsallcompare[0].legend()
        axsallcompare[0].set_title(f"p1: {float_to_scientific(p_values_all[0])}; p2: {float_to_scientific(p_values_all[1])}; p3: {float_to_scientific(p_value_one_sided_final)}")

        for correlation_index in correlation_index_lst:
            axsallcompare[1].plot([i+1 for i in range(len(metadata[f"{correlation_index}_angle"]))], metadata[f"{correlation_index}_angle"], \
                                    label=f"{correlation_index}", color=c_vals[correlation_index_lst.index(correlation_index)])
        axsallcompare[1].legend()
        axsallcompare[1].set_xlabel("Angle Index")
        axsallcompare[1].set_ylabel("Angle")

        figallcompare.tight_layout()
        figallcompare.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_offdiagcompare_all.png")
    
        hyp_name = f"./mds-results/Rmax_{R_max}_D_{embedding_dimension}_microns_{session_info}_{scan_info}_embed.mat"
        out_corr = scipy.io.loadmat(hyp_name)['Ddists'][0][1]
        out_corr = 1 - out_corr
        np.fill_diagonal(out_corr, 0)

        in_corr = scipy.io.loadmat(hyp_name)['Ddists'][0][2]
        in_corr = 1 - in_corr
        np.fill_diagonal(in_corr, 0)
        
        # sanity check
        np.allclose(out_corr, W_corrs_all_trc[1], rtol=1e-05, atol=1e-08) 
        np.allclose(in_corr, W_corrs_all_trc[0], rtol=1e-05, atol=1e-08) 

        hypembed_name = f"./mds-results/Rmax_{R_max}_D_{embedding_dimension}_microns_{session_info}_{scan_info}_embed_hypdist.mat"
        hypembed_connectome_distance_out = scipy.io.loadmat(hypembed_name)['hyp_dist'][0][1]

        rmax_quantile_out = activity_helper.find_quantile(hypembed_connectome_distance_out, float(R_max))
        metadata["rmax_quantile_out"] = rmax_quantile_out

        hypembed_connectome_corr_out = float(R_max) - hypembed_connectome_distance_out
        # hypembed_connectome_corr_out = np.max(hypembed_connectome_distance_out) - hypembed_connectome_distance_out
        np.fill_diagonal(hypembed_connectome_corr_out, 0)

        hypembed_connectome_distance_in = scipy.io.loadmat(hypembed_name)['hyp_dist'][0][2]

        rmax_quantile_in = activity_helper.find_quantile(hypembed_connectome_distance_in, float(R_max))
        metadata["rmax_quantile_in"] = rmax_quantile_in

        hypembed_connectome_corr_in = float(R_max) - hypembed_connectome_distance_in
        # hypembed_connectome_corr_in = np.max(hypembed_connectome_distance_in) - hypembed_connectome_distance_in
        np.fill_diagonal(hypembed_connectome_corr_in, 0)

        # load Euclidean embedding coordinate
        # calcualate the pairwise Euclidean distance afterward    

        eulembed_name = f"./mds-results/Rmax_{R_max}_D_{embedding_dimension}_microns_{session_info}_{scan_info}_embed_eulmds.mat"
        eulembed_connectome = scipy.io.loadmat(eulembed_name)['eulmdsembed'][0][1]
        eulembed_connectome_distance_out = squareform(pdist(eulembed_connectome, metric='euclidean'))
        metadata["rmax_eul_out_distance"] = np.max(eulembed_connectome_distance_out)
        # 
        eulembed_connectome_corr_out = activity_helper.find_value_for_quantile(eulembed_connectome_distance_out, rmax_quantile_out) - eulembed_connectome_distance_out
        # eulembed_connectome_corr_out = np.max(eulembed_connectome_distance_out) - eulembed_connectome_distance_out
        np.fill_diagonal(eulembed_connectome_corr_out, 0)

        eulembed_connectome = scipy.io.loadmat(eulembed_name)['eulmdsembed'][0][2]
        eulembed_connectome_distance_in = squareform(pdist(eulembed_connectome, metric='euclidean'))
        metadata["rmax_eul_in_distance"] = np.max(eulembed_connectome_distance_in)
        # 
        eulembed_connectome_corr_in = activity_helper.find_value_for_quantile(eulembed_connectome_distance_in, rmax_quantile_in) - eulembed_connectome_distance_in
        # eulembed_connectome_corr_in = np.max(eulembed_connectome_distance_in) - eulembed_connectome_distance_in
        np.fill_diagonal(eulembed_connectome_corr_in, 0)

        # soma distance (baseline)
        soma_distances_trc = activity_helper.find_value_for_quantile(soma_distances_trc, np.mean([rmax_quantile_in, rmax_quantile_out])) - soma_distances_trc
        np.fill_diagonal(soma_distances_trc, 0)

        input_matrices = [activity_correlation_all_trc, \
                                in_corr, hypembed_connectome_corr_in, eulembed_connectome_corr_in, \
                                out_corr, hypembed_connectome_corr_out, eulembed_connectome_corr_out, \
                                soma_distances_trc]

        # sanity check
        datas = [[out_corr, out_corr, in_corr, in_corr], \
                [hypembed_connectome_distance_out, eulembed_connectome_distance_out, hypembed_connectome_distance_in, eulembed_connectome_distance_in]]
        datas_names = ["out-hyp", "out-eul", "in-hyp", "in-eul"]

        figsanity, axssanity = plt.subplots(2,4,figsize=(4*4,4*2))
        for jj in range(4):
            xx = 1 - datas[0][jj]
            xx = np.triu(xx, k=1)
            xx = xx[xx!=0]
            yy = np.triu(datas[1][jj], k=1)
            yy = yy[yy!=0]
            axssanity[0,jj].scatter(xx, yy, alpha=0.1)

            rank_xx = rankdata(xx, method='dense')
            rank_yy = rankdata(yy, method='dense')
            axssanity[1,jj].scatter(rank_xx, rank_yy, alpha=0.1)

            slope, intercept, r_value, p_value, std_err = linregress(rank_xx, rank_yy)
            r_squared = r_value**2

            axssanity[0,jj].set_title(f"{datas_names[jj]}")
            axssanity[1,jj].set_title(f"{datas_names[jj]}: {np.round(r_squared,4)}")


        figsanity.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_sanity.png")

        # Top K neurons of consideration
        topK_values = ["all", int(W_corr.shape[0]/2)]
        print(f"topK_values: {topK_values}")

        # collection of all categories of reconstruction
        all_reconstruction_data = [
            [reconstruction(W, K, False)[0] for W in input_matrices] 
            for K in topK_values
        ]

        repeat_permute = 10
        all_reconstruction_data_permute_all = [
            [[reconstruction(W, K, "cellwise")[0] for W in input_matrices] for K in topK_values]
            for _ in range(repeat_permute)
        ]

        all_reconstruction_corr = [
            [reconstruction(W, K, False)[1] for W in input_matrices] 
            for K in topK_values
        ]

        reconstruction_names = ["Activity Correlation", \
                                "Connectome-In Correlation", "Connectome-In Hyp Embed", "Connectome-In Eul Embed", \
                                "Connectome-Out Correlation", "Connectome-Out Hyp Embed", "Connectome-Out Eul Embed", \
                                "Soma Distance"
                            ]

        reconstruction_corr_basic = all_reconstruction_corr[0]
        reconstruction_corr_basic_flatten = []

        # distribution plotting
        figcheck, axcheck = plt.subplots(1,len(input_matrices),figsize=(4*len(input_matrices),4))
        for j in range(len(input_matrices)):
            ccc = reconstruction_corr_basic[j]
            dd = np.triu_indices_from(ccc, k=1)
            upper_tri_values = ccc[dd].flatten()
            reconstruction_corr_basic_flatten.append(upper_tri_values)
            axcheck[j].hist(upper_tri_values, bins=50, density=False)
            padding = 0.05 * (max(upper_tri_values) - min(upper_tri_values))
            axcheck[j].set_xlim([min(upper_tri_values)-padding, max(upper_tri_values)+padding])
            axcheck[j].set_title(reconstruction_names[j])

        figcheck.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_checkcorr_D{embedding_dimension}_R{R_max}.png")

        # correlation of distribution plotting
        cc = len(reconstruction_corr_basic_flatten[:-1])
        input_matrics_corr, input_matrics_rank_diff = np.zeros((cc, cc)), np.zeros((cc, cc))
        for cc1 in range(cc):
            for cc2 in range(cc):
                mat1, mat2 = reconstruction_corr_basic_flatten[cc1], reconstruction_corr_basic_flatten[cc2]
                corr, _ = pearsonr(mat1, mat2)  
                rankdiff = np.mean(np.abs(list(rankdata(mat1, method='dense') - rankdata(mat2, method='dense'))))
                input_matrics_corr[cc1, cc2] = corr
                input_matrics_rank_diff[cc1, cc2] = rankdiff

        figcorr, axcorr = plt.subplots(1,2,figsize=(10*2,10))
        sns.heatmap(input_matrics_corr, ax=axcorr[0], cbar=True, square=True, center=0, cmap="coolwarm", annot=True, fmt=".2f", \
                    xticklabels=reconstruction_names[:-1], yticklabels=reconstruction_names[:-1])
        sns.heatmap(input_matrics_rank_diff, ax=axcorr[1], cbar=True, square=True, center=0, cmap="coolwarm", annot=True, fmt=".1f", \
                    xticklabels=reconstruction_names[:-1], yticklabels=reconstruction_names[:-1])
        figcorr.tight_layout()
        figcorr.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_corrcompare_D{embedding_dimension}_R{R_max}.png")

        # PSD
        def compute_psd(trace, fs):
            freqs, psd = welch(trace, fs, nperseg=1024)
            return freqs, psd

        fs = fps_value

        figtest, axtest = plt.subplots(len(selects),1,figsize=(10,len(selects)*1))

        for nn in selects:
            axtest[selects.index(nn)].plot(activity_extraction_extra[nn,:timecut], c=c_vals[selects.index(nn)], label='GroundTruth')
            # just use activity reconstruction for now (illustration purpose)
            all_reconstruction = [all_reconstruction_data[0][0]]
            for j in range(len(all_reconstruction)):
                axtest[selects.index(nn)].plot(all_reconstruction[j][nn,0:timecut], c=c_vals[0], label=reconstruction_names[j], linestyle='--')
                axtest[selects.index(nn)].set_title(f"Neuron {nn}")
                axtest[selects.index(nn)].set_xlim([-2, timecut+2])
        # for ax in axtest.flatten():
        #     ax.legend()
        figtest.tight_layout()
        figtest.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_test.png")

        timeuplst = [100,activity_extraction_extra_trc.shape[1]-1]
        # timeuplst = [100,1000,10000,20000,30000,activity_extraction_extra.shape[1]-1]
        ttlength = activity_extraction_extra_trc.shape[1]

        # add some random permuted data for the embedding results for sanity check
        # also add the correpsponding labels
        for kk in range(len(all_reconstruction_data)):
            # hyp in permute
            all_reconstruction_data[kk].append([all_reconstruction_data_permute_all[i][kk][2] for i in range(repeat_permute)])
            # eul in permute
            all_reconstruction_data[kk].append([all_reconstruction_data_permute_all[i][kk][3] for i in range(repeat_permute)])
            # hyp out permute
            all_reconstruction_data[kk].append([all_reconstruction_data_permute_all[i][kk][5] for i in range(repeat_permute)])
            # eu lout permute
            all_reconstruction_data[kk].append([all_reconstruction_data_permute_all[i][kk][6] for i in range(repeat_permute)])

        reconstruction_names.extend(["Connectome-In Hyp Embed Permuted", "Connectome-In Eul Embed Permuted", \
                                     "Connectome-Out Hyp Embed Permuted", "Connectome-Out Eul Embed Permuted"
                                ])

        # Approach: temporal moving window to calculate correlation
        KK = len(all_reconstruction_data)
        figact1, axsact1 = plt.subplots(KK,len(timeuplst),figsize=(4*len(timeuplst),4*KK))
        figact2, axsact2 = plt.subplots(KK,len(timeuplst),figsize=(4*len(timeuplst),4*KK))
        figact3, axsact3 = plt.subplots(KK,len(timeuplst),figsize=(4*len(timeuplst),4*KK))
        figact4, axsact4 = plt.subplots(KK,len(timeuplst),figsize=(4*len(timeuplst),4*KK))

        axsact1 = np.atleast_1d(axsact1)
        axsact2 = np.atleast_1d(axsact2)
        axsact3 = np.atleast_1d(axsact3)
        axsact4 = np.atleast_1d(axsact4)


        def vectorized_pearsonr(x, y):
            """Compute Pearson correlation coefficient vectorized over the first axis."""
            x_mean = x.mean(axis=1, keepdims=True)
            y_mean = y.mean(axis=1, keepdims=True)
            n = x.shape[1]
            cov = np.sum((x - x_mean) * (y - y_mean), axis=1)
            x_std = np.sqrt(np.sum((x - x_mean)**2, axis=1))
            y_std = np.sqrt(np.sum((y - y_mean)**2, axis=1))
            return cov / (x_std * y_std)

        allk_medians = []
        for k in range(KK):
            kk_medians = []
            for timeup in timeuplst:
                print(f"k:{k}; timeup: {timeup}")
                summ = []
                # iterate across neurons
                # unroll time dimension
                # we write this part explicitly (though redundant) for clarity
                for i in range(activity_extraction_extra_trc.shape[0]):
                    # print(i)
                    # Create a matrix of all windows for neuron i
                    gt = np.lib.stride_tricks.sliding_window_view(activity_extraction_extra_trc[i], window_shape=timeup)

                    gt_a = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][0][i], window_shape=timeup)
                    gt_c1 = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][1][i], window_shape=timeup)
                    gt_c1_hypembed = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][2][i], window_shape=timeup)
                    gt_c1_eulembed = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][3][i], window_shape=timeup)
                    gt_c2 = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][4][i], window_shape=timeup)
                    gt_c2_hypembed = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][5][i], window_shape=timeup)
                    gt_c2_eulembed = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][6][i], window_shape=timeup)
                    gt_soma = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][7][i], window_shape=timeup)

                    # Calculate correlations in a vectorized manner
                    corr_with_a = vectorized_pearsonr(gt, gt_a)
                    corr_with_c1 = vectorized_pearsonr(gt, gt_c1)
                    corr_with_c1_hypembed = vectorized_pearsonr(gt, gt_c1_hypembed)
                    corr_with_c1_eulembed = vectorized_pearsonr(gt, gt_c1_eulembed)
                    corr_with_c2 = vectorized_pearsonr(gt, gt_c2)
                    corr_with_c2_hypembed = vectorized_pearsonr(gt, gt_c2_hypembed)
                    corr_with_c2_eulembed = vectorized_pearsonr(gt, gt_c2_eulembed)
                    corr_with_soma = vectorized_pearsonr(gt, gt_soma)

                    # notice that the there are multiple choices in the permutation matrix
                    hypinpm, eulinpm, hypoutpm, euloutpm = [], [], [], []
                    for lll in range(repeat_permute):
                        gt_c1_hypembed_pm = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][8][lll][i], window_shape=timeup)
                        gt_c1_eulembed_pm = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][9][lll][i], window_shape=timeup)
                        gt_c2_hypembed_pm = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][10][lll][i], window_shape=timeup)
                        gt_c2_eulembed_pm = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][11][lll][i], window_shape=timeup)

                        corr_with_c1_hypembed_pm = vectorized_pearsonr(gt, gt_c1_hypembed_pm)
                        corr_with_c1_eulembed_pm = vectorized_pearsonr(gt, gt_c1_eulembed_pm)
                        corr_with_c2_hypembed_pm = vectorized_pearsonr(gt, gt_c2_hypembed_pm)
                        corr_with_c2_eulembed_pm = vectorized_pearsonr(gt, gt_c2_eulembed_pm)

                        hypinpm.append(corr_with_c1_hypembed_pm.mean())
                        eulinpm.append(corr_with_c1_eulembed_pm.mean())
                        hypoutpm.append(corr_with_c2_hypembed_pm.mean())
                        euloutpm.append(corr_with_c2_eulembed_pm.mean())

                    # Calculate mean of correlations across all windows for neuron i
                    summ.append([corr_with_a.mean(), \
                                corr_with_c1.mean(), corr_with_c1_hypembed.mean(), corr_with_c1_eulembed.mean(), \
                                corr_with_c2.mean(), corr_with_c2_hypembed.mean(), corr_with_c2_eulembed.mean(), \
                                corr_with_soma.mean(), \
                                np.mean(hypinpm), np.mean(eulinpm), np.mean(hypoutpm), np.mean(euloutpm)
                            ])

                summ = np.array(summ)
                medians = {}
                input_groups = [[0,1,4,7],[1,2,3],[4,5,6],[8,9,10,11]]
                all_use = list(range(0,12))
                axsacts = [axsact1, axsact2, axsact3, axsact4]
                for group in input_groups:
                    for j in group:  
                        mm = np.median(summ[:, j])
                        if j in all_use:
                            medians[j] = mm
                            all_use.remove(j)

                        # print(summ[:, j].shape)
                        axsacts[input_groups.index(group)][k,timeuplst.index(timeup)].hist(summ[:, j], bins=10, density=True, color=c_vals[j], alpha=0.7,
                                                            label=f"{reconstruction_names[j]}")
                        axsacts[input_groups.index(group)][k,timeuplst.index(timeup)].legend()

                for axsact in axsacts:
                    axsact[k,timeuplst.index(timeup)].set_title(f"Window Length {timeup}: K{topK_values[k]}")

                # Oct 7th
                # need sorting otherwise the order is inconsistent with the legend 
                # because the adding order is followed by input_groups
                medians = dict(sorted(medians.items()))
                medians = [medians[key] for key in medians]

                kk_medians.append(medians)
                
            allk_medians.append(kk_medians)

        for axsact in axsacts:
            for ax in axsact.flatten():
                ax.legend()
                ax.set_xlabel("Correlation")
                ax.set_ylabel("Frequency")

        figact1.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_actcompare1_D{embedding_dimension}_R{R_max}.png")
        figact2.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_actcompare2_D{embedding_dimension}_R{R_max}.png")
        figact3.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_actcompare3_D{embedding_dimension}_R{R_max}.png")
        figact4.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_actcompare4_D{embedding_dimension}_R{R_max}.png")


        figactshow, axsactshow = plt.subplots(1,len(allk_medians),figsize=(10*len(allk_medians),10))

        linestyles = ["-", "--", "-.", ":", (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (5, 10))]
        plotstyles = [[c_vals[0], linestyles[0]], \
                    [c_vals[1], linestyles[0]], [c_vals[1], linestyles[1]], [c_vals[1], linestyles[2]], \
                    [c_vals[2], linestyles[0]], [c_vals[2], linestyles[1]], [c_vals[2], linestyles[2]], \
                    [c_vals[3], linestyles[0]],
                    [c_vals[4], linestyles[0]], [c_vals[4], linestyles[1]], [c_vals[4], linestyles[2]], [c_vals[4], linestyles[3]]
                ]

        for i in range(len(allk_medians)):
            medians = np.array(allk_medians[i])
            for j in range(medians.shape[1]):
                axsactshow[i].plot(timeuplst, medians[:,j], color=plotstyles[j][0], linestyle=plotstyles[j][1], linewidth=3, label=reconstruction_names[j])
            axsactshow[i].set_title(f"K={topK_values[i]}")
            axsactshow[i].legend()

        for ax in axsactshow:
            ax.axhline(y=metadata["gt_median_corr"], color='black', linestyle='--')
            ax.set_xlabel("Window Length")
            ax.set_ylabel("Median Correlation")

        

        figactshow.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_actcompareshow_D{embedding_dimension}_R{R_max}.png")

        metadata["timeuplst"] = timeuplst
        metadata["allk_medians"] = allk_medians
        
        with open(f"./output/fromac_session_{session_info}_scan_{scan_info}_metadata_D{embedding_dimension}_R{R_max}.pkl", "wb") as pickle_file:
            pickle.dump(metadata, pickle_file)

    session_ds.close()

def benchmark_with_rnn(trial_index):

    activity_extraction_extra_trc = scipy.io.loadmat(f"./zz_data_rnn/rnn_activity_{trial_index}.mat")['activity']
    activity_extraction_extra = activity_extraction_extra_trc
    activity_correlation_all_trc = np.corrcoef(activity_extraction_extra_trc, rowvar=True)
    np.fill_diagonal(activity_correlation_all_trc, 0)

    metadata = {}

    def reconstruction(W, K="all", permute=False):
        """
        add option to only take top K components in reconstruction
        if K=None, then using all neurons (without any low-pass filter)
        """
        if K == "all":
            K = W.shape[0]

        filtered_matrix = np.zeros_like(W)
        for i in range(W.shape[0]):
            top_k_indices = np.argsort(W[i])[-K:]        
            filtered_matrix[i, top_k_indices] = W[i, top_k_indices]
        
        row_sums_a = np.sum(np.abs(filtered_matrix), axis=1)

        filtered_matrix_normalized = filtered_matrix / row_sums_a[:, np.newaxis]
        assert np.sum(np.diag(filtered_matrix_normalized)) == 0

        # permute the matrix but still keep it symmetric
        if permute == "rowwise":
            filtered_matrix_normalized = activity_helper.permute_symmetric_matrix(filtered_matrix_normalized)
        elif permute == "cellwise":
            filtered_matrix_normalized = activity_helper.permute_symmetric_matrix_cellwise(filtered_matrix_normalized)

        activity_reconstruct_a = filtered_matrix_normalized @ activity_extraction_extra_trc
        return activity_reconstruct_a, filtered_matrix_normalized

    R_max = 1

    hyp_name = f"./mds-rnn-results/rnn_embed_{trial_index}.mat"
    out_corr = scipy.io.loadmat(hyp_name)['Ddists'][0][1]
    out_corr = 1 - out_corr
    np.fill_diagonal(out_corr, 0)

    in_corr = scipy.io.loadmat(hyp_name)['Ddists'][0][2]
    in_corr = 1 - in_corr
    np.fill_diagonal(in_corr, 0)

    hypembed_name = f"./mds-rnn-results/rnn_embed_{trial_index}_hypdist.mat"
    hypembed_connectome_distance_out = scipy.io.loadmat(hypembed_name)['hyp_dist'][0][1]


    rmax_quantile_out = activity_helper.find_quantile(hypembed_connectome_distance_out, float(R_max))
    metadata["rmax_quantile_out"] = rmax_quantile_out

    hypembed_connectome_corr_out = float(R_max) - hypembed_connectome_distance_out
    # hypembed_connectome_corr_out = np.max(hypembed_connectome_distance_out) - hypembed_connectome_distance_out
    np.fill_diagonal(hypembed_connectome_corr_out, 0)

    hypembed_connectome_distance_in = scipy.io.loadmat(hypembed_name)['hyp_dist'][0][2]

    rmax_quantile_in = activity_helper.find_quantile(hypembed_connectome_distance_in, float(R_max))
    metadata["rmax_quantile_in"] = rmax_quantile_in

    hypembed_connectome_corr_in = float(R_max) - hypembed_connectome_distance_in
    # hypembed_connectome_corr_in = np.max(hypembed_connectome_distance_in) - hypembed_connectome_distance_in
    np.fill_diagonal(hypembed_connectome_corr_in, 0)

    # load Euclidean embedding coordinate
    # calcualate the pairwise Euclidean distance afterward    

    eulembed_name = f"./mds-rnn-results/rnn_embed_eulmds_{trial_index}.mat"
    eulembed_connectome = scipy.io.loadmat(eulembed_name)['eulmdsembed'][0][1]
    eulembed_connectome_distance_out = squareform(pdist(eulembed_connectome, metric='euclidean'))
    metadata["rmax_eul_out_distance"] = np.max(eulembed_connectome_distance_out)
    # 
    eulembed_connectome_corr_out = activity_helper.find_value_for_quantile(eulembed_connectome_distance_out, rmax_quantile_out) - eulembed_connectome_distance_out
    # eulembed_connectome_corr_out = np.max(eulembed_connectome_distance_out) - eulembed_connectome_distance_out
    np.fill_diagonal(eulembed_connectome_corr_out, 0)

    eulembed_connectome = scipy.io.loadmat(eulembed_name)['eulmdsembed'][0][2]
    eulembed_connectome_distance_in = squareform(pdist(eulembed_connectome, metric='euclidean'))
    metadata["rmax_eul_in_distance"] = np.max(eulembed_connectome_distance_in)
    # 
    eulembed_connectome_corr_in = activity_helper.find_value_for_quantile(eulembed_connectome_distance_in, rmax_quantile_in) - eulembed_connectome_distance_in
    # eulembed_connectome_corr_in = np.max(eulembed_connectome_distance_in) - eulembed_connectome_distance_in
    np.fill_diagonal(eulembed_connectome_corr_in, 0)


    input_matrices = [activity_correlation_all_trc, \
                            in_corr, hypembed_connectome_corr_in, eulembed_connectome_corr_in, \
                            out_corr, hypembed_connectome_corr_out, eulembed_connectome_corr_out
                    ]

    # sanity check
    datas = [[out_corr, out_corr, in_corr, in_corr], \
            [hypembed_connectome_distance_out, eulembed_connectome_distance_out, hypembed_connectome_distance_in, eulembed_connectome_distance_in]]
    datas_names = ["out-hyp", "out-eul", "in-hyp", "in-eul"]

    figsanity, axssanity = plt.subplots(2,4,figsize=(4*4,4*2))
    for jj in range(4):
        xx = 1 - datas[0][jj]
        xx = np.triu(xx, k=1)
        xx = xx[xx!=0]
        yy = np.triu(datas[1][jj], k=1)
        yy = yy[yy!=0]
        axssanity[0,jj].scatter(xx, yy, alpha=0.1)

        rank_xx = rankdata(xx, method='dense')
        rank_yy = rankdata(yy, method='dense')
        axssanity[1,jj].scatter(rank_xx, rank_yy, alpha=0.1)

        slope, intercept, r_value, p_value, std_err = linregress(rank_xx, rank_yy)
        r_squared = r_value**2

        axssanity[0,jj].set_title(f"{datas_names[jj]}")
        axssanity[1,jj].set_title(f"{datas_names[jj]}: {np.round(r_squared,4)}")


    figsanity.savefig(f"./output_rnn/fromac_session_rnn_{trial_index}_sanity.png")

    # Top K neurons of consideration
    topK_values = ["all", 30]
    print(f"topK_values: {topK_values}")

    # collection of all categories of reconstruction
    all_reconstruction_data = [
        [reconstruction(W, K, False)[0] for W in input_matrices] 
        for K in topK_values
    ]

    all_reconstruction_corr = [
        [reconstruction(W, K, False)[1] for W in input_matrices] 
        for K in topK_values
    ]

    reconstruction_names = ["Activity Correlation", \
                            "Connectome-In Correlation", "Connectome-In Hyp Embed", "Connectome-In Eul Embed", \
                            "Connectome-Out Correlation", "Connectome-Out Hyp Embed", "Connectome-Out Eul Embed", \
                        ]

    reconstruction_corr_basic = all_reconstruction_corr[0]
    reconstruction_corr_basic_flatten = []

    # distribution plotting
    figcheck, axcheck = plt.subplots(1,len(input_matrices),figsize=(4*len(input_matrices),4))
    for j in range(len(input_matrices)):
        ccc = reconstruction_corr_basic[j]
        dd = np.triu_indices_from(ccc, k=1)
        upper_tri_values = ccc[dd].flatten()
        reconstruction_corr_basic_flatten.append(upper_tri_values)
        axcheck[j].hist(upper_tri_values, bins=50, density=False)
        padding = 0.05 * (max(upper_tri_values) - min(upper_tri_values))
        axcheck[j].set_xlim([min(upper_tri_values)-padding, max(upper_tri_values)+padding])
        axcheck[j].set_title(reconstruction_names[j])

    figcheck.savefig(f"./output/fromac_session_rnn_{trial_index}_checkcorr.png")

    # correlation of distribution plotting
    cc = len(reconstruction_corr_basic_flatten[:-1])
    input_matrics_corr, input_matrics_rank_diff = np.zeros((cc, cc)), np.zeros((cc, cc))
    for cc1 in range(cc):
        for cc2 in range(cc):
            mat1, mat2 = reconstruction_corr_basic_flatten[cc1], reconstruction_corr_basic_flatten[cc2]
            corr, _ = pearsonr(mat1, mat2)  
            rankdiff = np.mean(np.abs(list(rankdata(mat1, method='dense') - rankdata(mat2, method='dense'))))
            input_matrics_corr[cc1, cc2] = corr
            input_matrics_rank_diff[cc1, cc2] = rankdiff

    figcorr, axcorr = plt.subplots(1,2,figsize=(10*2,10))
    sns.heatmap(input_matrics_corr, ax=axcorr[0], cbar=True, square=True, center=0, cmap="coolwarm", annot=True, fmt=".2f", \
                xticklabels=reconstruction_names[:-1], yticklabels=reconstruction_names[:-1])
    sns.heatmap(input_matrics_rank_diff, ax=axcorr[1], cbar=True, square=True, center=0, cmap="coolwarm", annot=True, fmt=".1f", \
                xticklabels=reconstruction_names[:-1], yticklabels=reconstruction_names[:-1])
    figcorr.tight_layout()
    figcorr.savefig(f"./output_rnn/fromac_session_rnn_{trial_index}_corrcompare.png")

    # PSD
    selects = [1,12,18,25,32,36]
    timecut = 100

    figtest, axtest = plt.subplots(len(selects),1,figsize=(10,len(selects)*1))

    for nn in selects:
        axtest[selects.index(nn)].plot(activity_extraction_extra[nn,:timecut], c=c_vals[selects.index(nn)], label='GroundTruth')
        # just use activity reconstruction for now (illustration purpose)
        all_reconstruction = [all_reconstruction_data[0][0]]
        for j in range(len(all_reconstruction)):
            axtest[selects.index(nn)].plot(all_reconstruction[j][nn,0:timecut], c=c_vals[0], label=reconstruction_names[j], linestyle='--')
            axtest[selects.index(nn)].set_title(f"Neuron {nn}")
            axtest[selects.index(nn)].set_xlim([-2, timecut+2])
    # for ax in axtest.flatten():
    #     ax.legend()
    figtest.tight_layout()
    figtest.savefig(f"./output/fromac_session_rnn_{trial_index}_test.png")

    timeuplst = [100,activity_extraction_extra_trc.shape[1]-1]
    # timeuplst = [100,1000,10000,20000,30000,activity_extraction_extra.shape[1]-1]
    ttlength = activity_extraction_extra_trc.shape[1]

    # Approach: temporal moving window to calculate correlation
    KK = len(all_reconstruction_data)
    figact1, axsact1 = plt.subplots(KK,len(timeuplst),figsize=(4*len(timeuplst),4*KK))
    figact2, axsact2 = plt.subplots(KK,len(timeuplst),figsize=(4*len(timeuplst),4*KK))
    figact3, axsact3 = plt.subplots(KK,len(timeuplst),figsize=(4*len(timeuplst),4*KK))

    axsact1 = np.atleast_1d(axsact1)
    axsact2 = np.atleast_1d(axsact2)
    axsact3 = np.atleast_1d(axsact3)

    def vectorized_pearsonr(x, y):
        """Compute Pearson correlation coefficient vectorized over the first axis."""
        x_mean = x.mean(axis=1, keepdims=True)
        y_mean = y.mean(axis=1, keepdims=True)
        n = x.shape[1]
        cov = np.sum((x - x_mean) * (y - y_mean), axis=1)
        x_std = np.sqrt(np.sum((x - x_mean)**2, axis=1))
        y_std = np.sqrt(np.sum((y - y_mean)**2, axis=1))
        return cov / (x_std * y_std)

    allk_medians = []
    for k in range(KK):
        kk_medians = []
        for timeup in timeuplst:
            print(f"k:{k}; timeup: {timeup}")
            summ = []
            # iterate across neurons
            # unroll time dimension
            # we write this part explicitly (though redundant) for clarity
            for i in range(activity_extraction_extra_trc.shape[0]):
                # print(i)
                # Create a matrix of all windows for neuron i
                gt = np.lib.stride_tricks.sliding_window_view(activity_extraction_extra_trc[i], window_shape=timeup)

                gt_a = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][0][i], window_shape=timeup)
                gt_c1 = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][1][i], window_shape=timeup)
                gt_c1_hypembed = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][2][i], window_shape=timeup)
                gt_c1_eulembed = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][3][i], window_shape=timeup)
                gt_c2 = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][4][i], window_shape=timeup)
                gt_c2_hypembed = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][5][i], window_shape=timeup)
                gt_c2_eulembed = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][6][i], window_shape=timeup)

                # Calculate correlations in a vectorized manner
                corr_with_a = vectorized_pearsonr(gt, gt_a)
                corr_with_c1 = vectorized_pearsonr(gt, gt_c1)
                corr_with_c1_hypembed = vectorized_pearsonr(gt, gt_c1_hypembed)
                corr_with_c1_eulembed = vectorized_pearsonr(gt, gt_c1_eulembed)
                corr_with_c2 = vectorized_pearsonr(gt, gt_c2)
                corr_with_c2_hypembed = vectorized_pearsonr(gt, gt_c2_hypembed)
                corr_with_c2_eulembed = vectorized_pearsonr(gt, gt_c2_eulembed)

                # Calculate mean of correlations across all windows for neuron i
                summ.append([corr_with_a.mean(), \
                            corr_with_c1.mean(), corr_with_c1_hypembed.mean(), corr_with_c1_eulembed.mean(), \
                            corr_with_c2.mean(), corr_with_c2_hypembed.mean(), corr_with_c2_eulembed.mean(), \
                        ])

            summ = np.array(summ)
            medians = {}
            input_groups = [[0,1,4],[1,2,3],[4,5,6]]
            all_use = list(range(0,12))
            axsacts = [axsact1, axsact2, axsact3]
            for group in input_groups:
                for j in group:  
                    mm = np.median(summ[:, j])
                    if j in all_use:
                        medians[j] = mm
                        all_use.remove(j)

                    # print(summ[:, j].shape)
                    axsacts[input_groups.index(group)][k,timeuplst.index(timeup)].hist(summ[:, j], bins=10, density=True, color=c_vals[j], alpha=0.7,
                                                        label=f"{reconstruction_names[j]}")
                    axsacts[input_groups.index(group)][k,timeuplst.index(timeup)].legend()

            for axsact in axsacts:
                axsact[k,timeuplst.index(timeup)].set_title(f"Window Length {timeup}: K{topK_values[k]}")

            # Oct 7th
            # need sorting otherwise the order is inconsistent with the legend 
            # because the adding order is followed by input_groups
            medians = dict(sorted(medians.items()))
            medians = [medians[key] for key in medians]

            kk_medians.append(medians)
            
        allk_medians.append(kk_medians)

    for axsact in axsacts:
        for ax in axsact.flatten():
            ax.legend()
            ax.set_xlabel("Correlation")
            ax.set_ylabel("Frequency")

    figact1.savefig(f"./output_rnn/fromac_session_rnn_{trial_index}_actcompare1.png")
    figact2.savefig(f"./output_rnn/fromac_session_rnn_{trial_index}_actcompare2.png")
    figact3.savefig(f"./output_rnn/fromac_session_rnn_{trial_index}_actcompare3.png")


    figactshow, axsactshow = plt.subplots(1,len(allk_medians),figsize=(10*len(allk_medians),10))

    linestyles = ["-", "--", "-.", ":", (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (5, 10))]
    plotstyles = [[c_vals[0], linestyles[0]], \
                [c_vals[1], linestyles[0]], [c_vals[1], linestyles[1]], [c_vals[1], linestyles[2]], \
                [c_vals[2], linestyles[0]], [c_vals[2], linestyles[1]], [c_vals[2], linestyles[2]], \
                [c_vals[3], linestyles[0]],
                [c_vals[4], linestyles[0]], [c_vals[4], linestyles[1]], [c_vals[4], linestyles[2]], [c_vals[4], linestyles[3]]
            ]

    for i in range(len(allk_medians)):
        medians = np.array(allk_medians[i])
        print(medians)
        for j in range(medians.shape[1]):
            axsactshow[i].plot(timeuplst, medians[:,j], color=plotstyles[j][0], linestyle=plotstyles[j][1], linewidth=3, label=reconstruction_names[j], alpha=0.7)
        axsactshow[i].set_title(f"K={topK_values[i]}")
        axsactshow[i].legend()

    for ax in axsactshow:
        ax.set_xlabel("Window Length")
        ax.set_ylabel("Median Correlation")

    figactshow.savefig(f"./output_rnn/fromac_session_rnn_{trial_index}_actcompareshow.png")

    metadata["timeuplst"] = timeuplst
    metadata["allk_medians"] = allk_medians
    
    with open(f"./output_rnn/fromac_session_rnn_{trial_index}_metadata.pkl", "wb") as pickle_file:
        pickle.dump(metadata, pickle_file)


def all_run(R_max, embedding_dimension, raw_data):
    # session_scan = [[4,7],[5,6],[5,7],[6,2],[6,4],[6,6],[7,4],[8,5],[9,3],[9,4]]
    session_scan = [[7,5],[6,7],[7,3],[5,3]]
    for_construction = 0
    for ss in session_scan:
        run(ss[0], ss[1], for_construction, R_max=R_max, embedding_dimension=embedding_dimension, raw_data=raw_data)
        gc.collect()

    gc.collect()
    

if __name__ == "__main__":

    # for trial_index in range(10):
    #     benchmark_with_rnn(trial_index)

    microns_across_scans.microns_across_scans_rnn(0)