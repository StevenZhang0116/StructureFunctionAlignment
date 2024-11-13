import numpy as np
import pandas as pd
import time
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import xarray as xr
from scipy.stats import pearsonr
import pickle
import gc
from os.path import join as pjoin

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
sys.path.append("/gscratch/amath/zihan-zhang/spatial/demo/mixture")

import helper
import activity_helper
import mix_helper

import microns_across_scans

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]

linestyles = ["-", "--", "-.", ":", (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (5, 10))]
plotstyles = [[c_vals[0], linestyles[0]], \
            [c_vals[1], linestyles[0]], [c_vals[1], linestyles[1]], [c_vals[1], linestyles[2]], \
            [c_vals[2], linestyles[0]], [c_vals[2], linestyles[1]], [c_vals[2], linestyles[2]], \
            [c_vals[3], linestyles[0]],
            [c_vals[4], linestyles[0]], [c_vals[4], linestyles[1]], [c_vals[4], linestyles[2]], [c_vals[4], linestyles[3]]
        ]


def all_run(R_max, embedding_dimension, raw_data, whethernoise, whetherconnectome):
    """
    """
    # session_scan = [[8,5],[4,7],[6,6],[5,3],[5,6],[5,7],[6,2],[6,4],[7,3],[7,5],[9,3],[9,4]]
    session_scan = [[8,5],[4,7],[6,6],[5,3],[5,6],[5,7],[6,2],[7,3],[7,5],[9,3],[9,4],[6,4]]
    # session_scan = [[5,3],[5,6],[5,7],[6,2],[7,3],[7,5],[9,3],[9,4],[6,4]]

    # session_scan = [[6,4]]
    for_construction = False
    for ss in session_scan:
        run(ss[0], ss[1], for_construction, R_max=R_max, embedding_dimension=embedding_dimension, raw_data=raw_data, \
                    whethernoise=whethernoise, whetherconnectome=whetherconnectome)
        gc.collect()

    gc.collect()

def run(session_info, scan_info, for_construction, R_max, embedding_dimension, raw_data, whethernoise, whetherconnectome):
    """
    By default, whethernoise == normal, whetherconnectome == count
    Others are considered as compartive/ablation study
    """
    assert whethernoise in ["normal", "glia", "noise", "all"]
    assert whetherconnectome in ["count", "binary", "psd"]

    metadata = {}

    metadata["session_info"] = session_info
    metadata["scan_info"] = scan_info
    metadata["whethernoise"] = whethernoise
    metadata["whetherconnectome"] = whetherconnectome

    # index
    sub_correlation_index = "nan_fill"

    # proofread & coregisteration overlap information
    prf_coreg = pd.read_csv("./microns/prf_coreg.csv")
    # take the information belong to that specific session
    prf_coreg = prf_coreg[(prf_coreg["session"] == session_info) & (prf_coreg["scan_idx"] == scan_info)]
    print(len(prf_coreg))

    # read in microns cell/synapse information
    cell_table = pd.read_feather("../microns_cell_tables/sven/microns_cell_annos_240524.feather")

    if whethernoise == "normal":
        matching_axon = cell_table.index[(cell_table["status_axon"].isin(["extended", "clean"])) & 
                                         (cell_table["classification_system"].isin(["excitatory_neuron", "inhibitory_neuron"]))
                                        ].tolist()
        
        matching_dendrite = cell_table.index[(cell_table["full_dendrite"] == True) & 
                                             (cell_table["classification_system"].isin(["excitatory_neuron", "inhibitory_neuron"]))
                                            ].tolist()

    elif whethernoise == "glia":
        matching_axon = cell_table.index[(cell_table["status_axon"].isin(["extended", "clean"]))].tolist()
        matching_dendrite = cell_table.index[(cell_table["full_dendrite"] == True)].tolist()

    elif whethernoise == "noise":
        matching_axon = cell_table.index[cell_table["classification_system"].isin(["excitatory_neuron", "inhibitory_neuron"])].tolist()
        matching_dendrite = cell_table.index[cell_table["classification_system"].isin(["excitatory_neuron", "inhibitory_neuron"])].tolist()

    elif whethernoise == "all":
        matching_axon = cell_table.index.tolist()
        matching_dendrite = cell_table.index.tolist()

    synapse_table = pd.read_feather("../microns_cell_tables/sven/synapses_minnie65_phase3_v1_943_combined_incl_trafo_240522.feather")

    # check which neurons are inhibitory
    classification_map = {
        "excitatory_neuron": 1,
        "nonneuron": 0,
        "inhibitory_neuron": -1
    }
    classification_list = cell_table["classification_system"].map(classification_map).tolist()

    # choose structural confident neurons
    # will be used for Betti curve analysis
    goodaxons = cell_table[(cell_table["status_axon"].isin(["extended", "clean"]))].index.tolist()
    gooddendrites = cell_table[(cell_table["full_dendrite"] == True)].index.tolist()

    good_ct = cell_table[(cell_table["status_axon"].isin(["extended", "clean"])) & (cell_table["full_dendrite"] == True)]
    good_ct = good_ct[good_ct["classification_system"] == "excitatory_neuron"]
    good_ct_indices = good_ct.index.tolist()

    save_filename = f"./microns/functional_xr/functional_session_{session_info}_scan_{scan_info}.nc"
    session_ds = xr.open_dataset(save_filename)

    # print(session_ds.stim_on.to_pandas().value_counts())
    # time.sleep(1000)

    prf_coreg_ids = set(prf_coreg["pt_root_id"])

    fps_value = session_ds.attrs.get("fps", None)

    selected_neurons = []
    neuron_types = []
    activity_extraction = []
    activity_extraction_extra = []

    for index, pr_root_id in enumerate(cell_table["pt_root_id"]):
        if pr_root_id in prf_coreg_ids:
            cell_row = cell_table[cell_table["pt_root_id"] == pr_root_id]
            # only select neurons that are excitatory 
            if cell_row["classification_system"].item() in ["excitatory_neuron"]:

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

    # selected neuron
    selectss_df = cell_table.loc[selected_neurons].reset_index(drop=True)

    assert all(selectss_df["classification_system"] == "excitatory_neuron"), "Not all values are 'excitatory_neuron'"

    soma_locations = selectss_df[['pt_position_x', 'pt_position_y', 'pt_position_z']].to_numpy()
    soma_distances = squareform(pdist(soma_locations, metric='euclidean'))

    # plot some activity trace
    selects = [1,12,18,25,32,34]
    ttt = activity_extraction_extra.shape[1]
    timecut = int(ttt/100)
    timechosen = [timecut/4, timecut/2, 3*timecut/4]

    stimulus_name = pjoin("./microns/",'movie_downsample_xr', f'movie_downsample_session_{session_info}_scan_{scan_info}.nc')
    stimulus_ds = xr.open_dataset(stimulus_name) # stimulus file
    # find the intervals where the stimulus is on
    stim_on_time = session_ds.stim_on.to_pandas().values
    stimon_intervals = activity_helper.find_intervals(stim_on_time)
    stimon_intervals = stimon_intervals["1"]
    # exclude the intervals that are too short
    stimon_intervals = [stimon_interval for stimon_interval in stimon_intervals if stimon_interval[1] - stimon_interval[0] > ttt/100]

    fignewact, axsnewact = plt.subplots(len(selects),1,figsize=(10,len(selects)*1))
    figstimulus, axsstimulus = plt.subplots(1,len(timechosen),figsize=(4*len(timechosen),4))
    for ni in selects:
        axsnewact[selects.index(ni)].plot([i / fps_value for i in range(timecut)], activity_extraction_extra[ni,:timecut], c=c_vals[selects.index(ni)])
        axsnewact[selects.index(ni)].set_title(f"Neuron {ni}")
        axsnewact[selects.index(ni)].set_xlim([-1, timecut/fps_value+1])
        for tc in timechosen:
            axsnewact[selects.index(ni)].axvline(tc / fps_value, color='black', linestyle='--')
            video_frame = stimulus_ds.isel(frame_times=int(tc)).movie.to_numpy()
            axsstimulus[timechosen.index(tc)].imshow(video_frame, cmap='gray')
            axsstimulus[timechosen.index(tc)].set_xticks([]) 
            axsstimulus[timechosen.index(tc)].set_yticks([])

    fignewact.tight_layout()
    figstimulus.tight_layout()
    # fignewact.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_activity_new.png")
    # figstimulus.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_astimulus_new.png")

    # calculate activity correlation
    metric_name = "correlation"
    metadata["metric_name"] = metric_name
    # activity_correlation_all = np.corrcoef(activity_extraction_extra, rowvar=True)
    activity_correlation_all = activity_helper.all_metric(activity_extraction_extra, metric_name)
    activity_cov_all = np.cov(activity_extraction_extra, rowvar=True)
    
    metadata["num_neurons"] = activity_extraction_extra.shape[0]

    dd = np.triu_indices_from(activity_correlation_all, k=1)
    upper_tri_values = activity_correlation_all[dd].flatten()
    gt_median_corr = np.median(upper_tri_values)

    metadata["gt_median_corr"] = gt_median_corr

    binary_count, psd_count, syn_count, _ = helper.create_connectivity_as_whole(cell_table, synapse_table)
    # # Oct 29th: Sanity check should work properly
    # helper.sanity_check_of_connectome(syn_count, "count", cell_table, synapse_table)
    # helper.sanity_check_of_connectome(binary_count, "binary", cell_table, synapse_table)
    # helper.sanity_check_of_connectome(psd_count, "psd", cell_table, synapse_table)

    if whetherconnectome == "count":
        used_structure = syn_count
    elif whetherconnectome == "binary":
        used_structure = binary_count
    elif whetherconnectome == "psd":
        used_structure = psd_count

    # make inhibitory neurons (output connection) to have negative sign
    inhindex = True
    metadata["inhindex"] = inhindex 
    if inhindex:
        print("Make inhibitory neurons to have negative sign")
        for i in range(len(classification_list)):
            if classification_list[i] == -1:
                used_structure[i,:] = -1 * used_structure[i,:]
    
    if whethernoise in ["noise", "glia"]: # 
        W_goodneurons_row_info = used_structure[good_ct_indices,:]
        W_goodneurons_col_info = used_structure[:,good_ct_indices]
    elif whethernoise == "normal":
        W_goodneurons_row_info = used_structure[np.ix_(good_ct_indices, gooddendrites)]
        W_goodneurons_col_info = used_structure[np.ix_(goodaxons,good_ct_indices)]
    
    print(f"W_goodneurons_row_info: {W_goodneurons_row_info.shape}")
    print(f"W_goodneurons_col_info: {W_goodneurons_col_info.shape}")

    W_goodneurons_row = np.corrcoef(W_goodneurons_row_info, rowvar=True)
    W_goodneurons_row = activity_helper.remove_nan_inf_union(W_goodneurons_row)

    W_goodneurons_col = np.corrcoef(W_goodneurons_col_info, rowvar=False)
    W_goodneurons_col = activity_helper.remove_nan_inf_union(W_goodneurons_col)

    print(f"Original selected_neurons: {len(selected_neurons)}")

    correlation_index_lst = ["column", "row"]
    W_corrs_all, diags, offdiags, W_samples, indices_delete_lst = [], [], [], [], []
    p_values_all = []
    
    for_metric = []

    fig, axs = plt.subplots(2,3,figsize=(4*3,4*2))
    fig_dist, ax_dist = plt.subplots(1,3,figsize=(4*3,4*1))

    for correlation_index in correlation_index_lst:
        corrind = correlation_index_lst.index(correlation_index)

        external_data = True
        if not external_data:
            W_corr = np.corrcoef(W, rowvar=False if correlation_index=="column" else True) 
        else:
            # calculated correlation using weighted connectome
            if correlation_index == "row":
                W_trc = used_structure[selected_neurons,:]
                W_trc = W_trc[:,matching_dendrite]
                W_corr = np.corrcoef(W_trc, rowvar=True)
                W_cov = np.cov(W_trc, rowvar=True)

            else:
                W_trc = used_structure[:,selected_neurons]
                W_trc = W_trc[matching_axon,:]
                W_corr = np.corrcoef(W_trc, rowvar=False)
                W_cov = np.cov(W_trc, rowvar=False)

            
            W_backupcheck = used_structure[np.ix_(selected_neurons, selected_neurons)]
            W_samples.append(W_trc)

        W_corr_all = W_corr
        W_corr, activity_correlation, indices_to_delete = activity_helper.sanity_check_W(W_corr_all, activity_correlation_all)

        W_cov = activity_helper.row_column_delete(W_cov, indices_to_delete)
        activity_cov = activity_helper.row_column_delete(activity_cov_all, indices_to_delete)
        assert W_cov.shape == activity_cov.shape

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
        # relation_matrix_compare = np.zeros((shape_check, shape_check))
        for i in range(shape_check):
            for j in range(shape_check):
                if sub_correlation_index == "zero_out":
                    correlation, _ = pearsonr(activity_correlation[i], W_corr[j])
                elif sub_correlation_index == "nan_fill":
                    correlation = activity_helper.pearson_correlation_with_nans(activity_correlation[i], W_corr[j])

                relation_matrix[i,j] = correlation

        # remove all null information in the whole column/row
        relation_matrix = activity_helper.remove_nan_inf_union(relation_matrix)
        change_indices = np.argwhere(np.isnan(relation_matrix) | np.isinf(relation_matrix))
        relation_matrix[change_indices[:, 0], change_indices[:, 1]] = 0
        print(f"change_indices: {change_indices}")
        
        sns.heatmap(relation_matrix, ax=axs[corrind,0], cbar=True, square=True, center=0, cmap="coolwarm", vmin=np.min(relation_matrix), vmax=np.max(relation_matrix))
        # sns.heatmap(relation_matrix_compare, ax=axs[corrind,1], cbar=True, square=True, cmap="coolwarm", vmin=np.min(relation_matrix_compare), vmax=np.max(relation_matrix_compare))
        sns.heatmap(activity_correlation, ax=axs[corrind,1], cbar=True, square=True)
        sns.heatmap(W_corr, ax=axs[corrind,2], cbar=True, square=True, vmin=0, vmax=1)
        axs[corrind,0].set_title(f"Corr(Corr(W), Corr(A)) - {correlation_index}")
        axs[corrind,1].set_title("Corr(A)")
        axs[corrind,2].set_title(f"Corr(W) - {correlation_index}")

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

        # for metric purpose
        for_metric.append(W_cov)
        for_metric.append(activity_cov)

        dim_loader, angle_loader = activity_helper.angles_between_flats_wrap(W_corr, activity_correlation)
        
        metadata[f"{correlation_index}_angle"] = angle_loader

        if correlation_index == "column":
            # do another benchmark plotting for random matrix
            # should have the same range of correlation, [-1,1]
            repeat_random = 1000
            angle_all = []
            for _ in range(repeat_random):
                random_matrix = 2 * np.random.rand(*W_corr.shape) - 1
                dim_loader, angle_loader = activity_helper.angles_between_flats_wrap(random_matrix, activity_correlation)
                angle_all.append(angle_loader)
            angle_all = np.array(angle_all)
            
            metadata["random_angle"] = np.mean(angle_all, axis=0)
            metadata["random_angle_std"] = np.std(angle_all, axis=0)

    fig.tight_layout()
    # fig.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_metric_{metric_name}_noise_{whethernoise}_cc_{whetherconnectome}_heatmap.png")

    if metadata["whethernoise"] == "normal" and metadata["whetherconnectome"] == "count" and R_max == "1":
        print("Save Data")
        np.savez(f"./for_metric/S{metadata['session_info']}s{metadata['scan_info']}_WandA.npz", W_in=for_metric[0], W_out=for_metric[2], A1=for_metric[1], A2=for_metric[3])

    # make sure neurons are aligned/matched correctly
    # delete all neurons if it is either nan or inf in any of the correlation matrices (row or column)
    neurons_tobe_deleted = activity_helper.merge_arrays(indices_delete_lst)
    print(f"Neurons union: {neurons_tobe_deleted}")

    activity_extraction_extra_trc = np.delete(activity_extraction_extra, neurons_tobe_deleted, axis=0)

    activity_per_section = [activity_extraction_extra_trc[:,interval[0]:interval[1]] for interval in stimon_intervals]
    activity_correlation_per_section = [np.corrcoef(activity_extraction_extra_trc[:,interval[0]:interval[1]], rowvar=True) for interval in stimon_intervals]
    for matrix in activity_correlation_per_section:
        np.fill_diagonal(matrix, 0)

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

    pendindex = f"noise_{whethernoise}_cc_{whetherconnectome}" 

    scipy.io.savemat(f"zz_data/microns_{session_info}_{scan_info}_{pendindex}_connectome_out.mat", {'connectome': out_sample_trc})
    scipy.io.savemat(f"zz_data/microns_{session_info}_{scan_info}_{pendindex}_connectome_in.mat", {'connectome': in_sample_trc})
    scipy.io.savemat(f"zz_data/microns_{session_info}_{scan_info}_activity.mat", {'activity': activity_extraction_extra_trc})

    # plot the connectome
    # first add further filtering to the neurons
    illindex = False
    if R_max == "1" and illindex: # only do it once
        selectss_df_drop = selectss_df.drop(index=neurons_tobe_deleted)
        selectss_df_drop = selectss_df_drop.reset_index(drop=True)
        
        cells_id = selectss_df_drop["pt_root_id"].to_numpy()
        synapse_lst = []
        for cell_id in cells_id:
            [_, _, pre_loc, syn1_size, _] = helper.extract_neuron_synaptic_info(cell_id, "pre", cell_table, synapse_table, "new")
            [_, _, post_loc, syn1_size, _] = helper.extract_neuron_synaptic_info(cell_id, "post", cell_table, synapse_table, "new")
            synapse_loc = np.vstack((pre_loc, post_loc))
            synapse_loc[:,[1,2]] = synapse_loc[:,[2,1]]
            synapse_lst.append(synapse_loc)
        mix_helper.plot_3d_gmm_diag_interactive(synapse_lst, None, f"S{session_info}s{scan_info}illustration.html")

    # Betti analysis
    bettiindex = False
    if bettiindex and whethernoise in ["normal", "noise"]: # only do it once
        doconnectome = False
        data_lst = [activity_correlation_all_trc, out_sample_corr_trc, in_sample_corr_trc, W_goodneurons_row, W_goodneurons_col]
        names = ["activity", "connectome_out", "connectome_in", "goodneurons_row", "goodneurons_col"]

        assert data_lst[0].shape == data_lst[1].shape == data_lst[2].shape
        activity_helper.betti_analysis(data_lst, names, metadata=metadata, doconnectome=doconnectome)

    # if not running Betti analysis, then the connectome must be negative
    if not bettiindex:
        assert metadata["inhindex"] 

    soma_distances_trc = np.delete(soma_distances, neurons_tobe_deleted, axis=0)  
    soma_distances_trc = np.delete(soma_distances_trc, neurons_tobe_deleted, axis=1)

    print(soma_distances_trc.shape)

    print("Off-Diagonal Comparison")
    # diagonal compare
    if len(diags[0]) != len(diags[1]):
        diag1, diag2 = activity_helper.match_list_lengths(diags[0], diags[1])
    else:
        diag1, diag2 = diags[0], diags[1]
    
    t_stat, p_value = stats.ttest_rel(diag1, diag2)
    p_value_one_sided = p_value / 2
    p_value_one_sided_final = p_value_one_sided if t_stat > 0 else 1 - p_value_one_sided

    figallcompare, axsallcompare = plt.subplots(2,1,figsize=(4,4)) # purposefully not square for paper purpose
    axsallcompare[0].hist(diags[0], bins=50, alpha=0.5, label='Column On-Diagonal', color=c_vals[1], density=True)
    axsallcompare[0].hist(diags[1], bins=50, alpha=0.5, label='Row On-Diagonal', color=c_vals_l[1], density=True)
    axsallcompare[0].hist(offdiags[0], bins=50, alpha=0.5, label='Column Off-Diagonal', color=c_vals[0], density=True)
    axsallcompare[0].hist(offdiags[1], bins=50, alpha=0.5, label='Row Off-Diagonal', color=c_vals_l[0], density=True)
    axsallcompare[0].legend()
    axsallcompare[0].set_title(f"p1: {activity_helper.float_to_scientific(p_values_all[0])}; p2: {activity_helper.float_to_scientific(p_values_all[1])}; p3: {activity_helper.float_to_scientific(p_value_one_sided_final)}")

    correlation_index_lst.append("random")

    for correlation_index in correlation_index_lst:
        axsallcompare[1].plot([i+1 for i in range(len(metadata[f"{correlation_index}_angle"]))], metadata[f"{correlation_index}_angle"], \
                                label=f"{correlation_index}", color=c_vals[correlation_index_lst.index(correlation_index)])
        if correlation_index == "random":
            axsallcompare[1].fill_between([i+1 for i in range(len(metadata[f"random_angle"]))], metadata[f"random_angle"] - metadata[f"random_angle_std"], \
                                    metadata[f"random_angle"] + metadata[f"random_angle_std"], color=c_vals_l[correlation_index_lst.index(correlation_index)])

    axsallcompare[1].legend()
    axsallcompare[1].set_ylabel("Angle")
    axsallcompare[1].set_title("Subspace Angle")
    axsallcompare[1].xaxis.set_major_locator(ticker.MaxNLocator(4)) 

    figallcompare.tight_layout()
    # figallcompare.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_noise_{whethernoise}_cc_{whetherconnectome}_offdiagcompare_all.png")

    if for_construction:

        hyp_name = f"./mds-results/Rmax_{R_max}_D_{embedding_dimension}_microns_{session_info}_{scan_info}__{pendindex}_embed.mat"

        out_corr = scipy.io.loadmat(hyp_name)['Ddists'][0][1]
        out_corr = 1 - out_corr
        np.fill_diagonal(out_corr, 0)

        in_corr = scipy.io.loadmat(hyp_name)['Ddists'][0][2]
        in_corr = 1 - in_corr
        np.fill_diagonal(in_corr, 0)
        
        # sanity check
        np.allclose(out_corr, W_corrs_all_trc[1], rtol=1e-05, atol=1e-08) 
        np.allclose(in_corr, W_corrs_all_trc[0], rtol=1e-05, atol=1e-08) 

        hypembed_name = f"./mds-results/Rmax_{R_max}_D_{embedding_dimension}_microns_{session_info}_{scan_info}__{pendindex}_embed_hypdist.mat"
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

        eulembed_name = f"./mds-results/Rmax_{R_max}_D_{embedding_dimension}_microns_{session_info}_{scan_info}__{pendindex}_embed_eulmds.mat"
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

        # # sanity check
        # datas = [[out_corr, out_corr, in_corr, in_corr], \
        #         [hypembed_connectome_distance_out, eulembed_connectome_distance_out, hypembed_connectome_distance_in, eulembed_connectome_distance_in]]
        # datas_names = ["out-hyp", "out-eul", "in-hyp", "in-eul"]

        # figsanity, axssanity = plt.subplots(2,4,figsize=(4*4,4*2))
        # for jj in range(4):
        #     xx = 1 - datas[0][jj]
        #     xx = np.triu(xx, k=1)
        #     xx = xx[xx!=0]
        #     yy = np.triu(datas[1][jj], k=1)
        #     yy = yy[yy!=0]
        #     axssanity[0,jj].scatter(xx, yy, alpha=0.1)

        #     rank_xx = rankdata(xx, method='dense')
        #     rank_yy = rankdata(yy, method='dense')
        #     axssanity[1,jj].scatter(rank_xx, rank_yy, alpha=0.1)

        #     slope, intercept, r_value, p_value, std_err = linregress(rank_xx, rank_yy)
        #     r_squared = r_value**2

        #     axssanity[0,jj].set_title(f"{datas_names[jj]}")
        #     axssanity[1,jj].set_title(f"{datas_names[jj]}: {np.round(r_squared,4)}")

        # figsanity.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_noise_{whethernoise}_cc_{whetherconnectome}_sanity.png")

        # Top K neurons of consideration
        topK_values = ["all", int(W_corr.shape[0]/2)]
        print(f"topK_values: {topK_values}")

        # collection of all categories of reconstruction
        all_reconstruction_data = [
            [activity_helper.reconstruction(W, activity_extraction_extra_trc, K, False)[0] for W in input_matrices] 
            for K in topK_values
        ]

        all_reconstruction_corr = [
            [activity_helper.reconstruction(W, activity_extraction_extra_trc, K, False)[1] for W in input_matrices] 
            for K in topK_values
        ]

        # 
        all_reconstruction_data_persession = []
        for interval in stimon_intervals:
            reconstruction_data = []
            for witer in range(len(input_matrices)):
                if witer == 0:
                    W = activity_correlation_per_section[stimon_intervals.index(interval)]
                else:
                    W = input_matrices[witer]
                reconstruction_data.append(activity_helper.reconstruction(W, activity_extraction_extra_trc[:,interval[0]:interval[1]], "all", False)[0])
            all_reconstruction_data_persession.append(reconstruction_data)

        period_results = []
        for ii in range(len(all_reconstruction_data_persession)):
            true_activity = activity_per_section[ii]
            session_data = all_reconstruction_data_persession[ii]
            reconstruct_corr = []
            for reconstruction_data in session_data:
                # corrs = [pearsonr(reconstruction_data[i,:], true_activity[i,:])[0] for i in range(activity_extraction_extra_trc.shape[0])]
                corrs = []
                for i in range(activity_extraction_extra_trc.shape[0]):
                    try:
                        corr = pearsonr(reconstruction_data[i, :], true_activity[i, :])[0]
                        corrs.append(corr)
                    except Exception as e:
                        corrs.append(np.nan)

                reconstruct_corr.append(corrs)
            period_results.append(reconstruct_corr)
        period_results = np.array(period_results)
        period_results_mean = np.nanmedian(period_results, axis=0)
        period_results_median_mean = np.nanmedian(period_results_mean, axis=1)
        print(period_results_median_mean)
        metadata["allk_medians_session"] = period_results_median_mean        

        repeat_permute = 10
        all_reconstruction_data_permute_all = [
            [[activity_helper.reconstruction(W, activity_extraction_extra_trc, K, "cellwise")[0] for W in input_matrices] for K in topK_values]
            for _ in range(repeat_permute)
        ]

        reconstruction_names = ["Activity Correlation", \
                                "Connectome-In Correlation", "Connectome-In Hyp Embed", "Connectome-In Eul Embed", \
                                "Connectome-Out Correlation", "Connectome-Out Hyp Embed", "Connectome-Out Eul Embed", \
                                "Soma Distance"
                            ]

        # # distribution plotting

        # reconstruction_corr_basic = all_reconstruction_corr[0]
        # reconstruction_corr_basic_flatten = []

        # figcheck, axcheck = plt.subplots(1,len(input_matrices),figsize=(4*len(input_matrices),4))
        # for j in range(len(input_matrices)):
        #     ccc = reconstruction_corr_basic[j]
        #     dd = np.triu_indices_from(ccc, k=1)
        #     upper_tri_values = ccc[dd].flatten()
        #     reconstruction_corr_basic_flatten.append(upper_tri_values)
        #     axcheck[j].hist(upper_tri_values, bins=50, density=False)
        #     padding = 0.05 * (max(upper_tri_values) - min(upper_tri_values))
        #     axcheck[j].set_xlim([min(upper_tri_values)-padding, max(upper_tri_values)+padding])
        #     axcheck[j].set_title(reconstruction_names[j])
        # figcheck.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_checkcorr_D{embedding_dimension}_R{R_max}.png")

        # # correlation of distribution plotting
        # cc = len(reconstruction_corr_basic_flatten[:-1])
        # input_matrics_corr, input_matrics_rank_diff = np.zeros((cc, cc)), np.zeros((cc, cc))
        # for cc1 in range(cc):
        #     for cc2 in range(cc):
        #         mat1, mat2 = reconstruction_corr_basic_flatten[cc1], reconstruction_corr_basic_flatten[cc2]
        #         try:
        #             corr, _ = pearsonr(mat1, mat2)  
        #         except Exception as e:
        #             corr = np.nan
        #         rankdiff = np.mean(np.abs(list(rankdata(mat1, method='dense') - rankdata(mat2, method='dense'))))
        #         input_matrics_corr[cc1, cc2] = corr
        #         input_matrics_rank_diff[cc1, cc2] = rankdiff

        # figcorr, axcorr = plt.subplots(1,2,figsize=(10*2,10))
        # sns.heatmap(input_matrics_corr, ax=axcorr[0], cbar=True, square=True, center=0, cmap="coolwarm", annot=True, fmt=".2f", \
        #             xticklabels=reconstruction_names[:-1], yticklabels=reconstruction_names[:-1])
        # sns.heatmap(input_matrics_rank_diff, ax=axcorr[1], cbar=True, square=True, center=0, cmap="coolwarm", annot=True, fmt=".1f", \
        #             xticklabels=reconstruction_names[:-1], yticklabels=reconstruction_names[:-1])
        # figcorr.tight_layout()
        # figcorr.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_noise_{whethernoise}_cc_{whetherconnectome}_corrcompare_D{embedding_dimension}_R{R_max}.png")

        figtest, axtest = plt.subplots(len(selects), 1, figsize=(10, len(selects)*1))

        for nn in selects:
            axtest[selects.index(nn)].plot([i / fps_value for i in range(timecut)],activity_extraction_extra[nn,:timecut], \
                        c=c_vals[selects.index(nn)], label='GroundTruth')
            # just use activity reconstruction for now (illustration purpose)
            all_reconstruction = [all_reconstruction_data[0][0]]
            for j in range(len(all_reconstruction)):
                axtest[selects.index(nn)].plot([i / fps_value for i in range(timecut)],all_reconstruction[j][nn,:timecut], \
                        c=c_vals[0], label=reconstruction_names[j], linestyle='--')
                axtest[selects.index(nn)].set_title(f"Neuron {nn}")
                axtest[selects.index(nn)].set_xlim([-1, timecut/fps_value+1])

        figtest.tight_layout()
        # figtest.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_test.png")

        timeuplst = [activity_extraction_extra_trc.shape[1]-1]

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
                    corr_with_a = activity_helper.vectorized_pearsonr(gt, gt_a)
                    corr_with_c1 = activity_helper.vectorized_pearsonr(gt, gt_c1)
                    corr_with_c1_hypembed = activity_helper.vectorized_pearsonr(gt, gt_c1_hypembed)
                    corr_with_c1_eulembed = activity_helper.vectorized_pearsonr(gt, gt_c1_eulembed)
                    corr_with_c2 = activity_helper.vectorized_pearsonr(gt, gt_c2)
                    corr_with_c2_hypembed = activity_helper.vectorized_pearsonr(gt, gt_c2_hypembed)
                    corr_with_c2_eulembed = activity_helper.vectorized_pearsonr(gt, gt_c2_eulembed)
                    corr_with_soma = activity_helper.vectorized_pearsonr(gt, gt_soma)

                    # notice that the there are multiple choices in the permutation matrix
                    hypinpm, eulinpm, hypoutpm, euloutpm = [], [], [], []
                    for lll in range(repeat_permute):
                        gt_c1_hypembed_pm = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][8][lll][i], window_shape=timeup)
                        gt_c1_eulembed_pm = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][9][lll][i], window_shape=timeup)
                        gt_c2_hypembed_pm = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][10][lll][i], window_shape=timeup)
                        gt_c2_eulembed_pm = np.lib.stride_tricks.sliding_window_view(all_reconstruction_data[k][11][lll][i], window_shape=timeup)

                        corr_with_c1_hypembed_pm = activity_helper.vectorized_pearsonr(gt, gt_c1_hypembed_pm)
                        corr_with_c1_eulembed_pm = activity_helper.vectorized_pearsonr(gt, gt_c1_eulembed_pm)
                        corr_with_c2_hypembed_pm = activity_helper.vectorized_pearsonr(gt, gt_c2_hypembed_pm)
                        corr_with_c2_eulembed_pm = activity_helper.vectorized_pearsonr(gt, gt_c2_eulembed_pm)

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
                all_use = list(range(0,12))
                medians = [np.nanmedian(summ[np.isfinite(summ[:, j]), j]) for j in sorted(all_use)]

                kk_medians.append(medians)
                
            allk_medians.append(kk_medians)

        # plot the final results
        figactshow, axsactshow = plt.subplots(1,len(allk_medians),figsize=(4*len(allk_medians),4))

        for i in range(len(allk_medians)):
            medians = np.array(allk_medians[i])
            for j in range(medians.shape[1]):
                if len(timeuplst) == 1:
                    axsactshow[i].scatter([0 for _ in range(len(medians[:,j]))], medians[:,j], label=reconstruction_names[j])
                else:
                    axsactshow[i].plot(timeuplst, medians[:,j], color=plotstyles[j][0], linestyle=plotstyles[j][1], linewidth=3, label=reconstruction_names[j])
            axsactshow[i].set_title(f"K={topK_values[i]}")
            axsactshow[i].legend()

        for ax in axsactshow:
            ax.axhline(y=metadata["gt_median_corr"], color='black', linestyle='--')
            ax.set_xlabel("Window Length")
            ax.set_ylabel("Median Correlation")

        figactshow.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_noise_{whethernoise}_cc_{whetherconnectome}_actcompareshow_D{embedding_dimension}_R{R_max}.png")

        metadata["timeuplst"] = timeuplst
        metadata["allk_medians"] = allk_medians        

        print(allk_medians)
        
        # extract the metadata
        with open(f"./output/fromac_session_{session_info}_scan_{scan_info}_noise_{whethernoise}_cc_{whetherconnectome}_metadata_D{embedding_dimension}_R{R_max}.pkl", "wb") as pickle_file:
            pickle.dump(metadata, pickle_file)

    session_ds.close()
    stimulus_ds.close()





def benchmark_with_rnn(trial_index):
    """
    """
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

    # hypembed_connectome_corr_out = float(R_max) - hypembed_connectome_distance_out
    hypembed_connectome_corr_out = np.max(hypembed_connectome_distance_out) - hypembed_connectome_distance_out
    np.fill_diagonal(hypembed_connectome_corr_out, 0)

    hypembed_connectome_distance_in = scipy.io.loadmat(hypembed_name)['hyp_dist'][0][2]

    rmax_quantile_in = activity_helper.find_quantile(hypembed_connectome_distance_in, float(R_max))
    metadata["rmax_quantile_in"] = rmax_quantile_in

    # hypembed_connectome_corr_in = float(R_max) - hypembed_connectome_distance_in
    hypembed_connectome_corr_in = np.max(hypembed_connectome_distance_in) - hypembed_connectome_distance_in
    np.fill_diagonal(hypembed_connectome_corr_in, 0)

    # load Euclidean embedding coordinate
    # calcualate the pairwise Euclidean distance afterward    

    eulembed_name = f"./mds-rnn-results/rnn_embed_eulmds_{trial_index}.mat"
    eulembed_connectome = scipy.io.loadmat(eulembed_name)['eulmdsembed'][0][1]
    eulembed_connectome_distance_out = squareform(pdist(eulembed_connectome, metric='euclidean'))
    metadata["rmax_eul_out_distance"] = np.max(eulembed_connectome_distance_out)
    # 
    # eulembed_connectome_corr_out = activity_helper.find_value_for_quantile(eulembed_connectome_distance_out, rmax_quantile_out) - eulembed_connectome_distance_out
    eulembed_connectome_corr_out = np.max(eulembed_connectome_distance_out) - eulembed_connectome_distance_out
    np.fill_diagonal(eulembed_connectome_corr_out, 0)

    eulembed_connectome = scipy.io.loadmat(eulembed_name)['eulmdsembed'][0][2]
    eulembed_connectome_distance_in = squareform(pdist(eulembed_connectome, metric='euclidean'))
    metadata["rmax_eul_in_distance"] = np.max(eulembed_connectome_distance_in)
    # 
    # eulembed_connectome_corr_in = activity_helper.find_value_for_quantile(eulembed_connectome_distance_in, rmax_quantile_in) - eulembed_connectome_distance_in
    eulembed_connectome_corr_in = np.max(eulembed_connectome_distance_in) - eulembed_connectome_distance_in
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

    figcheck.savefig(f"./output_rnn/fromac_session_rnn_{trial_index}_checkcorr.png")

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
    figtest.savefig(f"./output_rnn/fromac_session_rnn_{trial_index}_test.png")

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
                corr_with_a = activity_helper.vectorized_pearsonr(gt, gt_a)
                corr_with_c1 = activity_helper.vectorized_pearsonr(gt, gt_c1)
                corr_with_c1_hypembed = activity_helper.vectorized_pearsonr(gt, gt_c1_hypembed)
                corr_with_c1_eulembed = activity_helper.vectorized_pearsonr(gt, gt_c1_eulembed)
                corr_with_c2 = activity_helper.vectorized_pearsonr(gt, gt_c2)
                corr_with_c2_hypembed = activity_helper.vectorized_pearsonr(gt, gt_c2_hypembed)
                corr_with_c2_eulembed = activity_helper.vectorized_pearsonr(gt, gt_c2_eulembed)

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

    

if __name__ == "__main__":

    # all_run(2, "1",False)

    # for trial_index in range(50):
    #     try:
    #         benchmark_with_rnn(trial_index)
    #     except Exception as e:
    #         print(e)
    #         continue

    microns_across_scans.microns_across_scans_rnn(0)
    # microns_across_scans.microns_across_scans_rnn(1)