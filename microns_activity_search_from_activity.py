import numpy as np
import pandas as pd
import time
import seaborn as sns 
import matplotlib.pyplot as plt 
import xarray as xr
from scipy.stats import pearsonr
import dask.array as da
from dask.diagnostics import ProgressBar

import scipy
from scipy.sparse import csr_matrix
from scipy.linalg import subspace_angles
from scipy.spatial.distance import pdist, squareform

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

import sys
sys.path.append("../")
sys.path.append("../../")
import helper

import activity_helper

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e']

# index
session_info, scan_info = 8, 5
metric_compare = "correlation"
sub_correlation_index = "nan_fill"

# proofread & coregisteration overlap information
prf_coreg = pd.read_csv("./microns/prf_coreg.csv")
# take the information belong to that specific session
prf_coreg = prf_coreg[(prf_coreg["session"] == session_info) & (prf_coreg["scan_idx"] == scan_info)]
print(len(prf_coreg))

# read in microns cell/synapse information
cell_table = pd.read_feather("../microns_cell_tables/sven/microns_cell_annos_240524.feather")
# cell_table = cell_table.sort_values(by='cell_type')
synapse_table = pd.read_feather("../microns_cell_tables/sven/synapses_minnie65_phase3_v1_943_combined_incl_trafo_240522.feather")

save_filename = f"./microns/functional_xr/functional_session_{session_info}_scan_{scan_info}.nc"
session_ds = xr.open_dataset(save_filename)

selected_neurons = []
neuron_types = []
activity_extraction = []
activity_extraction_extra = []

for index, prf_coreg_id in enumerate(prf_coreg["pt_root_id"]):
    if prf_coreg_id in set(cell_table["pt_root_id"]):
        matching_indices = cell_table.index[cell_table["pt_root_id"] == prf_coreg_id].tolist()
        matching_row = prf_coreg[prf_coreg["pt_root_id"] == prf_coreg_id]

        # only take one neuron if there are multiple filtered results
        if len(matching_row) > 1:
            matching_row = matching_row.iloc[0]
            matching_indices = [matching_indices[0]]
        
        if matching_indices[0] not in selected_neurons:
            neuron_types.append(cell_table.iloc[matching_indices]["cell_type"].item())
            selected_neurons.extend(matching_indices)

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

# plot activity trace
fig, ax = plt.subplots(1,2,figsize=(2*6,6))
for i in range(0,10):
    ax[0].plot(activity_extraction[i,:])
    ax[1].plot(activity_extraction_extra[i,:])
fig.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_activity.png")

# calculate activity correlation
activity_correlation = np.corrcoef(activity_extraction, rowvar=True)
fig, ax = plt.subplots(1,1,figsize=(6,6))
dd = np.triu_indices_from(activity_correlation, k=1)
upper_tri_values = activity_correlation[dd].flatten()
median_corr = np.median(upper_tri_values)
ax.hist(upper_tri_values, bins=100)
ax.set_xlabel("Activity Correlation")
ax.set_ylabel("Frequency/Count")
ax.set_title(f"Median: {median_corr}")
fig.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_acthist.png")


W_all, totalSyn, synCount, _ = helper.create_connectivity_as_whole(cell_table, synapse_table)

correlation_index_lst = ["column", "row"]
W_corrs = []

fig, axs = plt.subplots(2,4,figsize=(4*6,6))
fig_dist, ax_dist = plt.subplots(2,2,figsize=(2*6,2*6))

for correlation_index in correlation_index_lst:
    corrind = correlation_index_lst.index(correlation_index)

    external_data = True
    if not external_data:
        W_corr = np.corrcoef(W, rowvar=False if correlation_index=="column" else True) 
    else:
        # # something wrong with that...
        # W_corr = np.fromfile(f'./corr_W_microns_{correlation_index}.dat', dtype=np.float64)
        # W_corr = W_corr.reshape(int(np.sqrt(W_corr.shape[0])), int(np.sqrt(W_corr.shape[0])))
        # W_corr = W_corr[np.ix_(selected_neurons, selected_neurons)]

        if correlation_index == "row":
            W_trc = synCount[selected_neurons,:]
            W_corr = np.corrcoef(W_trc, rowvar=True)
        else:
            W_trc = synCount[:,selected_neurons]
            W_corr = np.corrcoef(W_trc, rowvar=False)
        
        W_backupcheck = synCount[np.ix_(selected_neurons, selected_neurons)].astype(float)

    W_corr, activity_correlation = activity_helper.sanity_check_W(W_corr, activity_correlation)
    shape_check = W_corr.shape[0]
    print(f"Shape after sanity check (nan/inf): {shape_check}")

    if sub_correlation_index == "zero_out":
        np.fill_diagonal(activity_correlation, 0)
        np.fill_diagonal(W_corr, 0)

    elif sub_correlation_index == "nan_fill":
        np.fill_diagonal(activity_correlation, np.nan)
        np.fill_diagonal(W_corr, np.nan)
        # np.fill_diagonal(W_backupcheck, np.nan)

    print(W_corr.shape)
    print(activity_correlation.shape)

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


    sns.heatmap(relation_matrix, ax=axs[corrind,0], cbar=True, square=True, cmap="coolwarm", vmin=np.min(relation_matrix), vmax=np.max(relation_matrix))
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

    diagonal = np.diag(relation_matrix)
    i, j = np.indices(relation_matrix.shape)
    off_diagonal = relation_matrix[i != j]   

    ax_dist[0,corrind].hist(diagonal, bins=20, alpha=0.7, label='Diagonal Elements', color='blue', density=True)
    ax_dist[0,corrind].hist(off_diagonal, bins=20, alpha=0.7, label='Off-Diagonal Elements', color='red', density=True)
    ax_dist[0,corrind].set_title(f"p-value: {p_value} - {correlation_index}")
    ax_dist[0,corrind].set_xlabel('Value')
    ax_dist[0,corrind].set_ylabel('Frequency')
    ax_dist[0,corrind].legend()

    # subspace angle analysis
    # here we refill the diagonal to be 0 (instead of nan/inf as previous)
    np.fill_diagonal(W_corr, 0)
    W_corrs.append(W_corr)
    np.fill_diagonal(activity_correlation,0)
    U_connectome, S_connectome, Vh_connectome = np.linalg.svd(W_corr)
    U_activity, S_activity, Vh_activity = np.linalg.svd(activity_correlation)

    dim_loader, angle_loader = [], []
    for num_dimension in range(1,20):
        U_comps_activity = [U_activity[:,i] for i in range(num_dimension)]
        U_comps_connectome = [U_connectome[:,i] for i in range(num_dimension)]
        angle_in_bewteen = activity_helper.angles_between_flats(U_comps_activity, U_comps_connectome)
        dim_loader.append(num_dimension)
        angle_loader.append(angle_in_bewteen)

    ax_dist[1,corrind].plot(dim_loader, angle_loader, "-o")
    ax_dist[1,corrind].set_xlabel("Dimensionality")
    ax_dist[1,corrind].set_ylabel("Angle")


fig.tight_layout()
fig.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_heatmap.png")
fig_dist.tight_layout()
fig_dist.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_dist.png")

def reconstruction(W):
    row_sums_a = np.sum(W, axis=1)
    activity_correlation_normalized = W / row_sums_a[:, np.newaxis]
    activity_reconstruct_a = activity_correlation_normalized @ activity_extraction
    return activity_reconstruct_a
    

# reconstruction using correlation with other neuron's activity
activity_reconstruct_a = reconstruction(activity_correlation)

# reconstruction using correlation with other neuron's connectome
activity_reconstruct_c = reconstruction(W_corr)

# load embedding (hyp distance in hyp embedding)
R_max = 1
embedding_dimension = 2
hypembed_name = f"./mds-results/Rmax_{R_max}_D_{embedding_dimension}_microns_{session_info}_{scan_info}_embed_hypdist.mat"
hypembed_connectome_distance = scipy.io.loadmat(hypembed_name)['hyp_dist'][0][1]
hypembed_connectome_corr = np.max(hypembed_connectome_distance) - hypembed_connectome_distance
np.fill_diagonal(hypembed_connectome_corr, 0)

hypembed_name = f"./mds-results/Rmax_{R_max}_D_{embedding_dimension}_microns_{session_info}_{scan_info}_embed_eulmds.mat"
eulembed_connectome = scipy.io.loadmat(hypembed_name)['eulmdsembed'][0][1]
eulembed_connectome_distance = squareform(pdist(eulembed_connectome, metric='euclidean'))
eulembed_connectome_corr = np.max(eulembed_connectome_distance) - eulembed_connectome_distance
np.fill_diagonal(eulembed_connectome_corr, 0)

activity_reconstruct_c_hypembed = reconstruction(hypembed_connectome_corr)
activity_reconstruct_c_eulembed = reconstruction(eulembed_connectome_corr)


timeuplst = [100,1000,10000]
ttlength = activity_extraction.shape[1]
figact, axsact = plt.subplots(1,len(timeuplst),figsize=(6*len(timeuplst),6))
axsact = np.atleast_1d(axsact)

for timeup in timeuplst:
    summ = []
    for i in range(activity_extraction.shape[0]):
        summ_neuron = []
        for start in range(ttlength-timeup-1):
            print(f"Neuron {i} Start at {start}")
            gt = activity_extraction[i,start:start+timeup]
            gt_a = activity_reconstruct_a[i,start:start+timeup]
            gt_c = activity_reconstruct_c[i,start:start+timeup]
            gt_c_hypembed = activity_reconstruct_c_hypembed[i,start:start+timeup]
            gt_c_eulembed = activity_reconstruct_c_eulembed[i,start:start+timeup]

            corr_with_a, _ = pearsonr(gt, gt_a)
            corr_with_c, _ = pearsonr(gt, gt_c)
            corr_with_c_hypembed, _ = pearsonr(gt, gt_c_hypembed)
            corr_with_c_eulembed, _ = pearsonr(gt, gt_c_eulembed)

            summ_neuron.append([corr_with_a, corr_with_c, corr_with_c_hypembed, corr_with_c_eulembed])
        summ_neuron = np.array(summ_neuron)
        summ.append(list(np.mean(summ_neuron, axis=0)))

    summ = np.array(summ)
    axsact[timeuplst.index(timeup)].hist(summ[:,0], bins=10, color=c_vals[0], alpha=0.7, label=f"Activity Reconstruction: {round(np.median(summ[:,0]),5)}")
    axsact[timeuplst.index(timeup)].hist(summ[:,1], bins=10, color=c_vals[1], alpha=0.7, label=f"Connectome Reconstruction: {round(np.median(summ[:,1]),5)}")
    axsact[timeuplst.index(timeup)].hist(summ[:,2], bins=10, color=c_vals[2], alpha=0.7, label=f"Connectome Hyp {embedding_dimension} Embedding Reconstruction: {round(np.median(summ[:,2]),5)}")
    axsact[timeuplst.index(timeup)].hist(summ[:,3], bins=10, color=c_vals[3], alpha=0.7, label=f"Connectome Eul {embedding_dimension} Embedding Reconstruction: {round(np.median(summ[:,3]),5)}")

    axsact[timeuplst.index(timeup)].set_title(f"{timeup}")

for ax in axsact:
    ax.legend()
figact.savefig(f"./output/fromac_session_{session_info}_scan_{scan_info}_actcompare.png")

session_ds.close()
