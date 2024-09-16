import numpy as np
import pandas as pd
import time
import seaborn as sns 
import matplotlib.pyplot as plt 
import xarray as xr
from scipy.stats import pearsonr
import dask.array as da
import h5py
from dask.diagnostics import ProgressBar
from scipy.sparse import csr_matrix, save_npz

import sys
sys.path.append("../")
sys.path.append("../../")
import helper

import activity_helper

# index
session_info, scan_info = 4, 7
correlation_index = "column"
metric_compare = "correlation"

# proofread & coregisteration overlap information
prf_coreg = pd.read_csv("./microns/prf_coreg.csv")
# take the information belong to that specific session
prf_coreg = prf_coreg[(prf_coreg["session"] == session_info) & (prf_coreg["scan_idx"] == scan_info)]
print(f"Intersection between proofreading and coregisteration: {len(prf_coreg)}")

# read in microns cell/synapse information
cell_table = pd.read_feather("../microns_cell_tables/sven/microns_cell_annos_240524.feather")
print(f"Num of cells: {len(cell_table)}")
synapse_table = pd.read_feather("../microns_cell_tables/sven/synapses_minnie65_phase3_v1_943_combined_incl_trafo_240522.feather")

# cell_table = cell_table[(cell_table["status_axon"] == "extended")]
# extend_axon_neuron = cell_table.index.tolist()

external_data = False
if not external_data:
    # create connectivity matrix
    W, totalSyn, synCount, _ = helper.create_connectivity_as_whole(cell_table, synapse_table)
    print(f"W.shape: {synCount.shape}")
    if W.shape[0] > 90000:
        save_npz("microns_allW.npz", csr_matrix(synCount))
        print("Save W")
        time.sleep(1000)

    # only work for small scale
    W_corr = np.corrcoef(W, rowvar=False if correlation_index=="column" else True) 
else:
    # if using external data loader, the correlation must only compute row by row (extended axon issue)
    assert correlation_index == "row"
    W_corr = np.fromfile('corr_W.dat', dtype=np.float64)
    W_corr = W_corr.reshape(int(np.sqrt(W_corr.shape[0])), int(np.sqrt(W_corr.shape[0])))
    W_corr = W_corr[np.ix_(extend_axon_neuron, extend_axon_neuron)]
    print(f"W_corr.shape:{W_corr.shape}")

# session & scan information
save_filename = f"./microns/functional_xr/functional_session_{session_info}_scan_{scan_info}.nc"
session_ds = xr.open_dataset(save_filename)

# Convert prf_coreg["pt_root_id"] to a set for faster lookup
prf_coreg_ids = set(prf_coreg["pt_root_id"])

# Initialize a counter
count = 0
activity_extraction = []
select_cell = []

# Iterate over pt_root_id in cell_table and check for existence in prf_coreg_ids
for index, pr_root_id in enumerate(cell_table["pt_root_id"]):
    if pr_root_id in prf_coreg_ids:
        cell_row = cell_table[cell_table["pt_root_id"] == pr_root_id]

        select_cell.append(index)
        count += 1
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

# Print the count of matches
print(f"Total matches: {count}")

activity_extraction = np.array(activity_extraction)
activity_correlation = np.corrcoef(activity_extraction, rowvar=True)

select_cell = np.array(select_cell)
truncated_W_correlation = W_corr[np.ix_(select_cell, select_cell)]

assert activity_correlation.shape == truncated_W_correlation.shape

# delete row & column that are all nan or inf
truncated_W_correlation, activity_correlation = activity_helper.sanity_check_W(truncated_W_correlation, activity_correlation)

shape_check = activity_correlation.shape[0]
print(f"Shape after sanity check (nan/inf): {shape_check}")

# zero out diagonal elements
np.fill_diagonal(activity_correlation, 0)
np.fill_diagonal(truncated_W_correlation, 0)

relation_matrix = np.zeros((shape_check, shape_check))
for i in range(shape_check):
    for j in range(shape_check):
        if metric_compare == "correlation":
            correlation, _ = pearsonr(activity_correlation[i], truncated_W_correlation[j])
            relation_matrix[i,j] = correlation


fig, axs = plt.subplots(1,4,figsize=(4*4,4))
sns.heatmap(relation_matrix, ax=axs[0], cbar=True, square=True, cmap="coolwarm", vmin=np.min(relation_matrix), vmax=np.max(relation_matrix))
sns.heatmap(activity_correlation, ax=axs[1], cbar=True, square=True)
sns.heatmap(truncated_W_correlation, ax=axs[2], cbar=True, square=True)
sns.heatmap(W_corr, ax=axs[3], cbar=True, square=True)
axs[0].set_title("Microns Corr(Corr(W), Corr(A))")
axs[1].set_title("Corr(A)")
axs[2].set_title(f"Corr(W) - {correlation_index}")
axs[3].set_title(f"All Corr(W)")
fig.savefig(f"./output/session_{session_info}_scan_{scan_info}_{correlation_index}_heatmap_ext_{external_data}.png")


mean_diagonal, mean_off_diagonal, t_stat, p_value = activity_helper.test_diagonal_significance(relation_matrix)
print(f"Mean Diagonal: {mean_diagonal}, Mean Off-Diagonal: {mean_off_diagonal}")
print(f"T-statistic: {t_stat}, P-value: {p_value}")

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
fig_dist.savefig(f"./output/session_{session_info}_scan_{scan_info}_{correlation_index}_dist_ext_{external_data}.png")



session_ds.close()

