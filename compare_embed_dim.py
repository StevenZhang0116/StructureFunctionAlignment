import numpy as np
import json 
import scipy 
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

fig, axs = plt.subplots(1,2,figsize=(4*2,4))
figdist, axsdist = plt.subplots(1,2,figsize=(4*2,4))

names = ["in", "out"]
titles = ["Input", "Output"]
inindex, outindex = 1, 0
indexlst = [inindex, outindex]

for i in range(len(names)):

    dim1, dim2 = 2, 100
    data1_name = f"./perf_record/dim{dim1}_{names[i]}.json"
    data2_name = f"./perf_record/dim{dim2}_{names[i]}.json"

    mdsdata1 = scipy.io.loadmat(f"./mds-results-all/Rmax_1_D_{dim1}__noise_normal_cc_count_ss_all_embed_eulmds.mat")["eulmdsembed"][0][indexlst[i]]
    mdsdata1_dist = squareform(pdist(mdsdata1, metric='euclidean'))
    upper_tri_indices = np.triu_indices(mdsdata1_dist.shape[0], k=1)
    flattened_mdsdata1 = mdsdata1_dist[upper_tri_indices]

    mdsdata2 = scipy.io.loadmat(f"./mds-results-all/Rmax_1_D_{dim2}__noise_normal_cc_count_ss_all_embed_eulmds.mat")["eulmdsembed"][0][indexlst[i]]
    mdsdata2_dist = squareform(pdist(mdsdata2, metric='euclidean'))
    upper_tri_indices = np.triu_indices(mdsdata2_dist.shape[0], k=1)
    flattened_mdsdata2 = mdsdata2_dist[upper_tri_indices]

    with open(data1_name, "r") as f:
        data1 = json.load(f)

    with open(data2_name, "r") as f:
        data2 = json.load(f)

    data_compare = []
    for neuron_id in data1.keys():
        data_compare.append([data1[neuron_id], data2[neuron_id]])

    data_compare = np.array(data_compare)

    x = data_compare[:, 0]
    y = data_compare[:, 1]

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    ss_res = np.sum((y - y_pred) ** 2)     
    ss_tot = np.sum((y - np.mean(y)) ** 2)      
    r_squared = 1 - ss_res / ss_tot

    axs[i].scatter(x, y, s=5)
    axs[i].plot([min(x), max(x)], [min(x), max(x)], color="black", linestyle="--")
    axs[i].set_title(titles[i])
    axs[i].set_xlabel(f"Performance for Dim {dim1}")
    axs[i].set_ylabel(f"Performance for Dim {dim2}")

    axsdist[i].scatter(flattened_mdsdata1, flattened_mdsdata2, s=5)
    axsdist[i].plot([min(flattened_mdsdata1), max(flattened_mdsdata1)], [min(flattened_mdsdata1), max(flattened_mdsdata1)], color="black", linestyle="--")
    axsdist[i].set_title(titles[i])
    axsdist[i].set_xlabel(f"Pairwise Distance for Dim {dim1}")
    axsdist[i].set_ylabel(f"Pairwise Distance for Dim {dim2}")

fig.tight_layout()
fig.savefig(f"./perf_record/compare{dim1}_{dim2}.png", dpi=300)

figdist.tight_layout()
figdist.savefig(f"./perf_record/compare{dim1}_{dim2}_dist.png", dpi=300)