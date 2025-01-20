import os
import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy import stats
import time 

import sys
sys.path.append("../")
sys.path.append("../../")

import activity_helper
import comp_metrics

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]

file_lst = [file for file in os.listdir("./for_metric/") if file.endswith('.npz')]
result = []

corr = "corr"
lowrank = "cka"

for file in file_lst:
    data = np.load(f"./for_metric/{file}")

    if corr == "cov":
        W_in, W_out, A1, A2 = data[f'W_cov_column'], data[f'W_cov_row'], data[f'activity_cov_column'], data[f'activity_cov_row']
    elif corr == "corr":
        W_in, W_out, A1, A2 = data[f'W_corr_column'], data[f'W_corr_row'], data[f'activity_correlation_column'], data[f'activity_correlation_row']

    dim_loader1, angle_loader1, low_win, low_a1 = activity_helper.angles_between_flats_wrap(W_in, A1)
    dim_loader2, angle_loader2, low_wout, low_a2 = activity_helper.angles_between_flats_wrap(W_out, A2)

    if lowrank == "lowrank":
        ans_in = [comp_metrics.praveen_metric(low_win[i], low_a1[i]) for i in range(len(low_win))]
        ans_out = [comp_metrics.praveen_metric(low_wout[i], low_a2[i]) for i in range(len(low_wout))]
        result.append([np.mean(ans_in), np.mean(ans_out)])
    elif lowrank == "cka":
        ans_in = [comp_metrics.cka(comp_metrics.gram_linear(low_win[i]), comp_metrics.gram_linear(low_a1[i])) for i in range(len(low_win))]
        ans_out = [comp_metrics.cka(comp_metrics.gram_linear(low_wout[i]), comp_metrics.gram_linear(low_a2[i])) for i in range(len(low_wout))]

        random_all = []
        for _ in range(1000):
            random_matrix = 2 * np.random.rand(*W_in.shape) - 1
            _, _, low_random, _ = activity_helper.angles_between_flats_wrap(random_matrix, A1)
            random_whole = [comp_metrics.cka(comp_metrics.gram_linear(low_random[i]), comp_metrics.gram_linear(low_a1[i])) for i in range(len(low_random))]
            random_all.append(random_whole)
        
        ans_random = np.mean(random_all, axis=0)
        ans_random_std = np.std(random_all, axis=0)

        ans_in_all = comp_metrics.cka(comp_metrics.gram_linear(W_in), comp_metrics.gram_linear(A1))
        ans_out_all = comp_metrics.cka(comp_metrics.gram_linear(W_out), comp_metrics.gram_linear(A2))

        iii = file_lst.index(file)

        fig, axs = plt.subplots(figsize=(4,2))

        axs.plot(dim_loader1, ans_in, color=c_vals[0], label="Input")
        axs.plot(dim_loader1, ans_out, color=c_vals[1], label="Output")
        axs.plot(dim_loader1, ans_random, color=c_vals[2], label="Random")
        axs.fill_between(dim_loader1, ans_random - ans_random_std, ans_random + ans_random_std, color=c_vals_l[2], alpha=0.2)
        axs.legend()
        # axs.set_title(f"{file[:4]}-{W_in.shape[0]}")
        axs.set_xlabel("Dimension")
        axs.set_ylabel("CKA Score")

        fig.tight_layout()
        fig.savefig(f"./om/zz_{lowrank}_{file[:4]}.png")

    else:
        ans_in = metrics.praveen_metric(W_in, A1)
        ans_out = metrics.praveen_metric(W_out, A2)
        result.append([ans_in, ans_out])

