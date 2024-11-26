import os
import numpy as np
import math 
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy import stats

import sys
sys.path.append("../")
sys.path.append("../../")

import activity_helper

def praveen_metric(C,D):
    sqrt_C = sqrtm(C)
    sqrt_D = sqrtm(D)
    numerator = np.trace(np.dot(sqrt_C, sqrt_D))    
    denominator = np.sqrt(np.trace(C)) * np.sqrt(np.trace(D))    
    result = numerator / denominator
    return math.degrees(math.acos(result))

file_lst = [file for file in os.listdir("./for_metric/") if file.endswith('.npz')]
result = []
result_ang = []

corr = "corr"
lowrank = "lowrank"

for file in file_lst:
    data = np.load(f"./for_metric/{file}")

    if corr == "cov":
        W_in, W_out, A1, A2 = data[f'W_cov_column'], data[f'W_cov_row'], data[f'activity_cov_column'], data[f'activity_cov_row']
    elif corr == "corr":
        W_in, W_out, A1, A2 = data[f'W_corr_column'], data[f'W_corr_row'], data[f'activity_correlation_column'], data[f'activity_correlation_row']

    dim_loader1, angle_loader1, low_win, low_a1 = activity_helper.angles_between_flats_wrap(W_in, A1)
    dim_loader2, angle_loader2, low_wout, low_a2 = activity_helper.angles_between_flats_wrap(W_out, A2)

    if lowrank == "lowrank":
        ans_in = [praveen_metric(low_win[i], low_a1[i]) for i in range(len(low_win))]
        ans_out = [praveen_metric(low_wout[i], low_a2[i]) for i in range(len(low_wout))]
        result.append([np.mean(ans_in), np.mean(ans_out)])
    else:
        ans_in = praveen_metric(W_in, A1)
        ans_out = praveen_metric(W_out, A2)
        result.append([ans_in, ans_out])

    result_ang.append([np.mean(angle_loader1), np.mean(angle_loader2)])

result = np.array(result)
result_ang = np.array(result_ang)

result_diff = result[:,0] - result[:,1]
result_ang_diff = result_ang[:,0] - result_ang[:,1]

slope, intercept, r_value, p_value, std_err = stats.linregress(result_diff, result_ang_diff)
xx_line = np.linspace(np.min(result_diff), np.max(result_diff), 100)  
yy_line = slope * xx_line + intercept  

fig, axs = plt.subplots(figsize=(4,4))
axs.scatter(result_diff, result_ang_diff)
axs.plot(xx_line, yy_line, color='red', label=f"r={np.round(r_value,3)}")
axs.set_ylabel("In Angle - Out Angle")
axs.set_xlabel("In Praveen Metric - Out Praveen Metric")
axs.legend()
fig.savefig(f"praveen_{corr}_{lowrank}.png")