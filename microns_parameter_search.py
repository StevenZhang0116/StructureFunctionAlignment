import os
import glob
import numpy as np
import re
import matplotlib.pyplot as plt

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e']
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]

def microns_parameter_search(dimension, Kselect, whethernoise, whetherconnectome):
    
    def extract_R_value(file_name):
        match = re.search(r'_R([0-9\.e\+\-]+)_Tall', file_name)
        if match:
            return match.group(1)
        return None 

    directory = "./output/"
    notwantR = []
    npz_files = [
        file for file in glob.glob(os.path.join(directory, "*.npz"))
        if f"D{dimension}" in os.path.basename(file)
        and f"K{Kselect}" in os.path.basename(file)
        and f"noise_{whethernoise}" in os.path.basename(file)
        and f"cc_{whetherconnectome}" in os.path.basename(file)
        and not any(f"R{elem}" in os.path.basename(file) for elem in notwantR)
    ]

    print(npz_files)

    r_files_pairs = [(float(extract_R_value(file)), file) for file in npz_files]
    r_files_pairs.sort(key=lambda pair: pair[0])
    r_values, npz_files = zip(*r_files_pairs)
    r_values = list(r_values)
    npz_files = list(npz_files)

    hypin_data, hypout_data = [], []
    hypin_data_session, hypout_data_session = [], []
    inbase_data, outbase_data = [], []
    inbase_data_session, outbase_data_session = [], []

    figcomparer, axscomparer = plt.subplots(1,2,figsize=(4*2,4))  

    for i in range(len(npz_files)):
        file_path = npz_files[i]
        data = np.load(file_path, allow_pickle=True)["alldata"]

        hypin_data.append(data[0][:,4]/data[0][:,3])
        hypout_data.append(data[1][:,4]/data[1][:,3])
        hypin_data_session.append(data[0][:,11]/data[0][:,9])
        hypout_data_session.append(data[1][:,11]/data[1][:,9])
        inbase_data.append(data[0][:,2]/data[0][:,3])
        outbase_data.append(data[1][:,2]/data[1][:,3])
        inbase_data_session.append(data[0][:,10]/data[0][:,9])
        outbase_data_session.append(data[1][:,10]/data[1][:,9])

    log_r_values = np.log10(r_values)
    integer_power_indices = np.where(log_r_values == np.floor(log_r_values))[0] 
    r_values_power_of_10 = log_r_values[integer_power_indices] 
    tick_labels = [r'$10^{%d}$' % int(val) for val in log_r_values[integer_power_indices]]  

    meanin, stdin = np.array([np.median(dd) for dd in hypin_data]), np.array([np.std(dd) for dd in hypin_data])
    meanout, stdout = np.array([np.median(dd) for dd in hypout_data]), np.array([np.std(dd) for dd in hypout_data])
    meaninss, stdinss = np.array([np.median(dd) for dd in hypin_data_session]), np.array([np.std(dd) for dd in hypin_data_session])
    meanoutss, stdoutss = np.array([np.median(dd) for dd in hypout_data_session]), np.array([np.std(dd) for dd in hypout_data_session])

    axscomparer[0].plot(log_r_values, meanin, "-o", color=c_vals[0], label="Hyp2In")
    axscomparer[0].fill_between(log_r_values, meanin-stdin, meanin+stdin, color=c_vals_l[0], alpha=0.2)
    axscomparer[0].plot(log_r_values, meanout, "-o", color=c_vals[3], label="Hyp2Out")
    axscomparer[0].fill_between(log_r_values, meanout-stdout, meanout+stdout, color=c_vals_l[3], alpha=0.2)
    axscomparer[0].axhline(y=np.median(inbase_data), color=c_vals[0])
    axscomparer[0].axhline(y=np.median(outbase_data), color=c_vals[3])

    axscomparer[1].plot(log_r_values, meaninss, "--o", color=c_vals[0], label="Hyp2InSession")
    axscomparer[1].fill_between(log_r_values, meaninss-stdinss, meaninss+stdinss, color=c_vals_l[0], alpha=0.2)
    axscomparer[1].plot(log_r_values, meanoutss, "--o", color=c_vals[3], label="Hyp2OutSession")
    axscomparer[1].fill_between(log_r_values, meanoutss-stdoutss, meanoutss+stdoutss, color=c_vals_l[3], alpha=0.2)
    axscomparer[1].axhline(y=np.median(inbase_data_session), linestyle="--", color=c_vals[0])
    axscomparer[1].axhline(y=np.median(outbase_data_session), linestyle="--", color=c_vals[3])
    
    for ax in axscomparer:
        ax.set_xticks(r_values_power_of_10)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("R_max")
        ax.set_ylabel("Explanation Ratio to Activity")
        ax.legend()
        
    figcomparer.tight_layout()
    figcomparer.savefig(f"./zz_Rmax_D{dimension}_scan_K{Kselect}_noise_{whethernoise}_cc_{whetherconnectome}.png")

