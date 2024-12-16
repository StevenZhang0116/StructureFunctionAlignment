import numpy as np 
import matplotlib.pyplot as plt
import scipy
import os 
import re
from scipy import stats

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]

def get_number(s):
    match = re.match(r'rmax_(\d+)(?:_(\d))?', s)
    if match:
        main_number = int(match.group(1))
        decimal_part = int(match.group(2)) if match.group(2) else 0
    return (main_number + decimal_part / 10)

def connectome():
    fig, axs = plt.subplots(1,3,figsize=(4*3,4*1))

    subdatas = ["whether_noise_cc_binary", "whether_noise_cc_count"]
    append = "_conngood_connectome_primary"
    linestyles = [["-","--"], ["-.", ":"]]
    for iii in range(len(subdatas)):
        subdata = subdatas[iii]
        groundtruth = f"./zz_pyclique_results/{subdata}"
        fitting = f"{groundtruth}_fitting"
        groundtruth = np.load(groundtruth+append+".npz")["groundtruth_integratedbetti_save"][3:,:]

        rmax_best = [[18,14], [14,13]]

        fitting = np.load(fitting+".npz")["fake_integrated_bettis"]

        average_fakebetti = np.mean(fitting, axis=2)
        std_fakebetti = np.std(fitting, axis=2)

        readin_files = "./zz_pyclique/hyperbolic_dis_n=635_repeat=10_dim_2noise_0.0625minRatio_0.1.mat"
        data = scipy.io.loadmat(readin_files)
        data = data['distance_matrices']

        if isinstance(data, dict):
            data_keys = data.keys()
            fields = sorted([key for key in data_keys if not key.startswith('__')])
        else:
            fields = sorted(data.dtype.names)

        fields = sorted(fields, key=lambda x: int(x.split('_')[1]))
        fields = np.array([float(f[5:]) for f in fields])

        max_value_betti = [20,50,70]
        min_value_betti = [-0.5,-2,-2]

        for i in range(3):
            axs[i].plot(list(fields), list(average_fakebetti[:,i]), "-o", color=c_vals[i])
            axs[i].fill_between(list(fields), list(average_fakebetti[:,i]-std_fakebetti[:,i]), list(average_fakebetti[:,i]+std_fakebetti[:,i]), color=c_vals_l[i], alpha=0.2)
            axs[i].axhline(groundtruth[0,i], color=c_vals_l[i], linestyle=linestyles[iii][0])
            axs[i].axhline(groundtruth[1,i], color=c_vals_l[i], linestyle=linestyles[iii][1])
            axs[i].axvline(rmax_best[iii][0], color=c_vals_l[i], linestyle=linestyles[iii][0])
            axs[i].axvline(rmax_best[iii][1], color=c_vals_l[i], linestyle=linestyles[iii][1])

            axs[i].scatter(rmax_best[iii][0], groundtruth[0,i], color=c_vals_l[i], marker='x')
            axs[i].scatter(rmax_best[iii][1], groundtruth[1,i], color=c_vals_l[i], marker='o')
            axs[i].set_xlabel("Rmax")
            axs[i].set_ylabel(f"Betti {i+1} Value")
            axs[i].set_ylim([min_value_betti[i], max_value_betti[i]])

    fig.tight_layout()
    fig.savefig(f"./zz_pyclique_results/connectome_fakebetti.png")


def activity():
    session_scan = [[8,5],[4,7],[6,6],[5,3],[5,6],[5,7],[6,2],[7,3],[7,5],[9,3],[9,4],[6,4]]

    for ss in session_scan:
        subdata = f"S{ss[0]}s{ss[1]}{metric}_bettis_noise0.0625"
        groundtruth = f"./zz_pyclique_results/{subdata}"

        file = np.load(groundtruth+".npz")

        groundtruth = file["groundtruth_integratedbetti_save"][0,:]
        fitting = file["fake_integrated_bettis"]

        average_fakebetti = np.mean(fitting, axis=2)
        std_fakebetti = np.std(fitting, axis=2)

        readin_files = "./zz_pyclique/hyperbolic_dis_n=68_repeat=100_dim_2noise_0.0625minRatio_0.1.mat"
        data = scipy.io.loadmat(readin_files)
        data = data['distance_matrices']

        if isinstance(data, dict):
            data_keys = data.keys()
            fields = sorted([key for key in data_keys if not key.startswith('__')])
        else:
            fields = sorted(data.dtype.names)
        
        numbers = []
        for s in fields:
            numbers.append(get_number(s))

        numbers = sorted(numbers)
        fields = numbers
        print(fields)    

        fig, axs = plt.subplots(1,3,figsize=(4*3,4*1))
        for i in range(3):
            axs[i].plot(list(fields), list(average_fakebetti[:,i]), "-o", color=c_vals[i])
            axs[i].fill_between(list(fields), list(average_fakebetti[:,i]-std_fakebetti[:,i]), list(average_fakebetti[:,i]+std_fakebetti[:,i]), color=c_vals_l[i], alpha=0.2)
            axs[i].axhline(groundtruth[i], color=c_vals_l[i], linestyle='--')
            axs[i].set_title(f"Betti {i+1} Value")
            axs[i].set_xlabel("Rmax")

        fig.tight_layout()
        fig.savefig(f"./zz_pyclique_results/{subdata}_fakebetti.png")

def activity_all():
    metric = "correlation"

    dire = "./zz_pyclique_results/"
    files = [
        file for file in os.listdir(dire)
        if file.startswith("S") and file.endswith(".npz") and metric in file
    ]
    
    cc = []
    for file in files:
        data = np.load(f"{dire}{file}")
        bestR, size = get_number(str(data["bestR"])), data["size"]
        cc.append([bestR, size])
    cc = np.array(cc)

    slope, intercept, r_value, p_value, std_err = stats.linregress(cc[:,1], cc[:,0])
    xx_line = np.linspace(np.min(cc[:,1]), np.max(cc[:,1]), 100)  
    yy_line = slope * xx_line + intercept  

    fig, axs = plt.subplots(1,1,figsize=(4,4))
    axs.scatter(cc[:,1], cc[:,0])
    axs.plot(xx_line, yy_line, color='red', linestyle="--", label=f"r^2={np.round(r_value,3)}")
    axs.set_xlabel("# Neurons")
    axs.set_ylabel("Best Rmax")
    axs.legend()
    fig.tight_layout()
    fig.savefig(f"{dire}activity_{metric}_all.png")



if __name__ == "__main__":
    # activity()
    connectome()
    # activity_all()