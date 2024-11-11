import numpy as np 
import matplotlib.pyplot as plt
import scipy
import re

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]


def connectome():
    subdata = "whether_noise_cc_count"
    groundtruth = f"./zz_pyclique_results/{subdata}"
    fitting = f"{groundtruth}_fitting"

    groundtruth = np.load(groundtruth+".npz")["groundtruth_integratedbetti_save"][3:,:]
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

    fig, axs = plt.subplots(1,3,figsize=(4*3,4*1))
    for i in range(3):
        axs[i].plot(list(fields), list(average_fakebetti[:,i]), color=c_vals[i])
        axs[i].fill_between(list(fields), list(average_fakebetti[:,i]-std_fakebetti[:,i]), list(average_fakebetti[:,i]+std_fakebetti[:,i]), color=c_vals_l[i], alpha=0.2)
        axs[i].axhline(groundtruth[0,i], color=c_vals_l[i], linestyle='--')
        axs[i].axhline(groundtruth[1,i], color=c_vals_l[i], linestyle='-.')


    fig.savefig(f"./zz_pyclique_results/{subdata}_fakebetti.png")


def activity():
    subdata = "S8s5_bettis_noise0.0625"
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
        match = re.match(r'rmax_(\d+)(?:_(\d))?', s)
        if match:
            main_number = int(match.group(1))
            decimal_part = int(match.group(2)) if match.group(2) else 0
            numbers.append(main_number + decimal_part / 10)

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

if __name__ == "__main__":
    # activity()
    connectome()