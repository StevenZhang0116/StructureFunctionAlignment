import pickle 
import numpy as np 
from scipy.stats import ttest_ind, ttest_rel

import matplotlib.pyplot as plt 
import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']

with open(f"./structure_info/noise_normal_cc_binary_ss_all.pkl", "rb") as file:
    baseline = pickle.load(file)
    
with open(f"./structure_info/noise_normal_cc_binary_ss_all_perturb_0.01.pkl", "rb") as file:
    perturb1 = pickle.load(file)
    
with open(f"./structure_info/noise_normal_cc_binary_ss_all_perturb_0.1.pkl", "rb") as file:
    perturb2 = pickle.load(file)
    
with open(f"./structure_info/noise_normal_cc_binary_ss_all_perturb_0.2.pkl", "rb") as file:
    perturb3 = pickle.load(file)
    
with open(f"./structure_info/noise_normal_cc_binary_ss_all_perturb_0.3.pkl", "rb") as file:
    perturb4 = pickle.load(file)
    
with open(f"./structure_info/noise_normal_cc_binary_ss_all_perturb_0.4.pkl", "rb") as file:
    perturb5 = pickle.load(file)
    
figp, axsp = plt.subplots(2,2,figsize=(4*2,4*2))
names = ["structure_data", "structure_data_out"]

for name_index in range(len(names)):
    name = names[name_index]
    for i in [0,1]:
        hyp1, hyp2, hyp3, hyp4, hyp5, hyp6 = baseline[name][0][i], perturb1[name][0][i], perturb2[name][0][i], perturb3[name][0][i], perturb4[name][0][i], perturb5[name][0][i]
        groups = [hyp1, hyp2, hyp3, hyp4, hyp5, hyp6]

        xlst = [0,1,2,3,4,5]
        μlst = []
        for x, (data, color) in enumerate(zip(groups, c_vals)):
            data = np.asarray(data, dtype=float)          # make sure it’s numeric
            axsp[name_index,i].scatter(np.full(data.shape, x), data,     # raw points
                        color=color, alpha=0.6)

            μ, σ = data.mean(), data.std(ddof=1)          # mean and (sample) std-dev
            μlst.append(μ)
            axsp[name_index,i].errorbar(x, μ, yerr=σ, fmt='o',           # mean ± std marker
                        color=color, capsize=4,
                        elinewidth=2, markeredgewidth=2)
        axsp[name_index,i].set_xticks(xlst)
        print(μlst[-1]/μlst[0])
        axsp[name_index,i].plot(xlst, μlst, color='black', linewidth=2)
        axsp[name_index,i].set_ylabel("Activity Correlation", fontsize=12)
        axsp[name_index,i].set_xticklabels(["Baseline", "Perturb 0.01", "Perturb 0.1", "Perturb 0.2", "Perturb 0.3", "Perturb 0.4"], fontsize=12, rotation=45)
        axsp[name_index,i].set_ylim([0.00, 0.12])
        
axsp[0,0].set_title("HypEmbed Input Corr")
axsp[0,1].set_title("Input Corr")
axsp[1,0].set_title("HypEmbed Output Corr")
axsp[1,1].set_title("Output Corr")
figp.tight_layout()
figp.savefig("./structure_info/compare_perturb.png", dpi=300)