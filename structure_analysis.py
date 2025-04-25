import pickle 
import numpy as np 
from scipy.stats import ttest_ind, ttest_rel

import matplotlib.pyplot as plt 
import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']

# with open(f"./structure_info/noise_normal_cc_count_ss_all.pkl", "rb") as file:
#     indirect = pickle.load(file)
    
# with open(f"./structure_info/noise_normal_cc_count_ss_all_dirsyn.pkl", "rb") as file:
#     direct = pickle.load(file)
    
# indirect_in = indirect["structure_data"][0]
# indirect_out = indirect["structure_data"][1]
# direct_conn = direct["structure_data"][0]

# groups = [indirect_in, indirect_out, direct_conn]
# _, p_value = ttest_rel(groups[1], groups[2], alternative="greater")
# print(p_value)

# fig, axs = plt.subplots(1,1,figsize=(4,4))
# xlst = [0,1,2]
# μlst = []
# for x, (data, color) in enumerate(zip(groups, c_vals)):
#     data = np.asarray(data, dtype=float)          # make sure it’s numeric
#     axs.scatter(np.full(data.shape, x), data,     # raw points
#                 color=color, alpha=0.6)

#     μ, σ = data.mean(), data.std(ddof=1)          # mean and (sample) std-dev
#     μlst.append(μ)
#     axs.errorbar(x, μ, yerr=σ, fmt='o',           # mean ± std marker
#                  color=color, capsize=4,
#                  elinewidth=2, markeredgewidth=2)
# axs.set_xticks(xlst)
# axs.plot(xlst, μlst, color='black', linewidth=2)
# axs.set_ylabel("Activity Correlation", fontsize=12)
# axs.set_xticklabels(["Input Corr", "Output Corr", "Direct Syn"], fontsize=12)
# fig.savefig("./structure_info/compare_structure_bin.png", dpi=300)


with open(f"./structure_info/noise_normal_cc_count_ss_all.pkl", "rb") as file:
    indirect = pickle.load(file)
    
with open(f"./structure_info/noise_normal_cc_count_ss_all_dirsyn.pkl", "rb") as file:
    direct = pickle.load(file)

W = direct["structure_data"][0][1]
Win = indirect["structure_data"][0][1]
corrA = indirect["structure_data"][0][2]
hypWin = indirect["structure_data"][0][0]

groups = [corrA, W, Win, hypWin]

fig, axs = plt.subplots(1,1,figsize=(4,4))
xlst = [0,1,2,3]
μlst = []
for x, (data, color) in enumerate(zip(groups, c_vals)):
    data = np.asarray(data, dtype=float)          # make sure it’s numeric
    axs.scatter(np.full(data.shape, x), data,     # raw points
                color=color, alpha=0.6)

    μ, σ = data.mean(), data.std(ddof=1)          # mean and (sample) std-dev
    μlst.append(μ)
    axs.errorbar(x, μ, yerr=σ, fmt='o',           # mean ± std marker
                 color=color, capsize=4,
                 elinewidth=2, markeredgewidth=2)
axs.set_xticks(xlst)
axs.plot(xlst, μlst, color='black', linewidth=2)
axs.set_ylabel("Activity Correlation", fontsize=12)
axs.set_xticklabels(["Corr(A)", "W", "Cin", "Cin_hyp"], fontsize=12, rotation=45)

fig.savefig("./structure_info/stefan1.png", dpi=300)

with open(f"./structure_info/noise_normal_cc_count_ss_all.pkl", "rb") as file:
    gt = pickle.load(file)
    
with open(f"./structure_info/noise_normal_cc_count_ss_excitatory_neuron.pkl", "rb") as file:
    ext = pickle.load(file)
    
with open(f"./structure_info/noise_normal_cc_count_ss_inhibitory_neuron.pkl", "rb") as file:
    inh = pickle.load(file)

cin = gt["structure_data"][0][1]
cinhyp = gt["structure_data"][0][0]
cextin = ext["structure_data"][0][1]
cextinhyp = ext["structure_data"][0][0]
cinhin = inh["structure_data"][0][1]
cinhypinh = inh["structure_data"][0][0]

groups = [cin, cextin, cinhin, cinhyp, cextinhyp, cinhypinh]

fig, axs = plt.subplots(1,1,figsize=(4,4))
xlst = [0,1,2,3,4,5]
μlst = []
for x, (data, color) in enumerate(zip(groups, c_vals)):
    data = np.asarray(data, dtype=float)          # make sure it’s numeric
    axs.scatter(np.full(data.shape, x), data,     # raw points
                color=color, alpha=0.6)

    μ, σ = data.mean(), data.std(ddof=1)          # mean and (sample) std-dev
    μlst.append(μ)
    axs.errorbar(x, μ, yerr=σ, fmt='o',           # mean ± std marker
                 color=color, capsize=4,
                 elinewidth=2, markeredgewidth=2)
axs.set_xticks(xlst)
axs.plot(xlst, μlst, color='black', linewidth=2)
axs.set_ylabel("Activity Correlation", fontsize=12)
axs.set_xticklabels(["Cin", "Cin_ext", "Cin_inh", "Cin_hyp", "Cin_ext_hyp", "Cin_inh_hyp"], fontsize=12, rotation=45)

fig.savefig("./structure_info/stefan2.png", dpi=300)