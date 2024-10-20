import glob
import os
import numpy 
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']

def microns_across_scans(R_max, dimension, Kselect):
    def find_pkl_files(directory):
        
        # only select pkl files with desired dimension and R_max
        strname = f"D{dimension}_R{R_max}.pkl"
        all_pkl_files = glob.glob(os.path.join(directory, "**", "*.pkl"), recursive=True)
        matching_files = [f for f in all_pkl_files if strname in os.path.basename(f)]
        return matching_files

    directory_path = "./output/"
    pkl_files = find_pkl_files(directory_path)
    print(pkl_files)

    # assert len(pkl_files) == 14
    
    coldata, rowdata, somadata = [], [], []
    rmax_quantiles = []

    gt_median_corrs = []

    timeselect = "all"

    for pkl_file_path in pkl_files:
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)

        if timeselect == "all":
            ttind = -1
        else:
            ttind = data["timeuplst"].index(timeselect)

        cc = 1

        gt_median_corrs.append(data["gt_median_corr"])

        num_neurons = data["num_neurons"]
        rmax_quantiles.append([data["rmax_quantile_out"], data["rmax_quantile_in"]])

        column_primary_angle = np.mean(data["column_angle"][0:cc])
        row_primary_angle = np.mean(data["row_angle"][0:cc])
        allk_medians = data["allk_medians"][Kselect][ttind]
        column_explainratio = allk_medians[1]/allk_medians[0]
        row_explainratio = allk_medians[4]/allk_medians[0]

        soma_explainratio = allk_medians[7]/allk_medians[0]

        in_hyp_ratio = allk_medians[2]/allk_medians[1]
        in_eul_ratio = allk_medians[3]/allk_medians[1]
        in_hyp_ratio_pm = allk_medians[8]/allk_medians[1]
        in_eul_ratio_pm = allk_medians[9]/allk_medians[1]

        out_hyp_ratio = allk_medians[5]/allk_medians[4]
        out_eul_ratio = allk_medians[6]/allk_medians[4]
        out_hyp_ratio_pm = allk_medians[10]/allk_medians[4]
        out_eul_ratio_pm = allk_medians[11]/allk_medians[4]

        coldata.append([num_neurons, column_primary_angle, column_explainratio, in_hyp_ratio, in_eul_ratio, in_hyp_ratio_pm, in_eul_ratio_pm, soma_explainratio])
        rowdata.append([num_neurons, row_primary_angle, row_explainratio, out_hyp_ratio, out_eul_ratio, out_hyp_ratio_pm, out_eul_ratio_pm, soma_explainratio])

    coldata, rowdata = np.array(coldata), np.array(rowdata)
    alldata = [coldata, rowdata]
    allmarks = ["In-Correlation", "Out-Correlation"]

    fig, axs = plt.subplots(1,2,figsize=(4*2,4))
    figexp, axexp = plt.subplots(1,1,figsize=(4,4))
    figrmax, axrmax = plt.subplots(1,1,figsize=(4,4))

    showpermute = 0
    if showpermute:
        indices = [[0,1,2,3,4],[5,6,7,8,9]]
    else:
        indices = [[0,1,2],[3,4,5]]


    for i in range(len(alldata)):
        kk = 0
        xx, toactratio = alldata[i][:,kk].flatten(), alldata[i][:,2].flatten()

        slope, intercept, r_value, p_value, std_err = stats.linregress(xx, toactratio)
        print(f"p_value: {p_value}")

        xx_line = np.linspace(np.min(xx), np.max(xx), 100)  
        yy_line = slope * xx_line + intercept  
        
        axs[i].scatter(xx, toactratio)
        axs[i].plot(xx_line, yy_line, color='red', linestyle="--")
        axs[i].set_title(f"slope: {np.round(slope,3)}; r^2: {np.round(r_value**2,3)}")
        if kk == 1:
            axs[i].set_xlabel(f"{allmarks[i]} Primary Angle")
        elif kk == 0:
            axs[i].set_xlabel(f"Number of Neurons")
        axs[i].set_ylabel(f"{allmarks[i]} Explain Ratio")

        hypratio, eulratio, somaratio = alldata[i][:,3].flatten(), alldata[i][:,4].flatten(), alldata[i][:,7].flatten()
        hypratiopm, eulratiopm = alldata[i][:,5].flatten(), alldata[i][:,6].flatten()

        if showpermute:
            data = [hypratio, eulratio, toactratio, hypratiopm, eulratiopm]
        else:
            data = [hypratio, eulratio, toactratio]
        positions = [indices[i][j] for j in range(len(indices[i]))]

        violin_parts = axexp.violinplot(data, positions=positions, showmeans=False, showmedians=True)

        for j, body in enumerate(violin_parts['bodies']):
            body.set_facecolor(c_vals[indices[i][j]])  # Set color for each violin
            body.set_edgecolor('black')             # Optionally set edge color
            body.set_alpha(0.7)                     # Set transparency (optional)

    fig.tight_layout()
    fig.savefig(f"./output/zz_overall_D{dimension}_R{R_max}_T{timeselect}_K{Kselect}.png")

    if showpermute:
        names = ["Hyp2In", "Eul2In", "In2Act", "Hyp2pmIn", "Eul2pmIn", "Hyp2Out", "Eul2Out", "Out2Act", "Hyp2pmOut", "Eul2pmOut"]
    else:
        names = ["Hyp2In", "Eul2In", "In2Act", "Hyp2Out", "Eul2Out", "Out2Act"]

    axexp.set_xticks(range(len(names))) 
    axexp.set_xticklabels(names, rotation=45, ha='right')
    axexp.axhline(1, c='red', linestyle='--')

    axexp.set_ylabel("Explanation Ratio")
    figexp.savefig(f"./output/zz_overall_exp_D{dimension}_R{R_max}_T{timeselect}_K{Kselect}.png")

    rmax_quantiles = np.array(rmax_quantiles)
    axrmax.plot(rmax_quantiles[:,0], "-o", label="Out")
    axrmax.plot(rmax_quantiles[:,1], "-o", label="In")
    axrmax.legend()
    axrmax.set_xlabel("Trial")
    axrmax.set_ylabel("Rmax Quantile")
    figrmax.savefig(f"./output/zz_overall_rmax_D{dimension}_R{R_max}_T{timeselect}_K{Kselect}.png")

    np.savez(f"./output/zz_overall_D{dimension}_R{R_max}_T{timeselect}_K{Kselect}.npz", alldata=alldata)


def microns_across_scans_rnn(Kselect):
    def find_pkl_files(directory):
        
        strname = f".pkl"
        all_pkl_files = glob.glob(os.path.join(directory, "**", "*.pkl"), recursive=True)
        matching_files = [f for f in all_pkl_files if strname in os.path.basename(f)]
        return matching_files

    directory_path = "./output_rnn/"
    pkl_files = find_pkl_files(directory_path)
    print(pkl_files)
    
    coldata, rowdata, somadata = [], [], []
    rmax_quantiles = []

    timeselect = "all"

    for pkl_file_path in pkl_files:
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)

        if timeselect == "all":
            ttind = -1
        else:
            ttind = data["timeuplst"].index(timeselect)

        cc = 1

        num_neurons = None
        rmax_quantiles.append([data["rmax_quantile_out"], data["rmax_quantile_in"]])

        column_primary_angle = None
        row_primary_angle = None
        allk_medians = data["allk_medians"][Kselect][ttind]
        activity_ratio = allk_medians[0]
        column_explainratio = allk_medians[1]
        row_explainratio = allk_medians[4]

        soma_explainratio = None

        in_hyp_ratio = allk_medians[2]-allk_medians[1]
        in_eul_ratio = allk_medians[3]-allk_medians[1]

        out_hyp_ratio = allk_medians[5]-allk_medians[4]
        out_eul_ratio = allk_medians[6]-allk_medians[4]

        coldata.append([num_neurons, column_primary_angle, column_explainratio, in_hyp_ratio, in_eul_ratio, soma_explainratio, activity_ratio])
        rowdata.append([num_neurons, row_primary_angle, row_explainratio, out_hyp_ratio, out_eul_ratio, soma_explainratio, activity_ratio])

    coldata, rowdata = np.array(coldata), np.array(rowdata)
    alldata = [coldata, rowdata]
    allmarks = ["In-Correlation", "Out-Correlation"]

    fig, axs = plt.subplots(1,2,figsize=(4*2,4))
    figexp, axexp = plt.subplots(1,1,figsize=(4,4))
    figrmax, axrmax = plt.subplots(1,1,figsize=(4,4))

    indices = [[0,1,2,6],[3,4,5,6]]

    for i in range(len(alldata)):
        toactratio = alldata[i][:,2].flatten()

        hypratio, eulratio = alldata[i][:,3].flatten(), alldata[i][:,4].flatten()

        actratio = alldata[i][:,6].flatten()

        data = [list(hypratio), list(eulratio), list(toactratio), list(actratio)]

        positions = [indices[i][j] for j in range(len(indices[i]))]

        violin_parts = axexp.violinplot(data, positions=positions, showmeans=False, showmedians=True)

        for j, body in enumerate(violin_parts['bodies']):
            body.set_facecolor(c_vals[indices[i][j]])  # Set color for each violin
            body.set_edgecolor('black')             # Optionally set edge color
            body.set_alpha(0.7)                     # Set transparency (optional)

    fig.tight_layout()
    fig.savefig(f"./output/zz_overall_rnn_K{Kselect}.png")

    names = ["HypIn", "EulIn", "InAct", "HypOut", "EulOut", "OutAct", "Activity"]
    axexp.set_xticks(range(len(names))) 
    axexp.set_xticklabels(names, rotation=45, ha='right')
    # axexp.axhline(1, c='red', linestyle='--')
    # axexp.set_ylim([-1,1])

    axexp.set_ylabel("Explanation Ratio")
    figexp.savefig(f"./output_rnn/zz_overall_exp_rnn_K{Kselect}.png")

    colin, rowout = alldata[0][:,2].flatten(), alldata[1][:,2].flatten()
    t_stat, p_value = stats.ttest_rel(colin, rowout)
    p_value_one_sided = p_value / 2
    p_value_one_sided_final = p_value_one_sided if t_stat > 0 else 1 - p_value_one_sided
    print(p_value_one_sided_final)