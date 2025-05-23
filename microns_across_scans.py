import glob
import os
import numpy 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
import seaborn as sns
from scipy.stats import ttest_ind, ttest_rel
import pickle 

import activity_helper

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]

def analyze_pr(pkl_files):
    """"""
    alldata = []
    for pkl_file_path in pkl_files:
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)
        N = data["num_neurons"]
        pr = data["participation_ratio"]
        rpr = data["random_participation_ratio"]
        alldata.append([N, pr, rpr])

    alldata = np.array(alldata)

    data_for_violin = [alldata[:,1], alldata[:,2]]
    positions = [0, 1]
    fig, ax = plt.subplots(figsize=(4,4))
    violin_parts = ax.violinplot(data_for_violin, positions=positions, showmeans=False, showmedians=True)

    for j, body1 in enumerate(violin_parts['bodies']):
        color = c_vals[j] 
        body1.set_facecolor(color)      
        body1.set_edgecolor('black')    
        body1.set_alpha(0.7)

    for i, col_data in enumerate(data_for_violin):
        x_positions = np.full_like(col_data, i) + np.random.uniform(-0.1, 0.1, size=len(col_data))  # jitter for better visibility
        plt.scatter(x_positions, col_data, s=alldata[:,0]*0.5, alpha=0.6, color=c_vals_l[i])

    plt.xticks([0, 1], ['PR for Coregistered Neuron', 'PR for Random Neuron'])
    plt.ylabel('Normalized Participation Ratio')
    plt.savefig("participation_ratio.png", dpi=300)

def find_pkl_files(directory, dimension, R_max, pendindex, perturb, perturb_amount):
    """"""
    # only select pkl files with desired dimension and R_max
    strname1 = f"D{dimension}_R{R_max}.pkl"
    strname2 = f"{pendindex}_metadata"
    
    if perturb:
        strname2 = f"{pendindex}_perturb_{perturb_amount}_metadata"
    
    all_pkl_files = glob.glob(os.path.join(directory, "**", "*.pkl"), recursive=True)
    if "forall" in strname2:
        matching_files = [f for f in all_pkl_files if strname1 in os.path.basename(f) and strname2 in os.path.basename(f)]
    else:
        matching_files = [f for f in all_pkl_files if strname1 in os.path.basename(f) and strname2 in os.path.basename(f) and "forall" not in os.path.basename(f)]

    assert len(matching_files) > 10 and len(matching_files) <= 12
    
    return matching_files

def microns_across_scans(R_max, dimension, Kselect_lst, pendindex, scan_specific, perturb, perturb_amount):
    """
    """
    print(R_max)
    Krange_data = []

    for Kselect in range(Kselect_lst):
        if scan_specific:
            directory_path = "./output/"
        else:
            if not perturb: 
                directory_path = "./output-all/"
            else:
                directory_path = "./output-all-perturb/"
        
        if not perturb:
            perturb_amount = 0
        perturb_add_string = f"_perturb_{perturb_amount}" if perturb else ""
        
        pkl_files = find_pkl_files(directory_path, dimension, R_max, pendindex, perturb, perturb_amount)

        analyze_pr(pkl_files)
        
        coldata, rowdata, somadata = [], [], []
        rmax_quantiles = []

        gt_median_corrs = []

        timeselect = "all"

        showpermute = 0
        showactivity = 1
        showminimal = 0

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
            allk_medians_session = data["allk_medians_session"][Kselect]

            activity_explain = allk_medians[0]

            mean_value = np.mean(data["gt_median_corr"])

            if not showactivity:
                column_explainratio = allk_medians[1]/allk_medians[0]
                row_explainratio = allk_medians[4]/allk_medians[0]
                soma_explainratio = allk_medians[7]/allk_medians[0]
                in_hyp_ratio = allk_medians[2]/allk_medians[1]
                in_eul_ratio = allk_medians[3]/allk_medians[1]
                out_hyp_ratio = allk_medians[5]/allk_medians[4]
                out_eul_ratio = allk_medians[6]/allk_medians[4]
            else:
                column_explainratio = allk_medians[1]
                row_explainratio = allk_medians[4]
                soma_explainratio = allk_medians[7]
                in_hyp_ratio = allk_medians[2]
                in_eul_ratio = allk_medians[3]
                out_hyp_ratio = allk_medians[5]
                out_eul_ratio = allk_medians[6]

                session_act = allk_medians_session[0]
                session_in = allk_medians_session[1]
                session_inhyp = allk_medians_session[2]
                session_ineul = allk_medians_session[3]
                session_out = allk_medians_session[4]
                session_outhyp = allk_medians_session[5]
                session_outeul = allk_medians_session[6]
            
            coldata.append([num_neurons, column_primary_angle, column_explainratio, activity_explain, in_hyp_ratio, in_eul_ratio, mean_value, \
                            np.nan, np.nan, soma_explainratio, session_act, session_in, session_inhyp, session_ineul])
            rowdata.append([num_neurons, row_primary_angle, row_explainratio, activity_explain, out_hyp_ratio, out_eul_ratio, mean_value, \
                            np.nan, np.nan, soma_explainratio, session_act, session_out, session_outhyp, session_outeul])
            
        coldata, rowdata = np.array(coldata), np.array(rowdata)
        alldata = [coldata, rowdata]
        allmarks = ["In-Correlation", "Out-Correlation"]

        fig, axs = plt.subplots(1,2,figsize=(4*2,4))
        if showminimal:
            figexp, axexp = plt.subplots(1,1,figsize=(2,4))
        else:
            figexp, axexp = plt.subplots(1,2,figsize=(4*2,4))
        figrmax, axrmax = plt.subplots(1,1,figsize=(4,4))
        
        if showpermute:
            indices = [[0,1,2,3,4],[5,6,7,8,9]]
        else:
            if not showactivity:
                indices = [[0,1,2],[3,4,5]]
            else:
                if showminimal:
                    indices = [[0,1],[2,3]]
                else:
                    indices = [[0,1,2],[3,4,5]]

        structure_data = []
        structure_data_out = [] 
        
        for i in range(len(alldata)):
            kk = 0
            xx, fullconn = alldata[i][:,kk].flatten(), alldata[i][:,2].flatten()
            activity_base = alldata[i][:,3].flatten()

            ppt = fullconn

            slope, intercept, r_value, p_value, std_err = stats.linregress(xx, ppt)
            # print(f"p_value: {p_value}")

            xx_line = np.linspace(np.min(xx), np.max(xx), 100)  
            yy_line = slope * xx_line + intercept  
            
            axs[i].scatter(xx, ppt)
            axs[i].plot(xx_line, yy_line, color='red', linestyle="--", label=f"r^2={np.round(r_value,3)}")
            axs[i].set_title(f"p: {np.round(p_value,3)}")

            if kk == 1:
                axs[i].set_xlabel(f"{allmarks[i]} Primary Angle")
            elif kk == 0:
                axs[i].set_xlabel(f"Number of Neurons")
            axs[i].set_ylabel(f"{allmarks[i]} Explain Ratio" if not showactivity else f"{allmarks[i]} Correlation")
            # axs[i].set_ylabel(f"Activity Correlation")
            axs[i].legend()

            hypratio, eulratio, somaratio = alldata[i][:,4].flatten(), alldata[i][:,5].flatten(), alldata[i][:,9].flatten()
            hypratiopm, eulratiopm = alldata[i][:,7].flatten(), alldata[i][:,8].flatten()
            meansess = alldata[i][:,6].flatten()

            activity_session_base = alldata[i][:,10].flatten()
            activity_session_in = alldata[i][:,11].flatten()
            activity_session_inhyp = alldata[i][:,12].flatten()
            activity_session_ineul = alldata[i][:,13].flatten()

            if showpermute:
                data = [hypratio, eulratio, fullconn, hypratiopm, eulratiopm]
            else:
                if not showactivity:
                    data = [hypratio, eulratio, fullconn]
                else:
                    if showminimal:
                        data = [hypratio/activity_base, fullconn/activity_base]
                    else:
                        data = [hypratio/activity_base, eulratio/activity_base, fullconn/activity_base]
                        if Kselect == 0 and i == 0:
                            structure_data.append([hypratio, fullconn, activity_base])
                        if Kselect == 0 and i == 1:
                            structure_data_out.append([hypratio, fullconn, activity_base])
                        _, p_value = ttest_rel(data[0], data[2], alternative="greater")

                        if i == 0:
                            Krange_data.append([activity_base, hypratio, fullconn])
                        data_session = [activity_session_inhyp/activity_session_base, activity_session_ineul/activity_session_base, activity_session_in/activity_session_base]
                        _, p_value_session = ttest_rel(data_session[0], data_session[2], alternative="greater")

            benchmark_session_mean = meansess / activity_base
            mean_benchmark = np.mean(benchmark_session_mean)

            positions = [indices[i][j] for j in range(len(indices[i]))]
            
            violin_parts = axexp[0].violinplot(data, positions=positions, showmeans=False, showmedians=True)
            violin_parts_session = axexp[1].violinplot(data_session, positions=positions, showmeans=False, showmedians=True)

            for posindex in range(len(positions)):
                for datapt in data[posindex]:
                    axexp[0].scatter(positions[posindex], datapt, color=c_vals[positions[posindex]]) 
                for datapt in data_session[posindex]:
                    axexp[1].scatter(positions[posindex], datapt, color=c_vals[positions[posindex]])

            for ax in axexp:
                ax.axhline(mean_benchmark, c='red', linestyle='--')

            for j, (body1, body2) in enumerate(zip(violin_parts['bodies'], violin_parts_session['bodies'])):
                color = c_vals[indices[i][j]]  # Determine color for the current index
                body1.set_facecolor(color)      # Set color for violin in violin_parts
                body2.set_facecolor(color)      # Set same color for violin in violin_parts_session
                body1.set_edgecolor('black')    # Optionally set edge color
                body2.set_edgecolor('black')
                body1.set_alpha(0.7)
                body2.set_alpha(0.7)

        structure_dict = {
            "structure_data": structure_data,
            "structure_data_out": structure_data_out
        }
        
        if Kselect == 0:
            with open(f"structure_info/{pendindex}{perturb_add_string}.pkl", "wb") as file:
                pickle.dump(structure_dict, file)

        fig.tight_layout()
        fig.savefig(f"{directory_path}zz_overall_D{dimension}_R{R_max}_T{timeselect}_K{Kselect}_{pendindex}{perturb_add_string}.png", dpi=300)

        if showpermute:
            names = ["Hyp2In", "Eul2In", "In2Act", "Hyp2pmIn", "Eul2pmIn", "Hyp2Out", "Eul2Out", "Out2Act", "Hyp2pmOut", "Eul2pmOut"]
        else:
            if showminimal:
                names = ["HypInAct", "InAct", "HypOutAct", "OutAct"]
            else:
                names = ["Hyp2In", "Eul2In", "In2Act", "Hyp2Out", "Eul2Out", "Out2Act"]

        for ax in axexp:
            ax.set_xticks(range(len(names))) 
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylim([-0.2, 1.0])

            if not showactivity:
                ax.axhline(1, c='red', linestyle='--')

            if showactivity:
                ax.set_ylabel("Explanation Ratio to Activity")
            else:
                ax.set_ylabel("Explanation Ratio")

        figexp.tight_layout()
        figexp.savefig(f"{directory_path}zz_overall_exp_D{dimension}_R{R_max}_T{timeselect}_K{Kselect}_{pendindex}{perturb_add_string}.png", dpi=300)

        rmax_quantiles = np.array(rmax_quantiles)
        axrmax.plot(rmax_quantiles[:,0], "-o", label="Out")
        axrmax.plot(rmax_quantiles[:,1], "-o", label="In")
        axrmax.legend()
        axrmax.set_xlabel("Trial")
        axrmax.set_ylabel("Rmax Quantile")
        figrmax.savefig(f"./output/zz_overall_rmax_D{dimension}_R{R_max}_T{timeselect}_K{Kselect}_{pendindex}{perturb_add_string}.png", dpi=300)

        np.savez(f"{directory_path}zz_overall_D{dimension}_R{R_max}_T{timeselect}_K{Kselect}_{pendindex}{perturb_add_string}.npz", alldata=alldata)

    figacrossK, axsacrossK = plt.subplots(figsize=(4,4))
    x_ticks = ["All", "Top 50%", "Top 20%", "Random 50%", "Random 20%"]
    xxxx = [i for i in range(len(x_ticks))]
    
    for i, label, color in zip(range(3), ["Activity", "HypIn", "In"], c_vals):
        mean_values = [np.mean(Krange_data[j][i]) for j in range(len(Krange_data))]
        std_values = [np.std(Krange_data[j][i]) for j in range(len(Krange_data))]
        
        aaa = [Krange_data[j][1] for j in range(len(Krange_data))]
        bbb = [Krange_data[j][2] for j in range(len(Krange_data))]
        
        for tt in range(len(aaa)):
            _, p_value = ttest_rel(aaa[tt], bbb[tt], alternative="greater")
            # print(p_value)            
        
        axsacrossK.plot(xxxx, mean_values, "-o", label=label, color=color)
        
        axsacrossK.fill_between(
            xxxx,
            [m - s for m, s in zip(mean_values, std_values)],  
            [m + s for m, s in zip(mean_values, std_values)],  
            color=color,
            alpha=0.2, 
        )
    axsacrossK.legend()
    axsacrossK.set_xticks(ticks=xxxx, labels=x_ticks, rotation=20, ha='right')
    axsacrossK.set_ylabel("Reconstruction Accuracy")
    figacrossK.tight_layout()
    figacrossK.savefig(f"./Kacross_results/{pendindex}{perturb_add_string}.png", dpi=300)

    
def microns_across_scans_rnn(Kselect):
    """
    """
    def find_pkl_files(directory):
        strname = f".pkl"
        all_pkl_files = glob.glob(os.path.join(directory, "**", "*.pkl"), recursive=True)
        matching_files = [f for f in all_pkl_files if strname in os.path.basename(f)]
        return matching_files

    directory_path = "./output_rnn/"
    pkl_files = find_pkl_files(directory_path)
    print(pkl_files)

    showminimal = 1
    
    coldata, rowdata, somadata = [], [], []
    rmax_quantiles = []
    act_mean = []

    timeselect = "all"

    for pkl_file_path in pkl_files:
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)

        if timeselect == "all":
            ttind = -1
        else:
            ttind = data["timeuplst"].index(timeselect)

        cc = 1

        act_mean.append(data["mean_activity_corr"])

        num_neurons = None
        rmax_quantiles.append([data["rmax_quantile_out"], data["rmax_quantile_in"]])

        column_primary_angle = None
        row_primary_angle = None
        allk_medians = data["allk_medians"][Kselect][ttind]
        activity_ratio = allk_medians[0]
        column_explainratio = allk_medians[1]
        row_explainratio = allk_medians[4]

        soma_explainratio = None

        in_hyp_ratio = allk_medians[2]
        in_eul_ratio = allk_medians[3]

        out_hyp_ratio = allk_medians[5]
        out_eul_ratio = allk_medians[6]

        coldata.append([num_neurons, column_primary_angle, column_explainratio, in_hyp_ratio, in_eul_ratio, soma_explainratio, activity_ratio])
        rowdata.append([num_neurons, row_primary_angle, row_explainratio, out_hyp_ratio, out_eul_ratio, soma_explainratio, activity_ratio])

    act_corr_mean = np.mean(act_mean)

    coldata, rowdata = np.array(coldata), np.array(rowdata)
    alldata = [coldata, rowdata]
    allmarks = ["In-Correlation", "Out-Correlation"]

    fig, axs = plt.subplots(1,2,figsize=(4*2,4))
    if showminimal:
        figexp, axexp = plt.subplots(1,1,figsize=(4,4))
    else:
        figexp, axexp = plt.subplots(1,1,figsize=(4,4))
    figrmax, axrmax = plt.subplots(1,1,figsize=(4,4))

    if showminimal:
        indices = [[0,1,2],[3,4,5]]
    else:
        indices = [[0,1,2,6],[3,4,5,6]]

    for i in range(len(alldata)):
        fullconn = alldata[i][:,2].flatten()

        hypratio, eulratio = alldata[i][:,3].flatten(), alldata[i][:,4].flatten()

        actratio = alldata[i][:,6].flatten()

        if showminimal:
            data = [list(hypratio/actratio), list(eulratio/actratio), list(fullconn/actratio)]
        else:
            data = [list(hypratio), list(eulratio), list(fullconn), list(actratio)]

        positions = [indices[i][j] for j in range(len(indices[i]))]

        violin_parts = axexp.violinplot(data, positions=positions, showmeans=False, showmedians=True)

        for posindex in range(len(positions)):
            for datapt in data[posindex]:
                axexp.scatter(positions[posindex], datapt, color=c_vals[positions[posindex]]) 

        for j, body in enumerate(violin_parts['bodies']):
            body.set_facecolor(c_vals[indices[i][j]])  # Set color for each violin
            body.set_edgecolor('black')             # Optionally set edge color
            body.set_alpha(0.7)                     # Set transparency (optional)


    fig.tight_layout()
    fig.savefig(f"./output_rnn/zz_overall_rnn_K{Kselect}.png")

    if showminimal:
        names = ["Hyp2In", "Eul2In", "In2Act", "Hyp2Out", "Eul2Out", "Out2Act"]
    else:
        ames = ["HypIn", "EulIn", "InAct", "HypOut", "EulOut", "OutAct", "Activity"]
    axexp.set_xticks(range(len(names))) 
    axexp.set_xticklabels(names, rotation=45, ha='right')
    axexp.set_ylim([-0.2,1.0])
    axexp.axhline(np.mean(act_mean/actratio), c='red', linestyle='--')

    if not showminimal:
        axexp.set_ylabel("Correlation")
    figexp.tight_layout()
    figexp.savefig(f"./output_rnn/zz_overall_exp_rnn_K{Kselect}.png")

    colin, rowout = alldata[0][:,2].flatten(), alldata[1][:,2].flatten()
    t_stat, p_value = stats.ttest_rel(colin, rowout)
    p_value_one_sided = p_value / 2
    p_value_one_sided_final = p_value_one_sided if t_stat > 0 else 1 - p_value_one_sided
    print(p_value_one_sided_final)