import numpy as np 
import os 
import sys
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']

def fit_and_evaluate(x_list, y_list, num_points=100):
    """
    """
    x = np.array(x_list).reshape(-1, 1)
    y = np.array(y_list)

    # Fit model
    model = LinearRegression()
    model.fit(x, y)

    # R^2 score
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)

    # Generate line for plotting
    x_line = np.linspace(np.min(x), np.max(x), num_points).reshape(-1, 1)
    y_line = model.predict(x_line)

    return r2, x_line.flatten(), y_line


if __name__ == "__main__":
    directory = "./oracle_score_analysis/"
    
    all_oracle, all_activity, all_in, all_inhyp, all_out, all_outhyp, all_soma = [], [], [], [], [], [], []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            print(f"Reading {file_path}...")

            with open(file_path, "rb") as f:
                data = pickle.load(f)
                
            all_oracle.extend(data["oracle_scores"])
            all_activity.extend(data["summ"][:,0])
            all_in.extend(data["summ"][:,1])
            all_inhyp.extend(data["summ"][:,2])
            all_out.extend(data["summ"][:,4])
            all_outhyp.extend(data["summ"][:,5])
            all_soma.extend(data["summ"][:,6])

    all_oracle, all_activity, all_in, all_inhyp, all_out, all_outhyp, all_soma \
        = np.array(all_oracle), np.array(all_activity), np.array(all_in), np.array(all_inhyp), np.array(all_out), np.array(all_outhyp), np.array(all_soma)
    valid = ~np.isnan(all_oracle)
    print(np.sum(valid))

    fig, axs = plt.subplots(1,2,figsize=(4*2,4))
    axs[0].scatter(all_oracle, all_activity, alpha=0.5, color=c_vals[0])
    rr1, xx1, yy1 = fit_and_evaluate(all_oracle[valid], all_activity[valid])
    axs[0].plot(xx1, yy1, color="black", label=fr"$R^2 = {rr1:.2f}$", linestyle="--")
    axs[1].scatter(all_oracle, all_inhyp, alpha=0.5, color=c_vals[1])
    rr2, xx2, yy2 = fit_and_evaluate(all_oracle[valid], all_inhyp[valid])
    axs[1].plot(xx2, yy2, color="black", label=fr"$R^2 = {rr2:.2f}$", linestyle="--")
    axs[0].set_ylabel("Predictability Using Activity Corr")
    axs[1].set_ylabel("Predictability Using Hyp Embedding of Input Corr")
    for ax in axs:
        ax.set_xlabel("Oracle Scores")
        ax.legend()
    fig.tight_layout()
    fig.savefig("./figures/oracle.png", dpi=300)
    
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.hist(all_activity, bins=20, label="Activity", alpha=0.5, color=c_vals[0])
    ax.hist(all_in, bins=20, label="In-degree", alpha=0.5, color=c_vals[1])
    # ax.hist(all_out, bins=20, label="Out-degree", alpha=0.5, color=c_vals[2])
    ax.hist(all_inhyp, bins=20, label="In-Hyp", alpha=0.5, color=c_vals[3])
    # ax.hist(all_outhyp, bins=20, label="Out-Hyp", alpha=0.5, color=c_vals[4])
    ax.legend()
    fig.savefig("./figures/scattered_activity.png", dpi=300)
    
    fig, axs = plt.subplots(1,5,figsize=(4*5,4*1))
    axs[0].scatter(all_soma, all_activity, alpha=0.5, color=c_vals[0])
    axs[0].set_ylabel("Predictability Using Activity Corr")
    axs[1].scatter(all_soma, all_inhyp, alpha=0.5, color=c_vals[1])
    axs[1].set_ylabel("Predictability Using Hyp Embedding of Input Corr")
    axs[2].scatter(all_soma, all_in, alpha=0.5, color=c_vals[2])
    axs[2].set_ylabel("Predictability Using Input Corr")
    axs[3].scatter(all_soma, all_outhyp, alpha=0.5, color=c_vals[3])
    axs[3].set_ylabel("Predictability Using Hyp Embedding of Output Corr")
    axs[4].scatter(all_soma, all_out, alpha=0.5, color=c_vals[4])
    axs[4].set_ylabel("Predictability Using Out Corr")
    for ax in axs.flatten():
        ax.set_xlabel("Predictability Using Soma Distance")
        ax.set_xlim([-0.1, 0.6])
        ax.set_ylim([-0.1, 0.6])
    fig.tight_layout()
    fig.savefig("./figures/vs_soma.png", dpi=300)
