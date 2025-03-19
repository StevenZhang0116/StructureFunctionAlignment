import numpy as np
import json 
import matplotlib.pyplot as plt

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

dim1, dim2 = 2, 100
data1_name = f"./perf_record/dim{dim1}.json"
data2_name = f"./perf_record/dim{dim2}.json"

with open(data1_name, "r") as f:
    data1 = json.load(f)

with open(data2_name, "r") as f:
    data2 = json.load(f)

data_compare = []
for neuron_id in data1.keys():
    data_compare.append([data1[neuron_id], data2[neuron_id]])

data_compare = np.array(data_compare)

fig, ax = plt.subplots(1,1,figsize=(4,4))

x = data_compare[:, 0]
y = data_compare[:, 1]

slope, intercept = np.polyfit(x, y, 1)
y_pred = slope * x + intercept

ss_res = np.sum((y - y_pred) ** 2)     
ss_tot = np.sum((y - np.mean(y)) ** 2)      
r_squared = 1 - ss_res / ss_tot

ax.scatter(x, y, s=1)
fig.savefig(f"./perf_record/compare{dim1}_{dim2}.png", dpi=300)
