import os
import glob
import numpy as np
import re
import matplotlib.pyplot as plt

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

def extract_R_value(file_name):
    match = re.search(r'_R([0-9\.e\+\-]+)_Tall', file_name)
    if match:
        return match.group(1)
    return None 

directory = "./output/"
c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e']

dimension = 3

npz_files = [file for file in glob.glob(os.path.join(directory, "*.npz")) if f"D{dimension}" in os.path.basename(file)]
r_files_pairs = [(float(extract_R_value(file)), file) for file in npz_files]
r_files_pairs.sort(key=lambda pair: pair[0])
r_values, npz_files = zip(*r_files_pairs)
r_values = list(r_values)
npz_files = list(npz_files)

hypin_data = []
hypout_data = []

figcomparer, axscomparer = plt.subplots(1,2,figsize=(8,4))  

for i in range(len(npz_files)):
    file_path = npz_files[i]
    data = np.load(file_path, allow_pickle=True)["alldata"]
    hypin = data[0][:,3]
    hypout = data[1][:,3]
    hypin_data.append(hypin)
    hypout_data.append(hypout)

violin_parts_in = axscomparer[0].violinplot(hypin_data, positions=r_values, showmeans=False, showmedians=True, widths=0.1)
violin_parts_out = axscomparer[1].violinplot(hypout_data, positions=r_values, showmeans=False, showmedians=True, widths=0.1)

for violin_parts in [violin_parts_in, violin_parts_out]:
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(c_vals[i % len(c_vals)])  # Set the face color from the custom color list
        pc.set_edgecolor('black')  # Optional: set the edge color to black
        pc.set_alpha(0.8) 

    
for ax in axscomparer:
    ax.axhline(1, c='red', linestyle='--')
    ax.set_xlabel("R_max")
    ax.set_ylabel("Explanation Ratio")
    
axscomparer[0].set_title("HypIn")
axscomparer[1].set_title("HypOut")

figcomparer.tight_layout()
figcomparer.savefig(f"./zz_Rmax_D{dimension}_scan.png")

