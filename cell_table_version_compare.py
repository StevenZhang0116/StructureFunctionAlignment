import pandas as pd 
import numpy as np

cell_table1 = pd.read_feather("../microns_cell_tables/sven/microns_cell_annos_CV_240827.feather")
cell_table2 = pd.read_feather("../microns_cell_tables/sven/microns_cell_annos_240603.feather")
cell_tables = [cell_table1, cell_table2]
arrs = []

for cell_table in cell_tables:
    good_ct = cell_table[(cell_table["status_axon"].isin(["extended", "clean"])) & (cell_table["full_dendrite"] == True)]
    good_ct = good_ct[good_ct["classification_system"] == "excitatory_neuron"]
    good_ct_pt_rootids = good_ct["pt_root_id"].to_numpy()
    arrs.append(good_ct_pt_rootids)

print(len(arrs[0]), len(arrs[1]))
print(len(np.intersect1d(arrs[0], arrs[1])))