import numpy as np
import os 
import scipy 
import time

def summarize_data(ww, cc, index):
    assert index in ["in", "out", "activity"]

    output_directory = "./zz_data"

    if index in ["in", "out"]:
        search_string = f"noise_{ww}_cc_{cc}"
        dataname = "connectome"
    else:
        search_string = "activity"
        dataname = "activity"
        index = ""

    print(search_string)
    print(index)

    data_files = [
        f for f in os.listdir(output_directory)
        if all(cond in f for cond in [search_string, f"_{index}", "microns"]) and f.endswith(".mat")
    ]

    assert len(data_files) == 12

    connectome_lst, tag_lst = [], []

    for mat_file in data_files:
        data = scipy.io.loadmat(f"{output_directory}/{mat_file}")
        conn = data[dataname]
        if index == "in":
            conn = conn.T
        connectome_lst.append(conn)
        tag_lst.extend(list(data['tag'].reshape(-1,1)))

    connectome_array = np.concatenate(connectome_lst, axis=0)
    tag_lst = np.array(tag_lst).reshape(-1,1)

    def find_duplicate_indices(arr):
        # Flatten the array in case it's a column vector
        arr = arr.flatten()
        seen = {}
        duplicate_indices = []

        for i, value in enumerate(arr):
            if value in seen:
                duplicate_indices.append(i)
            else:
                seen[value] = True

        return duplicate_indices

    duplicates_index = find_duplicate_indices(tag_lst)
    tag_lst = np.delete(tag_lst, duplicates_index, axis=0)
    nonduplicate_connectome = np.delete(connectome_array, duplicates_index, axis=0)

    print(tag_lst[0:10])
    print(tag_lst.shape)
    print(nonduplicate_connectome.shape)

    scipy.io.savemat(f"{output_directory}/{search_string}_connectome_{index}.mat", {"connectome": nonduplicate_connectome, "tag": tag_lst})