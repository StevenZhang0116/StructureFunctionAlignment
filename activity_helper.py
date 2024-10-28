import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import time
import random
import gc

import scipy 
from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline

import sys
sys.path.append("/gscratch/amath/zihan-zhang/spatial/demo/pyclique")
import compute_betti_curves

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',]
colorset = [c_vals_l, c_vals_d]
lines = ["-.", "--"]

def float_to_scientific(value, n=4):
    return f"{value:.{n}e}"

def pearson_correlation_with_nans(arr1, arr2):
    """
    Calculate the Pearson correlation for two arrays where each array has exactly one NaN.
    The NaN elements are removed from both arrays before calculation.
    """
    if len(arr1) != len(arr2):
        raise ValueError("Both arrays must have the same length.")
    
    nan_index1 = np.where(np.isnan(arr1))[0][0]
    nan_index2 = np.where(np.isnan(arr2))[0][0]

    nan_num1 = len(np.where(np.isnan(arr1))[0])
    nan_num2 = len(np.where(np.isnan(arr2))[0])

    if nan_num1 != 1 or nan_num2 != 1:
        print(f"Nan: {nan_num1}; {nan_num2}")
        raise ValueError("Each array must contain exactly one NaN value.")
    
    indices_to_keep = np.setdiff1d(np.arange(len(arr1)), [nan_index1, nan_index2])
    truncated_arr1 = arr1[indices_to_keep]
    truncated_arr2 = arr2[indices_to_keep]

    # if len(indices_to_keep) == len(arr1) - 2:
    #     truncated_arr1 = np.nan_to_num(arr1, nan=0.0)
    #     truncated_arr2 = np.nan_to_num(arr2, nan=0.0)
    
    correlation, _ = pearsonr(truncated_arr1, truncated_arr2)
    
    return correlation

def remove_nan_inf_union(matrix):
    """
    """
    print(f"Before: {matrix.shape}")

    nan_inf_rows = np.all(np.isnan(matrix) | np.isinf(matrix), axis=1)
    nan_inf_columns = np.all(np.isnan(matrix) | np.isinf(matrix), axis=0)

    rows_to_remove = np.where(nan_inf_rows)[0]
    columns_to_remove = np.where(nan_inf_columns)[0]

    indices_to_remove = np.union1d(rows_to_remove, columns_to_remove)

    smaller_matrix = np.delete(matrix, indices_to_remove, axis=0)
    smaller_matrix = np.delete(smaller_matrix, indices_to_remove, axis=1)

    print(f"After: {smaller_matrix.shape}")

    return smaller_matrix

def cosine_similarity(arr1, arr2):
    """
    """
    dot_product = np.dot(arr1, arr2)
    norm_a = np.linalg.norm(arr1)
    norm_b = np.linalg.norm(arr2)
    return dot_product / (norm_a * norm_b)

def sanity_check_W(truncated_W_correlation, activity_correlation):
    """
    """
    mask_inf_nan = np.isinf(truncated_W_correlation) | np.isnan(truncated_W_correlation)
    rows_to_delete = np.all(mask_inf_nan, axis=1)
    cols_to_delete = np.all(mask_inf_nan, axis=0)
    indices_to_delete = np.where(rows_to_delete & cols_to_delete)[0]

    activity_correlation = np.delete(activity_correlation, indices_to_delete, axis=0)
    activity_correlation = np.delete(activity_correlation, indices_to_delete, axis=1)
    truncated_W_correlation = np.delete(truncated_W_correlation, indices_to_delete, axis=0)
    truncated_W_correlation = np.delete(truncated_W_correlation, indices_to_delete, axis=1)

    return truncated_W_correlation, activity_correlation, indices_to_delete

def match_list_lengths(list1, list2):
    """
    this function is not technically rigorous but since the lists length difference are minimal, the effect is minimal
    """
    if len(list1) > len(list2):
        indices_to_remove = random.sample(range(len(list1)), len(list1) - len(list2))
        list1 = [elem for i, elem in enumerate(list1) if i not in indices_to_remove]
    elif len(list2) > len(list1):
        indices_to_remove = random.sample(range(len(list2)), len(list2) - len(list1))
        list2 = [elem for i, elem in enumerate(list2) if i not in indices_to_remove]
    
    return list1, list2

def find_quantile(matrix, value):
    """
    """
    flattened = np.sort(matrix.flatten())            
    rank = np.searchsorted(flattened, value, side='right')            
    quantile = rank / len(flattened)
    return quantile

def find_value_for_quantile(matrix, quantile):
    """
    """
    flattened = np.sort(matrix.flatten())            
    index = int(np.clip(quantile * len(flattened), 0, len(flattened) - 1))            
    return flattened[index]

def permute_symmetric_matrix(matrix):
    """
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    N = matrix.shape[0]
    permutation = np.random.permutation(N)    
    permuted_matrix = matrix[permutation][:, permutation]
    
    return permuted_matrix

def permute_symmetric_matrix_cellwise(matrix):
    """
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    N = matrix.shape[0]    
    triu_indices = np.triu_indices(N, k=1)    
    upper_tri_values = matrix[triu_indices]    
    permuted_values = np.random.permutation(upper_tri_values)    
    permuted_matrix = np.zeros_like(matrix)    
    np.fill_diagonal(permuted_matrix, np.diag(matrix))    
    permuted_matrix[triu_indices] = permuted_values    
    permuted_matrix = permuted_matrix + permuted_matrix.T - np.diag(np.diag(permuted_matrix))
    
    return permuted_matrix

def angles_between_flats(v_lst, u_lst):
    """
    Calculate the smallest principal angle between two subspaces spanned by vectors in v_lst and u_lst.
    """
    # Stack vectors to form matrices V and U, vectors are reshaped to (N, 1) for proper matrix formation
    V = np.column_stack([v.reshape(-1, 1) for v in v_lst])
    U = np.column_stack([u.reshape(-1, 1) for u in u_lst])
    
    # Orthonormalize the columns of V and U using QR decomposition
    Q1, _ = np.linalg.qr(V)
    Q2, _ = np.linalg.qr(U)
    
    # Compute the matrix of cosines of angles (dot products) between the bases
    M = np.dot(Q1.T, Q2)
    
    # Use SVD to find the cosines of the principal angles
    _, sigma, _ = np.linalg.svd(M)
    
    # The smallest principal angle is the arccos of the largest singular value
    smallest_angle_degrees = np.degrees(np.arccos(np.max(sigma)))
    
    return smallest_angle_degrees

def angles_between_flats_wrap(W_corr, activity_correlation, angle_consideration=16):
    """
    """
    U_connectome, S_connectome, Vh_connectome = np.linalg.svd(W_corr)
    U_activity, S_activity, Vh_activity = np.linalg.svd(activity_correlation)

    dim_loader, angle_loader = [], []
    for num_dimension in range(1,angle_consideration):
        U_comps_activity = [U_activity[:,i] for i in range(num_dimension)]
        U_comps_connectome = [U_connectome[:,i] for i in range(num_dimension)]
        angle_in_bewteen = angles_between_flats(U_comps_activity, U_comps_connectome)
        dim_loader.append(num_dimension)
        angle_loader.append(angle_in_bewteen)

    return dim_loader, angle_loader

def test_diagonal_significance(matrix):
    """
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    
    diagonal = np.diag(matrix)
    
    i, j = np.indices(matrix.shape)
    off_diagonal = matrix[i != j]
    
    mean_diagonal = np.mean(diagonal)
    mean_off_diagonal = np.mean(off_diagonal)
    # t_stat, p_value = stats.ttest_1samp(diagonal, mean_off_diagonal)

    t_stat, p_value = stats.ttest_ind(diagonal, off_diagonal, equal_var=False)
    p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    
    return mean_diagonal, mean_off_diagonal, t_stat, p_value

def moving_average(data, window_size):
    """
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def spline_set(y):
    """
    """
    spt = 3000
    x = [(i+1)/len(y) for i in range(len(y))]
    x_new = np.linspace(0, 1, spt)
    cs = CubicSpline(x, y)
    y_new = cs(x_new)
    return y_new


def betti_analysis(data_lst, inputnames, label=""):
    """
    originally implemented in microns_activity_analysis.py
    data_lst: [activity_correlation, structure_correlation]
    """
    print(label)
    assert len(data_lst) == 5
    doconnectome = (label == "S8s5")
    Nneuron = data_lst[0].shape[0]
    NneuronWrow = data_lst[3].shape[0]
    NneuronWcol = data_lst[4].shape[0]

    if doconnectome:
        figgood, axsgood = plt.subplots(1,2,figsize=(4*2,4))

    fig, axs = plt.subplots(1,3,figsize=(4*3,4))

    groundtruth_bettis = [] # for 3 correlation matrix (3 bettis)
    groundtruth_integratedbettis = []

    for index in range(len(data_lst)):
        data = data_lst[index]
        print(f"Data Shape: {data.shape}")
        groundtruth_betti, groundtruth_integratedbetti = [], [] # for 1 correlation matrix (3 bettis)
        
        [betti_curves,edge_densities] = compute_betti_curves.compute_betti_curves(data)
        # dd = int(len(edge_densities)/100)
        dd = 1
    
        for i in range(3):
            curve = betti_curves[:,i+1]
            curve = moving_average(curve,dd)
            consecutive_differences = [edge_densities[i+1] - edge_densities[i] for i in range(len(edge_densities) - 1)]
            integrated_betti = np.sum([a * b for a, b in zip(curve, consecutive_differences)])
            groundtruth_integratedbetti.append(integrated_betti)
            if index <= 2:
                axs[index].plot(moving_average(edge_densities,dd), curve, c=c_vals[i], label=f"Betti {i+1}")
            elif doconnectome and index > 2: # only do this once for one scan
                axsgood[index-3].plot(moving_average(edge_densities,dd), curve, c=c_vals[i], label=f"Betti {i+1}")
            groundtruth_betti.append(curve)

        groundtruth_bettis.append(groundtruth_betti)
        groundtruth_integratedbettis.append(groundtruth_integratedbetti)

    repeat = 500
    dimension = 2
    noise = 0.2
    readin_hypfile = f"./zz_pyclique/hyperbolic_dis_n={Nneuron}_repeat={repeat}_dim_{dimension}noise_{noise}.mat"
    readin_files_lst = [readin_hypfile]
    names = ["Eul", "Hyp"]
    
    if doconnectome:
        readin_W_hypfiles = [f"./zz_pyclique/hyperbolic_dis_n={NneuronWrow}_repeat={repeat}_dim_{dimension}noise_{noise}.mat", \
                            f"./zz_pyclique/hyperbolic_dis_n={NneuronWcol}_repeat={repeat}_dim_{dimension}noise_{noise}.mat"]
        calculate_betti_for_connectome(axsgood, readin_W_hypfiles, groundtruth_bettis[3:], groundtruth_integratedbettis[3:], repeat, dd, noise)

        for ax in axsgood:
            ax.legend() 
        figgood.savefig(f"./zz_pyclique_results/gt_connectome.png")

        sys.exit()


    for iii in range(len(readin_files_lst)):
        readin_files = readin_files_lst[iii]
        select = ""

        data = scipy.io.loadmat(readin_files)
        data = data['distance_matrices'] if iii == 0 else data

        if isinstance(data, dict):
            data_keys = data.keys()
            fields = sorted([key for key in data_keys if not key.startswith('__')])
        else:
            fields = sorted(data.dtype.names)

        fields = sorted(fields, key=lambda x: int(x.split('_')[1]))
        print(f"fields: {fields}")

        allerrs = []
        allsynthetic = []
        fake_integrated_bettis = []

        for fieldNameIter in range(len(fields)):
            thisbetti = [[],[],[]]
            fieldName = fields[fieldNameIter]
            print(f"fieldName: {fieldName}")
            if isinstance(data, dict):
                currentMatrix = data[fieldName]
            else:
                currentMatrix = data[fieldName][0, 0]

            for i in range(repeat):
                # print(f'Number: {i+1}')
                squeeze_mat = np.squeeze(currentMatrix[i, :, :])
                [betti_curves,edge_densities] = compute_betti_curves.compute_betti_curves(squeeze_mat)
                for i in range(3):
                    thisbetti[i].append(betti_curves[:,i+1])

            thisbetti = [np.array(subbetti) for subbetti in thisbetti]
            meanbetti = [moving_average(np.mean(subbetti, axis=0), dd) for subbetti in thisbetti]
            fake_integrated_betti = []
            for jjj in range(3):
                consecutive_differences = [edge_densities[i+1] - edge_densities[i] for i in range(len(edge_densities) - 1)]
                integrated_betti = np.sum([a * b for a, b in zip(meanbetti[jjj], consecutive_differences)])
                fake_integrated_betti.append(integrated_betti)
            stdbetti = [moving_average(np.std(subbetti, axis=0),dd) for subbetti in thisbetti]

            fake_integrated_bettis.append(fake_integrated_betti)
            
            oneerr = []
            for j in range(3):
                errbetti = [mean_squared_error(spline_set(meanbetti[i]), spline_set(groundtruth_bettis[j][i])) for i in range(3)]
                oneerr.append(np.sum(errbetti))
            allerrs.append(oneerr)

            allsynthetic.append([meanbetti, stdbetti, moving_average(edge_densities,dd)])

        allerrs = np.array(allerrs)
        fakeallbettis = []
        for index in range(allerrs.shape[1]): # for each correlation matrix
            minerr_index = np.argmin(allerrs[:,index])
            synthetic_best = allsynthetic[minerr_index]
            axs[index].set_title(f"{inputnames[index]}; {fields[minerr_index]} ")
            realbetti, fakebetti = groundtruth_integratedbettis[index], fake_integrated_bettis[minerr_index]
            print(realbetti)
            print(fakebetti)
            fakeallbettis.append([realbetti, fakebetti])
            for i in range(3):
                edge_densities = synthetic_best[2]
                axs[index].plot(edge_densities, synthetic_best[0][i], c=colorset[iii][i], linestyle=lines[iii], label=f"{names[iii]} Betti {i+1}")
                axs[index].fill_between(edge_densities, synthetic_best[0][i]-synthetic_best[1][i], synthetic_best[0][i]+synthetic_best[1][i], color=c_vals_l[i], alpha=0.2)

        np.save(f'./zz_pyclique_results/{label}_bettis.npy', np.array(fakeallbettis))


    fig.savefig(f"./zz_pyclique_results/{label}.png")
    time.sleep(1000)


def calculate_betti_for_connectome(ax, readin_W_hypfiles, groundtruth_bettis, groundtruth_integratedbettis, repeat, dd, noise):
    """
    redundant to [betti_analysis] function
    separate for better readability
    """
    assert len(groundtruth_bettis) == len(groundtruth_integratedbettis) == 2
    names = ["row", "col"]

    for iii in range(len(readin_W_hypfiles)):
        readin_files = readin_W_hypfiles[iii]
        select = ""

        data = scipy.io.loadmat(readin_files)
        data = data['distance_matrices']

        if isinstance(data, dict):
            data_keys = data.keys()
            fields = sorted([key for key in data_keys if not key.startswith('__')])
        else:
            fields = sorted(data.dtype.names)

        fields = sorted(fields, key=lambda x: int(x.split('_')[1]))
        print(f"fields: {fields}")

        allerrs = []
        allsynthetic = []
        fake_integrated_bettis = []

        for fieldNameIter in range(len(fields)):
            thisbetti = [[],[],[]]
            fieldName = fields[fieldNameIter]
            print(f"fieldName: {fieldName}")
            if isinstance(data, dict):
                currentMatrix = data[fieldName]
            else:
                currentMatrix = data[fieldName][0, 0]

            for i in range(repeat):
                # print(f'Number: {i+1}')
                squeeze_mat = np.squeeze(currentMatrix[i, :, :])
                [betti_curves,edge_densities] = compute_betti_curves.compute_betti_curves(squeeze_mat)
                for i in range(3):
                    thisbetti[i].append(betti_curves[:,i+1])

                del squeeze_mat
                gc.collect()

            thisbetti = [np.array(subbetti) for subbetti in thisbetti]
            meanbetti = [moving_average(np.mean(subbetti, axis=0), dd) for subbetti in thisbetti]
            fake_integrated_betti = []
            for jjj in range(3):
                consecutive_differences = [edge_densities[i+1] - edge_densities[i] for i in range(len(edge_densities) - 1)]
                integrated_betti = np.sum([a * b for a, b in zip(meanbetti[jjj], consecutive_differences)])
                fake_integrated_betti.append(integrated_betti)
            stdbetti = [moving_average(np.std(subbetti, axis=0),dd) for subbetti in thisbetti]

            fake_integrated_bettis.append(fake_integrated_betti)
            
            oneerr = []
            errbetti = [mean_squared_error(spline_set(meanbetti[i]), spline_set(groundtruth_bettis[iii][i])) for i in range(3)]
            oneerr.append(np.sum(errbetti))
            allerrs.append(oneerr)

            allsynthetic.append([meanbetti, stdbetti, moving_average(edge_densities,dd)])

            del thisbetti, meanbetti, stdbetti 
            gc.collect()

        allerrs = np.array(allerrs)
        fakeallbettis = []
        minerr_index = np.argmin(allerrs)
        synthetic_best = allsynthetic[minerr_index]
        realbetti, fakebetti = groundtruth_integratedbettis[iii], fake_integrated_bettis[iii]
        fakeallbettis.append([realbetti, fakebetti])
        for i in range(3):
            edge_densities = synthetic_best[2]
            ax[iii].plot(edge_densities, synthetic_best[0][i], c=colorset[0][i], linestyle=lines[iii], label=f"{names[iii]} Betti {i+1}")
            ax[iii].fill_between(edge_densities, synthetic_best[0][i]-synthetic_best[1][i], synthetic_best[0][i]+synthetic_best[1][i], color=c_vals_l[i], alpha=0.2)

        np.save(f'./zz_pyclique_results/{names[iii]}_bettis_noise{noise}.npy', np.array(fakeallbettis))

        del allerrs, allsynthetic, fake_integrated_bettis 
        gc.collect()
