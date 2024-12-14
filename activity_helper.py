
import time
import random
import gc
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt

import scipy 
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, entropy 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mutual_info_score
from sklearn.cluster import KMeans, DBSCAN

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import sys
sys.path.append("/gscratch/amath/zihan-zhang/spatial/demo/pyclique")
sys.path.append("/gscratch/amath/zihan-zhang/spatial/demo/")
sys.path.append("/gscratch/amath/zihan-zhang/spatial/")
sys.path.append("/gscratch/amath/zihan-zhang/")

import compute_betti_curves
import metric
from netrep.metrics import LinearMetric

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

c_vals = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#dd6b20', '#319795', '#718096', '#d53f8c', '#d69e2e', '#ff6347', '#4682b4', '#32cd32', '#9932cc', '#ffa500']
c_vals_l = ['#feb2b2', '#90cdf4', '#9ae6b4', '#d6bcfa', '#fbd38d', '#81e6d9', '#e2e8f0', '#fbb6ce', '#faf089',]
c_vals_d = ['#9b2c2c', '#2c5282', '#276749', '#553c9a', '#9c4221', '#285e61', '#2d3748', '#97266d', '#975a16',]
colorset = [c_vals_l, c_vals_d]
lines = ["-.", "--"]

def reorder_matrix(matrix, priority_indices):
    """
    Reorders the rows and columns of a square matrix such that the 
    rows and columns with indices in `priority_indices` come first.

    Parameters:
    - matrix (np.ndarray): A square (N x N) NumPy array.
    - priority_indices (list or array-like): List of row/column indices to prioritize.

    Returns:
    - reordered_matrix (np.ndarray): The reordered square matrix.
    - new_order (list): The new ordering of indices.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("The matrix must be a NumPy ndarray.")
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The matrix must be a square (N x N) array.")
    
    N = matrix.shape[0]
    
    # Ensure priority_indices are unique and within the valid range
    priority_indices = list(dict.fromkeys(priority_indices))  # Remove duplicates, preserve order
    if any(idx < 0 or idx >= N for idx in priority_indices):
        raise IndexError("All priority indices must be within the range [0, N-1].")
    
    # Determine the remaining indices
    remaining_indices = [idx for idx in range(N) if idx not in priority_indices]
    
    # Combine to form the new order
    new_order = priority_indices + remaining_indices
    
    # Reorder the matrix
    reordered_matrix = matrix[np.ix_(new_order, new_order)]
    
    return reordered_matrix, new_order

def reverse_binary(array):
    """
    """
    if np.sum(array) > array.size / 2:
        array = 1 - array
    return array

def plot_soma_distribution(soma_data_dfs, output_name):
    """
    given a list of dataframes (subject to different selection criteria), plot the soma distribution for each
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}]
        ]
    )

    all_soma_positions = []

    for sdf, soma_data_d in enumerate(soma_data_dfs):
        scale_factor = 1 / 1000
        good_ct_all_soma_x = soma_data_d["pt_position_x_trafo"].to_numpy() * scale_factor
        good_ct_all_soma_y = soma_data_d["pt_position_y_trafo"].to_numpy() * scale_factor
        good_ct_all_soma_z = soma_data_d["pt_position_z_trafo"].to_numpy() * scale_factor

        soma_position = np.array([good_ct_all_soma_x, good_ct_all_soma_y, good_ct_all_soma_z]).T
        soma_position[:,[1,2]] = soma_position[:,[2,1]]
        all_soma_positions.append(soma_position)

        trace = go.Scatter3d(
            x=soma_position[:, 0],
            y=soma_position[:, 1],
            z=soma_position[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=c_vals[sdf],
                colorscale='Viridis',
                opacity=0.8
            )
        )

        fig.add_trace(trace, row=1, col=sdf+1)

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        scene2=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.write_html(output_name)

    return all_soma_positions

def clustering_by_soma(soma_positions, output_name):
    """
    """
    k = 2  
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(soma_positions)

    fig, ax = plt.subplots(figsize=(4,4))
    for i in range(k):
        cluster_points = soma_positions[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')

    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', 'box')
    fig.legend()
    fig.tight_layout()
    fig.savefig(output_name)

    return labels

def scaling_help(A):
    B = A.max() - A
    B_normalized = (B - B.min()) / (B.max() - B.min())
    return B_normalized

def shepard_fit(x, y, epsilon=1e-6):
    """
    Fits the Shepard diagram data to the equation y = a * (x - x0)^(k+1).
    """
    def model(x, a, k):
        x0 = np.min(x) - epsilon
        return a * (x - x0) ** (k + 1)
    
    initial_guess = [1.0, 1.0]
    
    popt, pcov = curve_fit(model, x, y, p0=initial_guess)
    
    return popt, pcov

def euclidean_connectome(matrix):
    """
    assume N*M shape; the output is technically distance so inversion is needed to transform
    """
    eul_mat = np.zeros((matrix.shape[0], matrix.shape[0]))
    matrix = (matrix >= 1).astype(int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            eul_mat[i,j] = np.linalg.norm(matrix[i] - matrix[j])
    return eul_mat

def variation_of_information(x, y):
    """
    Compute the Variation of Information (VI) between two binary vectors x and y.

    VI(X,Y) = H(X) + H(Y) - 2*I(X;Y)

    Where:
    - H(X) and H(Y) are the entropies of X and Y.
    - I(X;Y) is the mutual information between X and Y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Compute the joint frequencies to derive probabilities
    # For binary data, we have categories {0,1}, but this approach is general.
    contingency = np.zeros((2, 2))
    for xi, yi in zip(x, y):
        contingency[xi, yi] += 1
    
    # Convert counts to probabilities
    joint_prob = contingency / contingency.sum()
    
    # Compute marginals
    pX = joint_prob.sum(axis=1)
    pY = joint_prob.sum(axis=0)
    
    # Entropies
    Hx = entropy(pX, base=2)
    Hy = entropy(pY, base=2)
    
    # Mutual Information using sklearn (it expects discrete labels, so it can use counts internally)
    # Note: mutual_info_score does not directly take probabilities, it uses label frequencies.
    # We can just call it on x and y directly.
    Ixy = mutual_info_score(x, y) / np.log(2)  # mutual_info_score is in nats, convert to bits by dividing by ln(2)
    
    VI = Hx + Hy - 2 * Ixy
    return VI

def variation_of_information_matrix(matrix):
    """
    Compute the Variation of Information (VI) matrix for a binary matrix.
    """
    N = matrix.shape[0]
    return np.array([[variation_of_information(matrix[i], matrix[j]) for j in range(N)] for i in range(N)])

def other_diss_matrix(matrix, metric_name):
    """
    Use different dissimilarity metrics to compute the distance matrix.
    """
    N = matrix.shape[0]

    def czekanowski(u, v):
        """
        Czekanowski distance.
        """
        return np.sum(np.abs(u - v)) / np.sum(u + v)


    def dice(u, v):
        """
        Dice dissimilarity.
        """
        u_v = u - v
        return np.dot(u_v, u_v) / (np.dot(u, u) + np.dot(v, v))

    def jaccard(u, v):
        """
        Jaccard distance.
        """
        uv = np.dot(u, v)
        return 1 - (uv / (np.dot(u, u) + np.dot(v, v) - uv))

    if metric_name == "czekanowski":
        diss_func = czekanowski
    elif metric_name == "dice":
        diss_func = dice
    elif metric_name == "jaccard":
        diss_func = jaccard
    
    return 1 - np.array([[diss_func(matrix[i], matrix[j]) for j in range(N)] for i in range(N)])

def standard_metric(matrix, name):
    """
    """
    if name == "correlation":
        return np.corrcoef(matrix, rowvar=True)
    elif name == "cosine":
        return metric.cosine_distance_matrix(matrix)

def reconstruction(W, activity_extraction_extra_trc, K=None, random=False, permute=False):
    """
    add option to only take top K components in reconstruction
    if K=None, then using all neurons (without any low-pass filter)
    """

    if isinstance(K, str):
        if K[0] == "r":
            br = K.index("_")
            K = int(K[br+1:])
            # print(K)
            random = True

    filtered_matrix = np.zeros_like(W)
    for i in range(W.shape[0]):
        if not random:
            top_k_indices = np.argsort(W[i])[-K:]        
        else:
            top_k_indices = np.random.choice(len(W[i]), size=K, replace=False)

        filtered_matrix[i, top_k_indices] = W[i, top_k_indices]
    
    row_sums_a = np.sum(np.abs(filtered_matrix), axis=1)

    filtered_matrix_normalized = filtered_matrix / row_sums_a[:, np.newaxis]
    

    # # permute the matrix but still keep it symmetric
    # if permute == "rowwise":
    #     filtered_matrix_normalized = permute_symmetric_matrix(filtered_matrix_normalized)
    # elif permute == "cellwise":
    #     filtered_matrix_normalized = permute_symmetric_matrix_cellwise(filtered_matrix_normalized)

    activity_reconstruct_a = filtered_matrix_normalized @ activity_extraction_extra_trc
    return activity_reconstruct_a, filtered_matrix_normalized

def vectorized_pearsonr(x, y):
    """
    compute Pearson correlation coefficient vectorized over the first axis.
    multi-batches version of pearsonr for time efficiency
    """
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    n = x.shape[1]
    cov = np.sum((x - x_mean) * (y - y_mean), axis=1)
    x_std = np.sqrt(np.sum((x - x_mean)**2, axis=1))
    y_std = np.sqrt(np.sum((y - y_mean)**2, axis=1))
    return cov / (x_std * y_std)

def find_intervals(arr):
    """
    Detects all intervals of consecutive 1s and 0s in a NumPy array.
    """
    intervals = {'1': [], '0': []}
    start = 0 

    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:  
            end = i - 1
            intervals[str(arr[start])].append((start, end))  
            start = i  # Start a new interval

    intervals[str(arr[start])].append((start, len(arr) - 1))

    return intervals

def merge_arrays(arrays):
    """
    """
    union_result = arrays[0]    
    for arr in arrays[1:]:
        union_result = np.union1d(union_result, arr)
    return union_result

def float_to_scientific(value, n=4):
    """
    """
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

    smaller_matrix = row_column_delete(matrix, indices_to_remove)

    print(f"After: {smaller_matrix.shape}")

    return smaller_matrix

def cosine_similarity(arr1, arr2):
    """
    """
    dot_product = np.dot(arr1, arr2)
    norm_a = np.linalg.norm(arr1)
    norm_b = np.linalg.norm(arr2)
    return dot_product / (norm_a * norm_b)

def row_column_delete(activity_correlation, indices_to_delete):
    activity_correlation = np.delete(activity_correlation, indices_to_delete, axis=0)
    activity_correlation = np.delete(activity_correlation, indices_to_delete, axis=1)
    return activity_correlation

def sanity_check_W(truncated_W_correlation, activity_correlation):
    """
    """
    mask_inf_nan = np.isinf(truncated_W_correlation) | np.isnan(truncated_W_correlation)
    rows_to_delete = np.all(mask_inf_nan, axis=1)
    cols_to_delete = np.all(mask_inf_nan, axis=0)
    indices_to_delete = np.where(rows_to_delete & cols_to_delete)[0]

    activity_correlation = row_column_delete(activity_correlation, indices_to_delete)
    truncated_W_correlation = row_column_delete(truncated_W_correlation, indices_to_delete)

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
    V = np.column_stack([v.reshape(-1, 1) for v in v_lst])
    U = np.column_stack([u.reshape(-1, 1) for u in u_lst])
    Q1, _ = np.linalg.qr(V)
    Q2, _ = np.linalg.qr(U)
    M = np.dot(Q1.T, Q2)
    _, sigma, _ = np.linalg.svd(M)    
    smallest_angle_degrees = np.degrees(np.arccos(np.max(sigma)))
    
    return smallest_angle_degrees

def reconstruct_matrix(U, S, Vh, num_dimension):
    """
    Reconstruct the matrix using the top `num_dimension` singular values and vectors.
    """
    S_approx = np.diag(S[:num_dimension])  
    U_approx = U[:, :num_dimension]      
    Vh_approx = Vh[:num_dimension, :]   
    return U_approx @ S_approx @ Vh_approx

def angles_between_flats_wrap(W_corr, activity_correlation, angle_consideration=16):
    """
    """
    U_connectome, S_connectome, Vh_connectome = np.linalg.svd(W_corr)
    U_activity, S_activity, Vh_activity = np.linalg.svd(activity_correlation)

    dim_loader, angle_loader = [], []
    lower_approx_connectome, lower_approx_activity = [], []
    for num_dimension in range(1,angle_consideration):
        W_corr_approx = reconstruct_matrix(U_connectome, S_connectome, Vh_connectome, num_dimension)
        activity_correlation_approx = reconstruct_matrix(U_activity, S_activity, Vh_activity, num_dimension)

        lower_approx_connectome.append(W_corr_approx)
        lower_approx_activity.append(activity_correlation_approx)

        U_comps_activity = [U_activity[:,i] for i in range(num_dimension)]
        U_comps_connectome = [U_connectome[:,i] for i in range(num_dimension)]
        angle_in_bewteen = angles_between_flats(U_comps_activity, U_comps_connectome)

        dim_loader.append(num_dimension)
        angle_loader.append(angle_in_bewteen)

    return dim_loader, angle_loader, lower_approx_connectome, lower_approx_activity

def stats_test(array1, array2):
    t_stat, p_value = stats.ttest_ind(array1, array2, equal_var=False)
    p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    return p_value

def stats_test2(array1, array2):
    t_stat, p_value = stats.ttest_rel(array1, array2)
    p_value_one_sided_final = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    return p_value_one_sided_final

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


def betti_analysis(data_lst, inputnames, metadata=None, doconnectome=False):
    """
    originally implemented in microns_activity_analysis.py
    data_lst: [activity_correlation, structure_correlation]
    """
    label = f"S{metadata['session_info']}s{metadata['scan_info']}metric{metadata['metric_name']}"
    print(label)
    Nneuron = data_lst[0].shape[0]
    NneuronWrow = data_lst[3].shape[0]
    NneuronWcol = data_lst[4].shape[0]

    if doconnectome:
        figgood, axsgood = plt.subplots(2,3,figsize=(4*3,4*2))

    datanum = len(data_lst) if doconnectome else 1

    fig, axs = plt.subplots(1,datanum,figsize=(4*datanum,4*1))
    axs = np.atleast_1d(axs)

    dd = 1

    try:
        if doconnectome:
            # if already calculated in the previous run
            data = np.load(f"zz_pyclique_results/whether_{metadata['whethernoise']}_cc_{metadata['whetherconnectome']}_conn{metadata['connectome_name']}.npz", allow_pickle=True)
            densities = data['densities']    
            groundtruth_bettis = data['groundtruth_bettis_save']
            groundtruth_integratedbettis = data['groundtruth_integratedbetti_save']
            save_matrix = data['save_matrix']
            for ind in range(2):
                for i in range(3):
                    axsgood[ind,0].plot(densities[3+ind,i], groundtruth_bettis[3+ind,i], c=c_vals[i], label=f"Betti {i+1}")
                    axsgood[ind,1].plot(densities[3+ind,i], groundtruth_bettis[3+ind,i], c=c_vals[i], label=f"Betti {i+1}")
                    axsgood[ind,1].set_xlim([-0.01,0.21])
                sns.heatmap(save_matrix[ind], ax=axsgood[ind,2], cmap='coolwarm', cbar=True, square=True, center=0)
            print("period saving")

        else:
            raise Exception("Activity Calculation Only!")

    except Exception as e:
        print(e)

        densities, groundtruth_bettis, groundtruth_integratedbettis = [], [], [] # for 3 correlation matrix (3 bettis)
        save_matrix = []

        for index in range(datanum):
            data = data_lst[index]
            print(f"Data Shape: {data.shape}")
            density, groundtruth_betti, groundtruth_integratedbetti = [], [], [] # for 1 correlation matrix (3 bettis)
            
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
                    axsgood[index-3,0].plot(moving_average(edge_densities,dd), curve, c=c_vals[i], label=f"Betti {i+1}")
                    if i == 0: # do it once
                        sns.heatmap(data, ax=axsgood[index-3,2], cmap='coolwarm', cbar=True, square=True, center=0)
                        save_matrix.append(data)
                    
                groundtruth_betti.append(curve)
                density.append(moving_average(edge_densities,dd))

            groundtruth_bettis.append(groundtruth_betti)
            groundtruth_integratedbettis.append(groundtruth_integratedbetti)
            densities.append(density)

        groundtruth_bettis_save = np.array(groundtruth_bettis)
        groundtruth_integratedbetti_save = np.array(groundtruth_integratedbettis)
        densities = np.array(densities)
        if doconnectome:
            np.savez(f"zz_pyclique_results/whether_{metadata['whethernoise']}_cc_{metadata['whetherconnectome']}_conn{metadata['connectome_name']}.npz", \
                        densities=densities, groundtruth_bettis_save=groundtruth_bettis_save, groundtruth_integratedbetti_save=groundtruth_integratedbetti_save, save_matrix=save_matrix)

        figgood.savefig(f"./zz_pyclique_results/whether_{metadata['whethernoise']}_cc_{metadata['whetherconnectome']}_neg{metadata['inhindex']}_conn{metadata['connectome_name']}_backup.png")

    repeat = 100
    dimension = 2
    noise = 0.0625
    minRatio = 0.1
    print(f"Noise: {noise}: minRatio: {minRatio}; whether noise: {metadata['whethernoise']}; inhibitory: {metadata['inhindex']}; connectome type: {metadata['whetherconnectome']}")
    readin_hypfile = f"./zz_pyclique/hyperbolic_dis_n={Nneuron}_repeat={repeat}_dim_{dimension}noise_{noise}minRatio_{minRatio}.mat"
    readin_files_lst = [readin_hypfile]
    names = ["Eul", "Hyp"]
    
    if doconnectome:
        repeat = 10
        NneuronWselect = max(NneuronWrow, NneuronWcol)
        readin_W_hypfiles = [f"./zz_pyclique/hyperbolic_dis_n={NneuronWselect}_repeat={repeat}_dim_{dimension}noise_{noise}minRatio_{minRatio}.mat"]

        calculate_betti_for_connectome(axsgood, readin_W_hypfiles, groundtruth_bettis[3:], groundtruth_integratedbettis[3:], repeat, dd, noise, minRatio, metadata)

        for ax in axsgood.flatten():
            ax.legend()
        figgood.tight_layout()
        figgood.savefig(f"./zz_pyclique_results/gt_connectome_noise{noise}_minRatio_{minRatio}_whether_{metadata['whethernoise']}_cc_{metadata['whetherconnectome']}_neg{metadata['inhindex']}_conn{metadata['connectome_name']}.png")
        print("done")

        sys.exit()

    for iii in range(len(readin_files_lst)):
        readin_files = readin_files_lst[iii]

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
            stdbetti = [moving_average(np.std(subbetti, axis=0),dd) for subbetti in thisbetti]

            all_fake_integrated_betti = []
            for jjj in range(3):
                fake_integrated_betti = []
                for bettiselect in thisbetti[jjj]:
                    consecutive_differences = [edge_densities[i+1] - edge_densities[i] for i in range(len(edge_densities) - 1)]
                    integrated_betti = np.sum([a * b for a, b in zip(bettiselect, consecutive_differences)])
                    fake_integrated_betti.append(integrated_betti)
                all_fake_integrated_betti.append(fake_integrated_betti)
            
            fake_integrated_bettis.append(all_fake_integrated_betti)

            oneerr = []
            for j in range(datanum):
                errbetti = [mean_squared_error(spline_set(meanbetti[i]), spline_set(groundtruth_bettis[j][i])) for i in range(3)]
                oneerr.append(np.sum(errbetti))
            allerrs.append(oneerr)
            print(oneerr)

            allsynthetic.append([meanbetti, stdbetti, moving_average(edge_densities,dd)])

        allerrs = np.array(allerrs)
        fakeallbettis = []
        for index in range(allerrs.shape[1]): # for each correlation matrix
            minerr_index = np.argmin(allerrs[:,index])
            synthetic_best = allsynthetic[minerr_index]
            # axs[index].set_title(f"{inputnames[index]}; {fields[minerr_index]} ")
            realbetti, fakebetti = groundtruth_integratedbettis[index], fake_integrated_bettis[minerr_index]
            fakeallbettis.append([realbetti, fakebetti])
            for i in range(3):
                edge_densities = synthetic_best[2]
                axs[index].plot(edge_densities, synthetic_best[0][i], c=colorset[iii][i], linestyle=lines[iii], label=f"{names[iii]} Betti {i+1}")
                axs[index].fill_between(edge_densities, synthetic_best[0][i]-synthetic_best[1][i], synthetic_best[0][i]+synthetic_best[1][i], color=c_vals_l[i], alpha=0.2)
            axs[index].set_ylim([-2, 40])

    np.savez(f"./zz_pyclique_results/{label}_bettis_noise{noise}.npz", \
        fake_integrated_bettis=fake_integrated_bettis, groundtruth_integratedbetti_save=groundtruth_integratedbetti_save, bestR=fields[minerr_index], size=Nneuron)

    fig.tight_layout()
    fig.savefig(f"./zz_pyclique_results/{label}_noise{noise}.png")
    print(f"Done with {label}")


def calculate_betti_for_connectome(ax, readin_W_hypfiles, groundtruth_bettis, groundtruth_integratedbettis, repeat, dd, noise, minRatio, metadata=None):
    """
    redundant to [betti_analysis] function
    separate for better readability
    """
    assert len(groundtruth_bettis) == len(groundtruth_integratedbettis) == 2
    names = ["row", "col"]

    readin_files = readin_W_hypfiles[0]
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
        stdbetti = [moving_average(np.std(subbetti, axis=0),dd) for subbetti in thisbetti]
        
        all_fake_integrated_betti = []
        for jjj in range(3):
            fake_integrated_betti = []
            for bettiselect in thisbetti[jjj]:
                consecutive_differences = [edge_densities[i+1] - edge_densities[i] for i in range(len(edge_densities) - 1)]
                integrated_betti = np.sum([a * b for a, b in zip(bettiselect, consecutive_differences)])
                fake_integrated_betti.append(integrated_betti)
            all_fake_integrated_betti.append(fake_integrated_betti)
        
        print(all_fake_integrated_betti)

        fake_integrated_bettis.append(all_fake_integrated_betti)
        
        oneerr = []
        for iii in range(2):
            errbetti = [mean_squared_error(spline_set(meanbetti[i][:len(meanbetti[i])//2]), spline_set(groundtruth_bettis[iii][i][:len(groundtruth_bettis[iii][i])//2])) for i in range(3)]
            oneerr.append(np.sum(errbetti))
        allerrs.append(oneerr)
        print(allerrs)

        allsynthetic.append([meanbetti, stdbetti, moving_average(edge_densities,dd)])

        del thisbetti, meanbetti, stdbetti 
        gc.collect()

    allerrs = np.array(allerrs)
    
    for iii in range(2):
        allerr = allerrs[:,iii]
        minerr_index = np.argmin(allerr)
        synthetic_best = allsynthetic[minerr_index]
        realbetti, fakebetti = groundtruth_integratedbettis[iii], fake_integrated_bettis[iii]
        ax[iii,0].set_title(fields[minerr_index])
        for i in range(3):
            for dm in range(2):
                ax[iii,dm].plot(synthetic_best[2], synthetic_best[0][i], c=colorset[0][i], linestyle=lines[iii], label=f"{names[iii]} Betti {i+1}")
                ax[iii,dm].fill_between(synthetic_best[2], synthetic_best[0][i]-synthetic_best[1][i], synthetic_best[0][i]+synthetic_best[1][i], color=c_vals_l[i], alpha=0.2)

    np.savez(f"./zz_pyclique_results/whether_{metadata['whethernoise']}_cc_{metadata['whetherconnectome']}_fitting.npz", \
                fake_integrated_bettis=fake_integrated_bettis, allsynthetic=allsynthetic)

    gc.collect()
