import numpy as np 
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import time


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

def angles_between_flats_wrap(W_corr, activity_correlation, angle_consideration=20):
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