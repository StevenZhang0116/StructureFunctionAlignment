import os
import numpy as np
from scipy.linalg import sqrtm


def praveen_metric(C,D):
    top = np.trace(sqrtm(C) * sqrtm(D))
    bottom = np.sqrt(np.trace(C)) * np.sqrt(np.trace(D))
    return top / bottom

file_lst = [file for file in os.listdir("./for_metric/") if file.endswith('.npz')]
result = []

for file in file_lst:
    data = np.load(f"./for_metric/{file}")
    W_in, W_out, A = data['W_in'], data['W_out'], data['A']
    np.fill_diagonal(W_in,1)
    np.fill_diagonal(A,1)
    ans_in = praveen_metric(W_in, A)
    ans_out = praveen_metric(W_out, A)
    result.append([ans_in, ans_out])
