import scipy 
import numpy as np
import matplotlib.pyplot as plt

path = "mds-results-all/"
name = "stress_noise_normal_cc_count_ss_all.mat"
data = scipy.io.loadmat(path + name)

print(data)