import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, random
from scipy import stats
import time
import matplotlib.pyplot as plt


def sparse_pearson_correlation_1_on_1(matrix1, matrix2):
    n = matrix1.shape[1]

    sum1 = matrix1.sum()
    sum2 = matrix2.sum()

    mean1 = sum1 / n
    mean2 = sum2 / n

    product = matrix1.multiply(matrix2)

    covariance = product.sum() - n * mean1 * mean2

    variance1 = mean1 * (1 - mean1) * n
    variance2 = mean2 * (1 - mean2) * n
    std1 = np.sqrt(variance1)
    std2 = np.sqrt(variance2)

    correlation = covariance / (std1 * std2)

    return correlation


def sparse_pearson_correlation_N_on_N(matrix1, matrix2):
    n = matrix1.shape[1]
    m1 = matrix1.shape[0]
    m2 = matrix2.shape[0]

    sum1 = matrix1.sum(axis=1)
    sum2 = matrix2.sum(axis=1)

    mean1 = sum1 / n
    mean2 = sum2 / n

    mean1 = mean1.reshape(-1, 1)
    mean2 = mean2.reshape(1, -1)

    # Compute the covariance matrix
    product = matrix1.dot(matrix2.T)
    covariance = product - n * np.dot(mean1, mean2)

    variance1 = np.multiply(mean1, (1 - mean1)) * n
    variance2 = np.multiply(mean2, (1 - mean2)) * n
    std1 = np.sqrt(variance1)
    std2 = np.sqrt(variance2)

    std1 = np.array(std1).ravel()
    std2 = np.array(std2).ravel()
    std1[std1 == 0] = 1
    std2[std2 == 0] = 1

    std_matrix1 = np.tile(std1, (m2, 1)).T
    std_matrix2 = np.tile(std2, (m1, 1))

    correlation = covariance / (std_matrix1 * std_matrix2)

    return correlation

def test(side1, side2, density):
    print(f"=== Side1: {side1}; Side2: {side2}; Density: {density} ===")

    a = random(side1, side2, density=density, format='csr',
               data_rvs=stats.randint(1, 2).rvs)
    a.data = np.ones_like(a.data)
    b = random(side1, side2, density=density, format='csr',
               data_rvs=stats.randint(1, 2).rvs)
    b.data = np.ones_like(b.data)

    t1 = time.time()

    coeffs1 = sparse_pearson_correlation_N_on_N(a, b)

    t2 = time.time()
    coeffs2 = np.zeros((side1, side1))
    for ind1 in range(side1):
        for ind2 in range(side1):
            corr = np.corrcoef(a[ind1, :].toarray().flatten(),
                               b[ind2, :].toarray().flatten())[0, 1]
            coeffs2[ind1, ind2] = corr

    t3 = time.time()
    print(f"sparse:{t2-t1}")
    print(f"usual:{t3-t2}")

    print(np.allclose(coeffs1, coeffs2))

    return t2-t1, t3-t2


if __name__ == "__main__":
    densitylst = np.logspace(-4, -1, num=20)
    time_record = []
    for density in densitylst:
        sparse_time, np_time = test(side1=100, side2=100000, density=density)
        time_record.append([sparse_time, np_time])
    time_record = np.array(time_record)

    plt.figure()
    plt.plot(densitylst, time_record[:,0], "-o", label="sparse correlation")
    plt.plot(densitylst, time_record[:,1], "-o", label="numpy correlation")
    plt.xlabel("Density")
    plt.ylabel("Time")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("density_test.png")

    side1lst = np.logspace(np.log10(10), np.log10(1000), num=20)
    time_record = []
    for side1 in side1lst:
        sparse_time, np_time = test(side1=int(side1), side2=100000, density=0.01)
        time_record.append([sparse_time, np_time])
    time_record = np.array(time_record)

    plt.figure()
    plt.plot(side1lst, time_record[:,0], "-o", label="sparse correlation")
    plt.plot(side1lst, time_record[:,1], "-o", label="numpy correlation")
    plt.xlabel("Short Side Length")
    plt.ylabel("Time")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("side1_test.png")


