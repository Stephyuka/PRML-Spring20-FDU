import numpy as np
import time
from scipy import stats
from scipy import linalg
import matplotlib.pyplot as plt
import pandas as pd
import random

def generate_data(n, d, mean, cov, pr, file_path='./'):
    # if train_p is None:
    #     train_p = [0.6, 0.2, 0.2]
    mean, cov, pr = np.asarray(mean), np.asarray(cov), np.asarray(pr)
    data, label = np.zeros((n, d)), np.zeros(n, dtype=int)
    sample_dist = (n * pr[:]).astype(int)
    sample_dist[-1] = n - np.sum( (n * pr[:-1]).astype(int))
    for i, ni in enumerate(sample_dist):
        it, end = np.sum(sample_dist[:i]), np.sum(sample_dist[:i +1])
        label[it:end], data[it:end] = i * np.ones(end - it, dtype=int), np.random.multivariate_normal(mean[i], cov[i], (ni,))

    index = np.arange(n)
    np.random.shuffle(index)
    label, data = label[index,], data[index,]
    # train_n, test_n, valid_n = int(n * train_p[0]), int(n * train_p[1]), int(n * train_p[3])
    #
    df = ""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            df += str(data[i, j]) + ','
        df += str(label[i]) +'\n'
    file = open(file_path + 'sample' + '.dat', 'w')
    file.write(df)
    file.close()
    return label, data

mean = [[12, 5, -1], [0, -1, -5], [-2, 10, 5]]
cov = []
for i in range(3):
    a = np.random.rand(3,3)
    b = np.dot(a, a.transpose())
    b = b.tolist()
    print(b)
    cov.append(b)

# cov  = [[[1, 0, 0], [0, 2, 0], [2, 0, 3]],
#         [[1, 0, 0], [0, 2, 0], [2, 0, 3]],
#         [[1, 0, 0], [0, 2, 0], [2, 0, 3]]]
generate_data(500, 3, mean, cov, [.3333, .3333, 0.3334])
