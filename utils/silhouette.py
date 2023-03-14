import numpy as np
from numpy.linalg import norm



def silhouette_coefficient(x, label):
    # turn label into restrctly continuous (begin from 0)
    x = np.asarray(x)
    unique, label = np.unique(label, return_inverse=True)
    if unique.size == 1:
        raise ValueError('unique label should be > 2')
    assert x.shape[0] == label.size

    # distance between every two sample
    dist_matrix = norm(x[:, None, :] - x[None], axis=2)
    
    # distance between sample and cluster
    dist_matrix = np.asarray([np.ma.masked_equal(dist_matrix[:, label == i], 0).mean(1)
                              for i in range(len(unique))]).T

    # index of intra-distance
    idx = np.eye(len(unique), dtype=bool)[label]

    # result
    intra = dist_matrix[idx]
    inter = dist_matrix[~idx].reshape(label.size, -1)
    if inter.ndim > 1:
        inter = inter.min(1)
    sil = (inter - intra) / (np.maximum(intra, inter) + 1e-16)  # lest devided by 0
    sil[intra == 0] = 0  # when intra = 0, silhouette score should be 0

    return sil.mean(), sil
