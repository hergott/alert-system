import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.decomposition import PCA
from math import sqrt


def calc_cosine_similarity(data, folder=None):

    pca_1 = PCA(n_components=1)
    pca_1_transform = pca_1.fit_transform(data)

    pca_2 = PCA(n_components=2)
    pca_2_transform = pca_2.fit_transform(data)

    c = data.shape[0]

    c_sim = np.zeros((c, c))
    c_sim_mjh = np.zeros((c, c))

    c_dist = np.zeros((c, c))
    c_sim_scipy = np.zeros((c, c))

    euc = np.zeros((c, c))

    for i in range(c):
        for j in range(c):

            dot = np.dot(data[i], data[j])
            norm_i = np.linalg.norm(data[i])
            norm_j = np.linalg.norm(data[j])
            c_sim_mjh[i, j] = dot / (norm_i*norm_j)

            data_i = np.expand_dims(data[i], 0)
            data_j = np.expand_dims(data[j], 0)

            c_sim[i, j] = cosine_similarity(data_i, data_j)

            c_dist[i, j] = cosine_distance(data_i, data_j)
            c_sim_scipy[i, j] = 1. - c_dist[i, j]

            dist = data_i - data_j
            sqr = np.square(dist)
            sum_sqr = np.sum(sqr)
            euc[i, j] = sqrt(sum_sqr)

    if folder is not None:
        np.savetxt(f'{folder}/pca_transform_1.csv',
                   pca_1_transform, delimiter=", ")
        np.savetxt(f'{folder}/pca_transform_2.csv',
                   pca_2_transform, delimiter=", ")
        np.savetxt(f'{folder}/c_sim.csv', c_sim, delimiter=", ")
        np.savetxt(f'{folder}/c_sim_mjh.csv', c_sim_mjh, delimiter=", ")
        np.savetxt(f'{folder}/c_dist.csv', c_dist, delimiter=", ")
        np.savetxt(f'{folder}/c_sim_scipy.csv',
                   c_sim_scipy, delimiter=", ")
        np.savetxt(f'{folder}/euclidian.csv', euc, delimiter=", ")

    out = {'c_sim': c_sim, 'c_sim_mjh': c_sim_mjh, 'c_dist': c_dist, 'c_sim_scipy': c_sim_scipy,
           'euclidian': euc, 'pca_transform_1': pca_1_transform, 'pca_transform_2': pca_2_transform}

    return out
