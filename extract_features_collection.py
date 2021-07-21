import numpy as np
import pickle

from extract_features import extract_features
from calc_cosine_similarity import calc_cosine_similarity


def extract_features_collection(images, feature_layer, preprocess_func, output_folder=None, print_shape=True):
    out = []

    for img in images:
        features = extract_features(
            img, feature_layer, preprocess_func, print_shape=print_shape)
        out.append(features)

    features_mat = np.zeros((len(images), out[0].size))

    for i, o in enumerate(out):
        features_mat[i, :] = o[:]

    results = calc_cosine_similarity(features_mat, folder=output_folder)

    if output_folder is not None:
        # transpose output matrix because horizontal too large for Excel grid.
        np.savetxt(f'{output_folder}/all.csv',
                   features_mat.T, delimiter=',', fmt='%1.4f')

        pickle.dump(features_mat, open(
            f'{output_folder}/all.p', 'wb'))

    return results, features_mat
