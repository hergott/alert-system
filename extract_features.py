import numpy as np
import tensorflow as tf
# print(tf.__version__)


def extract_features(image, feature_layer, preprocess_func, print_shape=True):

    img = preprocess_func(image)
    img = np.expand_dims(img, axis=0)

    features_orig = feature_layer.predict(img)
    features = np.ravel(features_orig)

    if print_shape:
        print(
            f'features: {features.shape}, zero values {np.count_nonzero(features==0)}')

    return features
