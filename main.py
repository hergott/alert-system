import os
# GPU not necessary for this project (yet).
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'    # nopep8

import tensorflow as tf
import cv2
import numpy as np
from enum import Enum, auto
from collections import namedtuple
import matplotlib.pyplot as plt

from load_model import load_model
from extract_features_collection import extract_features_collection

ImageSize = namedtuple('ImageSize', ['width', 'height'])
vgg_img_size = ImageSize(224, 224)


class image_time(Enum):
    day = auto()
    night = auto()
    day_and_night = auto()


class image_type(Enum):
    chair = auto()
    gate = auto()


def load_image_vgg19(img_path, width, height, grayscale=False):
    if grayscale:
        load_color = cv2.IMREAD_GRAYSCALE
    else:
        load_color = cv2.IMREAD_COLOR

    img = cv2.imread(img_path, load_color)
    img = cv2.resize(img, (width, height))

    if grayscale:
        img = np.expand_dims(img, axis=-1)
        img = np.tile(img, 3)
    else:
        # VGG19 takes BGR as input, but TF preprocessing is RGB-->BGR
        # So we first have to convert images from CV2's BGR to RGB for the
        # TF preprocessing.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_cosine_similarity(c_sim):
    fig = plt.figure()
    fig.patch.set_facecolor('gray')
    fig.patch.set_alpha(0.1)

    ax = fig.add_subplot(111)
    ax.patch.set_facecolor('gray')
    ax.patch.set_alpha(0.15)
    ax.set_axisbelow(True)

    plt.scatter(range(len(c_sim)), c_sim, c=c_sim,
                s=300, cmap='viridis', alpha=1.0, edgecolors='darkslategrey')
    plt.grid(color='k', linestyle='-', linewidth=.3, alpha=0.7)
    plt.colorbar()

    plt.show()


def main(feature_layer_numbers=[-2], img_time=image_time.day_and_night, img_type=image_type.chair, plot_cos_sim=False):
    image_folder = 'input-images'

    output_folder = 'output'
    model_folder = 'pretrained-models'

    height = vgg_img_size.height

    if img_type == image_type.chair:
        image_subfolder = 'chair'

        width = vgg_img_size.width

        day_images = ["0_d", "1_d", "2_d", "3_d", "4_d", "5_d", "6_d"]
        night_images = ["0_n", "1_n", "2_n", "3_n", "4_n", "5_n", "6_n"]
        ref_images = ["ref_1", "ref_2"]
    else:
        image_subfolder = 'gate/horizontal'

        # pictures are landscape since they only show the top of the gate
        width = vgg_img_size.width*2

        day_images = ["0_d", "1_d", "2_d", "3_d",
                      "4_d", "5_d", "6_d", "7_d", "8_d"]
        night_images = ["0_n", "1_n", "2_n", "3_n",
                        "4_n", "5_n", "6_n", "7_n", "8_n"]
        ref_images = ["ref_1", "ref_2"]

    if img_time == image_time.day:
        image_names = day_images + ref_images
    elif img_time == image_time.night:
        image_names = night_images + ref_images
    else:
        image_names = day_images + night_images + ref_images

    images = []
    for name in image_names:
        img_path = f'{image_folder}/{image_subfolder}/{name}.jpg'

        images.append(load_image_vgg19(img_path, width, height))

    model, preprocess_func = load_model(
        'vgg19', model_folder=model_folder, input_shape=(height, width, 3),  print_model=True)

    results_collection = []

    for f in feature_layer_numbers:

        feature_layer = tf.keras.Model(model.inputs, model.layers[f].output)

        results, _ = extract_features_collection(
            images, feature_layer, preprocess_func, print_shape=True, output_folder=f'{output_folder}/{image_subfolder}')

        print(f'VGG-19 (no top) feature layer: {f}')
        print('cosine similarity:')
        c_sim = results['c_sim'][:, 0].tolist()
        for cs in c_sim:
            print(cs)

        if plot_cos_sim:
            plot_cosine_similarity(c_sim)

        results_collection.append(results)

    return results_collection


if __name__ == '__main__':
    results_collection = main(
        img_time=image_time.day_and_night, img_type=image_type.gate, plot_cos_sim=True)
