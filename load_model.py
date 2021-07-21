import tensorflow as tf

# model_types = 'inception', 'resnet50', 'vgg19'


def load_model(dl_model, model_folder, input_shape=(224, 224, 3),  print_model=True):

    if dl_model == 'inception':
        # https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3

        model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False, weights=f'{model_folder}/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', input_tensor=None,
            input_shape=input_shape, pooling='max')

        preprocess_func = tf.keras.applications.inception_v3.preprocess_input

    elif dl_model == 'resnet50':
        # https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
        # https://github.com/fchollet/deep-learning-models/releases/tag/v0.1

        model = tf.keras.applications.resnet50.ResNet50(
            include_top=False, weights=f'{model_folder}/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_tensor=None,
            input_shape=input_shape, pooling='max')

        preprocess_func = tf.keras.applications.resnet.preprocess_input

    elif dl_model == 'vgg19':
        # https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/VGG19
        # https://github.com/fchollet/deep-learning-models/releases/tag/v0.1

        model = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights=f'{model_folder}/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',  input_shape=input_shape, input_tensor=None, pooling='max')

        preprocess_func = tf.keras.applications.vgg19.preprocess_input

    else:
        raise ValueError('Please specify the neural network model.')

    if print_model:
        print(model.summary())

    return model, preprocess_func
