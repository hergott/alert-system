This project uses pretrained convolutional neural networks with "no top," meaning the final fully connected layers aren't included.

Here are two pages where you can download the TensorFlow weight file of your choice:

  [https://github.com/fchollet/deep-learning-models/releases/tag/v0.1](https://github.com/fchollet/deep-learning-models/releases/tag/v0.1)
  [https://github.com/fchollet/deep-learning-models/releases/tag/v0.2](https://github.com/fchollet/deep-learning-models/releases/tag/v0.2)

This project by default uses VGG-19:

  [vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5)

The "load_model.py" function can also use the ResNet50 or Inception-v3 models:

  [resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5)
  [inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5)
