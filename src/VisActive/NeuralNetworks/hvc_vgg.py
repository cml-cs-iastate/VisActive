import tensorflow as tf
from tensorflow.keras import models, regularizers, layers, initializers

from src.VisActive.Utils.utils import custom_weights_init
from src.VisActive.VCL.vcl import VCL


def hvc_vgg(num_classes, image_size, dropout=0.0, weight_decay=0.0, num_vc=0, p_h=1, p_w=1, p_d=128):
    """
    """

    input_shape = (image_size, image_size, 3)
    inputs = layers.Input(shape=input_shape, name='inputs')

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1', kernel_initializer=initializers.truncated_normal())(inputs)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)  # 64 --> 32
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)  # 32 --> 16
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)  # 16 --> 8
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)  # 8 --> 4
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5')(x)
    x = layers.AveragePooling2D((2, 2), strides=(1, 1), name='pool5')(x)  # 8 --> 4

    if num_vc > 0:
        # x = layers.Conv2D(p_d, (1, 1), activation='relu', padding='same', name='additional_layer_1')(x)
        x = layers.Conv2D(p_d, (1, 1), activation='sigmoid', padding='same', name='additional_layer_2')(x)

        vcl = VCL(num_vcl=num_vc, p_h=p_h, p_w=p_w)(x)

        output = layers.Dense(num_classes,
                              activation='softmax',
                              name='predictions',
                              kernel_initializer=custom_weights_init(weights_shape=[vcl[1].shape[-1], num_classes]),
                              kernel_regularizer=regularizers.l2(),
                              trainable=False,
                              use_bias=False)(vcl[0])

        model = models.Model(inputs=inputs, outputs=output, name='hvc_vgg16')

        return model, vcl

    else:
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dropout(dropout)(x)
        output = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

        model = models.Model(inputs=inputs, outputs=output, name='small_vgg16')

        return model
