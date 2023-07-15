import tensorflow as tf
from tensorflow.keras import models, layers


def resnet_block(inputs, channels, down_sample=False):
    strides = [2, 1] if down_sample else [1, 1]
    kernel_size = (3, 3)
    init_scheme = "he_normal"

    res = inputs

    x = layers.Conv2D(channels, strides=strides[0], kernel_size=kernel_size, padding="same", kernel_initializer=init_scheme)(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = layers.Conv2D(channels, strides=strides[1], kernel_size=kernel_size, padding="same", kernel_initializer=init_scheme)(x)
    x = layers.BatchNormalization()(x)

    if down_sample:
        res = layers.Conv2D(channels, strides=2, kernel_size=(1, 1), kernel_initializer=init_scheme, padding="same")(res)
        res = layers.BatchNormalization()(res)

    # if not perform down sample, then add a shortcut directly
    x = layers.Add()([x, res])
    out = tf.nn.relu(x)

    return out


def res_net_18(num_classes, image_size, dropout=0.0):
    """
    """

    input_shape = (image_size, image_size, 3)
    inputs = layers.Input(shape=input_shape, name='inputs')

    x = layers.Conv2D(64, (7, 7), strides=2, padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)
    res_1_1 = resnet_block(x, 64)                               # res_1_1
    res_1_2 = resnet_block(res_1_1, 64)                         # res_1_2
    res_2_1 = resnet_block(res_1_2, 128, down_sample=True)      # res_2_1
    res_2_2 = resnet_block(res_2_1, 128)                        # res_2_2
    res_3_1 = resnet_block(res_2_2, 256, down_sample=True)      # res_3_1
    res_3_2 = resnet_block(res_3_1, 256)                        # res_3_2
    res_4_1 = resnet_block(res_3_2, 512, down_sample=True)      # res_4_1
    res_4_2 = resnet_block(res_4_1, 512)                        # res_4_2

    x = layers.GlobalAveragePooling2D()(res_4_2)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model_ = models.Model(inputs=inputs, outputs=output, name='ResNet18')

    return model_
