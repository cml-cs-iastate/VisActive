import tensorflow as tf


def vgg_small(inputs, weight_decay=0.0):
    """
    """

    initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.1)

    net = tf.compat.v1.layers.conv2d(inputs, 16, [3, 3], activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L2(weight_decay), name='conv1')
    net = tf.compat.v1.layers.max_pooling2d(net, [2, 2], [2, 2], name='pool1')  # 64->32
    net = tf.compat.v1.layers.conv2d(net, 32, [3, 3], activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L2(weight_decay), name='conv2')
    net = tf.compat.v1.layers.max_pooling2d(net, [2, 2], [2, 2], name='pool2')  # 32->16
    net = tf.compat.v1.layers.conv2d(net, 64, [3, 3], activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L2(weight_decay), name='conv3')
    # TODO: remove this conv and pooling layer if dataset is CHIFAR-10
    # net = tf.compat.v1.layers.max_pooling2d(net, [2, 2], [2, 2], name='pool3')  # 16->8
    # net = tf.compat.v1.layers.conv2d(net, 128, [3, 3], activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L2(weight_decay), name='conv4')
    net = tf.compat.v1.layers.max_pooling2d(net, [2, 2], [2, 2], name='pool4')  # 8->4
    net = tf.compat.v1.layers.conv2d(net, 128, [4, 4], activation='relu', padding='valid', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L2(weight_decay), name='conv5')
    net = tf.compat.v1.layers.flatten(net)  # shape (None, 128)

    return net


def inference(images, weight_decay=0.0):
    return vgg_small(images, weight_decay)

#
# def resnet_block(inputs, channels, down_sample=False):
#     strides = [2, 1] if down_sample else [1, 1]
#     kernel_size = [3, 3]
#     init_scheme = tf.compat.v1.initializers.he_normal()
#
#     res = inputs
#
#     x = tf.compat.v1.layers.conv2d(inputs, channels, strides=strides[0], kernel_size=kernel_size, padding="same", kernel_initializer=init_scheme)
#     x = tf.compat.v1.layers.batch_normalization(x)
#     x = tf.compat.v1.nn.relu(x)
#     x = tf.compat.v1.layers.conv2d(x, channels, strides=strides[1], kernel_size=kernel_size, padding="same", kernel_initializer=init_scheme)
#     x = tf.compat.v1.layers.batch_normalization(x)
#
#     if down_sample:
#         res = tf.compat.v1.layers.conv2d(x, channels, strides=2, kernel_size=[1, 1], kernel_initializer=init_scheme, padding="same")
#         res = tf.compat.v1.layers.batch_normalization(res)
#
#     # if not perform down sample, then add a shortcut directly
#     x = tf.compat.v1.add(x, res)
#     x = tf.compat.v1.nn.relu(x)
#
#     return x
#
#
# def vgg_small(inputs,  weight_decay=0.0):
#     """
#     """
#
#     x = tf.compat.v1.layers.conv2d(inputs, 64, [7, 7], strides=2, padding="same", kernel_initializer=tf.compat.v1.initializers.he_normal())
#     x = tf.compat.v1.layers.batch_normalization(x)
#     x = tf.compat.v1.nn.relu(x)
#     x = tf.compat.v1.layers.max_pooling2d(x, pool_size=[2, 2], strides=2, padding="same")
#
#     res_1_1 = resnet_block(x, 64)                               # res_1_1
#     res_1_2 = resnet_block(res_1_1, 64)                         # res_1_2
#     res_2_1 = resnet_block(res_1_2, 128, down_sample=True)      # res_2_1
#     res_2_2 = resnet_block(res_2_1, 128)                        # res_2_2
#     res_3_1 = resnet_block(res_2_2, 256, down_sample=True)      # res_3_1
#     res_3_2 = resnet_block(res_3_1, 256)                        # res_3_2
#     res_4_1 = resnet_block(res_3_2, 512, down_sample=True)      # res_4_1
#     res_4_2 = resnet_block(res_4_1, 512)                        # res_4_2
#
#     x = tf.compat.v1.layers.average_pooling2d(res_4_2)
#     x = tf.compat.v1.layers.flatten(x)
#
#     return x
#
#
# def inference(images, weight_decay=0.0):
#     return vgg_small(images, weight_decay)
