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
    net = tf.compat.v1.layers.max_pooling2d(net, [2, 2], [2, 2], name='pool3')  # 16->8
    net = tf.compat.v1.layers.conv2d(net, 128, [3, 3], activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L2(weight_decay), name='conv4')
    net = tf.compat.v1.layers.max_pooling2d(net, [2, 2], [2, 2], name='pool4')  # 8->4
    net = tf.compat.v1.layers.conv2d(net, 128, [4, 4], activation='relu', padding='valid', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.L2(weight_decay), name='conv5')
    net = tf.compat.v1.layers.flatten(net)  # shape (None, 128)

    return net


def inference(images, weight_decay=0.0):
    return vgg_small(images, weight_decay)
