from tensorflow.keras import models, regularizers, layers, initializers


def small_vgg16(num_class_=2, image_size_=224, weight_decay_=0.0):
    """
    """
    input_shape = (image_size_, image_size_, 3)
    inputs = layers.Input(shape=input_shape, name='inputs')

    x = layers.Conv2D(16, (3, 3), kernel_initializer=initializers.truncated_normal(stddev=0.1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay_), name='conv1')(inputs)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    x = layers.Conv2D(32, (3, 3), kernel_initializer=initializers.truncated_normal(stddev=0.1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay_), name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    x = layers.Conv2D(64, (3, 3), kernel_initializer=initializers.truncated_normal(stddev=0.1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay_), name='conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    x = layers.Conv2D(128, (3, 3), kernel_initializer=initializers.truncated_normal(stddev=0.1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay_), name='conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    x = layers.Conv2D(128, (4, 4), kernel_initializer=initializers.truncated_normal(stddev=0.1), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(weight_decay_), name='conv5')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(num_class_, activation=None, name='predictions')(x)
    x = layers.Activation(activation='softmax', name='prob')(x)

    model = models.Model(inputs=inputs, outputs=x, name='small_vgg16')

    return model
