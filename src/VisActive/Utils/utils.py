import os
import numpy as np
import tensorflow as tf


# Tested
def custom_weights_init(weights_shape):
    """
    initiate the weights for the layer after the prototype layer
    :param weights_shape: the weights of the layer after the prototype layer        # shape [num_prototypes, num_classes]
    """
    if weights_shape[0] <= weights_shape[1]:
        print('ERROR: the prototypes layer has more neurones that the output layer!')

    weights = np.full(weights_shape, -0.5)
    prototypes_per_class = weights_shape[0] // weights_shape[1]

    for idx in range(weights_shape[0]):
        if idx // prototypes_per_class < weights_shape[1]:
            idx_one = idx // prototypes_per_class
        else:
            idx_one = weights_shape[1] - 1

        weights[idx, idx_one] = 1

    output = tf.constant_initializer(weights)

    return output


# Tested
def distance_to_similarity(distances, epsilon=0.1):
    """
    Similarity
    :param distances: TensorFlow tensor                             # shape = [batch_size, num_prototypes]
    :param epsilon: float stabilizer to avoid dividing by zero      # float
    :return: float tensor                                           # shape = [batch_size, num_prototypes]
    """
    loss = log10((distances + 1) / (distances + epsilon))
    # loss = 1/(1 + distances)
    return loss


# Tested
def log10(t):
    """
    TensorFlow function to calculate log to the base 10
    """
    numerator = tf.math.log(t)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


# Tested
def get_layer_names(model):
    """
    Return a list with the names of the keras model layers
    """
    layer_names = list()
    for i, layer in enumerate(model.layers):
        layer_names.append(layer.name)
    return layer_names


# Tested
def load_hvc_model(model_path, neural_network, num_classes, image_size, weight_decay, drop_out, num_vc, p_h, p_w, p_d):
    """
    load the custom heretical visual concept model
    """
    # Create the sub_model
    hvc_sub_mode, _ = neural_network(num_classes=num_classes,
                                     image_size=image_size,
                                     weight_decay=weight_decay,
                                     dropout=drop_out,
                                     num_vc=num_vc,
                                     p_h=p_h,
                                     p_w=p_w,
                                     p_d=p_d)

    hvc_sub_mode.load_weights(model_path)

    return hvc_sub_mode


# Tested
def get_ordered_file_names(generator):
    """
    Return the file names in a sorted based on their order when shuffle=True
    """
    current_index = ((generator.batch_index-1) * generator.batch_size)
    if current_index < 0:
        if generator.samples % generator.batch_size > 0:
            current_index = max(0, generator.samples - generator.samples % generator.batch_size)
        else:
            current_index = max(0, generator.samples - generator.batch_size)
    index_array = generator.index_array[current_index:current_index + generator.batch_size].tolist()

    image_file_names = [generator.filepaths[idx] for idx in index_array]

    return image_file_names


# Tested
def closest_patch_to_vc(batch_size, num_vc, distances):
    """
    return the (x, y) index of the closest patch for each vc
    :param batch_size: the batch size
    :param num_vc: the number of vc
    :param distances: distances of the vc with every patch  # shape [batch_size, height, width, num_vc]
    """

    # shape = [batch_size, num_vc, height, width]
    distances_transpose = tf.transpose(distances, perm=[0, 3, 1, 2])

    # Get the index of the closest activation vector to the vc (the index of tf.reduce_min(distances, axis=(2, 3))
    # sine the tf.argmin doesn't take the axis with 2d (e.g. (2, 3)), I had to solve it with the following code
    output_1 = tf.cast(tf.argmin(distances_transpose, axis=-1), dtype=tf.int32)
    output_2 = tf.cast(tf.argmin(tf.reduce_min(distances_transpose, axis=-1), axis=-1), dtype=tf.int32)
    range_vc = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(num_vc), axis=0), [batch_size, 1]), axis=2)
    range_batch = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, num_vc]), axis=2)
    m = tf.concat([range_batch, tf.concat([range_vc, tf.expand_dims(output_2, axis=2)], axis=2)], axis=2)
    o = tf.gather_nd(indices=m, params=output_1)
    indices = tf.concat([tf.expand_dims(output_2, axis=2), tf.expand_dims(o, axis=2)], axis=2)

    # The output hold the (x, y) of the closest convolutional layer output with each vc
    # shape [batch_size, num_vc, 2] (for x and y) the x and y range from 0 to height/width which equals to the height
    # and width of the visual concept kernel, i.e. the height and width of the convolutional layer output.
    return indices


# Tested
def output_labels(generator, iterations):
    """
    """
    labels = list()
    counter = 0
    while counter < iterations:
        gen_next = generator.next()
        # return image batch and 2 sets of labels
        labels.extend([l.tolist() for l in gen_next[1]])
        counter += 1

    return labels


# Tested
def get_file_name_from_path(path):
    head, tail = os.path.split(path)
    return tail
