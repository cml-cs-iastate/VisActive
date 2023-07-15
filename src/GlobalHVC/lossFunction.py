import tensorflow as tf

from tensorflow.keras import backend as K


# Tested
def cost(label, vc_distances):
    """
    """
    num_vc = tf.shape(vc_distances)[-1]
    num_classes = tf.shape(label)[0]  # e.g., label = [0., 0., 1.]
    vc_per_class = num_vc // num_classes

    # Get starting and ending index of the VC weights
    label = tf.cast(tf.argmax(label), dtype=tf.int32)
    start = label * vc_per_class
    end = (label + 1) * vc_per_class

    c_distance = vc_distances[:, :, start:end]
    s_distance = tf.concat([vc_distances[:, :, :start], vc_distances[:, :, end:]], axis=-1)

    c_cost = tf.reduce_min(c_distance)
    s_cost = tf.reduce_min(s_distance)

    # shape [[float], [float]]
    return [[c_cost], [s_cost]]


# Tested
def custom_loss(vc_distances, vc_weights, num_classes, feature_maps=None, triplet_loss_parameter=None):
    """
    :param vc_distances: TensorFlow vc tensor               # shape = [batch_size, height, width, num_vc]
    :param vc_weights: TensorFlow vc tensor                 # shape = [p_h, p_w, depth, num_vc]
    :param num_classes: TensorFlow constant
    :param feature_maps: feature_map                        # shape = [batch_size, 4, 4, 128]
    :param triplet_loss_parameter:
    """
    def loss(y_true, y_pred):
        batch_size = tf.shape(y_true)[0]

        def condition(start_, end_, g_loss, s_loss):
            return tf.less(start_, end_)

        def body(start_, end_, g_loss, s_loss):
            # Gathering Cost
            loss_ = cost(y_true[start_, :], vc_distances[start_])
            g_loss = tf.concat([g_loss, loss_[0]], axis=0)
            s_loss = tf.concat([s_loss, loss_[1]], axis=0)

            return [tf.add(start_, 1), end_, g_loss, s_loss]

        _, _, gather_, separate_ = tf.while_loop(condition,
                                                 body,
                                                 loop_vars=[tf.constant(0), batch_size, tf.constant([0.]), tf.constant([0.])],
                                                 shape_invariants=[tf.constant(0).get_shape(), batch_size.get_shape(), tf.TensorShape([None]), tf.TensorShape([None])])

        gather_ = gather_[1:]
        separate_ = separate_[1:]

        gathering_cost = tf.reduce_mean(gather_)
        separating_cost = tf.reduce_mean(separate_)

        crossentropy = K.mean(K.categorical_crossentropy(y_true, y_pred))

        final_loss = crossentropy + (0.8 * gathering_cost) - (0.1 * separating_cost)  # + (0.05 * divers_cost)

        return final_loss

    return loss
