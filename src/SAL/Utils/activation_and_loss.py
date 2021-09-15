import tensorflow as tf


# Tested
def cluster_separation_cost(label, vc_distances):
    """
    take one input tensor at a time and measure the clustering cost with the vc that belong to the same category
    :param label: int label index of the input sample class label       # shape = [1]
    :param vc_distances: TensorFlow vc tensor                           # shape = [batch_size, height, width, num_vc]
    :return: float clustering and separation cost
    """
    num_vc = tf.shape(vc_distances)[-1]
    num_classes = tf.shape(label)[0]
    vc_per_class = num_vc // num_classes

    # Get starting and ending index of the VC weights
    label = tf.cast(tf.argmax(label), dtype=tf.int32)
    start = label * vc_per_class
    end = (label + 1) * vc_per_class

    cluster_distance = vc_distances[:, :, :, start:end]
    separation_distance = tf.concat([vc_distances[:, :, :, :start], vc_distances[:, :, :, end:]], axis=-1)

    cluster_cost = tf.reduce_min(cluster_distance)
    separation_cost = tf.reduce_min(separation_distance)

    # shape [[float], [float]]
    return [[cluster_cost], [separation_cost]]


# Tested
def diversity_cost(vc_weights, num_classes):
    """
    :param vc_weights: TensorFlow vc tensor                             # shape = [p_h, p_w, depth, num_vc]
    :param num_classes: TensorFlow integer tensor                       # shape = [1]
    https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    """
    num_vc = tf.shape(vc_weights)[-1]
    vc_per_class = num_vc // num_classes

    # shape = [p_h * p_w * depth, num_vc]
    vc_weights_ = tf.reshape(vc_weights, [-1, num_vc])

    def condition(idx_, d_loss):
        return tf.less(idx_, num_vc)

    def body(idx_, d_loss):
        # shape = [vc_per_class, p_h * p_w * depth]
        p = tf.transpose(vc_weights_[:, idx_:idx_+vc_per_class])
        r = tf.reduce_sum(p * p, 1)

        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        # shape = [vc_per_class, vc_per_class]
        c = r - 2 * tf.matmul(p, tf.transpose(p)) + tf.transpose(r)

        # Because of some small values, we get distance values in negative, this just means it equals 0
        c = tf.maximum(c, 0)

        # cost_ = tf.reduce_min(tf.contrib.nn.nth_element(c, n=1))
        # Replace the diagonal value of 0 (distance between the vector and itself) with the max value for each row
        cost_ = tf.reduce_min(tf.linalg.set_diag(c, tf.reduce_max(c, axis=1)))

        d_loss = tf.concat([d_loss, [cost_]], axis=0)

        return [tf.add(idx_, vc_per_class), d_loss]

    _, diversity_distance = tf.while_loop(condition,
                                          body,
                                          loop_vars=[tf.constant(0), tf.constant([0.])],
                                          shape_invariants=[tf.constant(0).get_shape(), tf.TensorShape([None])])

    divers_cost = tf.reduce_mean(diversity_distance[1:])

    return divers_cost


# Tested
def custom_loss(vc_distances, vc_weights, batch_size, num_classes):
    """
    the gathering and membership loss functions that take the input samples in the batch with their labels
    :param vc_distances: TensorFlow vc tensor                                   # shape = [batch_size, height, width, num_vc]
    :param vc_weights: TensorFlow vc tensor                                     # shape = [p_h, p_w, depth, num_vc]
    :param batch_size: TensorFlow constant
    :param num_classes:
    """
    def loss(y_true, y_pred):

        def condition(start_, end_, g_loss, s_loss):
            return tf.less(start_, end_)

        def body(start_, end_, g_loss, s_loss):
            # Gathering Cost
            loss_ = cluster_separation_cost(y_true[start_, :], vc_distances)
            g_loss = tf.concat([g_loss, loss_[0]], axis=0)
            s_loss = tf.concat([s_loss, loss_[1]], axis=0)

            return [tf.add(start_, 1), end_, g_loss, s_loss]

        _, _, gather_, separate_ = tf.while_loop(condition,
                                                 body,
                                                 loop_vars=[tf.constant(0), batch_size, tf.constant([0.]), tf.constant([0.])],
                                                 shape_invariants=[tf.constant(0).get_shape(), batch_size.get_shape(), tf.TensorShape([None]), tf.TensorShape([None])])

        gathering_cost = tf.reduce_mean(gather_[1:])
        separating_cost = - tf.reduce_mean(separate_[1:])

        # Diversity Cost
        divers_cost = diversity_cost(vc_weights, num_classes)

        crossentropy = tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))

        final_loss = 1e-4 + crossentropy + (0.8 * gathering_cost) + (0.2 * divers_cost)  # + (0.2 * separating_cost)

        return final_loss

    # Return a function
    return loss
