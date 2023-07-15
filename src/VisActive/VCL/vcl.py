import tensorflow as tf

from tensorflow.keras import layers, backend as K, initializers

from src.VisActive.Utils.utils import distance_to_similarity, closest_patch_to_vc


class VCL(layers.Layer):
    def __init__(self, num_vcl=10, p_h=1, p_w=1, **kwargs):
        """
        :param num_vcl: the number of the VC for the VCL = num_classes * vc_per_class
        """
        super(VCL, self).__init__(**kwargs)
        self.num_vc = num_vcl
        self.p_h = p_h
        self.p_w = p_w

    def build(self, input_shape):
        """
        Create a trainable weight variable for this layer.
        :param input_shape: the shape of the input tensor [batch_size, height, width, depth]
        """
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.p_h, self.p_w, int(input_shape[-1]), self.num_vc),
                                      initializer=initializers.RandomUniform(minval=0., maxval=1.),
                                      trainable=True)

        self.ones = K.ones_like(self.kernel)

        return super(VCL, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        """
        Output always positive (distance --> similarity)
        :param inputs: input tensor [batch_size, height, width, depth]
        """
        batch_size = K.shape(inputs)[0]
        height = K.shape(inputs)[1]
        image_height = 32  # TODO: hardcoded

        # shape [batch_size, height, width, depth]
        inputs_square = inputs ** 2
        # shape [batch_size, height, width, num_vc]
        # inputs_patch_sum = layers.Conv2D(self.num_vc, (self.p_h, self.p_w), activation=None, kernel_initializer=initializers.Ones(), trainable=False, padding='same', name='patch_sum')(inputs_square)
        inputs_patch_sum = K.conv2d(inputs_square, kernel=self.ones, padding='same')

        # shape [p_h, p_w, depth, num_vc]
        p2 = self.kernel ** 2
        # shape [batch_size, p_h, p_w, num_vc]
        p2 = K.tile(K.expand_dims(K.sum(p2, axis=2), axis=0), [batch_size, 1, 1, 1])

        # shape [batch_size, 1, 1, num_vc]
        if self.p_h != 1 and self.p_w != 1:
            p2 = K.expand_dims(K.expand_dims(K.sum(p2, axis=[1, 2]), axis=1), axis=1)

        # shape [batch_size, height, width, num_vc]
        xp = K.conv2d(inputs, kernel=self.kernel, padding='same')

        # shape [batch_size, height, width, num_vc]
        intermediate_result = - 2 * xp + p2  # use broadcast

        # get the distances for each visual concept with convolutional patch output
        # shape [batch_size, height, width, num_vc]
        self.distances = K.relu(inputs_patch_sum + intermediate_result)

        # get most similar visual concept to the patch of the convolutional layer. This one should has the minimum distance.
        # global min pooling
        # shape [batch_size, 1, 1, num_vc]
        min_distances = -K.pool2d(-self.distances, pool_size=(self.distances.shape[1], self.distances.shape[2]))

        # shape [batch_size, num_vc]
        min_distances = K.reshape(min_distances, [-1, self.num_vc])
        # shape [batch_size, num_vc]
        # The activations are the similarity values for the set of visual concepts for each image in the batch
        vcl_activations = distance_to_similarity(min_distances)

        # shape [p_h, p_w, depth, num_vc]
        # the vc_weight is the actual visual concept representation that we are learning
        self.vc_weight = tf.tile(K.expand_dims(self.kernel, axis=0), [batch_size, 1, 1, 1, 1])

        # Get the coordinates of the patches in the scale of the image (x, y)
        # shape [batch_size, num_vc, 2] (for x1 and y1)  e.g. [6, 6]
        x_y = closest_patch_to_vc(batch_size, self.num_vc, self.distances)

        # [batch_size, num_vc, 4] (for x1, y1, x2, and y2)
        # the tf.minimum helps not to not exceed the limit of the image boundary
        # until this point, the coordinates are in the size of the visual concept kernel, e.g. [6, 6, 7, 7]
        self.coordinates = tf.minimum(tf.concat([x_y, x_y + self.p_h], axis=2), height)  # height and width are equal

        # up-sample the coordinates to the image size. Here we scale the size of the x and y to the size of the image.
        self.coordinates = tf.cast(self.coordinates * (image_height // height), dtype=tf.int32)

        return [vcl_activations, self.vc_weight, inputs, self.distances, self.coordinates]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.num_vc), self.vc_weight.shape, input_shape, self.distances.shape, self.coordinates.shape]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_vc': self.num_vc,
            'p_h': self.p_h,
            'p_w': self.p_w,
        })
        return config
