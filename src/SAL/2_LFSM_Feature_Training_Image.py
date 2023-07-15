import importlib
import numpy as np
import pandas as pd
import tensorflow as tf

import src.SAL.settings as settings
import src.SAL.Utils.data_process as data_process


def main():
    """
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Load Data and Model

    train_dataset = data_process.get_dataset(settings.training_path)

    network = importlib.import_module(settings.model_def, 'inference')

    # ------------------------------------------------------------------------------------------------------------------
    # Load Graph

    with tf.Graph().as_default():
        tf.compat.v1.set_random_seed(settings.seed)

        # Placeholder for input images
        images_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, settings.image_size, settings.image_size, 3), name='input')

        # Build the inference graph
        pre_embeddings = network.inference(images_placeholder, weight_decay=settings.weight_decay)

        # Split example embeddings into anchor, positive and negative and calculate triplet loss
        tf.identity(pre_embeddings, 'pre_embeddings')
        embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='embeddings')

        x_variable = tf.compat.v1.get_variable(name='matrix',
                                               shape=[128, 128],
                                               initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
                                               regularizer=tf.keras.regularizers.L2(settings.weight_decay))
        x_upper = tf.compat.v1.matrix_band_part(x_variable, 0, -1)
        similar_matrix = (x_upper + tf.transpose(x_upper))

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=10)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        with sess.as_default():

            saver.restore(sess, tf.train.latest_checkpoint(settings.models_base_dir))

            # write the similarity matrix to the file
            print('Loading training data')
            step = 0
            total_img = np.size(train_dataset, 0)
            print('calculate the accuracy on validation set')
            csv_train_feature_sm = open(settings.similarity_matrix, 'a+')
            matrix = sess.run([similar_matrix])
            print(matrix[0].shape)
            df = pd.DataFrame(matrix[0])
            df.to_csv(csv_train_feature_sm, header=False)
            csv_train_feature_sm.close()

            # write the training file names and the training features to the files
            csv_train_name = open(settings.train_name, 'a+')
            csv_train_feature = open(settings.train_feature, 'a+')

            while step < total_img:
                valid_x = data_process.load_data_v2(train_dataset, step, settings.image_size)
                feature_vector = sess.run([embeddings], feed_dict={images_placeholder: valid_x})
                if step % 5000 == 0:
                    # print(f'File name: {train_dataset[step].file_name}')
                    csv_train_name.close()
                    csv_train_feature.close()
                    csv_train_name = open(settings.train_name, 'a+')
                    csv_train_feature = open(settings.train_feature, 'a+')

                feature_vector = np.reshape(feature_vector, (1, 128))
                df = pd.DataFrame(feature_vector)

                csv_train_name.write(train_dataset[step].file_name + '\n')
                df.to_csv(csv_train_feature, header=False)
                step += 1

            csv_train_feature.close()
            csv_train_name.close()


if __name__ == '__main__':
    main()
