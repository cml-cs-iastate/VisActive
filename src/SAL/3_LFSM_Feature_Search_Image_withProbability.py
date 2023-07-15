import importlib
import numpy as np
import pandas as pd
import tensorflow as tf

import src.SAL.settings as settings
import src.SAL.Utils.facenet_sim_v2 as facenet
import src.SAL.Utils.data_process as data_process


def main():
    """
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Load Data and Model

    pretrained_model = '.\\CNN_model_training\\getFeatureVector\\caltech-256\\AIFT\\iter5_classifier\\20180514-155544'

    train_dataset = facenet.get_dataset(settings.training_path)

    network = importlib.import_module(settings.model_def, 'inference')

    # ------------------------------------------------------------------------------------------------------------------
    # Load Graph

    with tf.Graph().as_default():
        tf.compat.v1.set_random_seed(settings.seed)

        # Placeholder for input images
        images_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, settings.image_size, settings.image_size, 3), name='input')

        # Build the inference graph
        pre_embeddings = network.inference(images_placeholder, weight_decay=settings.weight_decay)
        tf.identity(pre_embeddings, name="pre_embeddings")

        output = tf.compat.v1.layers.dense(pre_embeddings, settings.num_class, activation_fn=None, reuse=False)
        prob = tf.nn.softmax(output, name='prob')

        # Create a saver
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=300)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        with sess.as_default():

            saver.restore(sess, tf.train.latest_checkpoint(settings.pretrained_model_dir))
            print(tf.train.latest_checkpoint(pretrained_model))

            # Keep training until reach max iterations
            print('Loading training data')
            step = 0
            total_img = np.size(train_dataset, 0)
            print('calculate the accuracy on validation set')
            csv_search_name = open(settings.search_name, 'a+')  # '.\\classifier\\search_name_caltech-256.csv'
            cvs_search_feature = open(settings.search_feature, 'a+')  # '.\\classifier\\search_prob_caltech-256.csv'
            shuffle = np.arange(total_img)
            np.random.shuffle(shuffle)

            while step < total_img:
                valid_x = data_process.load_data_v2(train_dataset, shuffle[step], settings.image_size)
                probability = sess.run([prob], feed_dict={images_placeholder: valid_x})
                probability = np.reshape(probability, (1, settings.num_class))

                if step % 500 == 0:
                    print(train_dataset[shuffle[step]].filename)
                    csv_search_name.close()
                    cvs_search_feature.close()
                    csv_search_name = open(settings.search_name, 'a+')
                    cvs_search_feature = open(settings.search_feature, 'a+')

                df = pd.DataFrame(probability)
                df.to_csv(cvs_search_feature, header=False)
                csv_search_name.write(train_dataset[shuffle[step]].filename + '\n')
                step += 1

            cvs_search_feature.close()
            csv_search_name.close()

    return step


if __name__ == '__main__':
    main()
