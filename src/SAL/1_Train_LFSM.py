import os
import time
import importlib
import numpy as np
import tensorflow as tf
from datetime import datetime

import src.SAL.settings as settings
import src.SAL.Utils.facenet_sim_v2 as facenet
from src.SAL.Utils.utils import save_variables_and_metagraph


def main():
    """
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Load Data and Model

    sub_dir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    train_dataset = facenet.get_dataset(settings.training_path)

    network = importlib.import_module(settings.model_def, 'inference')

    text_file_name = os.path.join(settings.models_base_dir, 'loss.txt')

    # ------------------------------------------------------------------------------------------------------------------
    # Load Graph

    with tf.Graph().as_default():
        tf.compat.v1.set_random_seed(settings.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for input images
        images_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, settings.image_size, settings.image_size, 3), name='input')

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

        # Build the inference graph
        pre_embeddings = network.inference(images_placeholder, weight_decay=settings.weight_decay)

        # Split example embeddings into anchor, positive and negative and calculate triplet loss
        tf.identity(pre_embeddings, 'pre_embeddings')
        embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='embeddings')

        anchor, positive, negative = tf.split(embeddings, 3, 0)

        x_variable = tf.compat.v1.get_variable(name='matrix',
                                               shape=[128, 128],
                                               initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
                                               regularizer=tf.keras.regularizers.L2(settings.weight_decay))
        x_upper = tf.linalg.band_part(x_variable, 0, -1)
        similar_matrix = (x_upper + tf.transpose(x_upper))

        triplet_loss = facenet.triplet_loss_similarity_learning(anchor, positive, negative, settings.alpha, similar_matrix)

        # Calculate the total losses
        regularization_losses = tf.add_n([tf.compat.v1.losses.get_regularization_loss()])

        total_loss = tf.add_n([triplet_loss]) + regularization_losses

        learning_rate = tf.compat.v1.train.exponential_decay(learning_rate_placeholder,
                                                             global_step,
                                                             settings.learning_rate_decay_epochs * settings.epoch_size,
                                                             settings.learning_rate_decay_factor,
                                                             staircase=True)
        tf.compat.v1.summary.scalar('learning_rate', learning_rate)

        # Create list with variables to restore
        restore_vars = tf.compat.v1.global_variables()
        update_gradient_vars = tf.compat.v1.global_variables()

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, settings.optimizer, learning_rate, settings.moving_average_decay, update_gradient_vars)
        
        # Create a saver
        restore_saver = tf.compat.v1.train.Saver(restore_vars)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=300)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.compat.v1.summary.merge_all()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        with sess.as_default():

            if settings.iteration > 0 and settings.pretrained_model_dir:
                restore_saver.restore(sess, tf.train.latest_checkpoint(settings.pretrained_model_dir))

            # Training and validation loop
            for epoch in range(settings.epochs):
                text_file = open(text_file_name, "a+")
                text_file.write('epoch' + str(epoch) + '\n')
                text_file.close()
                # Train for one epoch
                step = train(sess, train_dataset, epoch, images_placeholder, learning_rate_placeholder, global_step,
                             embeddings, total_loss, train_op, text_file_name, similar_matrix,
                             settings.image_size, settings.epoch_size, settings.alpha, settings.learning_rate_,
                             settings.batch_size, settings.num_classes_per_batch, settings.num_images_per_class)

                # Save the model checkpoint
                print('Saving checkpoint')
                save_variables_and_metagraph(sess, saver, settings.models_base_dir, sub_dir, step, epoch)
                
    return settings.models_base_dir


def train(sess, dataset, epoch, images_placeholder, learning_rate_placeholder, global_step,
          embeddings, loss, train_op, text_file_name, similar_matrix,
          image_size, epoch_size, alpha, learning_rate_,
          batch_size, people_per_batch, images_per_person):

    batch_number = 0
    total_err = 0
    max_err = 0
    count = 0

    if learning_rate_ > 0.0:
        lr = learning_rate_
    else:
        lr = facenet.get_learning_rate_from_file('../data/learning_rate_schedule.txt', epoch)

    while batch_number < epoch_size:
        print('Loading training data')
        # Sample people and load new data
        start_time = time.time()
        image_paths, num_per_class = facenet.sample_people(dataset, people_per_batch, images_per_person)
        # TODO:
        # print('image_paths')
        # for image_path in image_paths:
        #     print(image_path)
        # print('num_per_class')
        # print(num_per_class)

        image_data = facenet.load_data(image_paths, image_size)
        # TODO:
        # print('image_data')
        # print(f'image_data shape: {np.shape(image_data)}')

        load_time = time.time() - start_time
        print('Loaded %d images in %.2f seconds' % (image_data.shape[0], load_time))

        print('Selecting suitable triplets for training')
        start_time = time.time()
        emb_list = []
        # Run a forward pass for the sampled images
        num_examples_per_epoch = people_per_batch * images_per_person
        num_batches_per_epoch = int(np.floor(num_examples_per_epoch / batch_size))
        for i in range(num_batches_per_epoch):
            batch = facenet.get_batch(image_data, batch_size, i)
            # TODO:
            # print(f'batch shape: {np.shape(batch)}')

            feed_dict = {images_placeholder: batch, learning_rate_placeholder: lr}
            emb_list += sess.run([embeddings], feed_dict=feed_dict)

        # Stack the embeddings to a num_examples_per_epoch x 256 matrix
        emb_array = np.vstack(emb_list)

        matrix = sess.run(similar_matrix, feed_dict=None)

        # Select triplets based on the embeddings
        triplets, num_random_negs, num_triplets = facenet.select_triplets(emb_array, num_per_class, image_data, people_per_batch, alpha, matrix)
        # TODO:
        # print(f'triplets shape: {np.shape(triplets)}')
        # print(f'num_random_negs: {num_random_negs}')
        # print(f'num_triplets: {num_triplets}')

        selection_time = time.time() - start_time
        print('(num_random_negs, num_triplets) = (%d, %d): time=%.3f seconds' % (num_random_negs, num_triplets, selection_time))

        # Perform training on the selected triplets
        train_time = 0
        i = 0

        while i * batch_size < num_triplets * 3 and batch_number < epoch_size:
            start_time = time.time()
            batch = facenet.get_triplet_batch(triplets, i, batch_size)
            # print(f'batch shape: {np.shape(batch)}')
            feed_dict = {images_placeholder: batch, learning_rate_placeholder: lr}
            err, _, step = sess.run([loss, train_op, global_step], feed_dict=feed_dict)
            duration = time.time() - start_time

            text_file = open(text_file_name, "a+")

            text_file.write('Epoch: [%d][%d/%d] \tLoss %2.3f  ' % (epoch, batch_number+1, epoch_size, err))
            text_file.close()
            
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' % (epoch, batch_number+1, epoch_size, duration, err))
            batch_number += 1
            total_err += err
            max_err = max(max_err, err)
            i += 1
            count += 1
            train_time += duration

    text_file = open(text_file_name, "a+")
    text_file.write('\n Epoch: [%d] \tAverage Loss %2.4f  \tMax Loss %2.4f \n' % (epoch, total_err/count, max_err))
    text_file.close()

    return step


if __name__ == '__main__':
    main()
