"""
Functions for building the face recognition network.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

import src.SAL.settings as settings


def triplet_loss_similarity_learning(anchor, positive, negative, alpha, similar_matrix):
    """
    Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
      alpha:
      similar_matrix:
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.compat.v1.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.multiply(tf.matmul(anchor, similar_matrix), positive), 1)
        neg_dist = tf.reduce_sum(tf.multiply(tf.matmul(anchor, similar_matrix), negative), 1)
        basic_loss = tf.add(tf.subtract(neg_dist, pos_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss


def _add_loss_summaries(total_loss):
    """
    Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.compat.v1.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l_ in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.compat.v1.summary.scalar(l_.op.name + ' (raw)', l_)
        tf.compat.v1.summary.scalar(l_.op.name, loss_averages.average(l_))

    return loss_averages_op


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.compat.v1.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.compat.v1.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.compat.v1.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.compat.v1.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)
        
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.compat.v1.trainable_variables():
            tf.compat.v1.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.compat.v1.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_data(image_paths, image_size):
    num_samples = len(image_paths)
    images = np.zeros((num_samples, image_size, image_size, 3))
    for i in range(num_samples):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (settings.image_size, settings.image_size))
        img = img * 1.0/255
        images[i, :, :, :] = img
    return images


def get_batch(image_data, batch_size, batch_index):
    num_examples = np.size(image_data, 0)
    j = batch_index * batch_size % num_examples
    if j + batch_size <= num_examples:
        batch = image_data[j:j+batch_size, :, :, :]
    else:
        x1 = image_data[j:num_examples, :, :, :]
        x2 = image_data[0:num_examples-j, :, :, :]
        batch = np.vstack([x1, x2])
    batch_float = batch.astype(np.float32)
    return batch_float


def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch


def select_triplets(embeddings, num_per_class, image_data, people_per_batch, alpha, similar_matrix):
    """
    Select the triplets for training
    This is v1 of the triplet_selection function using pre-calculated distance matrix.
    """
    num_images = image_data.shape[0]
    # distance matrix
    dists = np.zeros((num_images, num_images))
    for i in np.arange(0, num_images):
        dists[i] = np.sum(np.multiply(np.dot(embeddings, similar_matrix), embeddings[i]), 1)

    num_triplets = num_images - people_per_batch
    shp = [num_triplets, image_data.shape[1], image_data.shape[2], image_data.shape[3]]
    as_arr = np.zeros(shp)
    ps_arr = np.zeros(shp)
    ns_arr = np.zeros(shp)
    
    trip_idx = 0
    # shuffle the triplets index
    shuffle = np.arange(num_triplets)
    np.random.shuffle(shuffle)
    emb_start_idx = 0
    num_random_negs = 0

    # Max int
    max_int = 2**31

    for i in range(people_per_batch):
        n = num_per_class[i]
        for j in range(1, n):
            a_idx = emb_start_idx
            p_idx = emb_start_idx + j
            as_arr[shuffle[trip_idx]] = image_data[a_idx]
            ps_arr[shuffle[trip_idx]] = image_data[p_idx]
      
            pos_dist = dists[a_idx, p_idx]
            sel_neg_idx = emb_start_idx

            # while sel_neg_idx in range(emb_start_idx, emb_start_idx + n):
            while sel_neg_idx >= emb_start_idx and sel_neg_idx <= emb_start_idx + n - 1:
                sel_neg_idx = (np.random.randint(1, max_int) % num_images) - 1

            sel_neg_dist = dists[a_idx, sel_neg_idx]

            random_neg = True
            for k in range(num_images):
                # skip if the index is within the positive (same person) class range.
                if k < emb_start_idx or k > emb_start_idx + n - 1:
                    neg_dist = dists[a_idx, k]
                    # if neg_dist in range(pos_dist + 1, sel_neg_dist) and np.abs(pos_dist - neg_dist) < alpha:
                    if pos_dist < neg_dist and neg_dist < sel_neg_dist and np.abs(pos_dist - neg_dist) < alpha:
                        random_neg = False
                        sel_neg_dist = neg_dist
                        sel_neg_idx = k

            if random_neg:
                num_random_negs += 1

            ns_arr[shuffle[trip_idx]] = image_data[sel_neg_idx]
            trip_idx += 1

        emb_start_idx += n

    triplets = (as_arr, ps_arr, ns_arr)

    return triplets, num_random_negs, num_triplets


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


class ImageClass:
    """
    Stores the paths to images for a given class
    """
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)


def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir, img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))

    return dataset


def sample_people(dataset, people_per_batch, images_per_person):
    num_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    num_classes = len(dataset)
    class_indices = np.arange(num_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < num_images:
        class_index = class_indices[i]
        num_images_in_class = len(dataset[class_index])
        image_indices = np.arange(num_images_in_class)
        np.random.shuffle(image_indices)
        num_images_from_class = min(num_images_in_class, images_per_person, num_images-len(image_paths))
        idx = image_indices[0:num_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * num_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(num_images_from_class)
        i += 1
    return image_paths, num_per_class


def load_model(model_dir, meta_file):
    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.compat.v1.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.compat.v1.get_default_session(), tf.train.latest_checkpoint(model_dir_exp))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt_files = [s for s in files if 'ckpt' in s]
    print(len(ckpt_files))
    if len(ckpt_files) == 0:
        raise ValueError('No checkpoint file found in the model directory (%s)' % model_dir)
    elif len(ckpt_files) > 1:
        ckpt_file = ckpt_files[1]
        print(ckpt_files)
    else:
        ckpt_iter = [(s, int(s.split('-')[-1])) for s in ckpt_files if 'ckpt' in s]
        sorted_iter = sorted(ckpt_iter, key=lambda tup: tup[1])
        ckpt_file = sorted_iter[-1][0]
    return meta_file, ckpt_file


# No Need for now
# def triplet_loss(anchor, positive, negative, alpha):
#     """
#     Calculate the triplet loss according to the FaceNet paper
#
#     Args:
#       anchor: the embeddings for the anchor images.
#       positive: the embeddings for the positive images.
#       negative: the embeddings for the negative images.
#
#     Returns:
#       the triplet loss according to the FaceNet paper as a float tensor.
#     """
#     with tf.compat.v1.variable_scope('triplet_loss'):
#         pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
#         neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
#         basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
#         loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
#     return loss
#
#
def select_triplets_org(embeddings, num_per_class, image_data, people_per_batch, alpha):
    """
    Select the triplets for training
    This is v1 of the triplet_selection function using pre-calculated distance matrix.
    """
    num_images = image_data.shape[0]
    # distance matrix
    dists = np.zeros((num_images, num_images))
    for i in np.arange(0, num_images):
        dists[i] = np.sum(np.square(np.subtract(embeddings, embeddings[i])), 1)

    num_triplets = num_images - people_per_batch
    shp = [num_triplets, image_data.shape[1], image_data.shape[2], image_data.shape[3]]
    as_arr = np.zeros(shp)
    ps_arr = np.zeros(shp)
    ns_arr = np.zeros(shp)

    trip_idx = 0
    # shuffle the triplets index
    shuffle = np.arange(num_triplets)
    np.random.shuffle(shuffle)
    emb_start_idx = 0
    num_random_negs = 0

    # Max int
    maxInt = 2 ** 32

    for i in range(people_per_batch):
        n = num_per_class[i]
        for j in range(1, n):
            a_idx = emb_start_idx
            p_idx = emb_start_idx + j
            as_arr[shuffle[trip_idx]] = image_data[a_idx]
            ps_arr[shuffle[trip_idx]] = image_data[p_idx]

            pos_dist = dists[a_idx, p_idx]
            sel_neg_idx = emb_start_idx

            while sel_neg_idx >= emb_start_idx and sel_neg_idx <= emb_start_idx + n - 1:
                sel_neg_idx = (np.random.randint(1, maxInt) % num_images) - 1

            sel_neg_dist = dists[a_idx, sel_neg_idx]

            random_neg = True
            for k in range(num_images):
                # skip if the index is within the positive (same person) class range.
                if k < emb_start_idx or k > emb_start_idx + n - 1:
                    neg_dist = dists[a_idx, k]
                    if pos_dist < neg_dist and neg_dist < sel_neg_dist and np.abs(pos_dist - neg_dist) < alpha:
                        random_neg = False
                        sel_neg_dist = neg_dist
                        sel_neg_idx = k

            if random_neg:
                num_random_negs += 1

            ns_arr[shuffle[trip_idx]] = image_data[sel_neg_idx]
            trip_idx += 1

        emb_start_idx += n

    triplets = (as_arr, ps_arr, ns_arr)

    return triplets, num_random_negs, num_triplets
#
#
# def prewhiten(x):
#     mean = np.mean(x)
#     std = np.std(x)
#     std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
#     y = np.multiply(np.subtract(x, mean), 1/std_adj)
#     return y
#
#
# def calculate_roc(thresholds, embeddings1, embeddings2, similar_matrix, actual_issame, seed, nrof_folds=10):
#     assert(embeddings1.shape[0] == embeddings2.shape[0])
#     assert(embeddings1.shape[1] == embeddings2.shape[1])
#     nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
#     nrof_thresholds = len(thresholds)
#     folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, random_state=seed)
#
#     tprs = np.zeros((nrof_folds, nrof_thresholds))
#     fprs = np.zeros((nrof_folds, nrof_thresholds))
#     accuracy = np.zeros((nrof_folds))
#     v1 = np.dot(embeddings1, similar_matrix)
#     v2 = np.multiply(v1, embeddings2)
#     dist = np.sum(v2, 1)
#
#     for fold_idx, (train_set, test_set) in enumerate(folds):
#         # Find the best threshold for the fold
#         acc_train = np.zeros((nrof_thresholds))
#         for threshold_idx, threshold in enumerate(thresholds):
#             _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
#         best_threshold_index = np.argmax(acc_train)
#         for threshold_idx, threshold in enumerate(thresholds):
#             tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
#         _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
#
#         tpr = np.mean(tprs, 0)
#         fpr = np.mean(fprs, 0)
#     return tpr, fpr, accuracy
#
#
# def calculate_accuracy(threshold, dist, actual_issame):
#     predict_issame = np.less(dist, threshold)
#     tp = np.sum(np.logical_and(predict_issame, actual_issame))
#     fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
#     tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
#     fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
#
#     tpr = 0 if (tp+fn == 0) else float(tp) / float(tp+fn)
#     fpr = 0 if (fp+tn == 0) else float(fp) / float(fp+tn)
#     acc = float(tp+tn)/dist.size
#     return tpr, fpr, acc
#
#
# def plot_roc(fpr, tpr, label):
#     plt.plot(fpr, tpr, label=label)
#     plt.title('Receiver Operating Characteristics')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend()
#     plt.plot([0, 1], [0, 1], 'g--')
#     plt.grid(True)
#     plt.show()
#
#
# def calculate_val(thresholds, embeddings1, embeddings2, similar_matrix, actual_issame, far_target, seed, nrof_folds=10):
#     assert(embeddings1.shape[0] == embeddings2.shape[0])
#     assert(embeddings1.shape[1] == embeddings2.shape[1])
#     nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
#     nrof_thresholds = len(thresholds)
#     folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, random_state=seed)
#
#     val = np.zeros(nrof_folds)
#     far = np.zeros(nrof_folds)
#
#     v1 = np.dot(embeddings1, similar_matrix)
#     v2 = np.multiply(v1, embeddings2)
#     dist = np.sum(v2, 1)
#
#     for fold_idx, (train_set, test_set) in enumerate(folds):
#         print(fold_idx)
#         # Find the threshold that gives FAR = far_target
#         far_train = np.zeros(nrof_thresholds)
#         for threshold_idx, threshold in enumerate(thresholds):
#             _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
#         if np.max(far_train) >= far_target:
#             f = interpolate.interp1d(far_train, thresholds, kind='slinear')
#             threshold = f(far_target)
#         else:
#             threshold = 0.0
#
#         val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
#
#     val_mean = np.mean(val)
#     far_mean = np.mean(far)
#     val_std = np.std(val)
#     return val_mean, val_std, far_mean
#
#
# def calculate_val_far(threshold, dist, actual_issame):
#     predict_issame = np.less(dist, threshold)
#     true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
#     false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
#     n_same = np.sum(actual_issame)
#     n_diff = np.sum(np.logical_not(actual_issame))
#     val = float(true_accept) / float(n_same)
#     far = float(false_accept) / float(n_diff)
#     return val, far
#
#
# def store_revision_info(src_path, output_dir, arg_string):
#     # Get git hash
#     gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE, cwd=src_path)
#     (stdout, _) = gitproc.communicate()
#     git_hash = stdout.strip()
#
#     # Get local changes
#     gitproc = Popen(['git', 'diff', 'HEAD'], stdout=PIPE, cwd=src_path)
#     (stdout, _) = gitproc.communicate()
#     git_diff = stdout.strip()
#
#     # Store a text file in the log directory
#     rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
#     with open(rev_info_filename, "w") as text_file:
#         text_file.write('arguments: %s\n--------------------\n' % arg_string)
#         text_file.write('git hash: %s\n--------------------\n' % git_hash)
#         text_file.write('%s' % git_diff)
#
#
# def list_variables(filename):
#     reader = training.NewCheckpointReader(filename)
#     variable_map = reader.get_variable_to_shape_map()
#     names = sorted(variable_map.keys())
#     return names
