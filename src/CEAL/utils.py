import os
import shutil
import numpy as np
from tensorflow.keras import models

import src.CEAL.settings as settings
from src.VisActive.Utils.plot import plot_loss_accuracy
from src.VisActive.Utils.utils import get_ordered_file_names, get_file_name_from_path


def initialize_model(x_initial, y_initial, x_test, y_test, checkpoint):
    model = settings.neural_network(num_classes=settings.num_classes,
                                    image_size=settings.image_shape,
                                    dropout=settings.drop_out)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_initial, y_initial,
                        validation_data=(x_test, y_test),
                        batch_size=settings.batch_size,
                        shuffle=True,
                        epochs=settings.epochs,
                        callbacks=[checkpoint],
                        verbose=settings.verbose_)
    print("Pre-training the baseline_model is over and the best model saved to disk!")

    plot_loss_accuracy(history.history,
                       settings.output_directory,
                       model_acc_file='pre_train_model_accuracy.png',
                       model_loss_file='pre_train_model_loss.png')

    # Testing
    new_model = models.load_model(settings.base_model_path)
    scores = new_model.evaluate(x_test, y_test, batch_size=settings.batch_size, verbose=settings.verbose_)
    print('Initial Test Loss: ', scores[0], ' Initial Test Accuracy: ', scores[1])

    return new_model


# Random sampling
def random_sampling(y_pred_prob, n_samples):
    return np.random.choice(range(len(y_pred_prob)), n_samples)


# Rank all the unlabeled samples in an ascending order according to the least confidence
def least_confidence(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)

    lci = np.column_stack((origin_index, max_prob, pred_label))
    lci = lci[lci[:, 1].argsort()]
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]


# Rank all the unlabeled samples in an ascending order according to the margin sampling
def margin_sampling(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    margim_sampling = np.diff(-np.sort(y_pred_prob)[:, ::-1][:, :2])
    pred_label = np.argmax(y_pred_prob, axis=1)
    msi = np.column_stack((origin_index, margim_sampling, pred_label))
    msi = msi[msi[:, 1].argsort()]
    return msi[:n_samples], msi[:, 0].astype(int)[:n_samples]


# Rank all the unlabeled samples in an descending order according to their entropy
def entropy(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)
    eni = np.column_stack((origin_index, entropy, pred_label))
    eni = eni[(-eni[:, 1]).argsort()]
    return eni[:n_samples], eni[:, 0].astype(int)[:n_samples]


def get_high_confidence_samples(y_pred_prob, delta):
    eni, eni_idx = entropy(y_pred_prob, len(y_pred_prob))
    hcs = eni[eni[:, 1] < delta]
    return hcs[:, 0].astype(int), hcs[:, 2].astype(int)


def get_uncertain_samples(y_pred_prob, n_samples, criteria):
    if criteria == 'lc':
        return least_confidence(y_pred_prob, n_samples)
    elif criteria == 'ms':
        return margin_sampling(y_pred_prob, n_samples)
    elif criteria == 'en':
        return entropy(y_pred_prob, n_samples)
    elif criteria == 'rs':
        return None, random_sampling(y_pred_prob, n_samples)
    else:
        raise ValueError('Unknown criteria value \'%s\', use one of [\'rs\',\'lc\',\'ms\',\'en\']' % criteria)


def generate_balanced_batch(generator, epoch_size):
    """
    Generate training data with class balance batches
    """
    # generate training data
    x_data = list()
    y_data = list()
    count = 0
    for batch_x_, batch_y_ in generator:
        x_data.append(batch_x_)
        y_data.append(batch_y_)
        count += 1
        if count >= epoch_size:
            break

    x_data_train = np.concatenate(x_data, axis=0)
    y_data_train = np.concatenate(y_data, axis=0)

    return x_data_train, y_data_train


def get_image_name(generator, epoch_size):
    # Get file name from file paths
    image_file_names = list()
    for _ in range(epoch_size + 1):
        input_x, _ = next(generator)
        image_file_names += [get_file_name_from_path(f) for f in get_ordered_file_names(generator)]

    return image_file_names


def copy_images(images_to_be_copies, iteration):
    save_dir_rare = os.path.join(settings.output_search_directory, f'iteration_{iteration}', '001')
    save_dir_common = os.path.join(settings.output_search_directory, f'iteration_{iteration}', '002')
    if not os.path.isdir(save_dir_rare):
        os.makedirs(save_dir_rare)
    if not os.path.isdir(save_dir_common):
        os.makedirs(save_dir_common)

    for image_file_name in images_to_be_copies:
        if image_file_name[:len(settings.rare_class_file_name)] == settings.rare_class_file_name:
            copy_to = save_dir_rare
            c_ = '001'
        else:
            copy_to = save_dir_common
            c_ = '002'
        current_image = os.path.join(settings.search_path, c_, image_file_name)
        copied_image = os.path.join(copy_to, image_file_name)
        shutil.copy(current_image, copied_image)

    return True


def count(images_to_be_copies):
    count_rare = 0
    count_common = 0
    for image_file_name in images_to_be_copies:
        if image_file_name[:len(settings.rare_class_file_name)] == settings.rare_class_file_name:
            count_rare += 1
        else:
            count_common += 1

    return count_rare, count_common
