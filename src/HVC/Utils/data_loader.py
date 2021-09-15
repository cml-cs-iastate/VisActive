import random
import numpy as np

from tensorflow.keras.preprocessing import image


# Tested
def create_data_generator(path, width, height, batch_size, validation_split=0.0, shuffle_=True):
    # create generator
    if validation_split > 0:
        data_generator = image.ImageDataGenerator(rescale=1./255, validation_split=validation_split)
        tran_data = data_generator.flow_from_directory(directory=path,
                                                       target_size=(width, height),
                                                       batch_size=batch_size,
                                                       subset='training',
                                                       shuffle=shuffle_)
        val_data = data_generator.flow_from_directory(directory=path,
                                                      target_size=(width, height),
                                                      batch_size=batch_size,
                                                      subset='validation',
                                                      shuffle=shuffle_)
        return tran_data, val_data

    else:
        data_generator = image.ImageDataGenerator(rescale=1./255)
        tran_data = data_generator.flow_from_directory(directory=path,
                                                       target_size=(width, height),
                                                       batch_size=batch_size,
                                                       shuffle=shuffle_)
        return tran_data


# Tested
def generator_loader(training_path, val_path, testing_path, image_width, image_height, batch_size):
    """
    Create Training, validation, and Testing generators
    """
    # Create Training and Validation Generators
    if val_path is None:
        train_generator, val_generator = create_data_generator(path=training_path,
                                                               width=image_width,
                                                               height=image_height,
                                                               batch_size=batch_size,
                                                               validation_split=0.2)
    else:
        train_generator = create_data_generator(path=training_path,
                                                width=image_width,
                                                height=image_height,
                                                batch_size=batch_size)

        val_generator = create_data_generator(path=val_path,
                                              width=image_width,
                                              height=image_height,
                                              batch_size=batch_size)

    # Create Testing Generator
    if testing_path is not None:
        testing_generator = create_data_generator(path=testing_path,
                                                  width=image_width,
                                                  height=image_height,
                                                  batch_size=batch_size,
                                                  shuffle_=False)
    else:
        testing_generator = None

    return train_generator, val_generator, testing_generator


# Not-Tested
def create_batch_triplets(x_train, y_train, batch_size, num_classes, input_h, input_w, input_d):
    """
    Modified version of
    https://zhangruochi.com/Create-a-Siamese-Network-with-Triplet-Loss-in-Keras/2020/08/11/
    """

    x_anchors = np.zeros((batch_size, input_h, input_w, input_d))
    x_positives = np.zeros((batch_size, input_h, input_w, input_d))
    x_negatives = np.zeros((batch_size, input_h, input_w, input_d))

    y_anchors = np.zeros((batch_size, num_classes))
    y_positives = np.zeros((batch_size, num_classes))
    y_negatives = np.zeros((batch_size, num_classes))

    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, x_train.shape[0] - 1)
        x_anchor = x_train[random_index]
        y_anchor = y_train[random_index]
        y = y_train[random_index]

        indices_for_pos = np.squeeze(np.where(np.argmax(y_train, axis=1) == np.argmax(y)))
        indices_for_neg = np.squeeze(np.where(np.argmax(y_train, axis=1) != np.argmax(y)))

        random_index_positive = random.randint(0, len(indices_for_pos) - 1)
        random_index_negative = random.randint(0, len(indices_for_neg) - 1)

        x_positive = x_train[indices_for_pos[random_index_positive]]
        x_negative = x_train[indices_for_neg[random_index_negative]]
        y_positive = y_train[indices_for_pos[random_index_positive]]
        y_negative = y_train[indices_for_neg[random_index_negative]]

        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative

        y_anchors[i] = y_anchor
        y_positives[i] = y_positive
        y_negatives[i] = y_negative

    x = np.concatenate([x_anchors, x_positives, x_negatives], axis=0)
    y = np.concatenate([y_anchors, y_positives, y_negatives], axis=0)

    return x, y


# Not-Tested
def data_generator_triplets(x_train, y_train, batch_size, num_classes):
    """
    """
    input_h = np.shape(x_train)[1]
    input_w = np.shape(x_train)[2]
    input_d = np.shape(x_train)[3]

    while True:
        x, y = create_batch_triplets(x_train, y_train, batch_size, num_classes, input_h, input_w, input_d)
        yield x, y


def generate_triplets(train_generator, epoch_size_training, val_generator, epoch_size_valid, batch_size, batch_size_triplets, num_classes):
    """
    Generate training data for triplet model
    """
    x_data_train = list()
    y_data_train = list()
    count = 0
    for batch_x_, batch_y_ in train_generator:
        if count < epoch_size_training:
            if len(batch_y_) == batch_size:
                x_data_train.append(batch_x_)
                y_data_train.append(batch_y_)
        else:
            break
        count += 1
    x_data_train = np.concatenate(x_data_train, axis=0)
    y_data_train = np.concatenate(y_data_train, axis=0)

    # Generate validation data for triplet model
    x_data_valid = list()
    y_data_valid = list()
    count = 0
    for batch_x_, batch_y_ in val_generator:
        if count < epoch_size_valid:
            if len(batch_y_) == batch_size:
                x_data_valid.append(batch_x_)
                y_data_valid.append(batch_y_)
        else:
            break
        count += 1
    x_data_valid = np.concatenate(x_data_valid, axis=0)
    y_data_valid = np.concatenate(y_data_valid, axis=0)

    triplets_generator_train = data_generator_triplets(x_data_train, y_data_train, batch_size_triplets, num_classes)
    triplets_generator_valid = data_generator_triplets(x_data_valid, y_data_valid, batch_size_triplets, num_classes)

    return triplets_generator_train, triplets_generator_valid


# Not-Tested
def create_balanced_batch(x_train, y_train, batch_size, num_classes, input_h, input_w, input_d):
    """
    This function will repeat some examples of the minor class. If the data is highly imbalance, this could be a problem.
    """

    indices_for_pos = np.squeeze(np.where(np.argmax(y_train, axis=1) == 0))
    indices_for_neg = np.squeeze(np.where(np.argmax(y_train, axis=1) != 0))

    x_positives = np.zeros((int(batch_size / 2), input_h, input_w, input_d))
    x_negatives = np.zeros((int(batch_size / 2), input_h, input_w, input_d))
    y_positives = np.zeros((int(batch_size / 2), num_classes))
    y_negatives = np.zeros((int(batch_size / 2), num_classes))

    for i in range(0, int(batch_size / 2)):
        random_index_positive = random.randint(0, len(indices_for_pos) - 1)
        random_index_negative = random.randint(0, len(indices_for_neg) - 1)

        x_positives[i] = x_train[indices_for_pos[random_index_positive]]
        x_negatives[i] = x_train[indices_for_neg[random_index_negative]]
        y_positives[i] = y_train[indices_for_pos[random_index_positive]]
        y_negatives[i] = y_train[indices_for_neg[random_index_negative]]

    x = np.concatenate([x_positives, x_negatives], axis=0)
    y = np.concatenate([y_positives, y_negatives], axis=0)

    # Shuffle data
    shuffle = np.arange(batch_size)
    np.random.shuffle(shuffle)
    x = x[shuffle]
    y = y[shuffle]

    return x, y


def data_generator_balanced_batch(x_train, y_train, batch_size, num_classes):
    """
    """
    input_h = np.shape(x_train)[1]
    input_w = np.shape(x_train)[2]
    input_d = np.shape(x_train)[3]

    while True:
        x, y = create_balanced_batch(x_train, y_train, batch_size, num_classes, input_h, input_w, input_d)
        yield x, y


def generate_balanced_batch(train_generator, epoch_size_training, val_generator, epoch_size_valid, batch_size, num_classes):
    """
    Generate training data with class balance batches
    """
    # generate training data
    x_data_train = list()
    y_data_train = list()
    count = 0
    for batch_x_, batch_y_ in train_generator:
        x_data_train.append(batch_x_)
        y_data_train.append(batch_y_)
        count += 1
        if count >= epoch_size_training:
            break

    x_data_train = np.concatenate(x_data_train, axis=0)
    y_data_train = np.concatenate(y_data_train, axis=0)

    # Generate validation data
    x_data_valid = list()
    y_data_valid = list()
    count = 0
    for batch_x_, batch_y_ in val_generator:
        x_data_valid.append(batch_x_)
        y_data_valid.append(batch_y_)
        count += 1
        if count >= epoch_size_valid:
            break

    x_data_valid = np.concatenate(x_data_valid, axis=0)
    y_data_valid = np.concatenate(y_data_valid, axis=0)

    balanced_generator_train = data_generator_balanced_batch(x_data_train, y_data_train, batch_size, num_classes)
    balanced_generator_valid = data_generator_balanced_batch(x_data_valid, y_data_valid, batch_size, num_classes)

    return balanced_generator_train, balanced_generator_valid
