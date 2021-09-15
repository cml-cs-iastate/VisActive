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
                                                  batch_size=batch_size)
    else:
        testing_generator = None

    return train_generator, val_generator, testing_generator


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


def generate_data(generator, iterations):
    """
    """
    counter = 0
    while counter < iterations:
        gen_next = generator.next()
        yield gen_next[0], gen_next[1]
