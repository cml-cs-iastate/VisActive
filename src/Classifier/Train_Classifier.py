import os
import time
import numpy as np
from sklearn import metrics
from tensorflow.keras import models, optimizers, callbacks, utils

import src.SAL.settings as settings
from src.Classifier.Utils.small_vgg16 import small_vgg16
from src.Classifier.Utils.plot import plot_loss_accuracy
from src.Classifier.Utils.data_loader import generator_loader, output_labels


def main():
    """
    """

    # ----------------------------------------------------------------------------------------------------------------------

    # Parameters
    iteration = 0
    plot_model = False

    # Hyper-parameters
    num_classes = 2
    image_size = 64
    num_epochs = 1
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 0.001

    # Path
    data_name = 'Forceps'
    train_data_dir = f'../../../Data/{data_name}/initial'
    validation_data_dir = f'../../../Data/{data_name}/test'
    test_data_dir = f'../../../Data/{data_name}/test'

    output_dir = f'Output/{data_name}/'
    models_base_dir = os.path.join(output_dir, f'iter_{iteration}')
    models_base_dir_previous = os.path.join(output_dir, f'iter_{iteration - 1}')

    base_model = os.path.join(models_base_dir, 'model.h5')
    pretrained_model = os.path.join(models_base_dir_previous, 'model.h5')

    # Create the model directory if it doesn't exist
    if not os.path.isdir(models_base_dir):
        os.makedirs(models_base_dir)

    # ----------------------------------------------------------------------------------------------------------------------
    # Loading data

    # Create Training, validation, and Testing generators
    train_generator, val_generator, testing_generator = generator_loader(train_data_dir,
                                                                         validation_data_dir,
                                                                         test_data_dir,
                                                                         image_size,
                                                                         image_size,
                                                                         batch_size)

    # ----------------------------------------------------------------------------------------------------------------------
    # Model

    model = small_vgg16(num_classes, image_size, weight_decay)
    print(model.summary())

    if iteration > 0:
        model.load_weights(pretrained_model)

    # Plot the sub_model
    if plot_model:
        utils.plot_model(model, to_file=os.path.join(models_base_dir, 'model_plot.png'))

    # Compile the model
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Checkpoint
    checkpoint = callbacks.ModelCheckpoint(base_model, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # ----------------------------------------------------------------------------------------------------------------------
    # Training

    start_time_training = time.time()

    # Train the model
    history = model.fit(train_generator,
                        steps_per_epoch=train_generator.samples // train_generator.batch_size,
                        validation_data=val_generator,
                        validation_steps=val_generator.samples // val_generator.batch_size,
                        epochs=num_epochs,
                        callbacks=[checkpoint],
                        verbose=1)

    print(f'Training is over, the best model is saved to disk.')
    print(f'The training time is {time.time() - start_time_training} seconds')

    # Plot the Training results
    plot_loss_accuracy(history.history, models_base_dir)

    # ----------------------------------------------------------------------------------------------------------------------
    # Testing

    test_model = models.load_model(base_model)

    # Evaluate the model with Testing data
    if testing_generator:
        results = test_model.evaluate(testing_generator)
    else:
        results = test_model.evaluate(val_generator)
    print(f'Testing is over')

    # ----------------------------------------------------------------------------------------------------------------------
    # Calculate Results

    # predict probabilities for test set
    if testing_generator:
        predictions = model.predict(testing_generator, verbose=1)
        labels = output_labels(testing_generator, iterations=testing_generator.samples // testing_generator.batch_size)
    else:
        predictions = model.predict(val_generator, verbose=1)
        labels = output_labels(val_generator, iterations=val_generator.samples // val_generator.batch_size)

    predictions = np.argmax(predictions, axis=-1)
    labels = np.argmax(labels, axis=-1)

    # f_names is all the file names used in testing
    f_names = testing_generator.filenames
    # miss-classified done on the test data where predictions is the predicted values
    errors = np.where(predictions != testing_generator.classes)[0]
    print(f'the number of incorrectly classified images : {len(errors)}')
    print('Incorrect predicted images file names:')
    incorrectly_classified_images = 'incorrectly_classified_images.csv'
    csv_f_name = open(incorrectly_classified_images, 'a+')
    for i in errors:
        print(f'File name: {f_names[i]},\tTrue Class: {testing_generator.classes[i]},\tPredicted Class: {predictions[i]}')
        csv_f_name.write(f_names[i] + '\n')
    csv_f_name.close()

    length = np.minimum(len(labels), len(predictions))
    predictions = predictions[:length]
    labels = labels[:length]

    # precision tp / (tp + fp)
    precision = metrics.precision_score(labels, predictions)
    # recall: tp / (tp + fn)
    recall = metrics.recall_score(labels, predictions)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(labels, predictions)

    print(f'Accuracy: {round(results[1], 2)}')
    print(f'Loss: {round(results[0], 2)}')
    print(f'Precision: {round(precision, 2)}')
    print(f'Recall: {round(recall, 2)}')
    print(f'F1 score: {round(f1, 2)}')

    # ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
