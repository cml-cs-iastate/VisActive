import os
import numpy as np
from sklearn import metrics

from tensorflow.keras import callbacks, utils, backend as K, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

import src.VisActive._settings as settings
from src.VisActive.Utils.utils import output_labels
from src.VisActive.Utils.plot import plot_loss_accuracy
from src.VisActive.Utils.data_loader import generator_loader, get_data_from_generator


def run():
    calculate_precision_recall = False

    # Loading data

    # Create training generator
    train_generator, val_generator, test_generator = generator_loader(settings.training_path,
                                                                      settings.val_path,
                                                                      settings.testing_path,
                                                                      settings.image_shape,
                                                                      settings.image_shape,
                                                                      settings.batch_size)

    epoch_size_training = int(train_generator.samples / train_generator.batch_size) + 1
    x_data_train, y_data_train = get_data_from_generator(train_generator, epoch_size_training)

    aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.05, height_shift_range=0.05)
    aug.fit(x_data_train)

    # ------------------------------------------------------------------------------------------------------------------
    # Model

    # build model
    # model = settings.neural_network(settings.num_classes)
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(settings.image_shape, settings.image_shape, 3))
    from tensorflow.keras import layers, models
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(2048, activation='relu')
    dense_layer_2 = layers.Dense(1024, activation='relu')
    prediction_layer = layers.Dense(8, activation='softmax')
    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    model.build(input_shape=(None, settings.image_shape, settings.image_shape, 3))
    print(model.summary())

    # Plot the sub_model
    if settings.plot_model:
        utils.plot_model(model, to_file=os.path.join(settings.output_directory, 'baseline_model.png'))

    # Optimizer configuration
    opt = optimizers.SGD(learning_rate=0.01, momentum=settings.momentum, decay=settings.weight_decay)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])

    # Checkpoint
    checkpoint = callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_accuracy")

    # ------------------------------------------------------------------------------------------------------------------
    # Training

    # Fit the model
    training_history = model.fit(aug.flow(x_data_train, y_data_train, batch_size=settings.batch_size),
                                 validation_data=val_generator,
                                 steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                 validation_steps=val_generator.samples // val_generator.batch_size,
                                 epochs=settings.epochs,
                                 callbacks=[checkpoint],
                                 verbose=settings.verbose_)

    print("Training the baseline_model is over and the best model saved to disk!")

    # Plot the results
    plot_loss_accuracy(training_history.history,
                       settings.output_directory,
                       model_acc_file='baseline_model_accuracy.png',
                       model_loss_file='baseline_model_loss.png')

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluation
    model_loss, model_accuracy = model.evaluate(test_generator)

    # print(f'Model Loss is {np.round(model_loss, 3)}')
    print(f'Model Accuracy is {np.round(model_accuracy, 3)}')

    # ------------------------------------------------------------------------------------------------------------------
    # Testing

    if calculate_precision_recall:
        # Predict probabilities for test set
        predictions = model.predict(test_generator, verbose=settings.verbose_)
        predictions = np.argmax(predictions, axis=-1)

        labels = output_labels(test_generator, iterations=test_generator.samples // test_generator.batch_size)
        labels = np.argmax(labels, axis=-1)

        l = np.minimum(len(labels), len(predictions))
        predictions = predictions[:l]
        labels = labels[:l]

        # macro accuracy
        macro_accuracy_sum = 0
        for c_ in range(settings.num_classes):
            predictions_m = np.where(predictions == c_, -1, 100)    # the 100 is any random number
            predictions_m = -1 * predictions_m
            labels_m = np.where(labels == c_, -1, 0)                # the 0 is any random number to be different from 100
            labels_m = -1 * labels_m
            if sum(labels_m) > 0:
                c_accuracy = sum(np.where(predictions_m == labels_m, 1, 0)) / sum(labels_m)
            else:
                c_accuracy = 0
            macro_accuracy_sum += c_accuracy
        macro_accuracy = macro_accuracy_sum / settings.num_classes

        print(f'Testing is over')
        print(f'macro_accuracy: {np.round(macro_accuracy, 3)}')
        print(metrics.classification_report(labels, predictions, digits=3, zero_division=0))

run()
# ======================================================================================================================

# To run this code for all the iterations, we need to start from iteration 1. The data for all the iterations should be
# generated already.

# Run for each dataset
# for d in ['CIFAR_10-10_truck_unbalanced']:
#     print(f'\n======================================================================================================\n')
#     print(f'Running Dataset: {d}')
#     settings.iteration = 5
#     settings.dataset = d
#     settings.test_search_data = test_search_data = settings.dataset.split('-')[0]
#     settings.training_path = os.path.join(settings.data_path, settings.dataset, f'iteration_{settings.iteration}')
#     settings.val_path = os.path.join(settings.data_path, settings.test_search_data, 'test')
#     settings.testing_path = os.path.join(settings.data_path, settings.test_search_data, 'test')
#     settings.image_shape = 32 if settings.dataset[:5] == 'CIFAR' else 64
#
#     settings.output_directory = os.path.join('Output', settings.dataset, f'iteration_{settings.iteration}')
#     settings.output_search_directory = os.path.join(settings.output_directory, 'Search_results')
#     settings.hvc_model_path = os.path.join(settings.output_directory, 'hvc_model.h5')
#     settings.pre_train_model_path = os.path.join(settings.output_directory, 'pre_train_model.h5')
#     settings.base_model_path = os.path.join(settings.output_directory, 'baseline_model.h5')
#     if not os.path.exists(settings.output_directory):
#         os.makedirs(settings.output_directory)
#     if not os.path.exists(settings.output_search_directory):
#         os.makedirs(settings.output_search_directory)
#
#     print(f'settings.iteration: {settings.iteration}')
#     print(f'settings.dataset: {settings.dataset}')
#     print(f'settings.test_search_data: {settings.test_search_data}')
#     print(f'settings.training_path: {settings.training_path}')
#     print(f'settings.val_path: {settings.val_path}')
#     print(f'settings.testing_path: {settings.testing_path}')
#     print(f'settings.image_shape: {settings.image_shape}')
#     print(f'settings.output_directory: {settings.output_directory}')
#     print(f'settings.output_search_directory: {settings.output_search_directory}')
#     print(f'settings.hvc_model_path: {settings.hvc_model_path}')
#     print(f'settings.pre_train_model_path: {settings.pre_train_model_path}')
#     print(f'settings.base_model_path: {settings.base_model_path}')
#
#     start_ = settings.iteration
#     # Run for each iteration
#     for _ in range(start_, 6, 1):
#         print(f'\nRunning iteration: {settings.iteration}\n')
#         for run_ in range(5):
#             print('\n---------------------------------------------------------------------------------------------------\n')
#             print(f'Running iteration: {settings.iteration} , model : {run_ + 1}')
#             run()
#             K.clear_session()
#
#         settings.iteration += 1
#         settings.training_path = os.path.join(settings.data_path, settings.dataset, f'iteration_{settings.iteration}')
#         settings.output_directory = os.path.join('Output', settings.dataset, f'iteration_{settings.iteration}')
#         settings.base_model_path = os.path.join(settings.output_directory, 'baseline_model.h5')
#         if not os.path.exists(settings.output_directory):
#             os.makedirs(settings.output_directory)
#
#         print(f'The next iteration: {settings.iteration}')
#         print(f'The next training_path : {settings.training_path}')
#         print(f'The next output_directory : {settings.output_directory}')
#         print(f'The next base_model_path : {settings.base_model_path}')
