import os
import numpy as np
from sklearn import metrics

from tensorflow.keras import optimizers, callbacks, utils, models

import src.HVC._settings as settings
from src.HVC.Utils.utils import output_labels
from src.HVC.Utils.plot import plot_loss_accuracy
from src.HVC.Utils.data_loader import generator_loader

testing_only = False


# ----------------------------------------------------------------------------------------------------------------------
# Loading data

# Create training generator
train_generator, val_generator, test_generator = generator_loader(settings.training_path,
                                                                  settings.testing_path,
                                                                  settings.testing_path,
                                                                  settings.image_shape,
                                                                  settings.image_shape,
                                                                  settings.batch_size)

# ----------------------------------------------------------------------------------------------------------------------
# Model

if not testing_only:

    model = settings.neural_network(num_classes=settings.num_classes,
                                    image_size=settings.image_shape,
                                    dropout=settings.drop_out,
                                    weight_decay=settings.weight_decay)
    print(model.summary())

    # Plot the sub_model
    if settings.plot_model:
        utils.plot_model(model, to_file=os.path.join(settings.output_directory, 'baseline_model.png'))

    # Optimizer configuration
    opt = optimizers.Adam(learning_rate=settings.learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Checkpoint
    checkpoint = callbacks.ModelCheckpoint(settings.base_model_path,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           mode='min')

# ----------------------------------------------------------------------------------------------------------------------
# Training

    # Fit the model
    training_history = model.fit(train_generator,
                                 validation_data=val_generator,
                                 steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                 validation_steps=val_generator.samples // val_generator.batch_size,
                                 epochs=settings.epochs,
                                 callbacks=[checkpoint],
                                 verbose=1)

    print("Training the baseline_model is over and the best model saved to disk!")

    # Plot the results
    plot_loss_accuracy(training_history.history, settings.output_directory, model_acc_file='baseline_model_accuracy.png', model_loss_file='baseline_model_loss.png')


# ----------------------------------------------------------------------------------------------------------------------
# Testing

new_model = models.load_model(settings.base_model_path)

# Evaluate the model with test data
results = new_model.evaluate(test_generator, verbose=1)

# Predict probabilities for test set
predictions = new_model.predict(test_generator, verbose=1)
predictions = np.argmax(predictions, axis=-1)

labels = output_labels(test_generator, iterations=test_generator.samples // test_generator.batch_size)
labels = np.argmax(labels, axis=-1)

l = np.minimum(len(labels), len(predictions))
predictions = predictions[:l]
labels = labels[:l]

# precision tp / (tp + fp)
precision = metrics.precision_score(labels, predictions, average='macro')
# recall: tp / (tp + fn)
recall = metrics.recall_score(labels, predictions, average='macro')
# f1: 2 tp / (2 tp + fp + fn)
f1 = metrics.f1_score(labels, predictions, average='macro')

print(f'Testing is over')
print(f'Accuracy: {results[1]}')
print(f'Loss: {results[0]}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 score: {f1}')
print(metrics.classification_report(labels, predictions, digits=3))
print(sum(np.where(predictions == labels, 1, 0))/l)
# ----------------------------------------------------------------------------------------------------------------------
