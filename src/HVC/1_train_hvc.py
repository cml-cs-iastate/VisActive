import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers, callbacks, utils, models

import src.HVC._settings as settings
from src.HVC.Utils.utils import get_layer_names
from src.HVC.Utils.plot import plot_loss_accuracy
from src.HVC.Utils.data_loader import generator_loader, generate_balanced_batch
from src.HVC.TrainingStage.activation_and_loss import custom_loss
from src.HVC.VCL.push_vcl import push

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# ----------------------------------------------------------------------------------------------------------------------
# Loading data

# Create training generator
train_generator, val_generator, _ = generator_loader(settings.training_path,
                                                     settings.val_path,
                                                     None,
                                                     settings.image_shape,
                                                     settings.image_shape,
                                                     settings.batch_size)

total_history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

global_min_vc_distance = np.full(settings.num_vc, np.inf)

# Calculate the epoch size in terms of number of batches
epoch_size_training = int(train_generator.samples / train_generator.batch_size) + 1
epoch_size_valid = int(val_generator.samples / val_generator.batch_size) + 1

# Generate balanced data batches
train_generator, val_generator = generate_balanced_batch(train_generator,
                                                         epoch_size_training,
                                                         val_generator,
                                                         epoch_size_valid,
                                                         settings.batch_size,
                                                         settings.num_classes)


# ----------------------------------------------------------------------------------------------------------------------
# Build the model

model, vcl = settings.neural_network(num_classes=settings.num_classes,
                                     image_size=settings.image_shape,
                                     weight_decay=settings.weight_decay,
                                     num_vc=settings.num_vc,
                                     p_h=settings.p_h,
                                     p_w=settings.p_w)
print(model.summary())

# Plot the sub_model
if settings.plot_model:
    utils.plot_model(model, to_file=os.path.join(settings.output_directory, 'hvc_model.png'))


# Load baseline model
baseline_model = models.load_model(settings.base_model_path)

# Get the layers name of the sub_model and base_model
model_layers = get_layer_names(model)
baseline_model_layers = get_layer_names(baseline_model)

# Copy the weights of the base model to the convolutional layers of the sub_model and set the trainable to False
for layer in model.layers:
    if layer.name in baseline_model_layers and layer.name != 'predictions':
        # print(f'Copy the weight of layer {layer.name} from the baseline model to the HVC model')
        layer.set_weights(baseline_model.get_layer(layer.name).get_weights())
        layer.trainable = False


# Optimizer configuration
opt = optimizers.Adam(learning_rate=settings.learning_rate)


# Compile the model
def compile_model():
    model.compile(optimizer=opt,
                  loss=custom_loss(vc_distances=vcl[3],
                                   vc_weights=vcl[1][0],
                                   num_classes=tf.constant(settings.num_classes)),
                  metrics=['accuracy'])


# Checkpoint
checkpoint = callbacks.ModelCheckpoint(settings.model_path,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='min')


# Fit the model
def fit_model(num_epoch_):
    training_history = model.fit(train_generator,
                                 validation_data=val_generator,
                                 steps_per_epoch=epoch_size_training,
                                 validation_steps=epoch_size_valid,
                                 epochs=num_epoch_,
                                 callbacks=[checkpoint],
                                 verbose=1)

    for key_, value_ in total_history.items():
        total_history[key_] += training_history.history[key_]


# ----------------------------------------------------------------------------------------------------------------------
# Training

iteration = 0
push_vc = False
train_cnn = False
trainable_softmax = False

"""
x epochs: Train the HVC layer only
y epochs: Train the entire model but the last layer
z epochs: Train the entire model
"""

while iteration < settings.hvc_epochs:
    if not push_vc:
        print(f'\nTrain the model for {settings.training_epochs} epochs.\n')

        if not train_cnn and iteration > settings.train_cnn_layers_after:
            for layer in model.layers:
                if layer.name != 'predictions':
                    layer.trainable = True
            train_cnn = True

        if not trainable_softmax and iteration > settings.train_fully_after:
            model.get_layer('predictions').trainable = True
            trainable_softmax = True

        compile_model()
        fit_model(settings.training_epochs)
        iteration += settings.training_epochs
        push_vc = True

    else:
        if iteration > settings.pushing:
            print(f'\nUpdate the VC to the closest patch\n')

            # Build hvc_model to get the VC output
            hvc_model = models.Model(inputs=model.input, outputs=model.get_layer('vcl').output)

            # Read the data and the labels from the generator to get the y_true,
            # needed for the VCL learning stage only
            count = 0
            for batch_x, batch_y in train_generator:
                # iterate for one epoch only
                if count < epoch_size_training:
                    predictions = hvc_model.predict(np.array(batch_x))
                    if count == 0:
                        output_vc_weights = predictions[1][0, :, :, :, :]
                    output_vc_weights, global_min_vc_distance = push(output_vc_weights,
                                                                     predictions[2],
                                                                     predictions[3],
                                                                     global_min_vc_distance,
                                                                     np.array(batch_y))
                    count += 1
                else:
                    break

            # Feed the pushed VC to the original model VCL layer
            model.get_layer('vcl').set_weights([output_vc_weights])

        push_vc = False

    print(f'\niteration: {iteration}')

model.save(os.path.join(settings.output_directory, 'hvc_last_model.h5'))

print("Training the hvc_model is over and the best model saved to disk!")

# Plot the results
plot_loss_accuracy(total_history,
                   settings.output_directory,
                   model_acc_file='hvc_model_accuracy.png',
                   model_loss_file='hvc_model_loss.png')
