import glob
from importlib import reload

from tensorflow.keras import callbacks, models, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import src.AIFT.settings as settings
from src.VisActive.Utils.data_loader import generator_loader
from src.AIFT.utils import get_result_dictionary, move_images_from_search, copy_previous_to_new_training_folder, replace_line


# ----------------------------------------------------------------------------------------------------------------------
# Read the data

image_size = (settings.image_shape, settings.image_shape)

# The set of augmentations to use in creating image patches
data_gen = ImageDataGenerator(rotation_range=15,
                              width_shift_range=0.02,
                              height_shift_range=0.02,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              zca_whitening=True,
                              fill_mode='nearest')

data_generator2 = ImageDataGenerator(rescale=1./255)

# Path to the unlabeled images
unlabeled_data_dir = settings.search_path + "/search"                   # The unlabeled dataset folder
unlabeled_image_list = glob.glob(unlabeled_data_dir + "/*")


# ----------------------------------------------------------------------------------------------------------------------
# Load and train the model

# If iteration 0, images are given already, train model with iteration 0 files
train_generator, val_generator, _ = generator_loader(settings.training_path,
                                                     settings.testing_path,
                                                     None,
                                                     settings.image_shape,
                                                     settings.image_shape,
                                                     settings.batch_size)

# Load the model we would be using
model = settings.neural_network(num_classes=settings.num_classes,
                                image_size=settings.image_shape,
                                dropout=settings.drop_out)

# Optimizer configuration
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint
checkpoint = callbacks.ModelCheckpoint(settings.base_model_path,
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only=True,
                                       mode='min')

model.fit(train_generator,
          validation_data=val_generator,
          steps_per_epoch=train_generator.samples // train_generator.batch_size,
          validation_steps=val_generator.samples // val_generator.batch_size,
          epochs=settings.epochs,
          callbacks=[checkpoint],
          verbose=0)


# ----------------------------------------------------------------------------------------------------------------------
# Iterate for each iteration

for iteration in range(settings.start_iteration, settings.total_iterations + 1):

    print(f"Iteration: {iteration}")

    if iteration > 0:
        # Load the best fine_tuned version of the model
        model = models.load_model(settings.base_model_path)

    # 1. Get next list of items to copy
    print(f'Running Step 1\n==========================================================================================')
    result_dictionary = get_result_dictionary(settings.search_images_to_check, image_size, data_gen, settings.search_path, model, data_generator2, settings.search_images_to_check)

    first_time = False if iteration > 0 else True

    # ------------------------------------------------------------------------------------------------------------------
    # 2. Move images to new training folder
    print(f'Running Step 2\n==========================================================================================')
    move_images_from_search(result_dictionary, settings.number_of_candidates_needed, settings.dataset_classes, first_time, settings.training_path, settings.rare_class_file_name)

    # ------------------------------------------------------------------------------------------------------------------
    # 3. Copy previous images to new folder
    print(f'Running Step 3\n==========================================================================================')
    copy_previous_to_new_training_folder(settings.dataset_classes, settings.training_path)

    # ------------------------------------------------------------------------------------------------------------------
    # 4. UPDATE settings file with iteration
    print(f'Running Step 4\n==========================================================================================')
    line_number_to_edit = settings.settings_iteration_line
    replace_line(settings.settings_file_path, line_number_to_edit, "iteration = " + str(iteration))
    # Reload the settings file
    reload(settings)

    # ------------------------------------------------------------------------------------------------------------------
    # Clear the not needed lists
    result_dictionary = None

    # ------------------------------------------------------------------------------------------------------------------
    # 5. Train the model

    print(f'Running Step 5\n==========================================================================================')

    K.clear_session()

    train_generator, val_generator, _ = generator_loader(settings.training_path,
                                                         settings.testing_path,
                                                         None,
                                                         settings.image_shape,
                                                         settings.image_shape,
                                                         settings.batch_size)

    # Load the model we would be using
    model = settings.neural_network(num_classes=settings.num_classes,
                                    image_size=settings.image_shape,
                                    dropout=settings.drop_out)

    # Optimizer configuration
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Checkpoint
    checkpoint = callbacks.ModelCheckpoint(settings.base_model_path,
                                           monitor='val_loss',
                                           verbose=0,
                                           save_best_only=True,
                                           mode='min')

    print("Fine_tuning the model **************************** ")
    model.fit(train_generator,
              validation_data=val_generator,
              steps_per_epoch=train_generator.samples // train_generator.batch_size,
              validation_steps=val_generator.samples // val_generator.batch_size,
              epochs=settings.epochs,
              callbacks=[checkpoint],
              verbose=0)
