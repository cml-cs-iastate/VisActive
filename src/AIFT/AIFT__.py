import os, glob, shutil
import numpy as np
from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Dense
from keras.preprocessing import image

import HVC.NeuralNetworks as NeuralNetworks
import HVC._settings as settings
from HVC.Utils.data_loader import generator_loader
from HVC.Utils.plot import plot_loss_accuracy
from HVC.Utils.utils import output_labels

# To reload the settings file
from importlib import reload

reload(settings)

from tensorflow.keras import callbacks, utils, models, backend as K

dataset_classes = settings.dataset_classes  # First class must be rare class
number_of_candidates_needed = settings.number_of_candidates_needed  # Number of images to copy search into training folder for both classes
search_images_to_check = settings.search_images_to_check  # Number of images in the search space to check
total_iterations = settings.total_iterations
start_iteration = settings.start_iteration

image_size = (settings.image_shape, settings.image_shape)

# The set of augmentations to use in creating image patches
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.02, height_shift_range=0.02, shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True, zca_whitening=True, fill_mode='nearest')

# Path to the unlabeled images
unlabeled_data_dir = settings.search_path + "/search"  # The unlabeled dataset folder
unlabeled_image_list = glob.glob(unlabeled_data_dir + "/*")


def create_image_patches_generator(file, image_size, number_of_patches):
    """
    Create image patches for a given image

    Patches are augmented images of the given image. For this project, we create
    15 augmented images for each image

    file: The path of the file to be loaded
    image_size: Size of the final image to be created
    number_of_patches: Total number of patches to be created
    """
    image_list = []

    # 1. Load image with PIL
    img = load_img(file)  # this is a PIL image

    # 2. Resize images to 1.0, 1.2, and 1.5
    img2 = img.resize((77, 77), resample=0)
    img3 = img.resize((96, 96), resample=0)

    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)
    img_flow = datagen.flow(img, batch_size=1)

    for i, new_imgs in enumerate(img_flow):

        new_img = array_to_img(new_imgs[0], scale=True)
        new_img = new_img.resize(image_size, resample=0)
        image_list.append(img_to_array(new_img))
        if i >= 4:
            break

    img = img_to_array(img2)
    img = img.reshape((1,) + img.shape)
    img_flow = datagen.flow(img, batch_size=1)

    for i, new_imgs in enumerate(img_flow):
        if i > 4:
            new_img = array_to_img(new_imgs[0], scale=True)

            # resize it back to 64*64 here
            new_img = new_img.resize(image_size, resample=0)
            image_list.append(img_to_array(new_img))

            if i >= 9:
                break

    img = img_to_array(img3)
    img = img.reshape((1,) + img.shape)
    img_flow = datagen.flow(img, batch_size=1)

    for i, new_imgs in enumerate(img_flow):
        if i > 9:
            new_img = array_to_img(new_imgs[0], scale=True)

            # resize it back to 64*64 here
            new_img = new_img.resize(image_size, resample=0)
            image_list.append(img_to_array(new_img))

            if i >= 14:
                break

    return image_list


def compute_R_score(w, h, probability_vector):
    """
    Compute the R score from the AIFT paper

    You can check the AIFT paper for more information

    w: width of the R matrix
    h: Height of the R matrix
    probability_vector: probability values for the patches of a candidate
    """

    # set lamda variables here
    lambda1 = 0.5
    lambda2 = 0.5

    # now that we have the ones we will be using, we can compute the R matrix (4x4)
    w, h = 4, 4
    sum = 0
    error = 1.0e-60

    # 1. Sort the probability values
    probability_vector = sorted(probability_vector)

    # 2. Compute mean value
    a = np.sum(probability_vector) / len(probability_vector)

    # 3. Check if greater than 0.5
    if a > 0.5:
        p = probability_vector[11:]
    else:
        p = probability_vector[0:4]

    # 4. Compute entropy and diversity
    R = [[0 for x in range(w)] for y in range(h)]
    for i in range(w):
        for j in range(h):

            if i == j:  # entropy computed for diagonal of R
                R[i][j] = -lambda1 * (((p[i] * np.log(p[i] + error))) + (((1 - p[i]) * np.log(1 - p[i] + error))))
            else:  # diversity computed for rest of R
                R[i][j] = lambda2 * (((p[i] - p[j])) * np.log((error + p[i]) / (p[j] + error)) +
                                     ((1 - p[i] - 1 + p[j])) * np.log((error + 1 - p[i]) / (1 - p[j] + error)))
                R[j][i] = R[i][j]

    return np.sum(R)


def make_prediction(model, data_generator):
    """
    Make prediction based on the given data generator

    model: The model to use to make prediction
    data_generator: The data to make predictions on
    """

    # Predict probabilities for patches set
    predictions = model.predict(data_generator, verbose=1)

    del data_generator

    # Return the predictions based for the first class
    return predictions[:, 0]


def delete_dir_if_exists(directory):
    """
    Remove a directory if it exists

    dir - Directory to remove
    """

    if os.path.exists(directory):
        shutil.rmtree(directory)


def create_dir(directory):
    """
    Create directory. Deletes and recreate directory if already exists

    Parameter:
    string - directory - name of the directory to create if it does not already exist
    """

    delete_dir_if_exists(directory)
    os.makedirs(directory)


def move_file(filename, destination_dir):
    """
    Move file to destination directory

    filename: The path of file to be moved
    destination_dir: Directory to move image into
    """

    os.rename(filename, destination_dir + "/" + filename.split("/")[-1])


def move_images_from_search(result_list, number_of_candidates_needed, dataset_classes, first_time):
    """
    Move images from the search folder to next iteration training folder

    result_dictionary: The list contains each file with its r score
    number_of_candidates_needed: Number of candidates to be chosen
    dataset_classes: List of the classes to be in the training dataset folder. First item in the list must be rare class
    first_time: First time to be moving images to training folder
    """

    candidate_list = result_list[:number_of_candidates_needed]

    if first_time:
        training_directory = settings.training_path
        print(training_directory)

    else:
        # Create a new training folder with the new images
        training_directory = ''.join(settings.training_path.rsplit("_", 1)[:-1]) + "_" + str(
            int(settings.training_path.rsplit("_", 1)[-1]) + 1)
        print(training_directory)

    create_dir(training_directory)

    # Create new classes in the training dataset folder
    for classes in dataset_classes:
        create_dir(training_directory + "/" + classes)

    # Iterate over the candidate list and move images appropriately
    for candidate in candidate_list:

        # If the candidate belongs to the rare class, move to rare class
        if candidate.split("/")[-1].startswith(settings.rare_class_file_name):
            os.rename(candidate, training_directory + "/" + dataset_classes[0] + "/" + candidate.split("/")[-1])
        else:
            os.rename(candidate, training_directory + "/" + dataset_classes[1] + "/" + candidate.split("/")[-1])


def copy_all_files(src_dir, dest_dir):
    """
    Copies all files from source directory to destination directory

    - src_dir - the source directory to copy from
    - dest_dir - the destination directory to copy into

    source - https://www.geeksforgeeks.org/copy-all-files-from-one-directory-to-another-using-python/
    """

    files = glob.glob(src_dir + "*")

    if not os.path.exists(dest_dir):
        create_dir(dest_dir)

    for f in files:
        shutil.copy(f, dest_dir)


def replace_line(file_name, line_num, text):
    """
    Source - StackOverflow
    Replace a line in a given file at a given line_num with given text

    file_name: File to be edited
    line_num: The line number to be edited, 0-based indexing
    text: The text to replace existing text
    """
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text + "\n"
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


def copy_previous_to_new_training_folder():
    """
    Copy previous training images into new training folder
    """
    for classes in dataset_classes:
        next_training_directory = ''.join(settings.training_path.rsplit("_", 1)[:-1]) + "_" + str(
            int(settings.training_path.rsplit("_", 1)[-1]) + 1)
        copy_all_files(settings.training_path + "/" + classes + "/", next_training_directory + "/" + classes + "/")


def get_result_dictionary(number_of_images_to_scan, image_size):
    """
    Get the r score for every unlabeled item

    number_of_images_to_scan: Number of images to scan
    image_size: size of image to create
    """

    unlabeled_data_dir = settings.search_path + "/search"  # The unlabeled dataset folder
    unlabeled_image_list = glob.glob(unlabeled_data_dir + "/*")

    result_list = []

    for items in range(0, len(unlabeled_image_list), number_of_images_to_scan):

        # 1. Get the list of files to work on for this count
        files_to_scan = unlabeled_image_list[items:items + number_of_images_to_scan]

        patches = []

        for file in files_to_scan:
            # 2. Generate the patches
            patches_generator = create_image_patches_generator(file, image_size, 15)
            patches.extend(patches_generator)

        data_generator = image.ImageDataGenerator(rescale=1. / 255)
        prediction_results = make_prediction(model, data_generator.flow(
            np.reshape(patches, (15 * len(files_to_scan), image_size[0], image_size[1], 3)), batch_size=128))
        patches.clear()

        del data_generator

        # Compute R score for each of the files
        for i, file in enumerate(files_to_scan):
            # 4. Compute R score
            r_score = compute_R_score(4, 4, prediction_results[i * 15: (i + 1) * 15])
            result_list.append(r_score)

    result_list = np.argsort(result_list)

    items_list = []
    for items in result_list:
        items_list.append(unlabeled_image_list[items])

    return items_list


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

# If iteration 0 images are given already, train model with iteration 0 files
train_generator, val_generator, test_generator = generator_loader(settings.training_path,
                                                                  settings.testing_path,
                                                                  settings.testing_path,
                                                                  settings.image_shape,
                                                                  settings.image_shape,
                                                                  settings.batch_size)

training_history = model.fit(train_generator,
                             validation_data=val_generator,
                             steps_per_epoch=train_generator.samples // train_generator.batch_size,
                             validation_steps=val_generator.samples // val_generator.batch_size,
                             epochs=settings.epochs,
                             callbacks=[checkpoint],
                             verbose=0)

for iteration in range(start_iteration, total_iterations + 1):

    print("Iteration: ", iteration)

    if iteration > 0:
        # Load the best finetuned version of the model
        model = models.load_model(settings.base_model_path)

    # Get next list of items to copy
    result_dictionary = get_result_dictionary(search_images_to_check, image_size)

    if iteration > 0:
        first_time = False
    else:
        first_time = True

    # 2. Move images to new training folder
    move_images_from_search(result_dictionary, number_of_candidates_needed, dataset_classes, first_time)

    # 3. Copy previous images to new folder
    copy_previous_to_new_training_folder()

    ## UPDATE settings file with iteration
    line_number_to_edit = settings.settings_iteration_line

    replace_line(settings.settings_file_path, line_number_to_edit, "iteration = " + str(iteration))

    # Reload the settings file
    reload(settings)

    # 5. Train the model
    train_generator, val_generator, test_generator = generator_loader(settings.training_path,
                                                                      settings.testing_path,
                                                                      settings.testing_path,
                                                                      settings.image_shape,
                                                                      settings.image_shape,
                                                                      settings.batch_size)

    # 6. Finetune the model
    print("Finetuning the model **************************** ")
    training_history = model.fit(train_generator,
                                 validation_data=val_generator,
                                 steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                 validation_steps=val_generator.samples // val_generator.batch_size,
                                 epochs=settings.epochs,
                                 callbacks=[checkpoint],
                                 verbose=0)
