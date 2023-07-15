import os
import glob
import shutil
import numpy as np

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img


def create_image_patches_generator(file, image_size, data_gen):
    """
    Create image patches for a given image

    Patches are augmented images of the given image. For this project, we create
    9 augmented images for each image

    file: The path of the file to be loaded
    image_size: Size of the final image to be created
    number_of_patches: Total number of patches to be created
    """
    image_list = list()

    # 1. Load image with PIL
    img = load_img(file)  # this is a PIL image

    # 2. Resize images to 1.0, 1.2, and 1.5
    img2 = img.resize((77, 77), resample=0)
    img3 = img.resize((96, 96), resample=0)

    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)
    # data_gen.fit(img)
    img_flow = [data_gen.flow(img, batch_size=1)]
    for i, new_images in enumerate(img_flow[0]):
        new_img = array_to_img(new_images[0], scale=True)
        new_img = new_img.resize(image_size, resample=0)
        image_list.append(img_to_array(new_img))
        if i >= 2:
            break
    img_flow.clear()

    img = img_to_array(img2)
    img = img.reshape((1,) + img.shape)
    # data_gen.fit(img)
    img_flow = [data_gen.flow(img, batch_size=1)]
    for i, new_images in enumerate(img_flow[0]):
        new_img = array_to_img(new_images[0], scale=True)
        # resize it back to 64*64 here
        new_img = new_img.resize(image_size, resample=0)
        image_list.append(img_to_array(new_img))
        if i >= 2:
            break
    img_flow.clear()

    img = img_to_array(img3)
    img = img.reshape((1,) + img.shape)
    # data_gen.fit(img)
    img_flow = [data_gen.flow(img, batch_size=1)]
    for i, new_images in enumerate(img_flow[0]):
        new_img = array_to_img(new_images[0], scale=True)
        # resize it back to 64 * 64 here
        new_img = new_img.resize(image_size, resample=0)
        image_list.append(img_to_array(new_img))
        if i >= 2:
            break
    img_flow.clear()

    return image_list


def compute_R_score(probability_vector, w=4, h=4):
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

            if i == j:
                # entropy computed for diagonal of R
                R[i][j] = -lambda1 * (((p[i] * np.log(p[i] + error))) + (((1 - p[i]) * np.log(1 - p[i] + error))))
            else:
                # diversity computed for rest of R
                R[i][j] = lambda2 * (((p[i] - p[j])) * np.log((error + p[i]) / (p[j] + error)) + ((1 - p[i] - 1 + p[j])) * np.log((error + 1 - p[i]) / (1 - p[j] + error)))
                R[j][i] = R[i][j]

    return np.sum(R)


def get_result_dictionary(number_of_images_to_scan, image_size, data_gen, search_path, model, data_generator2, search_images_to_check):
    """
    Get the r score for every unlabeled item

    number_of_images_to_scan: Number of images to scan
    image_size: size of image to create
    """

    unlabeled_data_dir = search_path + "/search"                           # The unlabeled dataset folder
    unlabeled_image_list = glob.glob(unlabeled_data_dir + "/*")

    result_dictionary = list()

    # 1. Generate the patches
    print(f'Generating the patches')
    patches = []
    for file in unlabeled_image_list[:number_of_images_to_scan]:
        image_list = create_image_patches_generator(file, image_size, data_gen)
        patches.extend(image_list)

    # 2. Predict the results
    print(f'Predict the results')
    data_generator2.fit(patches)
    prediction_results = make_prediction(model, data_generator2.flow(np.reshape(patches, (9 * len(unlabeled_image_list[:number_of_images_to_scan]), image_size[0], image_size[1], 3)), batch_size=128))
    patches.clear()

    # 4. Compute R score
    print(f'Compute R score')
    for i, file in enumerate(unlabeled_image_list[:search_images_to_check]):
        r_score = compute_R_score(prediction_results[i * 9: (i + 1) * 9], 4, 4)
        result_dictionary.append(np.round(r_score, 3))

    # Sort the dictionary based on r_scores
    print(f'Sorting the result_dictionary based on the R score')
    sorted_result_dictionary = np.argsort(result_dictionary)[::-1]

    output = np.array(unlabeled_image_list)[sorted_result_dictionary]

    return output


def make_prediction(model, data_generator):
    """
    Make prediction based on the given data generator

    model: The model to use to make prediction
    data_generator: The data to make predictions on
    """

    # Predict probabilities for patches set
    predictions = model.predict(data_generator, verbose=0)

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


def move_images_from_search(result_dictionary, number_of_candidates_needed, dataset_classes, first_time, training_path, rare_class_file_name):

    """
    Move images from the search folder to next iteration training folder

    result_dictionary: The dictionary contains each file with its r score
    number_of_candidates_needed: Number of candidates to be chosen
    dataset_classes: List of the classes to be in the training dataset folder. First item in the list must be rare class
    first_time: First time to be moving images to training folder
    """
    candidate_list = result_dictionary[:number_of_candidates_needed]

    if first_time:
        training_directory = training_path
        print(training_directory)
    else:
        # Create a new training folder with the new images
        training_directory = ''.join(training_path.rsplit("_", 1)[:-1]) + "_" + str(int(training_path.rsplit("_", 1)[-1]) + 1)
        print(training_directory)

    create_dir(training_directory)

    # Create new classes in the training dataset folder
    for classes in dataset_classes:
        create_dir(training_directory + "/" + classes)

    # Iterate over the candidate list and move images appropriately
    for candidate in candidate_list:
        # If the candidate belongs to the rare class, move to rare class
        if candidate.split("/")[-1].startswith(rare_class_file_name):
            os.rename(candidate, training_directory + "/" + dataset_classes[0] + "/" + candidate.split("\\")[-1])
        else:
            os.rename(candidate, training_directory + "/" + dataset_classes[1] + "/" + candidate.split("\\")[-1])


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


def copy_previous_to_new_training_folder(dataset_classes, training_path):
    """
    Copy previous training images into new training folder
    """
    for classes in dataset_classes:
        next_training_directory = ''.join(training_path.rsplit("_", 1)[:-1]) + "_" + str(int(training_path.rsplit("_", 1)[-1]) + 1)
        copy_all_files(training_path + "/" + classes + "/", next_training_directory + "/" + classes + "/")
