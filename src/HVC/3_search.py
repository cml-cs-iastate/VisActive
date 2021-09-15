import os
import shutil
import pickle as pkl
import numpy as np

from tensorflow.keras import models

import src.HVC._settings as settings
from src.HVC.Utils.utils import load_hvc_model, get_ordered_file_names, get_file_name_from_path
from src.HVC.Utils.data_loader import generator_loader

copy_correct_images = True
copy_correct_images_num = 1500


# ----------------------------------------------------------------------------------------------------------------------
# Loading data

# Create search generator
train_generator, _, search_generator = generator_loader(settings.training_path,
                                                        settings.training_path,
                                                        settings.search_path,
                                                        settings.image_shape,
                                                        settings.image_shape,
                                                        settings.batch_size)

# Calculate the epoch size in terms of number of batches
train_epoch_size = int(train_generator.samples / train_generator.batch_size)
search_epoch_size = int(search_generator.samples / search_generator.batch_size)

with open(os.path.join(settings.output_search_directory, f'coverage.pkl'), 'rb') as f:
    coverage = pkl.load(f, encoding='bytes')
with open(os.path.join(settings.output_search_directory, f'uniqueness.pkl'), 'rb') as f:
    uniqueness = pkl.load(f, encoding='bytes')

vc_of_class_0 = {i for i in range(0, settings.vc_per_class)}
vc_of_class_1 = {i for i in range(settings.vc_per_class, settings.vc_per_class * 2)}

# Get file name from file paths
training_image_file_names = list()
for _ in range(train_epoch_size + 1):
    input_x, _ = next(train_generator)
    training_image_file_names += [get_file_name_from_path(f) for f in get_ordered_file_names(train_generator)]


# ----------------------------------------------------------------------------------------------------------------------
# Model

# Load model
model = load_hvc_model(model_path=settings.model_path,
                       neural_network=settings.neural_network,
                       num_classes=settings.num_classes,
                       image_size=settings.image_shape,
                       weight_decay=settings.weight_decay,
                       num_vc=settings.num_vc,
                       p_h=settings.p_h,
                       p_w=settings.p_w)

# Build hvc_model to get the VC output
hvc_model = models.Model(inputs=model.input, outputs=model.get_layer('vcl').output)


# ----------------------------------------------------------------------------------------------------------------------
# Searching

print(f'Searching Started!')

# output = [('image_name', activation_score, VC_num, sorted_vc)]
output = list()

for counter in range(search_epoch_size + 1):

    input_x, _ = next(search_generator)
    # Get file name from file paths
    image_file_names = [get_file_name_from_path(f) for f in get_ordered_file_names(search_generator)]
    # Get the visual concept output
    predictions = hvc_model.predict(input_x)
    # shape [batch_size, num_vc]
    activations = predictions[0]

    for image_idx in range(search_generator.batch_size):
        # Get the minimum distance to a VC
        activation_score = np.max(activations[image_idx])
        # Get the VC number of the minimum distance
        max_activation_vc_num = np.argmax(activations[image_idx])
        # Get the sorted VCs based on their minimum score in descending order
        sorted_vc = np.argsort(activations[image_idx])[::-1]
        # Get the image file name
        image_file_name = image_file_names[image_idx]
        output.append((image_file_name, activation_score, max_activation_vc_num, sorted_vc))

with open(os.path.join(settings.output_search_directory, 'output.txt'), 'w') as f:
    for i in output:
        f.write(f'{i[0]}\t{i[1]}\t{i[2]}\t{i[3]} \n')
f.close()


# ----------------------------------------------------------------------------------------------------------------------
# Post-process the search

print(f'Post-process the search started!')

rare_class_common_vc = list()
rare_class_rare_vc = list()
rare_class_not_sure = list()
rare_class_unseen_samples = list()

common_class_common_vc = list()
common_class_rare_vc = list()
common_class_not_sure = list()
common_class_unseen_samples = list()

# Load image name from the training dataset
selected_images = set(training_image_file_names)

# Sort based on the minimum distance
output.sort(key=lambda x: x[1], reverse=True)

counter_ = 0

for image_file_name, activation_score, max_activation_vc_num, sorted_vc in output:

    if image_file_name in selected_images:
        continue

    counter_ += 1

    # Get the class
    class_ = 0 if max_activation_vc_num < settings.vc_per_class else 1

    # Calculate the not-sure images
    vc_belongs_to_class_0 = 0
    vc_belongs_to_class_1 = 0
    for vc in sorted_vc[:settings.gama]:
        if vc in vc_of_class_0:
            vc_belongs_to_class_0 += 1
        else:
            vc_belongs_to_class_1 += 1

    # 1) Samples with not-sure concepts
    if vc_belongs_to_class_0 == vc_belongs_to_class_1:
        if class_ == 0:
            rare_class_not_sure.append((image_file_name, activation_score, max_activation_vc_num))
            selected_images.add(image_file_name)
        else:
            common_class_not_sure.append((image_file_name, activation_score, max_activation_vc_num))
            selected_images.add(image_file_name)

    elif vc_belongs_to_class_0 == settings.gama or vc_belongs_to_class_1 == settings.gama:

        # 2) Rare class samples with common concepts
        if coverage[max_activation_vc_num] > settings.alpha:
            if class_ == 0:
                rare_class_common_vc.append((image_file_name, activation_score, max_activation_vc_num))
                selected_images.add(image_file_name)
            else:
                common_class_common_vc.append((image_file_name, activation_score, max_activation_vc_num))
                selected_images.add(image_file_name)

        # 3) Rare class samples with rare concepts
        else:  # elif uniqueness[max_activation_vc_num] == settings.beta:
            if class_ == 0:
                rare_class_rare_vc.append((image_file_name, activation_score, max_activation_vc_num))
                selected_images.add(image_file_name)
            else:
                common_class_rare_vc.append((image_file_name, activation_score, max_activation_vc_num))
                selected_images.add(image_file_name)

    # Stop the search if all the lists are filled
    if counter_ >= (search_generator.samples - train_generator.samples) * 0.8:
        break

# Sample with unseen concepts
counter_ = 0
for image_file_name, activation_score, max_activation_vc_num, sorted_vc in output[::-1]:

    if image_file_name in selected_images:
        continue

    counter_ += 1

    # Get the class
    class_ = 0 if max_activation_vc_num < settings.vc_per_class else 1

    # Calculate the not-sure images
    vc_belongs_to_class_0 = 0
    vc_belongs_to_class_1 = 0
    for vc in sorted_vc[:settings.gama]:
        if vc in vc_of_class_0:
            vc_belongs_to_class_0 += 1
        else:
            vc_belongs_to_class_1 += 1

    if vc_belongs_to_class_0 == vc_belongs_to_class_1:
        continue
    else:
        if class_ == 0:
            rare_class_unseen_samples.append((image_file_name, activation_score, max_activation_vc_num))
            selected_images.add(image_file_name)
        else:
            common_class_unseen_samples.append((image_file_name, activation_score, max_activation_vc_num))
            selected_images.add(image_file_name)

    if counter_ >= (search_generator.samples - train_generator.samples) * 0.2:
        break


# ----------------------------------------------------------------------------------------------------------------------
# Calculate the correctness of the selected image

counter = 0
for image_file_name, _, _ in rare_class_common_vc:
    if image_file_name[:3] == '001':
        counter += 1
print(f'The accuracy of rare_class_common_vc is: {round(np.divide(counter, len(rare_class_common_vc)), 2)}\t\t count {len(rare_class_common_vc)}')

counter = 0
for image_file_name, _, _ in rare_class_rare_vc:
    if image_file_name[:3] == '001':
        counter += 1
print(f'The accuracy of rare_class_rare_vc is: {round(np.divide(counter, len(rare_class_rare_vc)), 2)}\t\t count {len(rare_class_rare_vc)}')

counter = 0
for image_file_name, _, _ in rare_class_not_sure:
    if image_file_name[:3] == '001':
        counter += 1
print(f'The accuracy of rare_class_not_sure is: {round(np.divide(counter, len(rare_class_not_sure)), 2)}\t\t count {len(rare_class_not_sure)}')

counter = 0
for image_file_name, _, _ in rare_class_unseen_samples:
    if image_file_name[:3] == '001':
        counter += 1
print(f'The accuracy of rare_class_unseen_samples is: {round(np.divide(counter, len(rare_class_unseen_samples)), 2)}\t\t count {len(rare_class_unseen_samples)}')

counter = 0
for image_file_name, _, _ in common_class_common_vc:
    if image_file_name[:3] == '002':
        counter += 1
print(f'The accuracy of common_class_common_vc is: {round(np.divide(counter, len(common_class_common_vc)), 2)}\t\t count {len(common_class_common_vc)}')

counter = 0
for image_file_name, _, _ in common_class_rare_vc:
    if image_file_name[:3] == '002':
        counter += 1
print(f'The accuracy of common_class_rare_vc is: {round(np.divide(counter, len(common_class_rare_vc)), 2)}\t\t count {len(common_class_rare_vc)}')

counter = 0
for image_file_name, _, _ in common_class_not_sure:
    if image_file_name[:3] == '002':
        counter += 1
print(f'The accuracy of common_class_not_sure is: {round(np.divide(counter, len(common_class_not_sure)), 2)}\t\t count {len(common_class_not_sure)}')

counter = 0
for image_file_name, _, _ in common_class_unseen_samples:
    if image_file_name[:3] == '002':
        counter += 1
print(f'The accuracy of common_class_unseen_samples is: {round(np.divide(counter, len(common_class_unseen_samples)), 2)}\t\t count {len(common_class_unseen_samples)}')


# ----------------------------------------------------------------------------------------------------------------------
# Copy the selected images into directories

if copy_correct_images:

    print(f'Copying the rare_class_common_vc')
    counter = 0
    save_dir = os.path.join(settings.output_search_directory, 'rare_class_common_vc')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for image_file_name, _, _ in rare_class_common_vc:
        if image_file_name[:3] == '001':
            current_image = os.path.join(settings.search_path, 'search', image_file_name)
            copied_image = os.path.join(save_dir, image_file_name)
            shutil.copy(current_image, copied_image)
            counter += 1
        if counter >= copy_correct_images_num:
            break

    print(f'Copying the rare_class_rare_vc')
    counter = 0
    save_dir = os.path.join(settings.output_search_directory, 'rare_class_rare_vc')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for image_file_name, _, _ in rare_class_rare_vc:
        if image_file_name[:3] == '001':
            current_image = os.path.join(settings.search_path, 'search', image_file_name)
            copied_image = os.path.join(save_dir, image_file_name)
            shutil.copy(current_image, copied_image)
            counter += 1
        if counter >= copy_correct_images_num:
            break

    print(f'Copying the common_class_common_vc')
    counter = 0
    save_dir = os.path.join(settings.output_search_directory, 'common_class_common_vc')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for image_file_name, _, _ in common_class_common_vc:
        if image_file_name[:3] == '002':
            current_image = os.path.join(settings.search_path, 'search', image_file_name)
            copied_image = os.path.join(save_dir, image_file_name)
            shutil.copy(current_image, copied_image)
            counter += 1
        if counter >= copy_correct_images_num:
            break

    print(f'Copying the common_class_rare_vc')
    counter = 0
    save_dir = os.path.join(settings.output_search_directory, 'common_class_rare_vc')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for image_file_name, _, _ in common_class_rare_vc:
        if image_file_name[:3] == '002':
            current_image = os.path.join(settings.search_path, 'search', image_file_name)
            copied_image = os.path.join(save_dir, image_file_name)
            shutil.copy(current_image, copied_image)
            counter += 1
        if counter >= copy_correct_images_num:
            break

    print(f'Copying the not-sure samples')
    save_dir = os.path.join(settings.output_search_directory, 'not_sure')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    counter = 0
    for image_file_name, _, _ in rare_class_not_sure:
        if image_file_name[:3] == '001':
            current_image = os.path.join(settings.search_path, 'search', image_file_name)
            copied_image = os.path.join(save_dir, image_file_name)
            shutil.copy(current_image, copied_image)
            counter += 1
        if counter >= copy_correct_images_num:
            break

    counter = copy_correct_images_num + 1
    for image_file_name, _, _ in common_class_not_sure:
        if image_file_name[:3] == '002':
            current_image = os.path.join(settings.search_path, 'search', image_file_name)
            copied_image = os.path.join(save_dir, image_file_name)
            shutil.copy(current_image, copied_image)
            counter += 1
        if counter >= copy_correct_images_num * 2:
            break

    print(f'Copying the unseen samples')
    save_dir = os.path.join(settings.output_search_directory, 'unseen_samples')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    counter = 0
    for image_file_name, _, _ in rare_class_unseen_samples:
        if image_file_name[:3] == '001':
            current_image = os.path.join(settings.search_path, 'search', image_file_name)
            copied_image = os.path.join(save_dir, image_file_name)
            shutil.copy(current_image, copied_image)
            counter += 1
        if counter >= copy_correct_images_num:
            break

    counter = copy_correct_images_num + 1
    for image_file_name, _, _ in common_class_unseen_samples:
        if image_file_name[:3] == '002':
            current_image = os.path.join(settings.search_path, 'search', image_file_name)
            copied_image = os.path.join(save_dir, image_file_name)
            shutil.copy(current_image, copied_image)
            counter += 1
        if counter >= copy_correct_images_num * 2:
            break


# ----------------------------------------------------------------------------------------------------------------------
# # Copy the selected images into directories
#
# print(f'\n\nCopying the rare_class_common_vc')
# counter = 0
# save_dir = os.path.join(settings.output_search_directory, 'rare_class_common_vc')
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# for image_file_name, _, _ in rare_class_common_vc:
#     current_image = os.path.join(settings.search_path, 'search', image_file_name)
#     copied_image = os.path.join(save_dir, str(counter) + '_' + image_file_name)
#     shutil.copy(current_image, copied_image)
#     counter += 1
#
# print(f'\n\nCopying the rare_class_rare_vc')
# counter = 0
# save_dir = os.path.join(settings.output_search_directory, 'rare_class_rare_vc')
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# for image_file_name, _, _ in rare_class_rare_vc:
#     current_image = os.path.join(settings.search_path, 'search', image_file_name)
#     copied_image = os.path.join(save_dir, str(counter) + '_' + image_file_name)
#     shutil.copy(current_image, copied_image)
#     counter += 1
#
# print(f'\n\nCopying the common_class_common_vc')
# counter = 0
# save_dir = os.path.join(settings.output_search_directory, 'common_class_common_vc')
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# for image_file_name, _, _ in common_class_common_vc:
#     current_image = os.path.join(settings.search_path, 'search', image_file_name)
#     copied_image = os.path.join(save_dir, str(counter) + '_' + image_file_name)
#     shutil.copy(current_image, copied_image)
#     counter += 1
#
# print(f'\n\nCopying the common_class_rare_vc')
# counter = 0
# save_dir = os.path.join(settings.output_search_directory, 'common_class_rare_vc')
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# for image_file_name, min_dist, min_dist_vc_num in common_class_rare_vc:
#     current_image = os.path.join(settings.search_path, 'search', image_file_name)
#     copied_image = os.path.join(save_dir, str(counter) + '_' + image_file_name)
#     shutil.copy(current_image, copied_image)
#     counter += 1
#
# print(f'\n\nCopying the not-sure samples')
# save_dir = os.path.join(settings.output_search_directory, 'not_sure')
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# counter = 0
# for image_file_name, _, _ in rare_class_not_sure:
#     current_image = os.path.join(settings.search_path, 'search', image_file_name)
#     copied_image = os.path.join(save_dir, str(counter) + '_' + image_file_name)
#     shutil.copy(current_image, copied_image)
#     counter += 1
# counter = int(settings.delta_not_sure)
# for image_file_name, _, _ in common_class_not_sure:
#     current_image = os.path.join(settings.search_path, 'search', image_file_name)
#     copied_image = os.path.join(save_dir, str(counter) + '_' + image_file_name)
#     shutil.copy(current_image, copied_image)
#     counter += 1
#
# print(f'\n\nCopying the unseen samples')
# save_dir = os.path.join(settings.output_search_directory, 'unseen_samples')
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# counter = 0
# for image_file_name, _, _ in rare_class_unseen_samples:
#     current_image = os.path.join(settings.search_path, 'search', image_file_name)
#     copied_image = os.path.join(save_dir, str(counter) + '_' + image_file_name)
#     shutil.copy(current_image, copied_image)
#     counter += 1
# counter = int(settings.delta_not_seen)
# for image_file_name, _, _ in common_class_unseen_samples:
#     current_image = os.path.join(settings.search_path, 'search', image_file_name)
#     copied_image = os.path.join(save_dir, str(counter) + '_' + image_file_name)
#     shutil.copy(current_image, copied_image)
#     counter += 1
