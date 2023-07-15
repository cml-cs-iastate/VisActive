import os
import pickle as pkl
import numpy as np

from tensorflow.keras import models

import src.VisActive._settings as settings
from src.VisActive.Utils.utils import load_hvc_model, get_ordered_file_names, get_file_name_from_path
from src.VisActive.Utils.data_loader import generator_loader
from src.VisActive.Utils.copy_images import copy_images_multi, calculate_recomm_accuracy_multi

copy_correct_images = True


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


vc_of_class = {c: {i for i in range(c * settings.vc_per_class, (c + 1) * settings.vc_per_class)} for c in range(settings.num_classes)}


# Get file name from file paths
training_image_file_names = list()
for _ in range(train_epoch_size + 1):
    input_x, _ = next(train_generator)
    training_image_file_names += [get_file_name_from_path(f) for f in get_ordered_file_names(train_generator)]


# ----------------------------------------------------------------------------------------------------------------------
# Model

# Load model
model = load_hvc_model(model_path=settings.hvc_model_path,
                       neural_network=settings.neural_network_hvc,
                       num_classes=settings.num_classes,
                       image_size=settings.image_shape,
                       weight_decay=settings.weight_decay,
                       drop_out=0,
                       num_vc=settings.num_vc,
                       p_h=settings.p_h,
                       p_w=settings.p_w,
                       p_d=settings.p_d)

# Build hvc_model to get the VC output
hvc_model = models.Model(inputs=model.input, outputs=model.get_layer('vcl').output)


# ----------------------------------------------------------------------------------------------------------------------
# Searching

print(f'Searching Started!')

# output = [('image_name', activation_score, VC_num, sorted_vc, class_label, probability)]
output = list()

for counter in range(search_epoch_size):

    input_x, _ = next(search_generator)
    # Get file name from file paths
    image_file_names = [get_file_name_from_path(f) for f in get_ordered_file_names(search_generator)]
    # Get the class label for each image
    class_predictions = model.predict(input_x)
    class_probabilities = np.max(class_predictions, axis=-1)
    class_predictions = np.argmax(class_predictions, axis=-1)
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
        # get the prediction probability
        probability = class_probabilities[image_idx]
        class_label = class_predictions[image_idx]
        output.append((image_file_name, activation_score, max_activation_vc_num, sorted_vc, class_label, probability))

with open(os.path.join(settings.output_search_directory, 'output.txt'), 'w') as f:
    for i in output:
        f.write(f'{i[0]}\t{i[1]}\t{i[2]}\t{i[3]}\t{i[4]}\t{i[5]}\n')
f.close()


# ----------------------------------------------------------------------------------------------------------------------
# Post-process the search

print(f'Post-process the search started!')


common_vc = {c: [] for c in range(settings.num_classes)}
rare_vc = {c: [] for c in range(settings.num_classes)}
not_sure = {c: [] for c in range(settings.num_classes)}
unseen_samples = {c: [] for c in range(settings.num_classes)}

# Load image name from the training dataset
selected_images = set(training_image_file_names)

# Sort based on the minimum distance
output.sort(key=lambda x: x[-1], reverse=True)

counter_ = 0

for image_file_name, activation_score, max_activation_vc_num, sorted_vc, class_label, probability in output:

    if image_file_name in selected_images:
        continue

    counter_ += 1

    # Get the class
    class_ = int(max_activation_vc_num//settings.vc_per_class)

    # Calculate the not-sure images
    vc_belongs_to_class = {c: 0 for c in range(settings.num_classes)}
    for vc in sorted_vc[:settings.gama]:
        vc_belongs_to_class[int(vc//settings.vc_per_class)] += 1

    # 1) Samples with not-sure concepts
    unsure = False
    for c in range(settings.num_classes):
        if vc_belongs_to_class[c] == settings.gama / 2:
            unsure = True

    if unsure:
        not_sure[class_].append((image_file_name, activation_score, max_activation_vc_num))
        selected_images.add(image_file_name)

    else:
        if counter_ >= (search_generator.samples - train_generator.samples) * 0.6:
            continue

        # 2) Rare class samples with common concepts
        if coverage[max_activation_vc_num] >= settings.alpha:
            common_vc[class_].append((image_file_name, activation_score, max_activation_vc_num))
            selected_images.add(image_file_name)

        # 3) Rare class samples with rare concepts
        else:  # if uniqueness[max_activation_vc_num] >= settings.beta:
            rare_vc[class_].append((image_file_name, activation_score, max_activation_vc_num))
            selected_images.add(image_file_name)


# Sample with unseen concepts
counter_ = 0
for image_file_name, activation_score, max_activation_vc_num, sorted_vc, class_label, probability in output[::-1]:

    if image_file_name in selected_images:
        continue

    counter_ += 1

    # Get the class
    class_ = int(max_activation_vc_num // settings.vc_per_class)

    # Calculate the not-sure images
    vc_belongs_to_class = {c: 0 for c in range(settings.num_classes)}
    for vc in sorted_vc[:settings.gama]:
        vc_belongs_to_class[int(vc // settings.vc_per_class)] += 1

    # 1) Samples with not-sure concepts
    unsure = False
    for c in range(settings.num_classes):
        if vc_belongs_to_class[c] == settings.gama / 2:
            unsure = True

    if unsure:
        continue

    else:
        unseen_samples[class_].append((image_file_name, activation_score, max_activation_vc_num))
        selected_images.add(image_file_name)

    if counter_ >= (search_generator.samples - train_generator.samples) * 0.4:
        break


# ----------------------------------------------------------------------------------------------------------------------
# Calculate the correctness of the selected image
for c in range(settings.num_classes):
    calculate_recomm_accuracy_multi(common_vc[c], settings.class_file_name[c], f'class_{c}_common_vc')
    calculate_recomm_accuracy_multi(rare_vc[c], settings.class_file_name[c], f'class_{c}_rare_vc')
    calculate_recomm_accuracy_multi(not_sure[c], settings.class_file_name[c], f'class_{c}_not_sure')
    calculate_recomm_accuracy_multi(unseen_samples[c], settings.class_file_name[c], f'class_{c}_unseen_samples')


# ----------------------------------------------------------------------------------------------------------------------
# Copy the selected images into directories
# Ex. recommend 100 for rare 100 and 100 for common, and then all the 200 images will go the labeled dataset for both classes.
if copy_correct_images:

    for c in range(settings.num_classes):
        save_dir = os.path.join(settings.output_search_directory, f'class_{c}')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    class_file_name_reverse = {v: k for k, v in settings.class_file_name.items()}

    for c in range(settings.num_classes):

        print(f'Copying images with common_vc for class {c}')
        copy_images_multi(settings.output_search_directory,
                          common_vc[c],
                          settings.search_path,
                          settings.delta_per_set,
                          settings.class_names,
                          class_file_name_reverse)

        print(f'Copying images with rare_vc for class {c}')
        copy_images_multi(settings.output_search_directory,
                          rare_vc[c],
                          settings.search_path,
                          settings.delta_per_set,
                          settings.class_names,
                          class_file_name_reverse)

        print(f'Copying images with cross_vc for class {c}')
        copy_images_multi(settings.output_search_directory,
                          not_sure[c],
                          settings.search_path,
                          settings.delta_per_set,
                          settings.class_names,
                          class_file_name_reverse)

        print(f'Copying images with unseen_vc for class {c}')
        copy_images_multi(settings.output_search_directory,
                          unseen_samples[c],
                          settings.search_path,
                          settings.delta_per_set,
                          settings.class_names,
                          class_file_name_reverse)


# ----------------------------------------------------------------------------------------------------------------------

for c in range(settings.num_classes):
    list_len = settings.delta_per_set if len(common_vc[c]) > settings.delta_per_set else len(common_vc[c])
    print(f'len of common_vc:   {len([img[0] for img in common_vc[c][:list_len] if img[0][:len(settings.rare_class_file_name)] == settings.rare_class_file_name])}')
    list_len = settings.delta_per_set if len(rare_vc[c]) > settings.delta_per_set else len(rare_vc[c])
    print(f'len of rare_vc:     {len([img[0] for img in rare_vc[c][:list_len] if img[0][:len(settings.rare_class_file_name)] == settings.rare_class_file_name])}')
    list_len = settings.delta_per_set if len(not_sure[c]) > settings.delta_per_set else len(not_sure[c])
    print(f'len of cross_vc:    {len([img[0] for img in not_sure[c][:list_len] if img[0][:len(settings.rare_class_file_name)] == settings.rare_class_file_name])}')
    list_len = settings.delta_per_set if len(unseen_samples[c]) > settings.delta_per_set else len(unseen_samples[c])
    print(f'len of rare_unseen: {len([img[0] for img in unseen_samples[c][:list_len] if img[0][:len(settings.rare_class_file_name)] == settings.rare_class_file_name])}')
