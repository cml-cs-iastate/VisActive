import os
import csv
import shutil

from src.VisActive.Utils.data_loader import generator_loader
from src.VisActive.Utils.utils import get_ordered_file_names, get_file_name_from_path


import src.SAL.settings as settings


# ----------------------------------------------------------------------------------------------------------------------

# Get the training image file names
train_generator, _, _ = generator_loader(settings.training_path,
                                         settings.training_path,
                                         settings.training_path,
                                         settings.image_size,
                                         settings.image_size,
                                         settings.batch_size)
train_epoch_size = int(train_generator.samples / train_generator.batch_size)
training_image_file_names = list()
for _ in range(train_epoch_size + 1):
    input_x, _ = next(train_generator)
    training_image_file_names += [get_file_name_from_path(f) for f in get_ordered_file_names(train_generator)]
selected_images = set(training_image_file_names)


# ----------------------------------------------------------------------------------------------------------------------

f = open(settings.search_name)
reader = csv.reader(f)
name_search = [i[0] for i in reader]

save_dir_rare = os.path.join(settings.output_dir, f'Results_{settings.iteration}', 'Output_Images', f'{0}')
save_dir_common = os.path.join(settings.output_dir, f'Results_{settings.iteration}', 'Output_Images', f'{1}')

if not os.path.isdir(save_dir_rare):
    os.makedirs(save_dir_rare)
if not os.path.isdir(save_dir_common):
    os.makedirs(save_dir_common)

for class_ in {1, 2}:
    f = open(os.path.join(settings.output_dir, f'Results_{settings.iteration}', f'search_similarity_dist_for_class_{class_}.csv'))
    reader = csv.reader(f)
    similarity_dist = [(int(i[0]), float(i[1])) for i in list(reader) if len(i) > 0]
    similarity_dist = sorted(similarity_dist, key=lambda x: x[1], reverse=True)
    # Error: the same image is close to class 0 and 1
    idx = 0
    counter_ = 0
    for i in similarity_dist:
        image_file_name = name_search[i[0]]
        if image_file_name not in selected_images:
            if image_file_name[:len(settings.rare_class_file_name)] == settings.rare_class_file_name:
                if class_ == 1:
                    counter_ += 1
                # current_image = f'{settings.data_dir_search}/search/{image_file_name}'
                # copied_image = f'{save_dir_rare}/{image_file_name}'
                # shutil.copy(current_image, copied_image)
            else:
                if class_ == 2:
                    counter_ += 1
                # current_image = f'{settings.data_dir_search}/search/{image_file_name}'
                # copied_image = f'{save_dir_common}/{image_file_name}'
                # shutil.copy(current_image, copied_image)

            if class_ == 1:
                print(f'class {class_}, count: {counter_}')
            idx += 1
            selected_images.add(image_file_name)
            if idx >= settings.delta:
                break
