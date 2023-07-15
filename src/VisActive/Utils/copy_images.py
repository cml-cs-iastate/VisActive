import os
import shutil
import numpy as np


def copy_images(rare_list_dir, common_list_dir, output_search_directory, image_list, file_name, search_path, delta):
    """
    """
    counter = 0

    save_dir_rare = os.path.join(output_search_directory, rare_list_dir)
    save_dir_common = os.path.join(output_search_directory, common_list_dir)
    if not os.path.isdir(save_dir_rare):
        os.makedirs(save_dir_rare)
    if not os.path.isdir(save_dir_common):
        os.makedirs(save_dir_common)

    for image_file_name, _, _ in image_list:
        if image_file_name[:len(file_name)] == file_name:
            copy_to = save_dir_rare
            c_ = '001'
        else:
            copy_to = save_dir_common
            c_ = '002'
        current_image = os.path.join(search_path, c_, image_file_name)
        copied_image = os.path.join(copy_to, image_file_name)
        shutil.copy(current_image, copied_image)
        counter += 1
        if counter >= delta:
            break

    return True


def calculate_recomm_accuracy(recomm_list, recomm_list_name, vc_list_name, rare_class=True):
    if recomm_list:
        counter = 0
        for image_file_name, _, _ in recomm_list:
            if rare_class:
                if image_file_name[:len(recomm_list_name)] == recomm_list_name:
                    counter += 1
            else:
                if image_file_name[:len(recomm_list_name)] != recomm_list_name:
                    counter += 1
        print(f'The accuracy of {vc_list_name} is: {round(np.divide(counter, len(recomm_list)), 2)}\t\t count {len(recomm_list)}')
    else:
        print(f'The accuracy of {vc_list_name} is: None --- Noting in the list')


def copy_images_multi(output_search_directory, image_list, search_path, delta, class_names, class_file_name_reverse):
    """
    """
    counter = 0

    for image_file_name, _, _ in image_list:

        c = class_file_name_reverse[image_file_name[:3]]
        copy_to = os.path.join(output_search_directory, f'class_{c}')
        current_image = os.path.join(search_path, class_names[c], image_file_name)
        copied_image = os.path.join(copy_to, image_file_name)
        shutil.copy(current_image, copied_image)
        counter += 1
        if counter >= delta:
            break

    return True


def calculate_recomm_accuracy_multi(recomm_list, recomm_list_name, vc_list_name):
    if recomm_list:
        counter = 0
        for image_file_name, _, _ in recomm_list:
            if image_file_name[:len(recomm_list_name)] == recomm_list_name:
                counter += 1
        print(f'The accuracy of {vc_list_name} is: {round(np.divide(counter, len(recomm_list)), 2)}\t\t count {len(recomm_list)}')
    else:
        print(f'The accuracy of {vc_list_name} is: None --- Noting in the list')