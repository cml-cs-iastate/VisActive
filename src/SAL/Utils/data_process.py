import os
import cv2
import numpy as np

from src.SAL.Utils.facenet_sim_v2 import to_rgb


class ImageClass:
    """
    Stores the paths to images for a given class
    """
    def __init__(self, name, image_paths, file_name):
        self.name = name
        self.image_paths = image_paths
        self.file_name = file_name

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)


def get_dataset(paths):
    """
    """
    dataset = list()
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        num_classes = len(classes)
        for i in range(num_classes):
            class_name = classes[i]
            face_dir = os.path.join(path_exp, class_name)
            if os.path.isdir(face_dir):
                images = os.listdir(face_dir)
                images.sort()
                for img in images:
                    print(img)
                    image_paths = os.path.join(face_dir, img)
                    dataset.append(ImageClass(class_name, image_paths, img))
    return dataset


def load_data(dataset, batch_index, batch_size, class_num, image_size):
    """
    """
    num_dataset = np.size(dataset, 0)

    j = batch_index * batch_size % num_dataset

    if j+batch_size <= num_dataset:
        batch = dataset[j:j+batch_size]
    else:
        batch = dataset[j:j+batch_size]

    num_samples = np.size(batch, 0)
    images = np.zeros((num_samples, image_size, image_size, 3))
    labels = np.zeros((num_samples, class_num))
    label_matrix = np.identity(class_num)

    for i in range(num_samples):
        img = cv2.imread(batch[i].image_paths)
        if img.ndim == 2:
            img = to_rgb(img)
        img = cv2.resize(img, (image_size, image_size))
        img = img / 255
        images[i, :, :, :] = img
        labels[i, :] = label_matrix[int(batch[i].name)-1, :]

    return images, labels


def load_data_v2(dataset, index, image_size):
    """
    """
    img = cv2.imread(dataset[index].image_paths)
    img = cv2.resize(img, (image_size, image_size))
    images = np.zeros((1, image_size, image_size, 3))
    images[0, :, :, :] = img
    return images
