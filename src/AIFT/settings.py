import os
import tensorflow as tf

from src.VisActive.NeuralNetworks.hvc_vgg import hvc_vgg


print(f'TensorFlow Version: {tf.__version__}')
print(tf.config.list_physical_devices('GPU'))


# ----------------------------------------------------------------------------------------------------------------------

# Iterations
iteration = 6
# TODO Change the dataset name and total number of search images for each dataset
dataset = 'Caltech_256-AIFT'
search_images_to_check = 20400                                              # Number of images in the search space to check. Should be total images in search folder

rc = {'Force': '001', 'kvasi': '7_p', 'Calte': '001', 'CIFAR': 'aut'}       # Value must be 3 letters
rare_class_file_name = rc[dataset[:5]]

dr = {'Force': 476, 'kvasi': 100, 'Calte': 111, 'CIFAR': 500}               # key must be five letters
delta_rare = dr[dataset[:5]]


# AIFT parameters
dataset_classes = ['001', '002']                                            # First class must be rare class
number_of_candidates_needed = delta_rare * 2                                # Number of images to copy search into training folder for both classes
total_iterations = 6
start_iteration = 1
settings_file_path = 'settings.py'
settings_iteration_line = 13

# ----------------------------------------------------------------------------------------------------------------------
# Data

data_path = '../../../Data/AL/'
test_search_data = dataset.split('-')[0]
training_path = os.path.join(data_path, dataset, f'iteration_{iteration}')
val_path = os.path.join(data_path, test_search_data, 'test')
testing_path = os.path.join(data_path, test_search_data, 'test')
search_path = os.path.join(data_path, test_search_data, 'search')
l_ = search_path + '/search/'

num_classes = 2
image_shape = 32 if dataset[:5] == 'CIFAR' else 64
patches_batch_size = 5


# ----------------------------------------------------------------------------------------------------------------------
# Model hyper-parameters

neural_network = hvc_vgg
epochs = 30
drop_out = 0.5
batch_size = 16

# ----------------------------------------------------------------------------------------------------------------------
# Model path
base_model_path = os.path.join('Output', dataset, 'baseline_model.h5')
