import os
import tensorflow as tf

from src.VisActive.NeuralNetworks.hvc_vgg import hvc_vgg
from src.VisActive.NeuralNetworks.ResNet18 import ResNet18

print(f'TensorFlow Version: {tf.__version__}')
print(tf.config.list_physical_devices('GPU'))


# ----------------------------------------------------------------------------------------------------------------------
# Iterations

iteration = 'baseline'                                     # [0, 1, 2, 3, 4, 5, 'baseline']
dataset = 'kvasir'                                       # {'Forceps', 'kvasir', 'Caltech_256', 'CIFAR_10'}

rc = {'Force': '001', 'kvasi': '7_p', 'Calte': '001', 'CIFAR': 'aut'}       # Value must be 3 letters
rare_class_file_name = rc[dataset[:5]]
class_file_name = {0: 'air', 1: 'aut', 2: 'bir', 3: 'cat', 4: 'dee', 5: 'dog', 6: 'fro', 7: 'hor', 8: 'shi', 9: 'tru'}
class_names = {0: '01_airplane', 1: '02_automobile', 2: '03_bird', 3: '04_cat', 4: '05_deer', 5: '06_dog', 6: '07_frog', 7: '08_horse', 8: '09_ship', 9: '10_truck'}

nc = {'Force': 2, 'kvasi': 8, 'Calte': 10, 'CIFAR': 10}                     # key must be five letters
num_classes = nc[dataset[:5]]

dr = {'Force': 476, 'kvasi': 100, 'Calte': 111, 'CIFAR': 200}               # key must be five letters
delta_ = dr[dataset[:5]]
delta_per_set = int(delta_ / 4)

rare_class = 0              # The index of the rare class


# ----------------------------------------------------------------------------------------------------------------------
# Data

data_path = '../../../Data/kvasir'
test_search_data = dataset.split('-')[0]
training_path = data_path  # os.path.join(data_path, dataset, f'iteration_{iteration}')
val_path = data_path  # os.path.join(data_path, test_search_data, 'test')
testing_path = data_path  # os.path.join(data_path, test_search_data, 'test')
search_path = os.path.join(data_path, dataset, 'search')

# For binary
# num_classes = 2
image_shape = 32 if dataset[:5] == 'CIFAR' else 64


# ----------------------------------------------------------------------------------------------------------------------
# Model hyper-parameters

neural_network = ResNet18
epochs = 100
drop_out = 0.0
batch_size = 32
weight_decay = 0.0005
learning_rate = 0.0001
momentum = 0.9
verbose_ = 1
plot_model = False


# ----------------------------------------------------------------------------------------------------------------------
# VisActive model hyper-parameters

neural_network_hvc = hvc_vgg
hvc_epochs = 100
training_epochs = 5
vc_per_class = 10
num_vc = num_classes * vc_per_class
p_h = 1
p_w = 1
p_d = 128
train_softmax_after = 30
train_cnn_layers_after = hvc_epochs
pushing = 15


# ----------------------------------------------------------------------------------------------------------------------
# Active learning hyper-parameters

alpha = 0.1
beta = 1.0
gama = 4

# ----------------------------------------------------------------------------------------------------------------------
# Model path
output_directory = os.path.join('Output', dataset, f'iteration_{iteration}')
output_search_directory = os.path.join(output_directory, 'Search_results')

hvc_model_path = os.path.join(output_directory, 'hvc_model.h5')
pre_train_model_path = os.path.join(output_directory, 'pre_train_model.h5')
base_model_path = os.path.join(output_directory, 'baseline_model.h5')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if not os.path.exists(output_search_directory):
    os.makedirs(output_search_directory)
