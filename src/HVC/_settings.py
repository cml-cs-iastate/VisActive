import os
import tensorflow as tf

from src.HVC.NeuralNetworks.hvc_vgg import hvc_vgg

print(f'TensorFlow Version: {tf.__version__}')

# ----------------------------------------------------------------------------------------------------------------------
# Data

iteration = 'baseline'

dataset = 'AL_ImageNet'  # {'Forceps', 'EMIS', 'kvasir', 'Caltech-256', 'AL_ImageNet'}

data_path = '../../../Data/'

training_path = os.path.join(data_path, dataset, f'iteration_{iteration}')
val_path = os.path.join(data_path, dataset, 'test')
testing_path = os.path.join(data_path, dataset, 'test')
search_path = os.path.join(data_path, dataset, 'search')

num_classes = 2


# ----------------------------------------------------------------------------------------------------------------------
# Model hyper-parameters

batch_size = 16
epochs = 50
image_shape = 64
drop_out = 0.2
weight_decay = 0.1
learning_rate = 0.0001
plot_model = False


# ----------------------------------------------------------------------------------------------------------------------
# HVC model hyper-parameters

neural_network = hvc_vgg
hvc_epochs = 50
training_epochs = 5
vc_per_class = 20
num_vc = num_classes * vc_per_class
p_h = 1
p_w = 1
p_d = 128
train_cnn_layers_after = 30
train_fully_after = 40
pushing = 10


# ----------------------------------------------------------------------------------------------------------------------
# Active learning hyper-parameters

delta = 111
delta_not_sure = 100
delta_not_seen = 100
alpha = 1 / vc_per_class
beta = 1.0
gama = 4

# ----------------------------------------------------------------------------------------------------------------------
# Model path
output_directory = os.path.join('Output', dataset, f'iteration_{iteration}')
output_search_directory = os.path.join(output_directory, 'Search_results')

model_path = os.path.join(output_directory, 'hvc_model.h5')
base_model_path = os.path.join(output_directory, 'baseline_model.h5')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if not os.path.exists(output_search_directory):
    os.makedirs(output_search_directory)
