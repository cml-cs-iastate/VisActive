import os

from src.VisActive.NeuralNetworks.hvc_vgg import hvc_vgg

"""https://github.com/dhaalves/CEAL_keras"""


# ----------------------------------------------------------------------------------------------------------------------
# Iterations

iteration = 0                               # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'baseline']
dataset = 'kvasir-CEAL'                   # VisActive {'Forceps', 'kvasir', 'Caltech_256', 'CIFAR_10'}

rc = {'Force': '001', 'kvasi': '7_p', 'Calte': '001', 'CIFAR': 'aut'}       # Value must be 3 letters
rare_class_file_name = rc[dataset[:5]]

dr = {'Force': 476, 'kvasi': 100, 'Calte': 111, 'CIFAR': 500}               # key must be five letters
delta_rare = dr[dataset[:5]]

nc = {'Force': 1, 'kvasi': 7, 'Calte': 9, 'CIFAR': 9}                       # key must be five letters
num_common_classes = nc[dataset[:5]]


# ----------------------------------------------------------------------------------------------------------------------
# Data

data_path = '../../../Data/AL/'
test_search_data = dataset.split('-')[0]
training_path = os.path.join(data_path, dataset, f'iteration_{iteration}')
val_path = os.path.join(data_path, test_search_data, 'test')
testing_path = os.path.join(data_path, test_search_data, 'test')
search_path = os.path.join(data_path, test_search_data, 'search')

num_classes = 2
image_shape = 32 if dataset[:5] == 'CIFAR' else 64


# ----------------------------------------------------------------------------------------------------------------------
# Model hyper-parameters

neural_network = hvc_vgg
epochs = 50
drop_out = 0.5
batch_size = 16
weight_decay = 0.001
learning_rate = 0.0001


# ----------------------------------------------------------------------------------------------------------------------
# CEAL hyper-parameters
verbose_ = 0
fine_tunning_interval = 1                                 # Fine-tuning interval
maximum_iterations = 7                                    # Maximum iteration number
initial_annotated_perc = 0.1                              # Initial Annotated Samples Percentage. default: 0.1")
threshold_decay = 0.0033                                  # Threshold decay rate. default: 0.0033")
delta = 0.05                                              # High confidence samples selection threshold. default: 0.05")
uncertain_samples_size = delta_rare * num_classes         # Uncertain samples selection size. default: 1000")
uncertain_criteria = 'lc'                                 # Uncertain selection Criteria: \'rs\'(Random Sampling), \'lc\'(Least Confidence), \'ms\'(Margin Sampling), \'en\'(Entropy). default: lc")
cost_effective = True                                     # whether to use Cost Effective high confidence sample pseudo-labeling. default: True")

# ----------------------------------------------------------------------------------------------------------------------
# Model path
output_directory = os.path.join('Output', dataset, f'iteration_{iteration}')
output_search_directory = os.path.join(output_directory, 'Search_results')
base_model_path = os.path.join(output_directory, 'baseline_model.h5')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if not os.path.exists(output_search_directory):
    os.makedirs(output_search_directory)
