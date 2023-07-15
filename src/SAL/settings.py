import os
import tensorflow as tf

print(f'TensorFlow Version: {tf.__version__}')
print(tf.config.list_physical_devices('GPU'))

# ----------------------------------------------------------------------------------------------------------------------

data_name = 'kvasir-SAL'                             # {'Forceps', 'kvasir', 'Caltech-256', 'CIFAR-10'}

# These need to be change with every iteration
iteration = 5                                        # [0, 1, 2, 3, 4, 5, 'baseline']
actual_num_images_per_class = 2095                   # This should be the actual number of images of the rare classes at the current iteration

# ----------------------------------------------------------------------------------------------------------------------
# Data

data_path = '../../../Data/AL/'
training_path = os.path.join(data_path, data_name, f'iteration_{iteration}')
val_path = os.path.join(data_path, data_name, 'test')
testing_path = os.path.join(data_path, data_name, 'test')
data_dir_search = os.path.join(data_path, data_name, 'search')

output_dir = f'Output/{data_name}/LFSM_models/'
models_base_dir = os.path.join(output_dir, f'iter_{iteration}')
pretrained_model_dir = os.path.join(output_dir, f'iter_{iteration - 1}')

# Create the model directory if it doesn't exist
if not os.path.isdir(models_base_dir):
    os.makedirs(models_base_dir)

print('Model directory: %s' % models_base_dir)
print('Pre-trained model: %s' % os.path.expanduser(pretrained_model_dir))

if not os.path.isdir(os.path.join(output_dir, f'Results_{iteration}')):
    os.makedirs(os.path.join(output_dir, f'Results_{iteration}'))

search_feature = os.path.join(output_dir, f'Results_{iteration}', 'search_feature.csv')
search_name = os.path.join(output_dir, f'Results_{iteration}', 'search_name.csv')
train_feature = os.path.join(output_dir, f'Results_{iteration}', 'train_feature.csv')
train_name = os.path.join(output_dir, f'Results_{iteration}', 'train_name.csv')
similarity_matrix = os.path.join(output_dir, f'Results_{iteration}', 'similarity_matrix.csv')
similarity_name = os.path.join(output_dir, f'Results_{iteration}', 'similarity_name.csv')


# ----------------------------------------------------------------------------------------------------------------------

# Hyper-parameters
# These two hyper-parameters change per iteration
dr = {'Force': 476, 'kvasi': 100, 'Calte': 111, 'CIFAR': 500}
delta = dr[data_name[:5]]

rc = {'Force': '001', 'kvasi': '7_p', 'Calte': '001', 'CIFAR': 'aut'}
rare_class_file_name = rc[data_name[:5]]

num_class = 2
seed = 666
alpha = 0.2
image_size = 32 if data_name[:5] == 'CIFAR' else 64
epochs = 100
batch_size = 50
epoch_size = 100
num_classes_per_batch = 2
learning_rate_ = 0.01
weight_decay = 0.001
learning_rate_decay_epochs = 100
learning_rate_decay_factor = 1.0
moving_average_decay = 0.9999
optimizer = 'ADAGRAD'
model_def = 'Utils.vgg_small'
num_images_per_class = int(actual_num_images_per_class/batch_size) * batch_size


