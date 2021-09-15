import os

# Parameters
iteration = 0


# Hyper-parameters
actual_num_images_per_class = 476
num_images_per_class = 300          # Number of images per class of the training dataset. Should be dividable by 3

num_class = 2
seed = 666
alpha = 0.2
image_size = 64
epochs = 300
batch_size = 100
epoch_size = 400
num_classes_per_batch = 2
learning_rate_ = 0.1
weight_decay = 0.001
learning_rate_decay_epochs = 100
learning_rate_decay_factor = 1.0
moving_average_decay = 0.9999
optimizer = 'ADAGRAD'
model_def = 'Utils.vgg_small'


# Path
data_name = 'Caltech-256'  # {'Forceps', 'kvasir', 'Caltech-256', 'AL_ImageNet'}

data_dir = f'../../../Data/{data_name}/initial'  # initial

data_dir_search = f'../../../Data/{data_name}/search'
output_dir = f'Output/{data_name}/LFSM_models/'
models_base_dir = os.path.join(output_dir, f'iter_{iteration}')
pretrained_model_dir = os.path.join(output_dir, f'iter_{iteration - 1}')

# Create the model directory if it doesn't exist
if not os.path.isdir(models_base_dir):
    os.makedirs(models_base_dir)

print('Model directory: %s' % models_base_dir)
print('Pre-trained model: %s' % os.path.expanduser(pretrained_model_dir))


if not os.path.isdir(os.path.join(output_dir, 'Results')):
    os.makedirs(os.path.join(output_dir, 'Results'))

search_feature = os.path.join(output_dir, 'Results', 'search_feature.csv')
search_name = os.path.join(output_dir, 'Results', 'search_name.csv')
train_feature = os.path.join(output_dir, 'Results', 'train_feature.csv')
train_name = os.path.join(output_dir, 'Results', 'train_name.csv')
similarity_matrix = os.path.join(output_dir, 'Results', 'similarity_matrix.csv')
similarity_name = os.path.join(output_dir, 'Results', 'similarity_name.csv')
