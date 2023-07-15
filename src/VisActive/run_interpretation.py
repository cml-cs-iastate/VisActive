import os
import numpy as np
import pickle as pkl

import src.VisActive._settings as settings
from tensorflow.keras import models
from src.VisActive.Utils.utils import load_hvc_model
from src.VisActive.SearchingStage.visualize_vc import load_image, vc_interpretation


# Create output directory to store the rectangular images to the disk
save_directory = os.path.join(settings.output_search_directory, f'_hvc_model_interpretation_VC', f'common_class_unseen')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

with open(os.path.join(settings.output_search_directory, f'common_class_unseen.pkl'), 'rb') as f:
    rare_class_common_vc = pkl.load(f)

# Load model
model = load_hvc_model(model_path=settings.hvc_model_path,
                       neural_network=settings.neural_network,
                       num_classes=settings.num_classes,
                       image_size=settings.image_shape,
                       weight_decay=settings.weight_decay,
                       drop_out=settings.drop_out,
                       num_vc=settings.num_vc,
                       p_h=settings.p_h,
                       p_w=settings.p_w,
                       p_d=settings.p_d)

# Build hvc_model to get the VC output
hvc_model = models.Model(inputs=model.input, outputs=model.get_layer('vcl').output)

counter = 0
for image_file_name, activation_score, max_activation_vc_num in rare_class_common_vc:
    image_path = os.path.join(settings.search_path, 'search', image_file_name)

    image = load_image(image_path, 64)
    image = np.reshape(image, [1, 64, 64, 3])

    vc_interpretation(input_x=image,
                      image_file_name=image_file_name,
                      vc_idx=max_activation_vc_num,
                      save_directory=save_directory,
                      hvc_model=hvc_model,
                      counter=counter)
    counter += 1
