import os
import csv
import shutil

import src.SAL.settings as settings

# Path
data_name = 'Retroflexion'
test_data_dir = f'../../Data/{data_name}/test'

save_dir = os.path.join(settings.output_dir, 'Results', 'Incorrectly_Classified_Images')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

f = open('incorrectly_classified_images.csv')
reader = csv.reader(f)
image_file_path = [i[0] for i in reader]

for idx, i in enumerate(image_file_path):
    current_image = test_data_dir + "/" + i
    copied_image = save_dir + "/" + i.split("\\")[1]
    shutil.copy(current_image, copied_image)
