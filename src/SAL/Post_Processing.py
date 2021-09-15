import os
import csv
import shutil

import src.SAL.settings as settings

f = open(settings.search_name)
reader = csv.reader(f)
name_search = [i[0] for i in reader]

class_ = 1  # {1, 2}
top_k = 10
save_dir = os.path.join(settings.output_dir, 'Results', 'Output_Images', f'{class_}')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


f = open(os.path.join(settings.output_dir, 'Results', f'search_similarity_dist_for_class_{class_}.csv'))
reader = csv.reader(f)
similarity_dist = [(int(i[0]), float(i[1])) for i in list(reader) if len(i) > 0]
similarity_dist = sorted(similarity_dist, key=lambda x: x[1], reverse=True)
for idx, i in enumerate(similarity_dist[:top_k]):
    current_image = f'{settings.data_dir_search}/search/{name_search[i[0]]}.jpg'
    copied_image = f'{save_dir}/{idx}_{name_search[i[0]]}.jpg'
    shutil.copy(current_image, copied_image)
