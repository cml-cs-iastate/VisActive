import os
import csv
import numpy as np
import pandas as pd

import src.SAL.settings as settings


def main():
    """
    """

    matrix = np.genfromtxt(settings.similarity_matrix, delimiter=",")

    f = open(settings.search_name)
    reader = csv.reader(f)
    name_search = list(reader)

    result_fyle = open(settings.similarity_name, 'a+')
    wr = csv.writer(result_fyle, dialect='excel')
    for item in name_search:
        wr.writerow(item)
    
    csv_train_v1 = np.genfromtxt(settings.train_feature, delimiter=",")
    # print(f'csv_train_v1: \n{csv_train_v1} \nwith shape:  {np.shape(csv_train_v1)}')
    print('load training set successfully!')
    csv_search = np.genfromtxt(settings.search_feature, delimiter=",")
    # print(f'csv_search: \n{csv_search} \nwith shape:  {np.shape(csv_search)}')
    print('load search set 1 successfully!')
    print(np.size(csv_search, 0))
    print(np.size(csv_search, 1))
    center = np.zeros((2, 128))

    number = np.array([0, settings.actual_num_images_per_class, settings.actual_num_images_per_class * 2])

    for i in range(0, 2):
        center[i, :] = np.mean(csv_train_v1[number[i]:number[i+1], 1:129], axis=0)

    print(f'center: \n{center} \nwith shape: {np.shape(center)}')
    print(np.size(center))

    for i in range(0, 2):
        f = open(os.path.join(settings.output_dir, 'Results', f'search_similarity_dist_for_class_{str(i + 1)}.csv'), 'a+')
        dists = np.zeros((np.size(csv_search, 0), 2))

        for j in range(0, np.size(csv_search, 0)):
            if j % 1000 == 1:
                print(j)

            print(np.shape(csv_search[j, 1:129]), np.shape(matrix[:, 1:129]), np.shape(center[i, 0:128]))

            dists[j, 0] = np.sum(np.multiply(np.dot(csv_search[j, 1:129], matrix[:, 1:129]), center[i, 0:128]))
            dists[j, 1] = j

        df = pd.DataFrame(dists[:, 0])
        df.to_csv(f, header=False)
        f.close()


if __name__ == '__main__':
    main()
