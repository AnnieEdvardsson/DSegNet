import random
import os
import csv
import pandas as pd

###########################################################

dataset_vec = ['KITTI', 'CityScapes', 'BDD10k']

for dataset in dataset_vec:

    input_path = "/MLDatasetsStorage/exjobb/" + dataset + "/images/train"
    output_path = "/MLDatasetsStorage/exjobb/" + dataset + "/RandomBatches.cvs"

    img_list = os.listdir(input_path)
    img_indices = list(range(len(img_list)))

    epochs = range(1, 101)
    epochs_list = []

    raw_data = {}
    for i in epochs:
        # Shuffle vector
        random.shuffle(img_indices)

        # Save shuffled vector in dataframe
        raw_data[str(i)] = img_indices
        epochs_list.append(str(i))

    df = pd.DataFrame(raw_data, columns=epochs_list)
    print(df["2"].values)
    df.to_csv(output_path)

