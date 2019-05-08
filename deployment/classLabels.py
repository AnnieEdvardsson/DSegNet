import numpy as np


def update_label_classes(img, label_to_new_label):


    # Initialize new label image
    img_height, img_width = img.shape
    new_label = np.zeros((img_height, img_width))

    for key in label_to_new_label.keys():
        new_label[img == key] = label_to_new_label[key]

    return new_label
