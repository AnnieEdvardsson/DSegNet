import os
import numpy as np
import cv2

from deployment.preprocessors import InstanceToOneHot, OneHotEncoding


def generate_evaluation_batches(input_path, output_path, batch_size, input_shape, list_classes, nbr_classes):

    # create batch_name by taking random names,
    batch_list = os.listdir(input_path)
    if batch_size == "all":
        batch_size = len(batch_list)
        batch_indices = np.arange(0, batch_size)
    else:
        batch_indices = np.random.randint(low=0, high=len(batch_list), size=batch_size)

    # batch_img and batch_labels by reading corresponding images and labels. feed them into model
    batch_img = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], 3))
    batch_label = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], nbr_classes))
    preprocessor_inst = InstanceToOneHot(class_order=list_classes)
    preprocessor_one_hot = OneHotEncoding(total_number_classes=nbr_classes)

    for i in range(batch_size):
        img_name = batch_list[batch_indices[i]]
        label_name = batch_list[batch_indices[i]].split('.')[0] + '.png'
        image = cv2.imread(os.path.join(input_path, img_name))
        label = cv2.imread(os.path.join(output_path, label_name), 0)

        resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))
        resized_label = cv2.resize(label, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)
        index_based_label, _ = preprocessor_inst.pre_process(resized_label)
        one_hot_label, _ = preprocessor_one_hot.pre_process(index_based_label)
        batch_img[i] = resized_image
        batch_label[i] = one_hot_label

    return batch_img, batch_label


def generate_prediction_batch(input_path, batch_size, input_shape):
    # create batch_name by taking random names
    batch_list = os.listdir(input_path)
    if batch_size == "all":
        batch_size = len(batch_list)
        batch_indices = np.arange(0, batch_size)
    else:
        batch_indices = np.random.randint(low=0, high=len(batch_list), size=batch_size)

    # batch_img and batch_labels by reading corresponding images and labels. feed them into model
    batch_img = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], 3))

    for i in range(batch_size):
        img_name = batch_list[batch_indices[i]]
        image = cv2.imread(os.path.join(input_path, img_name))
        resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))
        batch_img[i] = resized_image

    return batch_img
