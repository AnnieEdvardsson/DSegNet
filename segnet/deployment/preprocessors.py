import numpy as np
import cv2
from typing import Tuple
from keras.utils import np_utils


class InstanceToOneHot(object):
    """
        Pre-processor that changes instances into index based labels

        :param class_order: list. The list should contain only the classes of the objects we want the network
        to be able to distinguish, everything else will be set to 0. (void, unlabelled, trash cans etc)

        (Can be used also to extract one class from a label, to create task specific labels,
         e.g. road -number 90 in master dataset- for perceptron)
    """

    def __init__(self, class_order: list):
        super().__init__()
        self.nbr_classes = len(class_order)
        self.class_order = class_order

    def pre_process(self, data, flag=False):
        new_data = np.zeros(data.shape)
        for i in range(self.nbr_classes):
            mask_i = data == self.class_order[i]
            new_data[mask_i] = i+1
            flag = False

        #print("old data" + str(data.shape))
        #print("new data" + str(new_data.shape))

        return new_data, flag



class ResizePreProcessor(object):
    """
    Preprocessor that resizes images to the new shape. config parameters: shape. 
    If the preprocessor is applied to an image the Flag has to be set as False. The bilinear interpolation will be used
    If the preprocessor is applied to a label the Flag has to be set as True. The nearest_neighbour interpolation will be used
    """
    def __init__(self, new_shape: Tuple):
        if len(new_shape) != 2:  # not only 3
            raise AssertionError("New shape has to be of type (y_dim, z_dim)")
        else:
            self.new_shape = tuple([new_shape[1], new_shape[0]])

    def pre_process(self, data, flag=False):
        if flag:  # Flag is true. assuming we are dealing with a label
            resized_data = cv2.resize(data, self.new_shape, interpolation=cv2.INTER_NEAREST)
        else:
            resized_data = cv2.resize(data, self.new_shape)
            flag = True  # report that a change geometry modification has been done to the image

        return resized_data, flag



class OneHotEncoding(object):
    """
    Pre-processor that changes integer-valued labels to one-hot encoding, i.e. index based labels

    Assumes tuples in batch are of type (input, label)
    """
    def __init__(self, total_number_classes: int):
        """
        :param logger:
        :param data_handling_generator_function:
        :param total_number_classes: The total number of classes in the training set
        """
        super().__init__()
        self.total_number_classes = total_number_classes

    def pre_process(self, data, flag=False):
        if np.max(data) >= self.total_number_classes:
            raise ValueError("Undefined behaviour: Class label greater than total number of classes")
        data = np.squeeze(data)
        label_cat = np_utils.to_categorical(data, self.total_number_classes)
        #print("new label_cat: " + str(label_cat))
        #print("new label_cat size: " + str(label_cat.shape))
        new_dimension = [dim for dim in data.shape]
        new_dimension.append(self.total_number_classes)
        new_label = np.reshape(label_cat, tuple(new_dimension)).astype('uint8')

        #print("new label: " + str(new_label))
        #print("new label size: " + str(new_label.shape))
        return new_label, flag


class ModifyLabels(object):
    """
        Pre-processor that changes the pixels in the labels to the defined ones in hyperparameters - label_to_new_label

        :param class_dict: A dictionary that translate originally class to new class
    """

    def __init__(self, class_dict: dict):
        super().__init__()
        self.class_dict = class_dict


    def pre_process(self, img):
        # Initialize new label image
        img_height, img_width = img.shape
        new_label = np.zeros((img_height, img_width))

        for key in self.class_dict.keys():
            new_label[img == key] = self.class_dict[key]

        return new_label
