""" Function that transforms one hot encoded into grayscale picture. 
 Inputs: 
  one_hot_softmax : ONE image(a numpy array) encoded with a probability distribution over classes
  - with shape (x_dim, y_dim, nbr_classes).
  class_list : a list of numbers - of length nbr_classes - to represent each class encoding in gray scale.
  (optionally: a probability threshold above which the probability is considered enough - generally consistently above 
  random chance level - to determine a preference of the network.

 Outputs: 
  a numpy arrays of dimensions (x_dim, y_dim) - a grayscale image. 
  
  
  this class_list has to be of the type [0, other class, other_class]
"""
from typing import Tuple
import numpy as np


class OneHotToInstanceConverter(object):
    # check right shape and number of classes
    def __init__(self, class_list: list, input_shape: Tuple, prob_threshold=None):
        self.class_list = class_list
        self.x_dim = input_shape[0]
        self.y_dim = input_shape[1]
        self.nbr_channels = input_shape[2]
        self.prob_threshold = prob_threshold

        if self.nbr_channels != len(self.class_list):
            raise AssertionError("Number of classes listed should be classes recognized. the rest is encoded as 0!")

    # set threshold probability to determine a class is actually consistently preferred over the others
        if self.prob_threshold is None:
            self.prob_threshold = (1 / self.nbr_channels)

    def convert_to_images(self, one_hot_softmax):
        instances_grayscale = np.zeros(shape=(self.x_dim, self.y_dim))

    # For every pixel check that the highest probability is higher than prob_threshold.
    # If not, then it will be encoded as undecided class. 0
    # cannot set to 0 everything that is below prob_threshold
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                m = np.argmax(one_hot_softmax[i, j, :])
                if one_hot_softmax[i, j, m] > self.prob_threshold:
                    instances_grayscale[i, j] = self.class_list[m]

        return instances_grayscale


def get_images_from_softmax(softmax_batch, class_list):
    x_dim = softmax_batch.shape[1]
    y_dim = softmax_batch.shape[2]
    batch_size = softmax_batch.shape[0]
    nbr_channels = softmax_batch.shape[3]

    all_images = np.zeros(shape=(batch_size, x_dim, y_dim))
    converter = OneHotToInstanceConverter(class_list, (x_dim, y_dim, nbr_channels))
    for i in range(batch_size):
        all_images[i] = converter.convert_to_images(softmax_batch[i])

    return all_images