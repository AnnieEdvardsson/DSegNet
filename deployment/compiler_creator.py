'''

    Compiler:
    Define optimizer and metrics (IoU)

'''

from keras import optimizers
from keras import backend as K
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
import numpy as np



class OptimizerCreator(object):
    def __init__(self, OPTIMIZER,  learning_rate, momentum):
        self.OPTIMIZER = OPTIMIZER
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = None


    def pick_opt(self):
        if self.OPTIMIZER == 'sgd':
            return self.create_sgd()

        if self.OPTIMIZER == 'adadelta':
            return self.create_adadelta()

        if self.OPTIMIZER == 'adam':
            return self.create_adam()

        else:
            raise NameError('Given optimizer is not implemented.')

    def create_sgd(self):
        if self.learning_rate is None: self.learning_rate = 0.01
        if self.momentum is None: self.momentum = 0.9

        return optimizers.SGD(lr=self.learning_rate, momentum=self.momentum), self.learning_rate, self.momentum

    def create_adadelta(self):
        if self.learning_rate is None: self.learning_rate = 1
        if self.rho is None: self.rho = 0.95
        self.momentum = None

        return optimizers.Adadelta(lr=self.learning_rate, rho=self.rho), self.learning_rate, self.momentum

    def create_adam(self):
        if self.learning_rate is None: self.learning_rate = 0.001
        self.momentum = None

        return optimizers.Adam(lr=self.learning_rate), self.learning_rate, self.momentum

# def Mean_iou(y_true, y_pred):
#     NUM_CLASSES = K.int_shape(y_pred)[-1]
#     score, up_opt = mean_iou(y_true, y_pred, NUM_CLASSES)
#     K.get_session().run(tf.local_variables_initializer())
#     with tf.control_dependencies([up_opt]):
#         score = tf.identity(score)
#     return score


def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true, axis=-1), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred, axis=-1), label), K.floatx())

    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection

    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    output = K.switch(K.equal(union, 0), 0.0, intersection / union)
    presence = K.switch(K.equal(union, 0), 0.0, 1.0)

    return output, presence

def Mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    total_presence = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        class_iou, presence = iou(y_true, y_pred, label)
        total_iou = total_iou + class_iou
        total_presence = total_presence + presence
    # divide total IoU by number of labels to get mean IoU
    return total_iou / total_presence


def Class_iou(label: int):
    def iou_label(y_true, y_pred):
        class_iou, _ = iou(y_true, y_pred, label)
        return class_iou
    return iou_label



