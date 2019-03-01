""" Script to check results of the training. The script should include a model, a set of weights and 
reference either to a batch of pictures or to a folder with pictures and predictions
assume you have a USB drive with models.py and the weights.hdf5. maybe extra script with the preprocessor needed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
import time
import argparse
from deployment.models import SegNetModel
from deployment.data_readers import generate_evaluation_batches, generate_prediction_batch
from deployment.postprocessors import get_images_from_softmax
from deployment.data_writer import save_predictions, create_video, save_images_with_predictions
from hyperparameters import *

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--Comp', dest='Comp', type=str,
                    default='ML', help='Which computer, ML/AE/MT - chooses path to weights')

args = parser.parse_args()
Comp = args.Comp

SEGNET_SAVED_WEIGHTS = {"ML": "/WeightModels/exjobb_SecretStuff_AnnieAndMartin/segnet_weights/Segnet_perceptron_general_gta_swap_weights-lowest_loss.hdf5",
                        "AE": "C:/Users/s26915/Documents/SegNet/weights/Segnet_perceptron_general_gta_swap_weights-lowest_loss.hdf5",
                        "MT": "weights/Segnet_perceptron_general_gta_swap_weights-lowest_loss.hdf5"}



# get images to predict
input_images = generate_prediction_batch(input_path=SEGNET_PREDICTION_IMAGES_ROOT_PATH[task],
                                         batch_size=SEGNET_PREDICTION_BATCH_SIZE[task],
                                         input_shape=SEGNET_INPUT_SHAPE)


# create model
model_instance = SegNetModel(shape=SEGNET_INPUT_SHAPE,
                             num_classes=nbr_classes[task],
                             pre_trained_encoder=True,
                             segned_indices=True,
                             weights=SEGNET_SAVED_WEIGHTS[Comp],
                             load_weights_by_name=True)

segnet = model_instance.create_model()
# problems with loading the network because unknown layers are used.
# segnet = keras.models.load_model(SEGNET_SAVED_MODEL)

# predictions
start_prediction_time = time.clock()
predictions_softmax = segnet.predict(input_images)
end_prediction_time = time.clock()
nbr_batch = len(input_images)
prediction_time_per_image = (end_prediction_time - start_prediction_time) / nbr_batch
new_list = [0]
new_list.extend(list_classes[task])
predictions = get_images_from_softmax(predictions_softmax, new_list)
print("prediction time per image in a batch of " + str(nbr_batch) + ": ")
print(prediction_time_per_image)


# save predictions, create video
save_predictions(predictions, SEGNET_SAVE_PATH[task])
save_images_with_predictions(input_images, predictions, SEGNET_SAVE_PATH[task])


# handle images for evaluation
batch_images, batch_labels = generate_evaluation_batches(input_path=SEGNET_EVALUATION_IMAGES_ROOT_PATH[task],
                                                         output_path=SEGNET_EVALUATION_LABELS_ROOT_PATH[task],
                                                         batch_size=SEGNET_EVALUATION_BATCH_SIZE[task],
                                                         input_shape=SEGNET_INPUT_SHAPE,
                                                         list_classes=list_classes[task],
                                                         nbr_classes=nbr_classes[task])

# compile and evaluate
segnet.compile(optimizer="Adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
start_evaluation_time = time.clock()
evaluation = segnet.evaluate(batch_images, batch_labels)
end_evaluation_time = time.clock()
nbr_batch = len(batch_images)
evaluation_time_per_image = (end_evaluation_time - start_evaluation_time) / nbr_batch
print("evaluation loss and accuracy in provided images, bacth size of " + str(nbr_batch) + ": ")
print(evaluation)
print("evaluation time per image in a batch of " + str(nbr_batch) + ": ")
print(evaluation_time_per_image)

