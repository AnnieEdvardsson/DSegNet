import tensorflow as tf
import argparse
import os
from hyperparameters import *
from builder import train_model, predict_model, evaluate_model, train_eval_model, \
    get_accurcay_distribution, segment_image
from deployment.compiler_creator import OptimizerCreator

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--cuda', dest='cuda', type=str, default='1', help='The CUDA we want to use, 0 or 1')
parser.add_argument('--model', dest='model', type=str, default='SegNetModel', help='The model we want to run')
parser.add_argument('--dataset', dest='dataset', type=str, default='CityScapes', help='The datset we want to use, if segment - the imgae "path"')
parser.add_argument('--datasetWeights', dest='datasetWeights', type=str, default='Same', help='The dataset weights')
parser.add_argument('--epochs', dest='EPOCHS', type=int, default=2, help='The number of epochs')
parser.add_argument('--train_batches', dest='TRAIN_BATCHES', type=int, default=1, help='The number of batches in train')
parser.add_argument('--predict_batches', dest='PREDICT_BATCHES', type=int, default=1, help='The number images to predict. Write "all" for all images in folder.')
parser.add_argument('--evaluate_batch', dest='EVALUATE_BATCH', type=int, default=1, help='The number of images used to evaluate. Write "all" for all images in folder.')
parser.add_argument('--optimizer', dest='OPTIMIZER', type=str, default='sgd', help='The type of optimizer used during training')
parser.add_argument('--class_group', dest='class_group', type=str, default='all', help='The way the classes are clustred together')
parser.add_argument('--task', dest='task', type=str, default='all', help='train/predict/evaluate/all')
parser.add_argument('--learning_rate', dest='learning_rate', type=int, default=None, help='The learning rate during training. None=defualt')
parser.add_argument('--momentum', dest='momentum', type=int, default=None, help='Momentum during training. None=default')
parser.add_argument('--indices', dest='indices', type=bool, default=False, help='Maxpooling indices')
parser.add_argument('--bestweights', dest='bestweights', type=str, default="True", help='If avaliabe choose weights from WorthyWeights, works if we dont run train')
parser.add_argument('--generate_test', dest='generate_test', type=bool, default=False, help='Run if you want to predict images from folder testImages')



args = parser.parse_args()

# forces tensorflow to run on GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

# Redefine the variable name for easier access
model = args.model
dataset = args.dataset
TRAINING_EPOCHS = args.EPOCHS
TRAINING_BATCHES = args.TRAIN_BATCHES
PREDICTION_BATCHES = args.PREDICT_BATCHES
EVALUATION_BATCHES = args.EVALUATE_BATCH
OPTIMIZER = args.OPTIMIZER
class_group = args.class_group
task = args.task
learning_rate = args.learning_rate
momentum = args.momentum
indices = args.indices
bestweights = args.bestweights
generate_test = args.generate_test
datasetWeights = args.datasetWeights

if datasetWeights == 'Same':
    datasetWeights = dataset

hyperdict, list_classes, label_to_color, class_names, class_names_dist = get_hyperparameter(model, dataset)


def main():
    print('\n\n\n\n\n\n\n\n\n###############################################################################')
    print('Running "{}" with model {} and dataset {} on CUDA {}.'.format(task, model, dataset, args.cuda))

    # Choose weights. Normal or weights in WorthyWeights.
    # Conditions: bestweights arg is True. No training is running. WorthyWeight file exists.
    weight_flag = False
    if (bestweights == "True" and task not in ['train', 'all', 'traineval']):
        print('Use weights from WorthyWeights folder')
        hyperdict['WEIGHT_PATH'] = hyperdict['BEST_WEIGHTS_PATH']
        weight_flag = True

    if generate_test:
        hyperdict['PREDICTION_IMAGES_PATH'] = "/MLDatasetsStorage/exjobb/testImages"
        hyperdict['PREDICTION_SAVE_PATH'] = "/MLDatasetsStorage/exjobb/prediction/testImages"
    print('###############################################################################\n')

    # Create optimizer
    opt_instance = OptimizerCreator(OPTIMIZER, args.learning_rate, args.momentum)
    opt, learning_rate, momentum = opt_instance.pick_opt()

    if task in ['train', 'all']:
        train_model(model=model,
                    INPUT_SHAPE=hyperdict['INPUT_SHAPE'],
                    dataset=dataset,
                    TRAINING_IMAGES_PATH=hyperdict['TRAINING_IMAGES_PATH'],
                    TRAINING_LABELS_PATH=hyperdict['TRAINING_LABELS_PATH'],
                    TRAINING_TEXTFILE_PATH=hyperdict['TRAINING_TEXTFILE_PATH'],
                    TRAINING_DEPTH_PATH=hyperdict['TRAINING_DEPTH_PATH'],
                    WEIGHTS_PATH=hyperdict['WEIGHT_PATH'],
                    TRAINING_BATCHES=TRAINING_BATCHES,
                    TRAINING_EPOCHS=TRAINING_EPOCHS,
                    OPTIMIZER=OPTIMIZER,
                    opt=opt,
                    list_classes=list_classes,
                    learning_rate=learning_rate,
                    momentum=momentum,
                    indices=indices,
                    class_names=class_names
                    )

    if task in ['traineval']:
        train_eval_model(model=model,
                         INPUT_SHAPE=hyperdict['INPUT_SHAPE'],
                         dataset=dataset,
                         TRAINING_IMAGES_PATH=hyperdict['TRAINING_IMAGES_PATH'],
                         TRAINING_LABELS_PATH=hyperdict['TRAINING_LABELS_PATH'],
                         TRAINING_TEXTFILE_PATH=hyperdict['TRAINING_TEXTFILE_PATH'],
                         TRAINING_DEPTH_PATH=hyperdict['TRAINING_DEPTH_PATH'],
                         WEIGHTS_PATH=hyperdict['WEIGHT_PATH'],
                         TRAINING_BATCHES=TRAINING_BATCHES,
                         TRAINING_EPOCHS=TRAINING_EPOCHS,
                         OPTIMIZER=OPTIMIZER,
                         opt=opt,
                         list_classes=list_classes,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         indices=indices,
                         EVALUATION_IMAGES_PATH=hyperdict['EVALUATION_IMAGES_PATH'],
                         EVALUATION_LABELS_PATH=hyperdict['EVALUATION_LABELS_PATH'],
                         EVALUATION_TEXTFILE_PATH=hyperdict['EVALUATION_TEXTFILE_PATH'],
                         EVALUATION_DEPTH_PATH=hyperdict['EVALUATION_DEPTH_PATH'],
                         EVALUATION_BATCHES=EVALUATION_BATCHES,
                         class_names=class_names)

    if task in ['predict', 'all']:
        predict_model(model=model,
                      dataset=dataset,
                      INPUT_SHAPE=hyperdict['INPUT_SHAPE'],
                      PREDICTION_IMAGES_PATH=hyperdict['PREDICTION_IMAGES_PATH'],
                      PREDICTION_SAVE_PATH=hyperdict['PREDICTION_SAVE_PATH'],
                      PREDICTION_TEXTFILE_PATH=hyperdict['PREDICTION_TEXTFILE_PATH'],
                      WEIGHTS_PATH=hyperdict['WEIGHT_PATH'],
                      list_classes=list_classes,
                      indices=indices,
                      label_to_color=label_to_color,
                      PREDICTION_BATCHES=PREDICTION_BATCHES,
                      weight_flag=weight_flag
                      )

    if task in ['segment']:
        segment_image(dataset_weights=datasetWeights,
                      dataset_images=dataset,
                      INPUT_SHAPE=hyperdict['INPUT_SHAPE'],
                      SEGMENT_IMAGES_PATH=hyperdict['SEGMENT_IMAGES_PATH'],
                      SEGMENT_SAVE_PATH=hyperdict['SEGMENT_SAVE_PATH'],
                      SEGMENT_DEPTH_PATH=hyperdict['SEGMENT_DEPTH_PATH'],
                      list_classes=list_classes,
                      indices=indices,
                      label_to_color=label_to_color,
                      weight_flag=weight_flag
                      )

    if task in ['evaluate', 'all']:
        evaluate_model(model=model,
                       dataset=dataset,
                       INPUT_SHAPE=hyperdict['INPUT_SHAPE'],
                       EVALUATION_IMAGES_PATH=hyperdict['EVALUATION_IMAGES_PATH'],
                       EVALUATION_LABELS_PATH=hyperdict['EVALUATION_LABELS_PATH'],
                       EVALUATION_TEXTFILE_PATH=hyperdict['EVALUATION_TEXTFILE_PATH'],
                       WEIGHTS_PATH=hyperdict['WEIGHT_PATH'],
                       opt=opt,
                       list_classes=list_classes,
                       indices=indices,
                       EVALUATION_BATCHES=EVALUATION_BATCHES,
                       class_names=class_names,
                       EVALUATION_DEPTH_PATH=hyperdict['EVALUATION_DEPTH_PATH']
                       )

    if task in ['dist']:
        get_accurcay_distribution(model=model,
                                   dataset=dataset,
                                   INPUT_SHAPE=hyperdict['INPUT_SHAPE'],
                                   EVALUATION_IMAGES_PATH=hyperdict['EVALUATION_IMAGES_PATH'],
                                   EVALUATION_LABELS_PATH=hyperdict['EVALUATION_LABELS_PATH'],
                                   DISTRUBUTATION_TEXTFILE_PATH=hyperdict['DISTRUBUTATION_TEXTFILE_PATH'],
                                   WEIGHTS_PATH=hyperdict['WEIGHT_PATH'],
                                   opt=opt,
                                   list_classes=list_classes,
                                   indices=indices,
                                  READABLE_DISTRUBUTATION_TEXTFILE_PATH=hyperdict['READABLE_DISTRUBUTATION_TEXTFILE_PATH'],
                                  class_names_dist=class_names_dist,
                                  EVALUATION_DEPTH_PATH=hyperdict['EVALUATION_DEPTH_PATH']
                                  )

    if task not in ['train', 'predict', 'segment', 'evaluate', 'traineval', 'dist', 'all']:
        raise ValueError('Define a correct task - (train/traineval/segment/predict/evaluate/dist/all).')


if __name__ == '__main__':
    main()


