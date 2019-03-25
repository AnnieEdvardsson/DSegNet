import tensorflow as tf
import argparse
import os
from hyperparameters import *
from builder import train_model, predict_model, evaluate_model

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--cuda', dest='cuda', type=str, default='1', help='The CUDA we want to use, 0 or 1')
parser.add_argument('--model', dest='model', type=str, default='SegNetModel', help='The model we want to run')
parser.add_argument('--dataset', dest='dataset', type=str, default='CityScapes', help='The datset we want to use')
parser.add_argument('--epochs', dest='EPOCHS', type=int, default=2, help='The number of epochs')
parser.add_argument('--batches', dest='BATCHES', type=int, default=4, help='The number of batches')
parser.add_argument('--optimizer', dest='OPTIMIZER', type=str, default='sgd', help='The type of optimizer used during training')
parser.add_argument('--class_group', dest='class_group', type=str, default='all', help='The way the classes are clustred together')
parser.add_argument('--task', dest='task', type=str, default='all', help='train/predict/evaluate/all')
parser.add_argument('--learning_rate', dest='learning_rate', type=int, default=None, help='The learning rate during training. None=defualt')
parser.add_argument('--momentum', dest='momentum', type=int, default=None, help='Momentum during training. None=default')
parser.add_argument('--indices', dest='indices', type=bool, default=False, help='Maxpooling indices')


args = parser.parse_args()

# forces tensorflow to run on GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

# Redefine the variable name for easier access
model = args.model
dataset = args.dataset
TRAINING_EPOCHS = args.EPOCHS
TRAINING_BATCHES = args.BATCHES
OPTIMIZER = args.OPTIMIZER
class_group = args.class_group
task = args.task
learning_rate = args.learning_rate
momentum = args.momentum
indices = args.indices


def main():
    print('\n#################################################################')
    print('Running "{}" with model {} and dataset {} on CUDA .'.format(task, model, dataset, args.cuda))
    print('#################################################################\n')

    if task in ['train', 'all']:
        train_model(model=model,
                    INPUT_SHAPE=INPUT_SHAPE,
                    dataset=dataset,
                    TRAINING_IMAGES_PATH=TRAINING_IMAGES_PATH[dataset],
                    TRAINING_LABELS_PATH=TRAINING_LABELS_PATH[dataset],
                    TRAINING_TEXTFILE_PATH=TRAINING_TEXTFILE_PATH[model+dataset],
                    WEIGHTS_PATH=WEIGHTS_PATH[model+dataset],
                    TRAINING_BATCHES=TRAINING_BATCHES,
                    TRAINING_EPOCHS=TRAINING_EPOCHS,
                    OPTIMIZER=OPTIMIZER,
                    list_classes=list_classes[dataset],
                    learning_rate=learning_rate,
                    momentum=momentum,
                    indices=indices
                    )

    if task in ['predict', 'all']:
        predict_model(model=model,
                      dataset=dataset,
                      INPUT_SHAPE=INPUT_SHAPE,
                      PREDICTION_IMAGES_PATH=PREDICTION_IMAGES_PATH[dataset],
                      PREDICTION_SAVE_PATH=PREDICTION_SAVE_PATH[model+dataset],
                      PREDICTION_TEXTFILE_PATH=PREDICTION_TEXTFILE_PATH[model+dataset],
                      PREDICTION_BATCH_SIZE=PREDICTION_BATCH_SIZE,
                      WEIGHTS_PATH=WEIGHTS_PATH[model + dataset],
                      list_classes=list_classes[dataset],
                      indices=indices,
                      label_to_color=label_to_color
                      )

    if task in ['evaluate', 'all']:
        evaluate_model(model=model,
                       INPUT_SHAPE=INPUT_SHAPE,
                       EVALUATION_IMAGES_PATH=EVALUATION_IMAGES_PATH[dataset],
                       EVALUATION_LABELS_PATH=EVALUATION_LABELS_PATH[dataset],
                       WEIGHTS_PATH=WEIGHTS_PATH[model + dataset],
                       EVALUATION_BATCH_SIZE=EVALUATION_BATCH_SIZE,
                       OPTIMIZER=OPTIMIZER,
                       list_classes=list_classes[dataset],
                       indices=indices,
                       learning_rate=learning_rate,
                       momentum=momentum,
                       )
    if task not in ['train', 'predict', 'evaluate', 'all']:
        raise ValueError('Define a correct task - (train/predict/evaluate/all).')


if __name__ == '__main__':
    main()
