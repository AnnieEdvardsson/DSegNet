import subprocess
import argparse

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--cuda', dest='cuda', type=str, default='1', help='The CUDA we want to use, 0 or 1')
parser.add_argument('--dataset', dest='dataset', type=str, default='CityScapes', help='The datset we want to use')
parser.add_argument('--epochs', dest='EPOCHS', type=str, default='2', help='The number of epochs')
parser.add_argument('--train_batches', dest='TRAIN_BATCHES', type=str, default='4', help='The number of batches in train')
parser.add_argument('--predict_batches', dest='PREDICT_BATCHES', type=str, default='4', help='The number images to predict. Write "all" for all images in folder.')
parser.add_argument('--evaluate_batch', dest='EVALUATE_BATCH', type=str, default='4', help='The number of images used to evaluate. Write "all" for all images in folder.')
parser.add_argument('--optimizer', dest='OPTIMIZER', type=str, default='sgd', help='The type of optimizer used during training')
parser.add_argument('--task', dest='task', type=str, default='all', help='train/predict/evaluate/all')
parser.add_argument('--learning_rate', dest='learning_rate', type=str, default='None', help='The learning rate during training. None=defualt')
parser.add_argument('--momentum', dest='momentum', type=str, default='None', help='Momentum during training. None=default')
parser.add_argument('--indices', dest='indices', type=str, default='False', help='Maxpooling indices')
parser.add_argument('--bestweights', dest='bestweights', type=str, default='False', help='If avaliabe choose weights from WorthyWeights, works if we dont run train')
parser.add_argument('--modelcombo', dest='modelcombo', type=int, default=0, help='The models we want to run 0: SegNetModel and DispSegNetModel. 1: dSegNetModel and DispSegNetBasicModel')
# Script to train all the different networks with the same arguments

args = parser.parse_args()

# Redefine the variable name for easier access
cuda = args.cuda
modelcombo = args.modelcombo
dataset = args.dataset
TRAINING_EPOCHS = args.EPOCHS
TRAINING_BATCHES = args.TRAIN_BATCHES
PREDICTION_BATCHES = args.PREDICT_BATCHES
EVALUATION_BATCHES = args.EVALUATE_BATCH
OPTIMIZER = args.OPTIMIZER
task = args.task
learning_rate = args.learning_rate
momentum = args.momentum
indices = args.indices
bestweights = args.bestweights

if modelcombo == 0:
    args1 = ['cd', '/home/exjobb/DSegNet']
    subprocess.call(args1)
    args2 = ['launcher.py', '--cuda', cuda, '--model', 'SegNetModel', '--dataset', dataset,
            '--epochs', TRAINING_EPOCHS, '--train_batches', TRAINING_BATCHES, '--predict_batches', PREDICTION_BATCHES,
            '--evaluate_batch', EVALUATION_BATCHES, '--optimizer', OPTIMIZER, '--task', task, '--learning_rate', learning_rate,
            '--momentum', momentum, '--indices', indices, '--bestweights', bestweights]
    subprocess.call(args)

    args = ['launcher.py', '--cuda', cuda, '--model', 'DispSegNetModel', '--dataset', dataset,
            '--epochs', TRAINING_EPOCHS, '--train_batches', TRAINING_BATCHES, '--predict_batches', PREDICTION_BATCHES,
            '--evaluate_batch', EVALUATION_BATCHES, '--optimizer', OPTIMIZER, '--task', task, '--learning_rate', learning_rate,
            '--momentum', momentum, '--indices', indices, '--bestweights', bestweights]
    subprocess.call(args)

elif modelcombo == 1:
    args = ['launcher.py', '--cuda', cuda, '--model', 'dSegNetModel', '--dataset', dataset,
            '--epochs', TRAINING_EPOCHS, '--train_batches', TRAINING_BATCHES, '--predict_batches', PREDICTION_BATCHES,
            '--evaluate_batch', EVALUATION_BATCHES, '--optimizer', OPTIMIZER, '--task', task, '--learning_rate', learning_rate,
            '--momentum', momentum, '--indices', indices, '--bestweights', bestweights]
    subprocess.call(args)

    args = ['launcher.py', '--cuda', cuda, '--model', 'DispSegNetBasicModel', '--dataset', dataset,
            '--epochs', TRAINING_EPOCHS, '--train_batches', TRAINING_BATCHES, '--predict_batches', PREDICTION_BATCHES,
            '--evaluate_batch', EVALUATION_BATCHES, '--optimizer', OPTIMIZER, '--task', task, '--learning_rate', learning_rate,
            '--momentum', momentum, '--indices', indices, '--bestweights', bestweights]
    subprocess.call(args)

else:
    args1 = ['cd', '/MLDatasetsStorage']
    subprocess.call(args1)
    args2 = ['ls']
    subprocess.call(args2)
    args = ['launcher.py']
    subprocess.call(args)

