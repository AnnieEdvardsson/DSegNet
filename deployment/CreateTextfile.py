import datetime
import csv
import pandas as pd
import numpy as np


def create_training_textfile(PATH, model, dataset, EPOCHS, BATCH, OPT, list_classes, hours, minutes, seconds,
                              learning_rate, momentum, indices, nr_imgs, History, hours_batch, minutes_batch,
                             seconds_batch, class_names):
    now = datetime.datetime.now()

    IoUlist = list(History.keys())[3:]

    acc_cat = round(History['categorical_accuracy'][-1] * 100, 1)

    acc_MIoU = round(History['Mean_iou'][-1] * 100, 1)

    loss = round(History['loss'][-1], 3)

    # Save textfile
    f = open(PATH, "w+")
    f.write("TRAINING PROPERTIES \r\n")
    f.write("##################################################### \r\n\r\n")

    f.write("Model: %s \r\n" % model)
    f.write("Dataset: %s \r\n" % dataset)
    f.write("Date & Time: %s \r\n" % (now.strftime("%Y-%m-%d %H:%M")))
    f.write("Run time: %s hours, %s minutes & %s seconds \r\n" % (hours, minutes, seconds))
    f.write("Average run time per epoch: %s hours, %s minutes & %s seconds \r\n" % (hours_batch, minutes_batch, seconds_batch))
    f.write("##################################################### \r\n\r\n")

    f.write("Number of images used in training: %s \r\n" % nr_imgs)
    f.write("Nr epochs: %i \r\n" % EPOCHS)
    f.write("Optimizer: %s \r\n" % OPT)
    f.write("Learning rate: %s \r\n" % learning_rate)
    if momentum is not None:
        f.write("Momentum: %s \r\n" % momentum)
    f.write("Batch size: %i \r\n" % BATCH)

    f.write("List of classes: %s \r\n" % list_classes)
    f.write("Usage of maxpooling indices: %s \r\n" % indices)
    f.write("##################################################### \r\n\r\n")
    f.write("Training loss: %s \r\n" % round(loss, 1))
    f.write("Training categorical accuracy: %s %%\r\n" % acc_cat)
    f.write("Training MIoU accuracy: %s %% \r\n\r\n" % acc_MIoU)

    f.write("Training IoU accuracy per class: \r\n")
    for name, acc in zip(class_names, IoUlist):
        modded_acc = round(History[acc][-1] * 100, 1)
        # f.write("%s %% for class: %s \r\n" % (modded_acc, name))
        f.write("{:5.1f}%: {} \r\n".format(modded_acc, name))

    f.close()


def create_training_eval_textfile(PATH, model, dataset, EPOCHS, BATCH, OPT, list_classes, hours, minutes, seconds,
                                  learning_rate, momentum, indices, nr_imgs, History, hours_batch, minutes_batch,
                                  seconds_batch, class_names, max_index):
    now = datetime.datetime.now()


    acc_cat = round(History['categorical_accuracy'][max_index] * 100, 1)
    val_acc_cat = round(History['val_categorical_accuracy'][max_index] * 100, 1)

    val_acc_MIoU = round(History['val_Mean_iou'][max_index] * 100, 1)
    acc_MIoU = round(History['Mean_iou'][max_index] * 100, 1)

    val_loss = round(History['val_loss'][max_index], 3)
    loss = round(History['loss'][max_index], 3)

    # Make a list out of the keys
    IoUlist = list(History.keys())
    IoUlist = IoUlist[:-1]

    # Take out all Training accuracies and then remove the one that aren't IoU
    IoUTrain = IoUlist[len(IoUlist) // 2:]
    IoUTrain = IoUTrain[len(IoUTrain)-(len(list_classes)+1):]

    # Take out all Evaluate accuracies and then remove the one that aren't IoU
    IoUEval = IoUlist[:len(IoUlist) // 2]
    IoUEval = IoUEval[len(IoUEval)-(len(list_classes)+1):]

    # Save textfile
    f = open(PATH, "w+")
    f.write("TRAINING PROPERTIES \r\n")
    f.write("##################################################### \r\n\r\n")

    f.write("Model: %s \r\n" % model)
    f.write("Dataset: %s \r\n" % dataset)
    f.write("Date & Time: %s \r\n" % (now.strftime("%Y-%m-%d %H:%M")))
    f.write("Run time: %s hours, %s minutes & %s seconds \r\n" % (hours, minutes, seconds))
    f.write("Average run time per epochs: %s hours, %s minutes & %s seconds \r\n" % (hours_batch, minutes_batch, seconds_batch))
    f.write("##################################################### \r\n\r\n")

    f.write("Number of images used in training: %s \r\n" % nr_imgs)
    f.write("Nr epochs: %i \r\n" % EPOCHS)
    f.write("Learning rate: %s \r\n" % learning_rate)
    f.write("Momentum: %s \r\n" % momentum)
    f.write("Batch size: %i \r\n" % BATCH)
    f.write("Optimizer: %s \r\n" % OPT)
    f.write("List of classes: %s \r\n" % list_classes)
    f.write("Usage of maxpooling indices: %s \r\n" % indices)
    f.write("##################################################### \r\n\r\n")

    f.write("Training loss: %s \r\n" % loss)
    f.write("Training categorical accuracy: %s %%\r\n" % acc_cat)
    f.write("Training MIoU accuracy: %s %% \r\n\r\n" % acc_MIoU)

    f.write("Training IoU accuracy per class: \r\n")
    for name, acc in zip(class_names, IoUTrain):
        modded_acc = round(History[acc][max_index] * 100, 1)
        f.write("{:5.1f}%: {} \r\n".format(modded_acc, name))

    f.write("##################################################### \r\n\r\n")

    f.write("Evaluation loss: %s \r\n" % val_loss)
    f.write("Evaluation categorical accuracy: %s %%\r\n" % val_acc_cat)
    f.write("Evaluation MIoU accuracy: %s %% \r\n\r\n" % val_acc_MIoU)

    f.write("Evaluation IoU accuracy per class: \r\n")
    for name, acc in zip(class_names, IoUEval):
        modded_acc = round(History[acc][max_index] * 100, 1)
        f.write("{:5.1f}%: {} \r\n".format(modded_acc, name))

    f.write("##################################################### \r\n\r\n")

    f.close()


def create_prediction_textfile(PATH, model, dataset, BATCH, list_classes, pre_trained_encoder, indices,
                               load_weights_by_name, hours, minutes, seconds, nr_imgs, weight_flag):
    now = datetime.datetime.now()
    # Save textfile
    f = open(PATH, "w+")
    f.write("PREDICTION PROPERTIES \r\n")
    f.write("##################################################### \r\n")

    f.write("Model: %s \r\n" % model)
    f.write("Dataset: %s \r\n" % dataset)

    f.write("Date & Time: %s \r\n" % (now.strftime("%Y-%m-%d %H:%M")))
    f.write("Run time: %s hours, %s minutes & %s seconds \r\n" % (hours, minutes, seconds))
    f.write("##################################################### \r\n")

    f.write("Number of predicted images: %s \r\n" % nr_imgs)
    f.write("WorthyWeight used: %s \r\n" % weight_flag)
    f.write("Batch size: %i \r\n" % BATCH)
    f.write("List classes: %s \r\n" % list_classes)
    f.write("Pre trained encoder: %s \r\n" % pre_trained_encoder)
    f.write("Pooling indices: %s \r\n" % indices)
    f.write("Load weights by name: %s \r\n" % load_weights_by_name)
    f.close()

def create_evaluation_textfile(PATH, model, dataset, BATCH, list_classes, pre_trained_encoder, indices,
                               load_weights_by_name, hours, minutes, seconds, evaluation, nr_imgs, seconds2, mseconds2,
                               class_names):
    now = datetime.datetime.now()
    acc_cat = round(evaluation[1] * 100, 1)
    acc_MIoU = round(evaluation[2] * 100, 1)
    IoUlist = evaluation[3:]

    # Save textfile
    f = open(PATH, "w+")
    f.write("PREDICTION PROPERTIES \r\n")
    f.write("##################################################### \r\n\r\n")

    f.write("Model: %s \r\n" % model)
    f.write("Dataset: %s \r\n" % dataset)
    f.write("Date & Time: %s \r\n" % (now.strftime("%Y-%m-%d %H:%M")))
    f.write("Run time: %s hours, %s minutes & %s seconds \r\n" % (hours, minutes, seconds))
    f.write("Run time for one image: %s seconds & %s milliseconds \r\n" % (seconds2, mseconds2))
    f.write("##################################################### \r\n\r\n")

    f.write("Number of images used to evaluate: %s \r\n" % nr_imgs)
    f.write("Batch size: %i \r\n" % BATCH)
    f.write("List classes: %s \r\n" % list_classes)
    f.write("Pre trained encoder: %s \r\n" % pre_trained_encoder)
    f.write("Pooling indices: %s \r\n" % indices)
    f.write("Load weights by name: %s \r\n" % load_weights_by_name)
    f.write("##################################################### \r\n\r\n")

    f.write("Evaluation loss: %s \r\n" % round(evaluation[0], 3))
    f.write("Evaluation categorical accuracy: %s %%\r\n" % acc_cat)
    f.write("Evaluation MIoU accuracy: %s %% \r\n" % acc_MIoU)

    f.write("Evaluation IoU accuracy per class: \r\n")
    for name, acc in zip(class_names, IoUlist):
        modded_acc = round(acc * 100, 1)
        # f.write("%s %% for class: %s \r\n" % (modded_acc, name))
        f.write("{:5.1f}%: {} \r\n".format(modded_acc, name))

    f.write("##################################################### \r\n\r\n")
    f.write("Mean IoU, then classes\r\n\r\n")
    f.write("{:5.1f} \r\n".format(acc_MIoU))
    for name, acc in zip(class_names, IoUlist):
        modded_acc = round(acc * 100, 1)
        # f.write("%s %% for class: %s \r\n" % (modded_acc, name))
        f.write("{:5.1f} \r\n".format(modded_acc))

    f.close()

def create_distribution_textfile(PATH, model, dataset, list_classes, hours, minutes, seconds, nr_imgs,
                                 seconds2, mseconds2, confusion):
    now = datetime.datetime.now()


    # Save textfile
    f = open(PATH, "w+")
    f.write("DISTRIBUTION PROPERTIES \r\n")
    f.write("##################################################### \r\n\r\n")

    f.write("Model: %s \r\n" % model)
    f.write("Dataset: %s \r\n" % dataset)
    f.write("Date & Time: %s \r\n" % (now.strftime("%Y-%m-%d %H:%M")))
    f.write("Run time: %s hours, %s minutes & %s seconds \r\n" % (hours, minutes, seconds))
    f.write("Run time for one image: %s seconds & %s milliseconds \r\n" % (seconds2, mseconds2))
    f.write("##################################################### \r\n\r\n")

    f.write("Number of images used to create distribution: %s \r\n" % nr_imgs)
    f.write("List classes: %s \r\n" % list_classes)
    f.write("##################################################### \r\n\r\n")

    f.write("Confusion matrix: \r\n")
    for i in range(len(list_classes)+1):
        f.write('{:7}'.format(i))

    f.write('\r\n')
    for _ in range(len(list_classes) + 1):
        f.write('-------')
    f.write('\r\n')
    for i, row in enumerate(confusion):
        f.write('{:2} |'.format(i))
        for j, num in enumerate(row):
            if i == j:
                number = '{:3.1f}* '.format(num)
            else:
                number = '{:3.1f} '.format(num)


            f.write('{:6} '.format(str(number)))
        f.write('\r\n\r\n')

    f.write("##################################################### \r\n\r\n")
    f.close()

    np.savetxt('result.txt', confusion, fmt='%.2e')


def create_readable_distribution_textfile(PATH, accuracy_dict, class_names_dist):
    raw_data = {}
    for i, classen in enumerate(class_names_dist):
        raw_data[str(classen)] = accuracy_dict[i]
    raw_data["mean_iou"] = accuracy_dict["mean_iou"]

    class_names_dist.append("mean_iou")
    df = pd.DataFrame(raw_data, columns=class_names_dist)
    df.to_csv(PATH)


def compile_row_string(a_row):
    return str(a_row).strip(']').strip('[').replace(' ', '')

