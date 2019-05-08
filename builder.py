
import time
import argparse
from keras import optimizers
from deployment.models import initilize_model
from deployment.data_readers import generate_evaluation_batches, generate_prediction_batch
from deployment.dataGenerator import *
from deployment.postprocessors import get_images_from_softmax
from deployment.data_writer import save_predictions, save_images_with_predictions
from deployment.CreateTextfile import create_training_textfile, create_prediction_textfile, create_evaluation_textfile, \
    create_training_eval_textfile, create_distribution_textfile, create_readable_distribution_textfile
from deployment.postprocessors import ColorConverter
from deployment.models import SegNetModel
from deployment.data_readers import * #get_imgs, get_imgs_and_depth
import os
from deployment.compiler_creator import Mean_iou, Class_iou  # , mean_iou
from keras.callbacks import History, ReduceLROnPlateau, EarlyStopping
from keras.callbacks import ModelCheckpoint
from DataHandler.create_plots import create_all_hist, loss_graph
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

#######################################################################################################################
''' TRAIN MODEL '''
#######################################################################################################################

def train_model(model,
                INPUT_SHAPE,
                dataset,
                TRAINING_IMAGES_PATH,
                TRAINING_LABELS_PATH,
                TRAINING_DEPTH_PATH,
                TRAINING_TEXTFILE_PATH,
                WEIGHTS_PATH,
                TRAINING_BATCHES,
                TRAINING_EPOCHS,
                OPTIMIZER,
                opt,
                list_classes,
                learning_rate,
                momentum,
                indices,
                class_names):

    print('\n###############################################################################')
    print('Start Training with {} epochs, {} batches and "{}" optimizer.'.format(TRAINING_EPOCHS, TRAINING_BATCHES, OPTIMIZER))
    print('###############################################################################\n')

    nbr_classes = len(list_classes) + 1

    nr_imgs = len(os.listdir(TRAINING_IMAGES_PATH))

    if nr_imgs == 0:
        raise ValueError('There is no images in the training folder: \n {}'.format(TRAINING_IMAGES_PATH))

    # Initialize model
    model_instance = initilize_model(model=model,
                                     INPUT_SHAPE=INPUT_SHAPE,
                                     nbr_classes=nbr_classes,
                                     pre_trained_encoder=False,
                                     indices=False,
                                     weights=None,
                                     load_weights_by_name=False)

    # Create model
    model_instance = model_instance.create_model()

    model_instance.summary()

    # Create the metric list (that include categorical accuracy and the Mean IoU for all classes)
    metric = ['categorical_accuracy', Mean_iou]
    for name, classes in zip(class_names, range(nbr_classes)):
        metric.append(Class_iou(classes))

    # Compile model
    model_instance.compile(optimizer=opt,
                           loss='categorical_crossentropy',
                           metrics=metric)

    # Train model
    # create batch_name by taking random names,
    trainGen = initialize_generator(task="train",
                                    model=model,
                                    dataset=dataset,
                                    input_path=TRAINING_IMAGES_PATH,
                                    output_path=TRAINING_LABELS_PATH,
                                    depth_path=TRAINING_DEPTH_PATH,
                                    batch_size=TRAINING_BATCHES,
                                    input_shape=INPUT_SHAPE,
                                    list_classes=list_classes,
                                    nbr_classes=nbr_classes)

    training_time = time.time()
    History = model_instance.fit_generator(trainGen,
                                           steps_per_epoch=len(os.listdir(TRAINING_LABELS_PATH)) // TRAINING_BATCHES,
                                           epochs=TRAINING_EPOCHS,
                                           verbose=1
                                           )


    training_time = (time.time() - training_time)
    m, s = divmod(training_time, 60)
    h, m = divmod(m, 60)
    h = round(h)
    m = round(m)
    s = round(s)

    m2, s2 = divmod(training_time / TRAINING_EPOCHS, 60)
    h2, m2 = divmod(m2, 60)
    h2 = round(h2)
    m2 = round(m2)
    s2 = round(s2)


    # Save weights and model
    model_instance.save_weights(WEIGHTS_PATH)

    nbr_epochs = len(History.history['loss'])

    # Save textfile
    create_training_textfile(PATH=TRAINING_TEXTFILE_PATH,
                             model=model,
                             dataset=dataset,
                             EPOCHS=nbr_epochs,
                             BATCH=TRAINING_BATCHES,
                             OPT=OPTIMIZER,
                             list_classes=list_classes,
                             hours=h, minutes=m, seconds=s,
                             hours_batch=h2, minutes_batch=m2, seconds_batch=s2,
                             learning_rate=learning_rate,
                             momentum=momentum,
                             indices=indices,
                             nr_imgs=nr_imgs,
                             History=History.history,
                             class_names=class_names)

    print("\n Training done. Run time: {} hours, {} minutes and {} seconds.".format(round(h), round(m), round(s)))
    print("Average run time per epoch: {} hours, {} minutes and {} seconds. \n".format(round(h2), round(m2), round(s2)))

    loss = round(History.history['loss'][-1], 3)
    print("Loss: {} \n".format(loss))
    acc_cat = round(History.history['categorical_accuracy'][-1] * 100, 1)
    print("Categorical accuracy: {}% \n".format(acc_cat))

    acc_MIoU = round(History.history['Mean_iou'][-1] * 100, 1)
    print("Mean IoU: {} \n".format(acc_MIoU))

    IoUlist = list(History.history.keys())[3:]
    print("Training IoU accuracy per class:")
    for name, acc in zip(class_names, IoUlist):
        modded_acc = round(History.history[acc][-1] * 100, 1)

        print("{:5.1f}%: {}".format(modded_acc, name))


#######################################################################################################################
''' TRAIN EVALUATE MODEL '''
#######################################################################################################################


def train_eval_model(model,
                     INPUT_SHAPE,
                     dataset,
                     TRAINING_IMAGES_PATH,
                     TRAINING_LABELS_PATH,
                     TRAINING_TEXTFILE_PATH,
                     TRAINING_DEPTH_PATH,
                     WEIGHTS_PATH,
                     TRAINING_BATCHES,
                     TRAINING_EPOCHS,
                     OPTIMIZER,
                     opt,
                     list_classes,
                     learning_rate,
                     momentum,
                     indices,
                     EVALUATION_IMAGES_PATH,
                     EVALUATION_LABELS_PATH,
                     EVALUATION_DEPTH_PATH,
                     EVALUATION_BATCHES,
                     EVALUATION_TEXTFILE_PATH,
                     class_names
                     ):

    print('\n###############################################################################')
    print('Start Training with evaluation {} epochs, {} batches and "{}" optimizer.'.format(TRAINING_EPOCHS, TRAINING_BATCHES, OPTIMIZER))
    print('###############################################################################\n')

    nbr_classes = len(list_classes) + 1

    nr_imgs = len(os.listdir(TRAINING_IMAGES_PATH))

    if nr_imgs == 0:
        raise ValueError('There is no images in the training folder: \n {}'.format(TRAINING_IMAGES_PATH))

    #if model in ["SegNetModel", "dSegNetModel", "DispSegNetModel", "DispSegNetBasicModel", "EncFuseModel"]:
    #    pre_trained_encoder = True
    #else:
    pre_trained_encoder = False
    # Initialize model
    model_instance = initilize_model(model=model,
                                     INPUT_SHAPE=INPUT_SHAPE,
                                     nbr_classes=nbr_classes,
                                     pre_trained_encoder=pre_trained_encoder,
                                     indices=False,
                                     weights=None,
                                     load_weights_by_name=False)



    # Create model
    Model = model_instance.create_model()

    #segnet.summary()

    # Create the metric list (that include categorical accuracy and the Mean IoU for all classes)
    metric = ['categorical_accuracy', Mean_iou]
    for classes in range(nbr_classes):
        metric.append(Class_iou(classes))

    # Compile model
    Model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=metric)

    # Train model
    # create batch_name by taking random names,
    trainGen = initialize_generator(task="train",
                                    model=model,
                                    dataset=dataset,
                                    input_path=TRAINING_IMAGES_PATH,
                                    output_path=TRAINING_LABELS_PATH,
                                    depth_path=TRAINING_DEPTH_PATH,
                                    batch_size=TRAINING_BATCHES,
                                    input_shape=INPUT_SHAPE,
                                    list_classes=list_classes,
                                    nbr_classes=nbr_classes
                                    )

    evalGen = initialize_generator(task="eval",
                                   model=model,
                                   dataset=dataset,
                                   input_path=EVALUATION_IMAGES_PATH,
                                   output_path=EVALUATION_LABELS_PATH,
                                   depth_path=EVALUATION_DEPTH_PATH,
                                   batch_size=EVALUATION_BATCHES,
                                   input_shape=INPUT_SHAPE,
                                   list_classes=list_classes,
                                   nbr_classes=nbr_classes
                                   )

    checkpoint_callback = ModelCheckpoint(filepath=WEIGHTS_PATH, monitor='val_loss', verbose=1,
                                          save_best_only=True, save_weights_only=True, mode='min')
    # if pre_trained_encoder:
    #     reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=6, min_lr=0.00001, verbose=1, factor=0.1)
    #     earlystopper = EarlyStopping(patience=15, verbose=1, monitor='val_loss')
    # else:
    if dataset == "KITTI":
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=6, min_lr=0.00001, verbose=1, factor=0.1)
        earlystopper = EarlyStopping(patience=18, verbose=1, monitor='val_loss')
    elif dataset == "CityScapes":
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, min_lr=0.00001, verbose=1, factor=0.1)
        earlystopper = EarlyStopping(patience=10, verbose=1, monitor='val_loss')
    elif dataset == "BDD10k":
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, min_lr=0.00001, verbose=1, factor=0.1)
        earlystopper = EarlyStopping(patience=6, verbose=1, monitor='val_loss')


    training_time = time.time()
    History = Model.fit_generator(trainGen,
                                  steps_per_epoch=len(os.listdir(TRAINING_LABELS_PATH)) // TRAINING_BATCHES,
                                  validation_data=evalGen,
                                  validation_steps=len(os.listdir(EVALUATION_LABELS_PATH)) // EVALUATION_BATCHES,
                                  epochs=TRAINING_EPOCHS, verbose=1,
                                  callbacks=[checkpoint_callback, reduce_lr, earlystopper])

    Model.save_weights("/WeightModels/exjobb/" + model + dataset + "/weights_full_epoch.hdf5")



    training_time = (time.time() - training_time)
    m, s = divmod(training_time, 60)
    h, m = divmod(m, 60)
    h = round(h)
    m = round(m)
    s = round(s)

    m2, s2 = divmod(training_time/TRAINING_EPOCHS, 60)
    h2, m2 = divmod(m2, 60)
    h2 = round(h2)
    m2 = round(m2)
    s2 = round(s2)

    # Take out the History that is best according to val_Mean_iou
    my_list = History.history['val_loss']
    max_value = min(my_list)
    max_index = my_list.index(max_value)
    print('Max index: {}'.format(max_index))
    nbr_epochs = len(my_list)

    # Save textfile
    create_training_eval_textfile(PATH=TRAINING_TEXTFILE_PATH,
                                 model=model,
                                 dataset=dataset,
                                 EPOCHS=nbr_epochs,
                                 BATCH=TRAINING_BATCHES,
                                 OPT=OPTIMIZER,
                                 list_classes=list_classes,
                                 hours=h, minutes=m, seconds=s,
                                 hours_batch=h2, minutes_batch=m2, seconds_batch=s2,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 indices=indices,
                                 nr_imgs=nr_imgs,
                                 History=History.history,
                                 class_names=class_names,
                                  max_index=max_index)

    loss_graph(History.history['val_loss'], model, dataset)

    PATH = "/WeightModels/exjobb/" + model + dataset

    hist_val_loss = History.history['val_loss']
    hist_loss = History.history['loss']

    np.save(PATH + '/val_loss', hist_val_loss)
    np.save(PATH + '/loss', hist_loss)

    print("Training done. Run time: {} hours, {} minutes and {} seconds.".format(round(h), round(m), round(s)))
    print("Average run time per epoch: {} hours, {} minutes and {} seconds.\n".format(round(h2), round(m2), round(s2)))

    loss_train = round(History.history['loss'][max_index], 3)
    loss_eval = round(History.history['val_loss'][max_index], 3)
    print("Loss-> Train: {}, Eval: {}\n".format(loss_train, loss_eval))

    # acc_cat_train = round(History.history['categorical_accuracy'][-1] * 100, 1)
    # print("Categorical accuracy: {}% \n".format(acc_cat))

    acc_MIoU_train = round(History.history['Mean_iou'][max_index] * 100, 1)
    acc_MIoU_eval = round(History.history['val_Mean_iou'][max_index] * 100, 1)
    print("Mean IoU-> Train: {}%,   Eval: {}%\n".format(acc_MIoU_train, acc_MIoU_eval))

    # Make a list out of the keys
    IoUlist = list(History.history.keys())
    IoUlist = IoUlist[:-1]

    # Take out all Training accuracies and then remove the one that aren't IoU
    IoUTrain = IoUlist[len(IoUlist) // 2:]
    IoUTrain = IoUTrain[len(IoUTrain)-(len(list_classes)+1):]

    # Take out all Evaluate accuracies and then remove the one that aren't IoU
    IoUEval = IoUlist[:len(IoUlist) // 2]
    IoUEval = IoUEval[len(IoUEval)-(len(list_classes)+1):]

    print("Training and evaluation IoU accuracy per class:")
    for name, acc_train, acc_eval in zip(class_names, IoUTrain, IoUEval):
        modded_acc_train = round(History.history[acc_train][max_index] * 100, 1)
        modded_acc_eval = round(History.history[acc_eval][max_index] * 100, 1)
        print('{:13}-> Train: {:5.1f}%, Eval: {:5.1f}%'.format(name, modded_acc_train, modded_acc_eval))

#######################################################################################################################
''' PREDICT MODEL '''
#######################################################################################################################


def predict_model(model,
                  dataset,
                  INPUT_SHAPE,
                  PREDICTION_IMAGES_PATH,
                  PREDICTION_SAVE_PATH,
                  PREDICTION_TEXTFILE_PATH,
                  WEIGHTS_PATH,
                  list_classes,
                  indices,
                  label_to_color,
                  PREDICTION_BATCHES,
                  weight_flag):

    nr_imgs = len(os.listdir(PREDICTION_IMAGES_PATH))

    if nr_imgs == 0:
        raise ValueError('There is no images in the prediction folder: \n {}'.format(PREDICTION_IMAGES_PATH))

    print('\n###############################################################################')
    print('Start Prediction with {} images and a batch size {}.'.format(nr_imgs, PREDICTION_BATCHES))
    print('###############################################################################\n')

    nbr_classes = len(list_classes) + 1

    # get images to predict
    # input_images = generate_prediction_batch(input_path=PREDICTION_IMAGES_PATH,
    #                                          batch_size=PREDICT_BATCHES,
    #                                          input_shape=INPUT_SHAPE)

    # create batch_name by taking random names,
    dataGen = initialize_generator(task="predict",
                                   model=model,
                                   dataset=dataset,
                                   input_path=PREDICTION_IMAGES_PATH,
                                   batch_size=PREDICTION_BATCHES,
                                   input_shape=INPUT_SHAPE
                                   )

    # Create model instance
    pre_trained_encoder = True
    load_weights_by_name = False
    model_instance = initilize_model(model=model,
                                     INPUT_SHAPE=INPUT_SHAPE,
                                     nbr_classes=nbr_classes,
                                     pre_trained_encoder=pre_trained_encoder,
                                     indices=indices,
                                     weights=WEIGHTS_PATH,
                                     load_weights_by_name=load_weights_by_name)

    # Create model
    Model = model_instance.create_model()



    prediction_time = time.time()

    stepsize = nr_imgs // PREDICTION_BATCHES
    predictions_softmax = Model.predict_generator(dataGen,
                                                  steps=stepsize,
                                                  verbose=1)

    new_list = label_to_color
    new_list[0] = [0, 0, 0]

    predictions = get_images_from_softmax(predictions_softmax, new_list)

    # Convert predictions to rgb
    rgb_converter = ColorConverter(new_list)
    predictions_rgb = rgb_converter.convert_all_images(predictions)

    # Save predictions
    save_predictions(predictions_rgb, PREDICTION_SAVE_PATH)
    #save_images_with_predictions(input_images, predictions, PREDICTION_SAVE_PATH)

    prediction_time = (time.time() - prediction_time)
    m, s = divmod(prediction_time, 60)
    h, m = divmod(m, 60)
    h = round(h)
    m = round(m)
    s = round(s)

    # Save textfile
    create_prediction_textfile(PATH=PREDICTION_TEXTFILE_PATH,
                               model=model,
                               dataset=dataset,
                               BATCH=PREDICTION_BATCHES,
                               list_classes=list_classes,
                               pre_trained_encoder=pre_trained_encoder,
                               indices=indices,
                               load_weights_by_name=load_weights_by_name,
                               hours=h, minutes=m, seconds=s,
                               nr_imgs=nr_imgs,
                               weight_flag=weight_flag
                               )

    print("Prediction done. It took {} hours, {} minutes and {} seconds.".format(round(h), round(m), round(s)))


#######################################################################################################################
''' EVALUATE MODEL '''
#######################################################################################################################


def evaluate_model(model,
                   dataset,
                   INPUT_SHAPE,
                   EVALUATION_IMAGES_PATH,
                   EVALUATION_LABELS_PATH,
                   EVALUATION_TEXTFILE_PATH,
                   WEIGHTS_PATH,
                   EVALUATION_DEPTH_PATH,
                   opt,
                   list_classes,
                   indices,
                   EVALUATION_BATCHES,
                   class_names):

    nr_imgs = len(os.listdir(EVALUATION_IMAGES_PATH))

    print('\n###############################################################################')
    print('Start Evaulation with {} images and a batch size of {}.'.format(nr_imgs, EVALUATION_BATCHES))
    print('###############################################################################\n')

    nbr_classes = len(list_classes) + 1


    pre_trained_encoder = False
    load_weights_by_name = False
    # Create model instance
    model_instance = initilize_model(model=model,
                                     INPUT_SHAPE=INPUT_SHAPE,
                                     nbr_classes=nbr_classes,
                                     pre_trained_encoder=pre_trained_encoder,
                                     indices=indices,
                                     weights=WEIGHTS_PATH,
                                     load_weights_by_name=load_weights_by_name)

    # Create model
    Model = model_instance.create_model()

    dataGen = initialize_generator(task="eval",
                                   model=model,
                                   dataset=dataset,
                                   input_path=EVALUATION_IMAGES_PATH,
                                   output_path=EVALUATION_LABELS_PATH,
                                   batch_size=EVALUATION_BATCHES,
                                   input_shape=INPUT_SHAPE,
                                   list_classes=list_classes,
                                   nbr_classes=nbr_classes,
                                   depth_path=EVALUATION_DEPTH_PATH
                                   )

    # Create the metric list (that include categorical accuracy and the Mean IoU for all classes)
    metric = ["accuracy", Mean_iou]
    for classes in range(nbr_classes):
        metric.append(Class_iou(classes))

    # Compile
    Model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=metric) # Mean_iou

    evaluation_time = time.time()



    stepsize = nr_imgs // EVALUATION_BATCHES
    evaluation = Model.evaluate_generator(dataGen,
                                          steps=stepsize,
                                          verbose=1)


    evaluation_time = (time.time() - evaluation_time)
    m, s = divmod(evaluation_time, 60)
    h, m = divmod(m, 60)
    h = round(h)
    m = round(m)
    s = round(s)

    s2, ms2 = divmod(1000*evaluation_time/nr_imgs, 1000)

    s2 = round(s2)
    ms2 = round(ms2)


    # Save textfile
    create_evaluation_textfile(PATH=EVALUATION_TEXTFILE_PATH,
                               model=model,
                               dataset=dataset,
                               BATCH=EVALUATION_BATCHES,
                               list_classes=list_classes,
                               pre_trained_encoder=pre_trained_encoder,
                               indices=indices,
                               load_weights_by_name=load_weights_by_name,
                               evaluation=evaluation,
                               hours=h, minutes=m, seconds=s,
                               seconds2=s2, mseconds2=ms2,
                               nr_imgs=nr_imgs,
                               class_names=class_names)

    print("Prediction done. It took {} hours, {} minutes and {} seconds.".format(round(h), round(m), round(s)))
    print("Average run time per image: {} seconds and {} milliseconds.\n".format(round(s2), round(ms2)))
    print("Evaluation loss: {:.3}".format(evaluation[0]))
    print("Evaluation categorical accuracy: {:.3}% \n".format(evaluation[1]*100))
    print("Evaluation MIoU accuracy: {:.3f}%".format(evaluation[2] * 100))

    print("Evaluation IoU accuracy per class:")
    for name, acc in zip(class_names, evaluation[3:]):
        print("{:5.1f}%: {}".format(acc * 100, name))

#######################################################################################################################
''' PREDICT IMAGES FOR ALL MODELS '''
#######################################################################################################################


def segment_image(dataset_weights,
                  dataset_images,
                  INPUT_SHAPE,
                  SEGMENT_IMAGES_PATH,
                  SEGMENT_DEPTH_PATH,
                  SEGMENT_SAVE_PATH,
                  list_classes,
                  indices,
                  label_to_color,
                  weight_flag):

    imgs_names = os.listdir(SEGMENT_IMAGES_PATH)
    nr_imgs = len(imgs_names)


    if nr_imgs == 0:
        raise ValueError('There is no images in the prediction folder: \n {}'.format(SEGMENT_IMAGES_PATH))

    print('\n###############################################################################')
    print('Start Segmentation with {} images \n For weights from dataset "{}" \n On images from dataset "{}".'.format(nr_imgs, dataset_weights, dataset_images))
    print('###############################################################################\n')

    nbr_classes = len(list_classes) + 1
    prediction_tot = 0

    models = ['SegNetModel', 'dSegNetModel', 'DispSegNetModel', 'DispSegNetBasicModel', 'PydSegNetModel', 'EncFuseModel']

    for i, img_name in enumerate(imgs_names):

        for model in models:
            if weight_flag == 'True':
                WEIGHTS_PATH = "/WeightModels/exjobb/WorthyWeights/" + model + dataset_weights + "/weights.hdf5"
            else:
                WEIGHTS_PATH = "/WeightModels/exjobb/" + model + dataset_weights + "/weights.hdf5"


            # Create model instance
            pre_trained_encoder = True
            load_weights_by_name = False
            model_instance = initilize_model(model=model,
                                             INPUT_SHAPE=INPUT_SHAPE,
                                             nbr_classes=nbr_classes,
                                             pre_trained_encoder=pre_trained_encoder,
                                             indices=indices,
                                             weights=WEIGHTS_PATH,
                                             load_weights_by_name=load_weights_by_name)

            # Create model
            Model = model_instance.create_model()

            prediction_time = time.time()

            if model != 'SegNetModel':
                img, depth = get_imgs_and_depth_seg(img_name, SEGMENT_IMAGES_PATH,
                                                       SEGMENT_DEPTH_PATH, INPUT_SHAPE)

                predictions_softmax = Model.predict([img, depth])

            else:
                img = get_imgs_seg(img_name, SEGMENT_IMAGES_PATH, INPUT_SHAPE)

                predictions_softmax = Model.predict(img)

            new_list = list_classes
            new_list[0] = [0, 0, 0]

            predictions = get_images_from_softmax(predictions_softmax, new_list)

            # Convert predictions to rgb
            rgb_converter = ColorConverter(new_list)
            predictions_rgb = rgb_converter.convert_all_images(predictions)

            # Save predictions
            cv2.imwrite(SEGMENT_SAVE_PATH + '/' + model + '_' + dataset_images + 'ImgsOn' + dataset_weights + 'Weights_' + img_name, predictions_rgb[0])

            prediction_time = (time.time() - prediction_time)

            prediction_tot = prediction_tot + prediction_time

        m, s = divmod(prediction_tot, 60)
        h, m = divmod(m, 60)
        h = round(h)
        m = round(m)
        s = round(s)

        print("Prediction done for img {}/{}. It took {} hours, {} minutes and {} seconds.".format(i+1, nr_imgs, round(h), round(m), round(s)))
        prediction_tot = 0






#######################################################################################################################
''' EVALUATE DISTRIBUTION '''
#######################################################################################################################


def get_accurcay_distribution(model,
                              dataset,
                              INPUT_SHAPE,
                              EVALUATION_IMAGES_PATH,
                              EVALUATION_LABELS_PATH,
                              DISTRUBUTATION_TEXTFILE_PATH,
                              WEIGHTS_PATH,
                              opt,
                              list_classes,
                              indices,
                              READABLE_DISTRUBUTATION_TEXTFILE_PATH,
                              class_names_dist,
                              EVALUATION_DEPTH_PATH):

    imgs_names = os.listdir(EVALUATION_IMAGES_PATH)
    nr_imgs = len(imgs_names)

    if nr_imgs == 0:
        raise ValueError('There is no images in the evaluation folder: \n {}'.format(EVALUATION_IMAGES_PATH))


    print('\n###############################################################################')
    print('Start accuracy distribution with {} images.'.format(nr_imgs))
    print('###############################################################################\n')

    nbr_classes = len(list_classes) + 1

    pre_trained_encoder = False
    load_weights_by_name = False
    # Create model instance
    model_instance = initilize_model(model=model,
                                     INPUT_SHAPE=INPUT_SHAPE,
                                     nbr_classes=nbr_classes,
                                     pre_trained_encoder=pre_trained_encoder,
                                     indices=indices,
                                     weights=WEIGHTS_PATH,
                                     load_weights_by_name=load_weights_by_name)

    # Create model
    Model = model_instance.create_model()

    # Create the metric list (that include categorical accuracy and the Mean IoU for all classes)
    metric = [Mean_iou]

    for classes in range(nbr_classes):
        metric.append(Class_iou(classes))

    # Compile
    Model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=metric) # Mean_iou

    evaluation_time = time.time()

    confusion = np.zeros((nbr_classes, nbr_classes))
    confusion_ratio = np.zeros((nbr_classes, nbr_classes))

    class_accuracy = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],

        5: [],
        6: [],
        7: [],
        8: [],
        9: [],

        10: [],
        11: [],
        12: [],
        13: [],
        14: [],

        15: [],
        16: [],
        17: [],
        18: [],
        'mean_iou': [],
        'confusion_iou': []
    }
    metric_iou = []

    for i, img_name in enumerate(imgs_names):

        if np.mod(i+1, 10) == 0:
            print('Image {}/{}'.format(i+1, len(imgs_names)))

        if model != 'SegNetModel':
            img, label, depth = get_imgs_and_depth(img_name, EVALUATION_IMAGES_PATH, EVALUATION_LABELS_PATH,
                                                   EVALUATION_DEPTH_PATH, INPUT_SHAPE, list_classes, nbr_classes)

            # Evaluation part
            evaluation = Model.evaluate(x=[img, depth],
                                        y=label,
                                        verbose=0)

            label_pred = Model.predict([img, depth])


        else:
            img, label = get_imgs(img_name, EVALUATION_IMAGES_PATH, EVALUATION_LABELS_PATH, INPUT_SHAPE, list_classes, nbr_classes)

            # Evaluation part
            evaluation = Model.evaluate(x=img,
                                        y=label,
                                        verbose=0)

            label_pred = Model.predict(img)

        # Confusion matrix
        label_pred = np.argmax(label_pred, axis=-1)
        label_true = np.argmax(label, axis=-1)
        confusion_img = confusion_matrix(label_true.flatten(), label_pred.flatten(), labels=range(nbr_classes))
        confusion = confusion + confusion_img

        # Create bool matrix that shows the presence of classes in the data
        class_presence = np.union1d(np.unique(label_true.flatten()), np.unique(label_pred.flatten()))
        class_accuracy['mean_iou'].append(evaluation[1])

        summed_iou = 0
        summed_metric_iou = 0
        for classen in range(nbr_classes):
            if classen in class_presence:
                TF = confusion_img[classen, classen]
                denominator = np.sum(confusion_img[:, classen]) + np.sum(confusion_img[classen, :]) - TF
                iou_class = TF / denominator
                summed_iou = summed_iou + iou_class
                class_accuracy[classen].append(evaluation[classen + 2])
                summed_metric_iou = summed_metric_iou + evaluation[classen + 2]
            else:
                class_accuracy[classen].append(float('nan'))

        conf_iou = summed_iou / len(class_presence)

        # IoU based on evaluation class iou
        metric_iou.append(summed_metric_iou / len(class_presence))
        # IoU based on confusion matrix
        class_accuracy['confusion_iou'].append(conf_iou)

        # print("Mean Confusion iou for current img: {}".format(conf_iou))
        # print("Mean iou for current img: {}".format(evaluation[1]))


    print("Mean Iou:")
    print(np.round(np.nanmean(class_accuracy['mean_iou'])*100, decimals=2))
    print("Confusion Iou:")
    print(np.round(np.nanmean(class_accuracy['confusion_iou'])*100, decimals=2))
    print("Metric Iou:")
    print(np.round(np.nanmean(metric_iou)*100, decimals=2))


    for i in range(nbr_classes):
        row = np.sum(confusion[i, :])
        if row != 0:
            confusion_ratio[i, :] = confusion[i, :] / row

    print('Confusion ratio matrix')
    print(np.int32(confusion_ratio*100))
    confusion_ratio = np.round(confusion_ratio*100, decimals=2)
    evaluation_time = (time.time() - evaluation_time)
    m, s = divmod(evaluation_time, 60)
    h, m = divmod(m, 60)
    h = round(h)
    m = round(m)
    s = round(s)

    s2, ms2 = divmod(1000*evaluation_time/nr_imgs, 1000)

    s2 = round(s2)
    ms2 = round(ms2)


    # Save textfile
    create_distribution_textfile(PATH=DISTRUBUTATION_TEXTFILE_PATH,
                                 model=model,
                                 dataset=dataset,
                                 list_classes=list_classes,
                                 hours=h, minutes=m, seconds=s,
                                 seconds2=s2, mseconds2=ms2,
                                 nr_imgs=nr_imgs,
                                 confusion=confusion_ratio)

    create_readable_distribution_textfile(PATH=READABLE_DISTRUBUTATION_TEXTFILE_PATH, accuracy_dict=class_accuracy,
                                          class_names_dist=class_names_dist)


    print("Distribution calculation done. It took {} hours, {} minutes and {} seconds.".format(round(h), round(m), round(s)))
    print("Average run time per image: {} seconds and {} milliseconds.\n".format(round(s2), round(ms2)))

    create_all_hist(model, dataset, DISTRUBUTATION_TEXTFILE_PATH)







