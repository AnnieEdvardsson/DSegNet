
import time
import argparse
from keras import optimizers
from deployment.models import initilize_model
from deployment.data_readers import generate_evaluation_batches, generate_prediction_batch
from deployment.dataGenerator import data_generator
from deployment.postprocessors import get_images_from_softmax
from deployment.data_writer import save_predictions, save_images_with_predictions
from deployment.CreateTextfile import create_prediction_textfile, create_training_textfile
from deployment.CreateOptimizer import OptimizerCreator
from deployment.postprocessors import ColorConverter
from deployment.models import SegNetModel
import os


#######################################################################################################################
''' TRAIN MODEL '''
#######################################################################################################################


def train_model(model,
                INPUT_SHAPE,
                dataset,
                TRAINING_IMAGES_PATH,
                TRAINING_LABELS_PATH,
                TRAINING_TEXTFILE_PATH,
                WEIGHTS_PATH,
                TRAINING_BATCHES,
                TRAINING_EPOCHS,
                OPTIMIZER,
                list_classes,
                learning_rate,
                momentum,
                indices):

    nbr_classes = len(list_classes) + 1

    # Initialize model
    model_instance = initilize_model(model=model,
                                     INPUT_SHAPE=INPUT_SHAPE,
                                     nbr_classes=nbr_classes,
                                     pre_trained_encoder=True,
                                     indices=False,
                                     weights=None,
                                     load_weights_by_name=False)

    # Create model
    model_instance = model_instance.create_model()

    #segnet.summary()

    # Compile model
    opt_instance = OptimizerCreator(OPTIMIZER, learning_rate, momentum)
    opt, learning_rate, momentum = opt_instance.pick_opt()

    model_instance.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    # Train model
    # create batch_name by taking random names,
    trainGen = data_generator(input_path=TRAINING_IMAGES_PATH,
                              output_path=TRAINING_LABELS_PATH,
                              batch_size=TRAINING_BATCHES,
                              input_shape=INPUT_SHAPE,
                              list_classes=list_classes,
                              nbr_classes=nbr_classes,
                              )

    print('\n#################################################################')
    print("Start Training..")
    print('#################################################################\n')

    training_time = time.clock()
    model_instance.fit_generator(trainGen,
                                 steps_per_epoch=len(os.listdir(TRAINING_LABELS_PATH)) // TRAINING_BATCHES,
                                 epochs=TRAINING_EPOCHS)

    training_time = (time.clock() - training_time)
    m, s = divmod(training_time, 60)
    h, m = divmod(m, 60)
    print("Training done. It took {} hours, {} minutes and {} seconds.".format(h, m, s))

    # Save weights and model
    model_instance.save_weights(WEIGHTS_PATH)

    # Save textfile
    create_training_textfile(PATH=TRAINING_TEXTFILE_PATH,
                             model=model,
                             dataset=dataset,
                             EPOCHS=TRAINING_EPOCHS,
                             BATCH=TRAINING_BATCHES,
                             OPT=OPTIMIZER,
                             list_classes=list_classes,
                             hours=h, minutes=m, seconds=s,
                             learning_rate=learning_rate,
                             momentum=momentum,
                             indices=indices)


#######################################################################################################################
''' PREDICT MODEL '''
#######################################################################################################################


def predict_model(model,
                dataset,
                INPUT_SHAPE,
                PREDICTION_IMAGES_PATH,
                PREDICTION_SAVE_PATH,
                PREDICTION_TEXTFILE_PATH,
                PREDICTION_BATCH_SIZE,
                WEIGHTS_PATH,
                list_classes,
                indices,
                  label_to_color):

    nbr_classes = len(list_classes) + 1

    # get images to predict
    input_images = generate_prediction_batch(input_path=PREDICTION_IMAGES_PATH,
                                             batch_size=PREDICTION_BATCH_SIZE,
                                             input_shape=INPUT_SHAPE)


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

    # Predictions
    print('\n#################################################################')
    print("Start Prediction..")
    print('#################################################################\n')

    prediction_time = time.clock()
    predictions_softmax = Model.predict(input_images)
    new_list = [0]
    new_list.extend(list_classes)
    predictions = get_images_from_softmax(predictions_softmax, new_list)

    # Save textfile
    create_prediction_textfile(PATH=PREDICTION_TEXTFILE_PATH,
                               model=model,
                               dataset=dataset,
                               BATCH=PREDICTION_BATCH_SIZE,
                               list_classes=list_classes,
                               pre_trained_encoder=pre_trained_encoder,
                               indices=indices,
                               load_weights_by_name=load_weights_by_name
                               )

    # Convert predictions to rgb
    rgb_converter = ColorConverter(label_to_color)
    predictions_rgb = rgb_converter.convert_all_images(predictions)

    # Save predictions
    save_predictions(predictions_rgb, PREDICTION_SAVE_PATH)
    save_images_with_predictions(input_images, predictions, PREDICTION_SAVE_PATH)

    prediction_time = (time.clock() - prediction_time)
    m, s = divmod(prediction_time, 60)
    h, m = divmod(m, 60)
    print("Prediction done. It took {} hours, {} minutes and {} seconds.".format(h, m, s))


#######################################################################################################################
''' EVALUATE MODEL '''
#######################################################################################################################


def evaluate_model(model,
                   INPUT_SHAPE,
                   EVALUATION_IMAGES_PATH,
                   EVALUATION_LABELS_PATH,
                   WEIGHTS_PATH,
                   EVALUATION_BATCH_SIZE,
                   OPTIMIZER,
                   list_classes,
                   indices,
                   learning_rate,
                   momentum,):
    nbr_classes = len(list_classes) + 1

    # Create model instance
    model_instance = initilize_model(model=model,
                                     INPUT_SHAPE=INPUT_SHAPE,
                                     nbr_classes=nbr_classes,
                                     pre_trained_encoder=True,
                                     indices=indices,
                                     weights=WEIGHTS_PATH,
                                     load_weights_by_name=False)

    # Create model
    Model = model_instance.create_model()

    # handle images for evaluation
    batch_images, batch_labels = generate_evaluation_batches(input_path=EVALUATION_IMAGES_PATH,
                                                             output_path=EVALUATION_LABELS_PATH,
                                                             batch_size=EVALUATION_BATCH_SIZE,
                                                             input_shape=INPUT_SHAPE,
                                                             list_classes=list_classes,
                                                             nbr_classes=nbr_classes)

    # Define optimizer
    opt_instance = OptimizerCreator(OPTIMIZER, learning_rate, momentum)
    opt, learning_rate, momentum = opt_instance.pick_opt()

    # Compile
    Model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    # Evaluate
    print('\n#################################################################')
    print("Start Evaulation..")
    print('#################################################################\n')

    evaluation_time = time.clock()
    evaluation = Model.evaluate(batch_images, batch_labels)

    evaluation_time = (time.clock() - evaluation_time)
    nbr_batch = len(batch_images)
    m, s = divmod(evaluation_time, 60)
    h, m = divmod(m, 60)
    print("Evaluation done.")
    print("Total time for batch size {}: {} hours, {} minutes and {} seconds.".format(nbr_batch, h, m, s))
    print("Evaluation loss: {:.3}%".format(evaluation[0]*100))
    print("Evaluation accuracy: {:.3}%".format(evaluation[1]*100))

