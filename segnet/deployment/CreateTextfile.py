import datetime


def create_textfile(SEGNET_SAVED_TEXTFILE_TRAIN, dataset):

    # Save textfile
    f= open(SEGNET_SAVED_TEXTFILE_TRAIN, "w+")
    f.write("Model: \r\n")
    f.write("Dataset: %s \r\n" % dataset)
    f.write("#Epochs: \r\n")
    f.write("Batch size: \r\n")
    f.write("Optimizer: \r\n")
    f.write("List of classes: \r\n")
    f.close()


def create_training_textfile(PATH, model, dataset, EPOCHS, BATCH, OPT, list_classes,
                             hours, minutes, seconds, learning_rate, momentum, indices):
    now = datetime.datetime.now()
    # Save textfile
    f = open(PATH, "w+")
    f.write("TRAINING PROPERTIES \r\n")
    f.write("##################################################### \r\n")
    f.write("Model: %s \r\n" % model)
    f.write("Dataset: %s \r\n" % dataset)
    f.write("Date & Time: %s \r\n" % (now.strftime("%Y-%m-%d %H:%M")))
    f.write("Run time: %s hours, %s minutes & %s seconds \r\n" % (hours, minutes, seconds))
    f.write("Nr epochs: %i \r\n" % EPOCHS)
    f.write("Learning rate: %i \r\n" % learning_rate)
    f.write("Momentum: %i \r\n" % momentum)
    f.write("Batch size: %i \r\n" % BATCH)
    f.write("Optimizer: %s \r\n" % OPT)
    f.write("List of classes: %s \r\n" % list_classes)
    f.write("Usage of maxpooling indices: %s \r\n" % indices)
    f.close()


def create_prediction_textfile(PATH, model, dataset, BATCH, list_classes, pre_trained_encoder, indices, load_weights_by_name):
    now = datetime.datetime.now()
    # Save textfile
    f = open(PATH, "w+")
    f.write("PREDICTION PROPERTIES \r\n")
    f.write("##################################################### \r\n")
    f.write("Model: %s \r\n" % model)
    f.write("Dataset: %s \r\n" % dataset)
    f.write("Time: %s \r\n" % (now.strftime("%Y-%m-%d %H:%M")))
    f.write("Batch size: %i \r\n" % BATCH)
    f.write("List classes: %s \r\n" % list_classes)
    f.write("Pre trained encoder: %s \r\n" % pre_trained_encoder)
    f.write("Pooling indices: %s \r\n" % indices)
    f.write("Load weights by name: %s \r\n" % load_weights_by_name)
    f.close()
