import os
import numpy as np
import pandas as pd
import cv2
import keras
import random
import sys
import tensorflow as tf
from deployment.pydnet.utils import *
from deployment.pydnet.pydnet import *

from deployment.preprocessors import InstanceToOneHot, OneHotEncoding, InstanceToOneHot_rgb


# Loads the image and label, pre-process the labels to right classes and one-hot-encoding
def segnet_train_data_generator(input_path, output_path, batch_size, input_shape, list_classes, nbr_classes, dataset, task):

    # Create list of directory
    img_list = os.listdir(input_path)
    nbr_images = len(img_list)
    img_indices = list(range(len(img_list)))

    # Initialize the pre-process classes
    # if task == "train":
    #     print("Using rgb list")
    #preprocessor_inst = InstanceToOneHot_rgb(class_order=list_classes)
    # else:
    preprocessor_inst = InstanceToOneHot(class_order=list_classes)

    preprocessor_one_hot = OneHotEncoding(total_number_classes=nbr_classes)

    # Load the ordering of the dataset
    index_path = "/MLDatasetsStorage/exjobb/" + dataset + "/RandomBatches.cvs"
    index_df = pd.read_csv(index_path)
    epoch = 1
    counter = 0
    index_list = index_df[str(epoch)].values
    while True:
        # Indexes for batch

        if task == "eval":
            batch_indices = random.sample(img_indices, batch_size)
            for delete_ind in batch_indices:
                img_indices.remove(delete_ind)
            if len(img_indices) < batch_size:
                img_indices = list(range(len(img_list)))
            elif len(img_indices) is None:
                img_indices = list(range(len(img_list)))
            #print("img_name eval:" + img_list[batch_indices])
        else:
            if counter + 1 == nbr_images:
                print("Counter value: {} \n nbr_images: {}".format(counter+1, nbr_images))
                epoch = epoch + 1
                counter = 0
                index_list = index_df[str(epoch)].values
                print("Next epoch initialized")

            lim = counter + batch_size
            batch_indices = index_list[counter:lim]
            counter = counter + batch_size


        # initialize batch_img and batch_labels to the correct shape
        batch_img = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], 3))
        batch_label = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], nbr_classes))

        for i in range(batch_size):
            # Extract image name and generate it's corresponding label name
            img_name = img_list[batch_indices[i]]
            if dataset == "CityScapes":
                label_name = img_name.replace("leftImg8bit.jpg", "gtFine_labelIds.png")
            else:
                label_name = img_name.replace(".jpg", ".png")


            # Read image and label
            # if task == "train":
            #label_bgr = cv2.imread(os.path.join(output_path, label_name))
            #label = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2RGB)
            # else:
            label = cv2.imread(os.path.join(output_path, label_name), 0)
            image = cv2.imread(os.path.join(input_path, img_name))

            # Resize image and label
            new_image = cv2.resize(image, (input_shape[1], input_shape[0]))
            new_label = cv2.resize(label, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)

            # Call pre-processes
            new_label, _ = preprocessor_inst.pre_process(new_label)
            # print("Size of label shape before one_hot: {}".format(np.shape(new_label)))
            new_label, _ = preprocessor_one_hot.pre_process(new_label)

            # Add updated image and label to the return parameters
            batch_img[i] = new_image
            batch_label[i] = new_label

        yield batch_img, batch_label


def dsegnet_train_data_generator(input_path, output_path, depth_path, batch_size, input_shape, list_classes, nbr_classes, dataset, task):
    #PYDNET_SAVED_WEIGHTS = '/WeightModels/exjobb/OldStuff/pydnet_weights/pydnet_org/pydnet'
    # Create list of directory
    img_list = os.listdir(input_path)
    img_indices = list(range(len(img_list)))
    nbr_images = len(img_list)

    # Initialize the pre-process classes
    preprocessor_inst = InstanceToOneHot(class_order=list_classes)
    preprocessor_one_hot = OneHotEncoding(total_number_classes=nbr_classes)

    # Load the ordering of the dataset
    index_path = "/MLDatasetsStorage/exjobb/" + dataset + "/RandomBatches.cvs"
    index_df = pd.read_csv(index_path)
    epoch = 1
    counter = 0
    index_list = index_df[str(epoch)].values

    # Predict disparity-map from pydnet
    # with tf.Graph().as_default():
    #     placeholders = {'im0': tf.placeholder(tf.float32, [None, None, None, 3], name='im0')}
    #
    #     with tf.variable_scope("model") as scope:
    #         model = pydnet(placeholders)
    #
    #     init = tf.group(tf.global_variables_initializer(),
    #                     tf.local_variables_initializer())
    #
    #     loader = tf.train.Saver()

    # Pre-process output labels for the batches
    while True:
        # Indexes for batch
        if task == "eval":
            batch_indices = random.sample(img_indices, batch_size)
            for delete_ind in batch_indices:
                img_indices.remove(delete_ind)
            if len(img_indices) < batch_size:
                img_indices = list(range(len(img_list)))
            elif len(img_indices) is None:
                img_indices = list(range(len(img_list)))
            #print("img_name eval:" + img_list[batch_indices])
        else:
            if counter + 1 == nbr_images:
                print("Counter value: {} \n nbr_images: {}".format(counter+1, nbr_images))
                epoch = epoch + 1
                counter = 0
                index_list = index_df[str(epoch)].values
                print("Next epoch initialized")

            lim = counter + batch_size
            batch_indices = index_list[counter:lim]
            counter = counter + batch_size

        # initialize batch_img and batch_labels to the correct shape
        batch_img = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], 3))
        batch_label = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], nbr_classes))
        batch_disp_img = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], 1))

        for i in range(batch_size):
            # Extract image name and generate it's corresponding label name
            img_name = img_list[batch_indices[i]]
            label_name = img_name.replace(".jpg", ".png")

            # Read image and label
            label = cv2.imread(os.path.join(output_path, label_name), 0)
            image = cv2.imread(os.path.join(input_path, img_name))
            left_disp = cv2.imread(os.path.join(depth_path, img_name), 0)

            # with tf.Session() as sess:
            #     sess.run(init)
            #     loader.restore(sess, PYDNET_SAVED_WEIGHTS)
            #     pyd_image = cv2.resize(image, (input_shape[1], input_shape[0])).astype(np.float32)
            #     pyd_image = np.expand_dims(pyd_image, 0)
            #     disp = sess.run(model.results[0], feed_dict={placeholders['im0']: pyd_image})
            #     left_disp = disp[0, :, :, 0]*3.33
            #     left_disp = np.expand_dims(left_disp, 3)

            # Resize image and label
            new_image = cv2.resize(image, (input_shape[1], input_shape[0]))
            new_label = cv2.resize(label, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)
            new_disp = cv2.resize(left_disp, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)
            new_disp = np.expand_dims(new_disp, 2)

            # Call pre-processes
            # new_label = preprocessor_class.pre_process(new_label)
            new_label, _ = preprocessor_inst.pre_process(new_label)
            new_label, _ = preprocessor_one_hot.pre_process(new_label)


            # Add updated image and label to the return parameters
            batch_img[i] = new_image
            batch_label[i] = new_label
            batch_disp_img[i] = new_disp

        yield [batch_img, batch_disp_img], batch_label

# Loads the image and label, pre-process the labels to right classes and one-hot-encoding
def segnet_pred_data_generator(input_path, batch_size, input_shape):

    # Create list of left and right images
    list_img = os.listdir(input_path)

    nbr_img = len(list_img)
    img_indices = list(range(nbr_img))

    while True:

        # Generate random batches and delete the images from img_indices so they cannot be drawn again
        batch_indices = random.sample(img_indices, batch_size)

        for delete_ind in batch_indices:
            img_indices.remove(delete_ind)
        if len(img_indices) < batch_size:
            img_indices = list(range(nbr_img))
        elif len(img_indices) is None:
            img_indices = list(range(nbr_img))

        # Preallocate tensors for input, output pairs
        input_img = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], 3))

        for i in range(batch_size):
            # Get image name in batch
            name = list_img[batch_indices[i]]

            # Read img
            img = cv2.imread(os.path.join(input_path, name))

            # Resize image [height, width, colour channel]
            new_img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)

            # Load into input_img batch
            input_img[i] = new_img

        yield input_img

def dsegnet_pred_data_generator(input_path, batch_size, input_shape):
    PYDNET_SAVED_WEIGHTS = '/WeightModels/exjobb/OldStuff/pydnet_weights/pydnet_org/pydnet'

    # Create list of left and right images
    list_img = os.listdir(input_path)

    nbr_img = len(list_img)
    img_indices = list(range(nbr_img))

    with tf.Graph().as_default():
        placeholders = {'im0': tf.placeholder(tf.float32, [None, None, None, 3], name='im0')}

        with tf.variable_scope("model") as scope:
            model = pydnet(placeholders)

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        loader = tf.train.Saver()

        while True:

            # Generate random batches and delete the images from img_indices so they cannot be drawn again
            batch_indices = random.sample(img_indices, batch_size)

            for delete_ind in batch_indices:
                img_indices.remove(delete_ind)
            if len(img_indices) < batch_size:
                img_indices = list(range(nbr_img))
            elif len(img_indices) is None:
                img_indices = list(range(nbr_img))

            # Preallocate tensors for input, output pairs
            input_img = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], 3))
            disp_img = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], 1))

            for i in range(batch_size):
                # Get image name in batch
                name = list_img[batch_indices[i]]

                # Read img
                img = cv2.imread(os.path.join(input_path, name))

                with tf.Session() as sess:
                    sess.run(init)
                    loader.restore(sess, PYDNET_SAVED_WEIGHTS)
                    pyd_image = cv2.resize(img, (input_shape[1], input_shape[0])).astype(np.float32)
                    pyd_image = np.expand_dims(pyd_image, 0)
                    disp = sess.run(model.results[0], feed_dict={placeholders['im0']: pyd_image})
                    left_disp = disp[0, :, :, 0]
                    left_disp = np.expand_dims(left_disp, 3)
                    # print("Max value from PydNet: {}".format(np.max(left_disp)))
                    # print("Min value from PydNet: {}".format(np.min(left_disp)))

                # Resize image [height, width, colour channel]
                new_img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)

                # Load into input_img batch
                input_img[i] = new_img
                disp_img[i] = left_disp

            yield [input_img, disp_img]


def initialize_generator(task, model, dataset, input_path, batch_size, input_shape, depth_path=str, output_path=str, list_classes=list, nbr_classes=int):
    if task == 'train':
        if model == 'SegNetModel':
            dataGenerator = segnet_train_data_generator(input_path,
                                                        output_path,
                                                        batch_size,
                                                        input_shape,
                                                        list_classes,
                                                        nbr_classes,
                                                        dataset,
                                                        task
                                                        )
        elif model in ['dSegNetModel', 'DispSegNetModel', 'DispSegNetBasicModel', 'PydSegNetModel', 'EncFuseModel']:
            dataGenerator = dsegnet_train_data_generator(input_path,
                                                         output_path,
                                                         depth_path,
                                                         batch_size,
                                                         input_shape,
                                                         list_classes,
                                                         nbr_classes,
                                                         dataset,
                                                         task
                                                         )
        else:
            raise NameError('Input a Model which are defined.. pleb')
    elif task == 'predict':
        if model == 'SegNetModel':
            dataGenerator = segnet_pred_data_generator(input_path,
                                                       batch_size,
                                                       input_shape
                                                       )
        elif model in ['dSegNetModel', 'DispSegNetModel', 'DispSegNetBasicModel', 'PydSegNetModel', 'EncFuseModel']:
            dataGenerator = dsegnet_pred_data_generator(input_path,
                                                        batch_size,
                                                        input_shape
                                                        )
        else:
            raise NameError('Input a Model which are defined.. pleb')
    elif task == "eval":
        if model == 'SegNetModel':
            dataGenerator = segnet_train_data_generator(input_path,
                                                        output_path,
                                                        batch_size,
                                                        input_shape,
                                                        list_classes,
                                                        nbr_classes,
                                                        dataset,
                                                        task
                                                        )
        elif model in ['dSegNetModel', 'DispSegNetModel', 'DispSegNetBasicModel', 'PydSegNetModel', 'EncFuseModel']:
            dataGenerator = dsegnet_train_data_generator(input_path,
                                                         output_path,
                                                         depth_path,
                                                         batch_size,
                                                         input_shape,
                                                         list_classes,
                                                         nbr_classes,
                                                         dataset,
                                                         task
                                                         )
        else:
            raise NameError('Input a Model which are defined.. pleb')

    return dataGenerator
