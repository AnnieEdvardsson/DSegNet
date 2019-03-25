import os
import numpy as np
import cv2
import keras
import random

from deployment.preprocessors import InstanceToOneHot, OneHotEncoding


# Loads the image and label, pre-process the labels to right classes and one-hot-encoding
def data_generator(input_path, output_path, batch_size, input_shape, list_classes, nbr_classes):

    # Create list of directory
    img_list = os.listdir(input_path)
    img_indices = list(range(len(img_list)))

    # Initialize the pre-process classes
    preprocessor_inst = InstanceToOneHot(class_order=list_classes)
    preprocessor_one_hot = OneHotEncoding(total_number_classes=nbr_classes)

    # Pre-process output labels for the batches
    while True:
        # Indexes for batch
        # batch_indices = np.random.randint(low=0, high=len(img_list), size=batch_size)
        batch_indices = random.sample(img_indices, batch_size)
        for delete_ind in batch_indices:
            img_indices.remove(delete_ind)
        if len(img_indices) < batch_size:
            print(" Nice, indices reset for next epoch!")
            img_indices = list(range(len(img_list)))
        elif len(img_indices) is None:
            print("Nice")
            img_indices = list(range(len(img_list)))

        # initialize batch_img and batch_labels to the correct shape
        batch_img = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], 3))
        batch_label = np.zeros(shape=(batch_size, input_shape[0], input_shape[1], nbr_classes))

        for i in range(batch_size):
            # Extract image name and generate it's corresponding label name
            img_name = img_list[batch_indices[i]]
            label_name = img_name.replace(".jpg", ".png")

            # Read image and label
            label = cv2.imread(os.path.join(output_path, label_name), 0)
            image = cv2.imread(os.path.join(input_path, img_name))

            # Resize image and label
            new_image = cv2.resize(image, (input_shape[1], input_shape[0]))
            new_label = cv2.resize(label, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)

            # Call pre-processes
            # new_label = preprocessor_class.pre_process(new_label)
            new_label, _ = preprocessor_inst.pre_process(new_label)
            new_label, _ = preprocessor_one_hot.pre_process(new_label)


            # Add updated image and label to the return parameters
            batch_img[i] = new_image
            batch_label[i] = new_label

        yield batch_img, batch_label
