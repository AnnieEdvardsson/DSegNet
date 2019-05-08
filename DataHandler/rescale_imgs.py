
import numpy as np
import cv2
import os

dataset = 'CityScapes'
SEGMENT_IMAGES_PATH = "/MLDatasetsStorage/exjobb/" + dataset + "/imgs2seg/images"
SEGMENT_DEPTH_PATH = "/MLDatasetsStorage/exjobb/" + dataset + "/imgs2seg/depth"

imgs_names = os.listdir(SEGMENT_IMAGES_PATH)
nr_imgs = len(imgs_names)
input_shape = (512, 384, 3)


for img_name in imgs_names:

    image = cv2.imread(os.path.join(SEGMENT_IMAGES_PATH, img_name))
    depth = cv2.imread(os.path.join(SEGMENT_DEPTH_PATH, img_name), 0)

    resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))
    resized_depth = cv2.resize(depth, (input_shape[1], input_shape[0]))

    cv2.imwrite(SEGMENT_IMAGES_PATH + '/resized_Img_' + img_name, resized_image)
    cv2.imwrite(SEGMENT_DEPTH_PATH + '/resized_Depth_' + img_name, resized_depth)
