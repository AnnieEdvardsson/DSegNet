import cv2
import os
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

NumbersALL = []

def update(KITTI):
    if KITTI is True:
        INPUT_DATA_ROOT_PATH = "/MLDatasetsStorage/exjobb/KITTI/images"
        OUTPUT_DATA_ROOT_PATH = "/MLDatasetsStorage/exjobb/KITTI/labels"
        END_PATH = ["/train", "/val", "/test"]
        END_PATH_DEPTH = ["/traindepth", "/valdepth", "/testdepth"]

        # Gray scale image, boolean, input, depth, output
        GRAYSCALE = False, True, True

         Input, depth, output
        FILE_TYPE = ".jpg", ".jpg", ".jpg"
    else:
        INPUT_DATA_ROOT_PATH = "/MLDatasetsStorage/Perceptron_Master_Dataset/Perceptron_Coarse_Training/images"
        OUTPUT_DATA_ROOT_PATH = "/MLDatasetsStorage/Perceptron_Master_Dataset/Perceptron_Coarse_Training/labels"
        END_PATH = ["/train/BDD100k", "/val/BDD100k", "/test/BDD100k"]
        #END_PATH_DEPTH = ["/traindepth/BDD100k", "/valdepth/BDD100k", "/testdepth/BDD100k"]

        # Gray scale image, boolean, input, depth, output
        GRAYSCALE = False, True, True

        # Input, depth, output
        FILE_TYPE = ".jpg", ".jpg", ".png"

    #return INPUT_DATA_ROOT_PATH, OUTPUT_DATA_ROOT_PATH, END_PATH, END_PATH_DEPTH, GRAYSCALE, FILE_TYPE
    return INPUT_DATA_ROOT_PATH, OUTPUT_DATA_ROOT_PATH, END_PATH,  GRAYSCALE, FILE_TYPE

# BDD100
KITTI = False
INPUT_DATA_ROOT_PATH, OUTPUT_DATA_ROOT_PATH, END_PATH, END_PATH_DEPTH, GRAYSCALE, FILE_TYPE = update(KITTI)
INPUT_DATA_ROOT_PATH, OUTPUT_DATA_ROOT_PATH, END_PATH, GRAYSCALE, FILE_TYPE = update(KITTI)

image_name = os.listdir(INPUT_DATA_ROOT_PATH + END_PATH[0])
image = os.path.splitext(image_name[1])[0]
BDD100_INPUT = cv2.imread(INPUT_DATA_ROOT_PATH + END_PATH[0] + "/" + image + FILE_TYPE[0])

image_name = os.listdir(INPUT_DATA_ROOT_PATH + END_PATH_DEPTH[1])
image = os.path.splitext(image_name[1])[0]
BDD100_DEPTH = cv2.imread(INPUT_DATA_ROOT_PATH + END_PATH_DEPTH[1] + "/" + image + FILE_TYPE[1])

image_name = os.listdir(OUTPUT_DATA_ROOT_PATH + END_PATH[2])
image = os.path.splitext(image_name[1])[0]
BDD100_OUTPUT = cv2.imread(OUTPUT_DATA_ROOT_PATH + END_PATH[2] + "/" + image + FILE_TYPE[2])


# KITTI
KITTI = True
INPUT_DATA_ROOT_PATH, OUTPUT_DATA_ROOT_PATH, END_PATH, END_PATH_DEPTH, GRAYSCALE, FILE_TYPE = update(KITTI)
INPUT_DATA_ROOT_PATH, OUTPUT_DATA_ROOT_PATH, END_PATH, GRAYSCALE, FILE_TYPE = update(KITTI)


image_name = os.listdir(INPUT_DATA_ROOT_PATH + END_PATH[0])
for i in range(len(image_name)):

    image = os.path.splitext(image_name[i])[0]
    KITTI_INPUT = cv2.imread(INPUT_DATA_ROOT_PATH + END_PATH[0] + "/" + image + FILE_TYPE[0])

    if len(KITTI_INPUT.shape) is not len(BDD100_INPUT.shape):
        print("For image i={}, INPUT image size KITTI: {}, BDD100: {}.".format(i, KITTI_INPUT.shape, BDD100_INPUT.shape))


image_name = os.listdir(INPUT_DATA_ROOT_PATH + END_PATH_DEPTH[1])
for i in range(len(image_name)):

    image = os.path.splitext(image_name[i])[0]
    KITTI_DEPTH = cv2.imread(INPUT_DATA_ROOT_PATH + END_PATH_DEPTH[1] + "/" + image + FILE_TYPE[1])

    if len(KITTI_DEPTH.shape) is not len(BDD100_DEPTH.shape):
        print("For i={}, DEPTH image size KITTI: {}, BDD100: {}.".format(i, KITTI_DEPTH.shape, BDD100_DEPTH.shape))


image_name = os.listdir(OUTPUT_DATA_ROOT_PATH + END_PATH[2])
for i in range(len(image_name)):

    image = os.path.splitext(image_name[i])[0]
    KITTI_OUTPUT = cv2.imread(OUTPUT_DATA_ROOT_PATH + END_PATH[2] + "/" + image + FILE_TYPE[2], 0)

    if len(KITTI_OUTPUT.shape) is not len(BDD100_OUTPUT.shape):
        print("For i={}, OUTPUT image size KITTI: {}, BDD100: {}.".format(i, KITTI_OUTPUT.shape, BDD100_OUTPUT.shape))


    Numbers = np.unique(KITTI_OUTPUT)

    NumbersALL = np.hstack([NumbersALL, Numbers])

NumbersALL = np.unique(NumbersALL)
print(NumbersALL)

