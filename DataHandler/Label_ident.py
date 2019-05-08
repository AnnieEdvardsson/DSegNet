import cv2
import os
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

NumbersALL = []

def update(DATASET):

    OUTPUT_DATA_ROOT_PATH = "/MLDatasetsStorage/exjobb/" + DATASET + "/labels/labelId"
    END_PATH = "/train"
    #END_PATH_DEPTH = ["/traindepth", "/valdepth", "/testdepth"]

    # Gray scale image, boolean, input, depth, output
    GRAYSCALE = True

    # Input, depth, output
    FILE_TYPE = ".png"

    return OUTPUT_DATA_ROOT_PATH, END_PATH,  GRAYSCALE, FILE_TYPE


DATASET = 'CityScapes'
OUTPUT_DATA_ROOT_PATH, END_PATH, GRAYSCALE, FILE_TYPE = update(DATASET)


image_name = os.listdir(OUTPUT_DATA_ROOT_PATH + END_PATH)

OUTPUT = cv2.imread(OUTPUT_DATA_ROOT_PATH + END_PATH + "/" + image_name[0])
#print(OUTPUT.shape)
uni = np.unique(OUTPUT)
#print("unique numbers in RGB read:" + str(uni))

OUTPUT_GRAY = cv2.imread(OUTPUT_DATA_ROOT_PATH + END_PATH + "/" + image_name[0], 0)
#print(OUTPUT_GRAY.shape)
uni_GRAY = np.unique(OUTPUT_GRAY)
#print("unique numbers in GRAY read:" + str(uni_GRAY))

for image in image_name:
    # OUTPUT_RGB = cv2.imread(OUTPUT_DATA_ROOT_PATH + END_PATH + "/" + image)
    # print("unique numbers in RGB read:" + str(np.unique(OUTPUT_RGB)))

    OUTPUT = cv2.imread(OUTPUT_DATA_ROOT_PATH + END_PATH + "/" + image, 0)
    # print("unique numbers in GRAY read:" + str(np.unique(OUTPUT)))
    Numbers = np.unique(OUTPUT)
    print("Unique numbers for current img: {}".format(Numbers))

    NumbersALL = np.hstack([NumbersALL, Numbers])


NumbersALL = np.unique(NumbersALL)
print("All the unique values in " + DATASET + " is: " + str(NumbersALL))

