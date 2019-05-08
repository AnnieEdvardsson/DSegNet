
import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
import random
import cv2
#from utils import *
#from pydnet import *
import numpy as np

# forces Tensorflow to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
dataset = 'KITTI'

INPUT_DATA_PATH_IMAGE = '/MLDatasetsStorage/exjobb/' + dataset + '/KITTI_seg/train/image_2/'
OUTPUT_DATA_PATH_IMAGE = '/MLDatasetsStorage/exjobb/' + dataset +'/images/'

INPUT_DATA_PATH_LABEL = '/MLDatasetsStorage/exjobb/' + dataset + '/KITTI_seg/train/semantic_rgb/'
OUTPUT_DATA_PATH_LABEL = '/MLDatasetsStorage/exjobb/' + dataset + '/labels/'


def main():
    print("Running splitting..")
    type = ['train/', 'val/']
    ratio = [0.8, 0.2, 0]
    image_names = os.listdir(INPUT_DATA_PATH_IMAGE)
    random.shuffle(image_names)
    numbers = len(image_names)

    # Remove all pre-existing images in output directory
    for name in type:
        for file in os.listdir(OUTPUT_DATA_PATH_IMAGE + name):
            os.remove(OUTPUT_DATA_PATH_IMAGE + name + file)

        for file in os.listdir(OUTPUT_DATA_PATH_LABEL + name):
            os.remove(OUTPUT_DATA_PATH_LABEL + name + file)

    print("Removed all pre-existing images in INPUT and LABELS folders (train/test/val)")

    for i, image in enumerate(image_names):

        # Define end of path, train, test or val
        if i < numbers * ratio[0]:
            ind = 0
        elif i < numbers * (ratio[0] + ratio[1]):
            ind = 1
        else:
            ind = 2

        # Load png image and save jpg image
        img = cv2.imread(INPUT_DATA_PATH_IMAGE + image)
        cv2.imwrite(OUTPUT_DATA_PATH_IMAGE + type[ind] + image, img)

        label = cv2.imread(INPUT_DATA_PATH_LABEL + image)
        cv2.imwrite(OUTPUT_DATA_PATH_LABEL + type[ind] + image, label)

        del img, label
    print("Splitting completed.")

if __name__ == '__main__':
    main()
