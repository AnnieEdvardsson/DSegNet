
import numpy as np
import cv2
import os
from monodepth_loss import MonoLoss

num_scale = 4
alpha = 0.85

input_path_left = "/MLDatasetsStorage/exjobb/test_stero/left/"
input_path_right = "/MLDatasetsStorage/exjobb/test_stero/right/"

batch_list = os.listdir(input_path_left)
batch_size = len(batch_list)
batch_indices = np.arange(0, batch_size)
w = 375
h = 1242
batch_img_left = np.zeros(shape=(batch_size, w, h, 3))
batch_img_right = np.zeros(shape=(batch_size, w, h, 3))
batch_img_left_gray = np.zeros(shape=(batch_size, w, h))
batch_img_right_gray = np.zeros(shape=(batch_size, w, h))

for i in range(batch_size):
    img_name = batch_list[batch_indices[i]]
    image_left = cv2.imread(os.path.join(input_path_left, img_name))
    batch_img_left[i] = image_left

    image_left = cv2.imread(os.path.join(input_path_left, img_name), 0)
    batch_img_left_gray[i] = image_left


batch_list = os.listdir(input_path_right)
for i in range(batch_size):
    img_name = batch_list[batch_indices[i]]
    image_right = cv2.imread(os.path.join(input_path_right, img_name))
    batch_img_right[i] = image_right

    image_right = cv2.imread(os.path.join(input_path_right, img_name), 0)
    batch_img_right_gray[i] = image_left

def main():
    #scale_pyramid(image_left, num_scale)
    MonoLoss(batch_img_left, batch_img_right, batch_img_left_gray, batch_img_right_gray)

# /MLDatasetsStorage/exjobb/test_stero/left


#def disparity_smoothness():

if __name__ == '__main__':
    main()