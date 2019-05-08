import os
import numpy as np
import cv2
import keras
import tensorflow as tf
from deployment.pydnet.utils import *
from deployment.pydnet.pydnet import *
from hyperparameters import *
from deployment.preprocessors import InstanceToOneHot, OneHotEncoding
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def get_imgs_and_depth(img_name, input_path, depth_path, input_shape):
    PYDNET_SAVED_WEIGHTS = '/WeightModels/exjobb/OldStuff/pydnet_weights/pydnet_org/pydnet'

    batch_disp_img = np.zeros(shape=(1, input_shape[0], input_shape[1], 1))


    image = cv2.imread(os.path.join(input_path, img_name))
    depth = cv2.imread(os.path.join(depth_path, img_name), 0)

    resized_depth = cv2.resize(depth, (input_shape[1], input_shape[0]))

    #Predict disparity-map from pydnet
    with tf.Graph().as_default():
        placeholders = {'im0': tf.placeholder(tf.float32, [None, None, None, 3], name='im0')}

        with tf.variable_scope("model") as scope:
            model = pydnet(placeholders)

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        loader = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            loader.restore(sess, PYDNET_SAVED_WEIGHTS)
            pyd_image = cv2.resize(image, (input_shape[1], input_shape[0])).astype(np.float32) / 255.
            pyd_image = np.expand_dims(pyd_image, 0)
            disp = sess.run(model.results[0], feed_dict={placeholders['im0']: pyd_image})
            #left_disp = disp[0, :, :, 0]*3.33
            disp_color = applyColorMap(disp[0, :, :, 0] * 20, 'plasma')
            gray_image = cv2.cvtColor(disp_color, cv2.COLOR_BGR2GRAY)

            # height, width = img.shape[:2]
            toShow = cv2.resize(gray_image * 255, (input_shape[1], input_shape[0]))

    disp_difference = np.sum(np.abs(resized_depth - np.round(toShow)))
    print(resized_depth)
    print(np.round(toShow))

    return disp_difference

model = "SegNetModel"
dataset = "KITTI"
hyperdict, _, _, _, _ = get_hyperparameter(model, dataset)
input_path = hyperdict['EVALUATION_IMAGES_PATH']
depth_path = hyperdict['EVALUATION_DEPTH_PATH']
input_shape = hyperdict['INPUT_SHAPE']
img_names = os.listdir(input_path)
disp_difference = []
for i in range(1):
    diff = get_imgs_and_depth(img_names[i], input_path, depth_path, input_shape)
    disp_difference.append(diff)

print(disp_difference)


