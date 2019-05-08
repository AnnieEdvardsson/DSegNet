import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils import *
from training_code.pydnet import *

# forces tensorflow to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

resolution = 1
DATASET = 'CityScapes'

PYDNET_SAVED_WEIGHTS = "/WeightModels/exjobb/OldStuff/pydnet_weights/pydnet_org/pydnet"

INPUT_DATA_PATH = ['/MLDatasetsStorage/exjobb/' + DATASET + '/images/train/',
                   '/MLDatasetsStorage/exjobb/' + DATASET + '/images/test/',
                   '/MLDatasetsStorage/exjobb/' + DATASET + '/images/val/']

OUTPUT_DATA_PATH = ['/MLDatasetsStorage/exjobb/' + DATASET + '/depth/train/',
                   '/MLDatasetsStorage/exjobb/' + DATASET + '/depth/test/',
                   '/MLDatasetsStorage/exjobb/' + DATASET + '/depth/val/']

type = ['train', 'test', 'val']


def main(_):

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

      for i, (INPUT_PATH, OUTPUT_PATH) in enumerate(zip(INPUT_DATA_PATH, OUTPUT_DATA_PATH)):
        image_name = os.listdir(INPUT_PATH)

        # Removes all files in output folders
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        for file in os.listdir(OUTPUT_PATH):
            os.remove(OUTPUT_PATH + file)

        print('Removed all pre-existing images in: ' + type[i] + ' folder')

        for image in image_name:
          width = 512
          height = 384

          img = cv2.imread(INPUT_PATH + image)

          img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
          img = np.expand_dims(img, 0)
          disp = sess.run(model.results[resolution - 1], feed_dict={placeholders['im0']: img})

          disp_color = applyColorMap(disp[0, :, :, 0] * 20, 'plasma')
          gray_image = cv2.cvtColor(disp_color, cv2.COLOR_BGR2GRAY)

          #height, width = img.shape[:2]
          toShow = cv2.resize(gray_image*255, (width, height))



          cv2.imwrite(OUTPUT_PATH + image, toShow)

        print('Estimated depth for images in: ' + type[i] + ' folder \n')


if __name__ == '__main__':
    tf.app.run()



