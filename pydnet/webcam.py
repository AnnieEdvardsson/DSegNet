#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils import *
from pydnet import *

# forces tensorflow to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--resolution', dest='resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
# parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/IROS18/pydnet', help='checkpoint directory')
parser.add_argument('--case', dest='case', type=int, default='2', help='case=1 image / case=2 cam')
parser.add_argument('--scale', dest='scale', type=int, default='2', help='scale size ')
parser.add_argument('--image', dest='NR', type=str, default='2', help='image nr')
parser.add_argument('--Comp', dest='Comp', type=str,
                    default='ML', help='Which computer, ML/AE/MT - chooses path to weights')
parser.add_argument('--Cuda', dest='Cuda', type=int,
                    default='1', help='Which cuda to run on ')

args = parser.parse_args()

PYDNET_SAVED_WEIGHTS = {"ML": "/WeightModels/exjobb_SecretStuff_AnnieAndMartin/pydnet_weights/pydnet",
                        "AE": "C:/Users/s26915/Documents/pydnet/checkpoint/IROS18/pydnet",
                        "MT": "weights/Segnet_perceptron_general_gta_swap_weights-lowest_loss.hdf5"}
#weights=PYDNET_SAVED_WEIGHTS[Comp]

def main(_):

  with tf.Graph().as_default():
    height = args.height
    width = args.width
    case = args.case
    scale = args.scale
    NR = args.NR
    Comp = args.Comp
    Cuda = args.Cuda


    if Comp=='ML':
      import os
      os.environ["CUDA_VISIBLE_DEVICES"] = str(Cuda)

    placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}

    with tf.variable_scope("model") as scope:
      model = pydnet(placeholders)

    init = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

    loader = tf.train.Saver()
    saver = tf.train.Saver()
    if case == 1:

      with tf.Session() as sess:
        sess.run(init)
        loader.restore(sess, PYDNET_SAVED_WEIGHTS[Comp])
        while True:
          link = 'Road_ex' + NR + '.jpg'
          img = cv2.imread(link)

          img = cv2.resize(img, (width*scale, height*scale)).astype(np.float32) / 255.
          img = np.expand_dims(img, 0)
          disp = sess.run(model.results[args.resolution - 1], feed_dict={placeholders['im0']: img})

          disp_color = applyColorMap(disp[0, :, :, 0] * 20, 'plasma')
          img = np.squeeze(img, axis=0)

          toShow = (np.concatenate((img, disp_color), 0) * 255.).astype(np.uint8)
          toShow = cv2.resize(toShow, (round(width * 2.5 // 2), round(height * 2.5)))

          cv2.imshow('pydnet', toShow)
          k = cv2.waitKey(1)
          if k == 1048603 or k == 27:
            break  # esc to quit

          del img
          del disp
          del toShow

    else:
      cam = cv2.VideoCapture(0)

    with tf.Session() as sess:
        sess.run(init)
        loader.restore(sess, PYDNET_SAVED_WEIGHTS[Comp])
        while True:
          for i in range(4):
            cam.grab()
          ret_val, img = cam.read()
          img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
          img = np.expand_dims(img, 0)
          start = time.time()
          disp = sess.run(model.results[args.resolution-1], feed_dict={placeholders['im0']: img})
          end = time.time()

          disp_color = applyColorMap(disp[0,:,:,0]*20, 'plasma')
          img = np.squeeze(img, axis=0)
          toShow = (np.concatenate((img, disp_color), 0)*255.).astype(np.uint8)
          toShow = cv2.resize(toShow, (width//2, height))

          cv2.imshow('pydnet', toShow)
          k = cv2.waitKey(1)
          if k == 1048603 or k == 27:
            break  # esc to quit
          if k == 1048688:
            cv2.waitKey(0) # 'p' to pause

          print("Time: " + str(end - start))
          del img
          del disp
          del toShow

        cam.release()

if __name__ == '__main__':
    tf.app.run()
