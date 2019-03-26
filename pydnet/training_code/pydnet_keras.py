from keras.applications.vgg16 import VGG16
from keras.backend import gradients, sum, repeat_elements, shape, get_session
from keras.engine.topology import Layer
from keras.engine.training import Model as tModel
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D

import os
from keras import layers
from keras.applications.resnet50 import ResNet50
from typing import Tuple, Any, AnyStr, List
from keras.layers import Input, LeakyReLU, Concatenate, Conv2DTranspose, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D, Deconvolution2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.activations import relu, sigmoid
from layers import deconv2d_leaky
import tensorflow as tf


class PydNetModel(object):

    def __init__(self, shape=None, include_bias=True, weights=None, load_weights_by_name=False, kernel_init='he_normal', padding='same'):
        """
        :param shape: Shape of input images (h,w,channels)
        :param include_bias: Boolean, True will include bias in the nodes, False wont
        :param weights: The path of the weights in a .hdf5 file
        :param load_weights_by_name: Boolean, True to load weights
        :param kernel_init: Method to initialize the weights
        :param padding: How to pad, either "same" or "valid" (case-insensitive)
        """
        super().__init__()
        self.input_shape = shape
        self.weights_init = kernel_init
        self.bias = include_bias
        self.pre_trained_weights = weights  # path to a .hdf5 file with the weights that we want to use.
        # If None the weights are initialized according to "kernel_init"
        self.load_weights_by_name = load_weights_by_name
        self.padding = padding

    def bilinear_upsampling_by_deconvolution(self, x):
        """
        Upsample by deconvolution with a kernel size 2x2 with stride 2
        :param x: The layers which will be upsampled
        :return: Upsampled image which doubles in resolution
        """
        # Upsampling method
        upsample = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding=self.padding)(x)
        return upsample

    def pyramid_decoder(self, encoder_output, layer_name, decoder_from_lower_level=None):
        """
        Building the decoder for PydNet which have the same properties in every level, 4 convolutional layers with
        a decreasing number of filters, starting from 96, 64, 32 and 8. Every conv layers has a kernel size of 3x3
        and strides 1. If the function gets a decoder_from_lower_level it also upsamples the decoder and concatenate
        it with the encoder.

        :param encoder_output:              Estimates from encoder in a level in PydNet
        :param layer_name:                  The level in the pyramid
        :param decoder_from_lower_level:    Estimates from decoder in previous level
        :return:                            Decoded image in the scale-level
        """
        if decoder_from_lower_level is not None:
            x = self.bilinear_upsampling_by_deconvolution(decoder_from_lower_level)
            x = Concatenate()([encoder_output, x])
        else:
            x = encoder_output

        x = BatchNormalization()(Conv2D(96, 3, strides=1, kernel_initializer=self.weights_init, padding=self.padding,
                                        name="decoder" + layer_name + "_conv1", use_bias=self.bias)(x))
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(Conv2D(64, 3, strides=1, kernel_initializer=self.weights_init, padding=self.padding,
                                        name="decoder" + layer_name + "_conv2", use_bias=self.bias)(x))
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(Conv2D(32, 3, strides=1, kernel_initializer=self.weights_init, padding=self.padding,
                                        name="decoder" + layer_name + "_conv3", use_bias=self.bias)(x))
        x = LeakyReLU(0.2)(x)
        decoded_image = BatchNormalization()(Conv2D(8, 3, strides=1, kernel_initializer=self.weights_init, padding=self.padding,
                                        name="decoder" + layer_name + "_conv4", use_bias=self.bias)(x))

        return decoded_image

    def pyramid_encoder(self, input_tensor, layer_name, features):
        """
        Builds the encoder in PydNet for a specific level. Consist of two conv layers with kernel 3x3 but strides 2
        through the first and strides 1 through the second
        :param input_tensor: Input tensor
        :param layer_name:   The level in the pyramid
        :param features:     Amount of features/filters in the conv layers
        :return:             Encoded tensor in the pyramid level
        """
        x = BatchNormalization()(Conv2D(features, 3, strides=2, kernel_initializer=self.weights_init, padding=self.padding,
                                        name="encoder" + layer_name + "_conv1", use_bias=self.bias)(input_tensor))
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(Conv2D(features, 3, strides=1, kernel_initializer=self.weights_init, padding=self.padding,
                                        name="encoder" + layer_name + "_conv2", use_bias=self.bias)(x))
        encoded_image = LeakyReLU(0.2)(x)

        return encoded_image

    @staticmethod
    def get_disp(x):
        """
        Slice the tensor to get a disparity map for the left and right image and further processed through a sigmoid
        function.
        :param x: Tensor to slice
        :return:  Left and right disparity map
        """
        disp = 0.3 * sigmoid(tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, 2]))
        return disp

    def build_model(self):
        """
        Allocate a PydNet model by calling .build_model() to train weights. If pre_trained_weights is True then a PydNet
        model is loaded instead.
        :return: PydNet with Keras layers
        """
        # Get a tensor placeholder
        input_img = Input(shape=self.input_shape)

        # Encode the different levels from L1 to L6
        encoder_L1 = self.pyramid_encoder(input_img, "L1", 16)

        encoder_L2 = self.pyramid_encoder(encoder_L1, "L2", 32)

        encoder_L3 = self.pyramid_encoder(encoder_L2, "L3", 64)

        encoder_L4 = self.pyramid_encoder(encoder_L3, "L4", 96)

        encoder_L5 = self.pyramid_encoder(encoder_L4, "L5", 128)

        encoder_L6 = self.pyramid_encoder(encoder_L5, "L6", 192)

        # Decode and concatenate from L6 to L1
        decoder_L6 = self.pyramid_decoder(encoder_L6, "L6")

        decoder_L5 = self.pyramid_decoder(encoder_L5, "L5", decoder_from_lower_level=decoder_L6)

        decoder_L4 = self.pyramid_decoder(encoder_L4, "L4", decoder_from_lower_level=decoder_L5)

        decoder_L3 = self.pyramid_decoder(encoder_L3, "L3", decoder_from_lower_level=decoder_L4)

        decoder_L2 = self.pyramid_decoder(encoder_L2, "L2", decoder_from_lower_level=decoder_L3)

        decoder_L1 = self.pyramid_decoder(encoder_L1, "L1", decoder_from_lower_level=decoder_L2)

        # Get the different resolutions from L1 to L4
        self.disp1 = Lambda(self.get_disp)(decoder_L1)
        self.disp2 = Lambda(self.get_disp)(decoder_L2)
        self.disp3 = Lambda(self.get_disp)(decoder_L3)
        self.disp4 = Lambda(self.get_disp)(decoder_L4)

        predictions = [self.disp1, self.disp2, self.disp3, self.disp4]

        pydnet = Model(inputs=input_img, outputs=predictions)

        if self.pre_trained_weights is not None:
            print(self.pre_trained_weights)
            if os.path.exists(self.pre_trained_weights):
                pydnet.load_weights(self.pre_trained_weights, by_name=self.load_weights_by_name)
            else:
                raise AssertionError('Not able to load pydnet weights. File does not exist')

        return pydnet


model = PydNetModel(shape=(512, 384, 3),
                    )
pydnet = model.build_model()
pydnet.summary()
Annies_amazeballs_epic_mono_loss = MonoLoss()
pydnet.compile(loss=Annies_amazeballs_epic_mono_loss,
               optimizer="sgd")

pydnet.fit_generator