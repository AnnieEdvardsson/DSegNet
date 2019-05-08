from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Concatenate
from keras.layers import Conv2D, Cropping2D
from keras.layers import MaxPooling2D, UpSampling2D, Activation, BatchNormalization
from keras.layers.core import Dropout
from keras.layers.convolutional import ZeroPadding2D
from typing import Tuple, AnyStr, Any
#from models.keras_models.keras_model_template import KerasModelTemplate
import numpy as np


class DePool2D(UpSampling2D):
    '''
    https://github.com/nanopony/keras-convautoencoder/blob/c8172766f968c8afc81382b5e24fd4b57d8ebe71/autoencoder_layers.py#L24
    Simplar to UpSample, yet traverse only maxpooled elements.
    '''

    def __init__(self, pool2d_layer: MaxPooling2D, *args, **kwargs):
        self._pool2d_layer = pool2d_layer
        super().__init__(*args, **kwargs)

    def get_output(self, train: bool = False) -> Any:
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = repeat_elements(X, self.size[0], axis=2)
            output = repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = repeat_elements(X, self.size[0], axis=1)
            output = repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        return gradients(
            sum(
                self._pool2d_layer.get_output(train)
            ),
            self._pool2d_layer.get_input(train)
        ) * output


class DepSegNetModel(object):
    """
    Depth-assisted SegNet
    """

    def __init__(self, shape: Tuple[int, int, int], num_classes: int, pre_trained_encoder: bool, dsegnet_indices: bool,
                 weights: Any, load_weights_by_name: bool, kernel_init='he_normal'):
        #super().__init__()
        self.input_shape = shape
        self.num_classes = num_classes
        self.pre_trained_encoder = 'imagenet' if pre_trained_encoder else None
        self.dsegnet_indices = dsegnet_indices
        self.weights_init = kernel_init
        self.pre_trained_weights = None  # weights  # path to a .hdf5 file with the weights that we want to use.
        # If None the weights are initialized according to "kernel_init"
        self.load_weights_by_name = load_weights_by_name

    def create_model(self):
        input_tensor = Input(shape=self.input_shape)
        h, w, c = self.input_shape
        depth_tensor = Input(shape=(h, w, 1))  # Already prepared depthmap with PyDNet

        # VGG16 encodes with 5 successives blocks of Conv2D and 2x2 MaxPooling (s=2)
        # So it divides input size by 2**5 = 32
        # We zero pad image so that both sides are multiples of 32
        # If we don't, information is lost at encoding step
        # TODO: Investigate Mirror Padding instead of Zero Padding
        modVGG16 = 32
        hpad = (modVGG16 - w % modVGG16) % modVGG16
        vpad = (modVGG16 - h % modVGG16) % modVGG16

        ceil_hpad = int(np.ceil(hpad / 2))
        floor_hpad = int(np.floor(hpad / 2))

        ceil_vpad = int(np.ceil(vpad / 2))
        floor_vpad = int(np.floor(vpad / 2))

        encoder_input_tensor = ZeroPadding2D(padding=((ceil_vpad, floor_vpad), (ceil_hpad, floor_hpad)))(input_tensor)

        concat_depth_tensor = ZeroPadding2D(padding=((ceil_vpad, floor_vpad), (ceil_hpad, floor_hpad)))(depth_tensor)

        encoder = VGG16(
            include_top=False,
            weights=self.pre_trained_encoder,
            input_tensor=encoder_input_tensor,
            input_shape=self.input_shape,
            pooling="None")  # type: tModel

        L = [layer for i, layer in enumerate(encoder.layers)]  # type: List[Layer]
        for layer in L: layer.trainable = False  # freeze VGG16
        L.reverse()

        x = encoder.output
        x = Dropout(0.2)(x)

        # Block 5
        if self.dsegnet_indices:
            x = DePool2D(L[0], size=L[0].pool_size, input_shape=encoder.output_shape[1:], name='DePool5')(x)
        else:
            x = UpSampling2D(size=L[0].pool_size, input_shape=encoder.output_shape[1:], name='UpSampling5')(x)

        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[1].filters, L[1].kernel_size, padding=L[1].padding, kernel_initializer=self.weights_init,
                   name='block5_deconv3')(x)))
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[2].filters, L[2].kernel_size, padding=L[2].padding, kernel_initializer=self.weights_init,
                   name='block5_deconv5')(x)))
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[3].filters, L[3].kernel_size, padding=L[3].padding, kernel_initializer=self.weights_init,
                   name='block5_deconv1')(x)))
        x = Dropout(0.2)(x)
        # Block 4
        if self.dsegnet_indices:
            x = DePool2D(L[4], size=L[4].pool_size, name='DePool4')(x)
        else:
            x = UpSampling2D(size=L[4].pool_size, name='UpSampling4')(x)

        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[5].filters, L[5].kernel_size, padding=L[5].padding, kernel_initializer=self.weights_init,
                   name='block4_deconv3')(x)))
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[6].filters, L[6].kernel_size, padding=L[6].padding, kernel_initializer=self.weights_init,
                   name='block4_deconv2')(x)))
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[7].filters, L[7].kernel_size, padding=L[7].padding, kernel_initializer=self.weights_init,
                   name='block4_deconv1')(x)))
        x = Dropout(0.3)(x)
        # Block 3
        if self.dsegnet_indices:
            x = DePool2D(L[8], size=L[8].pool_size, name='DePool3')(x)
        else:
            x = UpSampling2D(size=L[8].pool_size, name='UpSampling3')(x)

        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[10].filters, L[10].kernel_size, padding=L[10].padding, kernel_initializer=self.weights_init,
                   name='block3_deconv2')(x)))
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[11].filters, L[11].kernel_size, padding=L[11].padding, kernel_initializer=self.weights_init,
                   name='block3_deconv1')(x)))
        x = Dropout(0.3)(x)
        # Block 2
        if self.dsegnet_indices:
            x = DePool2D(L[12], size=L[12].pool_size, name='DePool2')(x)
        else:
            x = UpSampling2D(size=L[12].pool_size, name='UpSampling2')(x)

        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[13].filters, L[13].kernel_size, padding=L[13].padding, kernel_initializer=self.weights_init,
                   name='block2_deconv2')(x)))

        # Next layer drops a filters to make sure we have a power of two filters after concatenation
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[14].filters - 1, L[14].kernel_size, padding=L[14].padding, kernel_initializer=self.weights_init,
                   name='block2_deconv1')(x)))
        # Block 1
        if self.dsegnet_indices:
            x = DePool2D(L[15], size=(L[15].pool_size), name='DePool1')(x)
        else:
            x = UpSampling2D(size=L[15].pool_size, name='UpSampling1')(x)

        # Plugging in the depth
        x = Concatenate()([x, concat_depth_tensor])

        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[16].filters, L[16].kernel_size, padding=L[16].padding, kernel_initializer=self.weights_init,
                   name='block1_deconv2')(x)))

        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[17].filters, L[17].kernel_size, padding=L[17].padding, kernel_initializer=self.weights_init,
                   name='block1_deconv1')(x)))

        x = Conv2D(self.num_classes, (1, 1), padding='valid', kernel_initializer=self.weights_init,
                   name='block1_deconv_classes' + str(self.num_classes))(x)

        x = Cropping2D(cropping=(vpad // 2, hpad // 2))(x)

        x = Activation('softmax')(x)
        predictions = x

        segnet = Model(inputs=[encoder.inputs[0], depth_tensor], outputs=predictions)  # type: tModel

        if self.pre_trained_weights is not None:
            if os.path.exists(self.pre_trained_weights):
                segnet.load_weights(self.pre_trained_weights, by_name=self.load_weights_by_name)
            else:
                raise AssertionError('Not able to load weights. File does not exist')

        return segnet
