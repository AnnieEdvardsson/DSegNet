
from keras.applications.vgg16 import VGG16
from keras.backend import gradients, sum, repeat_elements
from keras.engine.topology import Layer
from keras.engine.training import Model as tModel
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D

import os
from keras import layers
from keras.applications.resnet50 import ResNet50
from typing import Tuple, Any, AnyStr, List
from keras.layers import Input, Concatenate, LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model


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

class SegNetModel(object):
    """
    base on https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/segnet_train.prototxt
    and https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/segnet_model_driving_webdemo.prototxt

    """

    def __init__(self, shape: Tuple[int, int, int], num_classes: int, pre_trained_encoder: bool, segnet_indices: bool,
                 weights: Any, load_weights_by_name: bool, kernel_init='he_normal'):
        super().__init__()
        self.input_shape = shape
        self.num_classes = num_classes
        self.pre_trained_encoder = 'imagenet' if pre_trained_encoder else None
        self.segnet_indices = segnet_indices
        self.weights_init = kernel_init
        self.pre_trained_weights = weights  # path to a .hdf5 file with the weights that we want to use.
        # If None the weights are initialized according to "kernel_init"
        self.load_weights_by_name = load_weights_by_name

    def create_model(self):

        input_tensor = Input(shape=self.input_shape)  # type: object

        x = VGG16_encoder(input_tensor, self.weights_init, "Img")
        x = VGG16_decoder(x, self.weights_init, "Img")

        x = Conv2D(self.num_classes, (1, 1), padding='valid', kernel_initializer=self.weights_init,
                   name='block1_deconv_classes' + str(self.num_classes))(x)
        x = Activation('softmax')(x)
        predictions = x

        segnet = Model(inputs=input_tensor, outputs=predictions)  # type: tModel

        if self.pre_trained_weights is not None:
            if os.path.exists(self.pre_trained_weights):
                segnet.load_weights(self.pre_trained_weights, by_name=self.load_weights_by_name)
            else:
                raise AssertionError('Not able to load weights. File does not exist')

        return segnet

class dSegNetModel(object):
    """
    base on https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/segnet_train.prototxt
    and https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/segnet_model_driving_webdemo.prototxt

    """

    def __init__(self, shape: Tuple[int, int, int], num_classes: int, pre_trained_encoder: bool, segnet_indices: bool,
                 weights: Any, load_weights_by_name: bool, kernel_init='he_normal'):
        super().__init__()
        self.input_shape = shape
        self.num_classes = num_classes
        self.pre_trained_encoder = 'imagenet' if pre_trained_encoder else None
        self.segnet_indices = segnet_indices
        self.weights_init = kernel_init
        self.pre_trained_weights = weights  # path to a .hdf5 file with the weights that we want to use.
        # If None the weights are initialized according to "kernel_init"
        self.load_weights_by_name = load_weights_by_name

    def create_model(self):
        # 1 concatenate disp map in the end, 2 concatenate disp map in input similar to RGB-D files
        h, w, c = self.input_shape
        input_tensor = Input(shape=self.input_shape)
        disp_tensor = Input(shape=(h, w, 1))
        weights_init = self.weights_init
        # Encoder with VGG16 + + BN
        x = VGG16_encoder(input_tensor, self.weights_init, "Img")
        # Decoder block 5
        x = UpSampling2D(size=(2, 2), name='UpSampling5')(x)
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
                   name='block5_deconv1')(x)))
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
                   name='block5_deconv2')(x)))
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
                   name='block5_deconv3')(x)))
        # x = Dropout(0.2)(x)
        # Block 4
        x = UpSampling2D(size=(2, 2), name='UpSampling4')(x)
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
                   name='block4_deconv1')(x)))
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
                   name='block4_deconv2')(x)))
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
                   name='block4_deconv3')(x)))
        # x = Dropout(0.2)(x)
        # Block 3
        x = UpSampling2D(size=(2, 2), name='UpSampling3')(x)
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(256, 3, padding="same", kernel_initializer=weights_init,
                   name='block3_deconv1')(x)))
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(256, 3, padding="same", kernel_initializer=weights_init,
                   name='block3_deconv2')(x)))
        # x = Dropout(0.3)(x)
        # Block 2
        x = UpSampling2D(size=(2, 2), name='UpSampling2')(x)
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(128, 3, padding="same", kernel_initializer=weights_init,
                   name='block2_deconv1')(x)))
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(128, 3, padding="same", kernel_initializer=weights_init,
                   name='block2_deconv2')(x)))
        # x = Dropout(0.5)(x)
        # Block 1
        x = UpSampling2D(size=(2, 2), name='UpSampling1')(x)
        # Concatenate depth information
        x = Concatenate()([x, disp_tensor])

        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(64, 3, padding="same", kernel_initializer=weights_init,
                   name='block1_deconv1')(x)))
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(64, 3, padding="same", kernel_initializer=weights_init,
                   name='block1_deconv2')(x)))
        # encoder = VGG16(
        #     include_top=False,
        #     weights=self.pre_trained_encoder,
        #     input_tensor=input_tensor,
        #     input_shape=self.input_shape,
        #     pooling="None")  # type: tModel
        #
        # L = [layer for i, layer in enumerate(encoder.layers)]  # type: List[Layer]
        # # for layer in L: layer.trainable = False # freeze VGG16
        # L.reverse()
        #
        # x = encoder.output
        #
        # #x = Dropout(0.2)(x)
        #
        # # Block 5
        # if self.segnet_indices:
        #     x = DePool2D(L[0], size=L[0].pool_size, input_shape=encoder.output_shape[1:], name='DePool5')(x)
        # else:
        #     x = UpSampling2D(size=L[0].pool_size, input_shape=encoder.output_shape[1:], name='UpSampling5')(x)
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[1].filters, L[1].kernel_size, padding=L[1].padding, kernel_initializer=self.weights_init,
        #            name='block5_deconv3')(x)))
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[2].filters, L[2].kernel_size, padding=L[2].padding, kernel_initializer=self.weights_init,
        #            name='block5_deconv5')(x)))
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[3].filters, L[3].kernel_size, padding=L[3].padding, kernel_initializer=self.weights_init,
        #            name='block5_deconv1')(x)))
        # #x = Dropout(0.2)(x)
        # # Block 4
        # if self.segnet_indices:
        #     x = DePool2D(L[4], size=L[4].pool_size, name='DePool4')(x)
        # else:
        #     x = UpSampling2D(size=L[4].pool_size, name='UpSampling4')(x)
        # # x = ZeroPadding2D(padding=(0, 1))(x)
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[5].filters, L[5].kernel_size, padding=L[5].padding, kernel_initializer=self.weights_init,
        #            name='block4_deconv3')(x)))
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[6].filters, L[6].kernel_size, padding=L[6].padding, kernel_initializer=self.weights_init,
        #            name='block4_deconv2')(x)))
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[7].filters, L[7].kernel_size, padding=L[7].padding, kernel_initializer=self.weights_init,
        #            name='block4_deconv1')(x)))
        # #x = Dropout(0.3)(x)
        # # Block 3
        # if self.segnet_indices:
        #     x = DePool2D(L[8], size=L[8].pool_size, name='DePool3')(x)
        # else:
        #     x = UpSampling2D(size=L[8].pool_size, name='UpSampling3')(x)
        # #x = ZeroPadding2D(padding=(0, 1))(x)
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[10].filters, L[10].kernel_size, padding=L[10].padding, kernel_initializer=self.weights_init,
        #            name='block3_deconv2')(x)))
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[11].filters, L[11].kernel_size, padding=L[11].padding, kernel_initializer=self.weights_init,
        #            name='block3_deconv1')(x)))
        # #x = Dropout(0.3)(x)
        # # Block 2
        # if self.segnet_indices:
        #     x = DePool2D(L[12], size=L[12].pool_size, name='DePool2')(x)
        # else:
        #     x = UpSampling2D(size=L[12].pool_size, name='UpSampling2')(x)
        #
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[13].filters, L[13].kernel_size, padding=L[13].padding, kernel_initializer=self.weights_init,
        #            name='block2_deconv2')(x)))
        #
        # # Next layer drops a filters to make sure we have a power of two filters after concatenation
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[14].filters-1, L[14].kernel_size, padding=L[14].padding, kernel_initializer=self.weights_init,
        #            name='block2_deconv1')(x)))
        #
        # # Block 1
        # if self.segnet_indices:
        #     x = DePool2D(L[15], size=L[15].pool_size, name='DePool1')(x)
        # else:
        #     x = UpSampling2D(size=L[15].pool_size, name='UpSampling1')(x)
        #
        # x = Concatenate()([x, disp_tensor])
        #
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[16].filters, L[16].kernel_size, padding=L[16].padding, kernel_initializer=self.weights_init,
        #            name='block1_deconv2')(x)))
        # x = Activation('relu')(BatchNormalization()(
        #     Conv2D(L[17].filters, L[17].kernel_size, padding=L[17].padding, kernel_initializer=self.weights_init,
        #            name='block1_deconv1')(x)))
        # #x = Dropout(0.5)(x)
        x = Conv2D(self.num_classes, (1, 1), padding='valid', kernel_initializer=self.weights_init,
                   name='block1_deconv_classes' + str(self.num_classes))(x)
        x = Activation('softmax')(x)
        predictions = x

        dsegnet = Model(inputs=[input_tensor, disp_tensor], outputs=predictions)  # type: tModel

        if self.pre_trained_weights is not None:
            if os.path.exists(self.pre_trained_weights):
                dsegnet.load_weights(self.pre_trained_weights, by_name=self.load_weights_by_name)
            else:
                raise AssertionError('Not able to load weights. File does not exist')

        return dsegnet

class DispSegNetModel(object):

    def __init__(self, shape: Tuple[int, int, int], num_classes: int, pre_trained_encoder: bool, segnet_indices: bool,
                 weights: Any, load_weights_by_name: bool, kernel_init='he_normal'):
        super().__init__()
        self.input_shape = shape
        self.num_classes = num_classes
        self.pre_trained_encoder = 'imagenet' if pre_trained_encoder else None
        self.segnet_indices = segnet_indices
        self.weights_init = kernel_init
        self.pre_trained_weights = weights  # path to a .hdf5 file with the weights that we want to use.
        # If None the weights are initialized according to "kernel_init"
        self.load_weights_by_name = load_weights_by_name

    # def segnet_model(self, input_tensor, input_shape):
    #     if input_shape[2] == 3:
    #         encoder = VGG16(
    #             include_top=False,
    #             weights=self.pre_trained_encoder,
    #             input_tensor=input_tensor,
    #             input_shape=input_shape,
    #             pooling="None")  # type: tModel
    #         name = 'Image'
    #         for i, layer in enumerate(encoder.layers):
    #             layer.name = 'Layer_' + str(i) + name
    #     else:
    #         encoder = VGG16(
    #             include_top=False,
    #             weights=None,
    #             input_tensor=input_tensor,
    #             input_shape=input_shape,
    #             pooling="None")  # type: tModel
    #         name = 'Disparity'
    #         for i, layer in enumerate(encoder.layers):
    #             layer.name = 'Layer_' + str(i) + name
    #
    #     L = [layer for i, layer in enumerate(encoder.layers)]  # type: List[Layer]
    #     # for layer in L: layer.trainable = False # freeze VGG16
    #     L.reverse()
    #
    #     x = encoder.output
    #
    #     #x = Dropout(0.2)(x)
    #
    #     # Block 5
    #     if self.segnet_indices:
    #         x = DePool2D(L[0], size=L[0].pool_size, input_shape=encoder.output_shape[1:], name='DePool5' + name)(x)
    #     else:
    #         x = UpSampling2D(size=L[0].pool_size, input_shape=encoder.output_shape[1:], name='UpSampling5' + name)(x)
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[1].filters, L[1].kernel_size, padding=L[1].padding, kernel_initializer=self.weights_init,
    #                name='block5_deconv3' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[2].filters, L[2].kernel_size, padding=L[2].padding, kernel_initializer=self.weights_init,
    #                name='block5_deconv5' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[3].filters, L[3].kernel_size, padding=L[3].padding, kernel_initializer=self.weights_init,
    #                name='block5_deconv1' + name)(x)))
    #     #x = Dropout(0.2)(x)
    #     # Block 4
    #     if self.segnet_indices:
    #         x = DePool2D(L[4], size=L[4].pool_size, name='DePool4' + name)(x)
    #     else:
    #         x = UpSampling2D(size=L[4].pool_size, name='UpSampling4' + name)(x)
    #     # x = ZeroPadding2D(padding=(0, 1))(x)
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[5].filters, L[5].kernel_size, padding=L[5].padding, kernel_initializer=self.weights_init,
    #                name='block4_deconv3' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[6].filters, L[6].kernel_size, padding=L[6].padding, kernel_initializer=self.weights_init,
    #                name='block4_deconv2' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[7].filters, L[7].kernel_size, padding=L[7].padding, kernel_initializer=self.weights_init,
    #                name='block4_deconv1' + name)(x)))
    #     #x = Dropout(0.3)(x)
    #     # Block 3
    #     if self.segnet_indices:
    #         x = DePool2D(L[8], size=L[8].pool_size, name='DePool3' + name)(x)
    #     else:
    #         x = UpSampling2D(size=L[8].pool_size, name='UpSampling3' + name)(x)
    #     # x = ZeroPadding2D(padding=(0, 1))(x)
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[10].filters, L[10].kernel_size, padding=L[10].padding, kernel_initializer=self.weights_init,
    #                name='block3_deconv2' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[11].filters, L[11].kernel_size, padding=L[11].padding, kernel_initializer=self.weights_init,
    #                name='block3_deconv1' + name)(x)))
    #     #x = Dropout(0.3)(x)
    #     # Block 2
    #     if self.segnet_indices:
    #         x = DePool2D(L[12], size=L[12].pool_size, name='DePool2' + name)(x)
    #     else:
    #         x = UpSampling2D(size=L[12].pool_size, name='UpSampling2' + name)(x)
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[13].filters, L[13].kernel_size, padding=L[13].padding, kernel_initializer=self.weights_init,
    #                name='block2_deconv2' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[14].filters, L[14].kernel_size, padding=L[14].padding, kernel_initializer=self.weights_init,
    #                name='block2_deconv1' + name)(x)))
    #     # Block 1
    #     if self.segnet_indices:
    #         x = DePool2D(L[15], size=L[15].pool_size, name='DePool1' + name)(x)
    #     else:
    #         x = UpSampling2D(size=L[15].pool_size, name='UpSampling1' + name)(x)
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[16].filters, L[16].kernel_size, padding=L[16].padding, kernel_initializer=self.weights_init,
    #                name='block1_deconv2' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[17].filters, L[17].kernel_size, padding=L[17].padding, kernel_initializer=self.weights_init,
    #                name='block1_deconv1' + name)(x)))
    #
    #     #x = Dropout(0.3)(x)
    #
    #     return x

    def create_model(self):
        # 1 concatenate disp map in the end, 2 concatenate disp map in input similar to RGB-D files
        h, w, c = self.input_shape
        image_tensor = Input(shape=self.input_shape)
        disp_tensor = Input(shape=(h, w, 1))

        #image_prediction = self.segnet_model(image_tensor, self.input_shape)
        #disp_prediction = self.segnet_model(disp_tensor, (h, w, 1))
        img_encoder = VGG16_encoder(image_tensor, self.weights_init, "Img")
        img_decoder = VGG16_decoder(img_encoder, self.weights_init, "Img")

        disp_encoder = VGG16_encoder(disp_tensor, self.weights_init, "Disp")
        disp_decoder = VGG16_decoder(disp_encoder, self.weights_init, "Disp")

        x = Concatenate(name="concat")([img_decoder, disp_decoder])
        x = Conv2D(self.num_classes, (1, 1), padding='valid', kernel_initializer=self.weights_init,
                   name='block1_deconv_classes' + str(self.num_classes))(x)
        x = Activation('softmax')(x)

        predictions = x

        dispsegnet = Model(inputs=[image_tensor, disp_tensor], outputs=predictions)  # type: tModel

        if self.pre_trained_weights is not None:
            if os.path.exists(self.pre_trained_weights):
                dispsegnet.load_weights(self.pre_trained_weights, by_name=self.load_weights_by_name)
            else:
                raise AssertionError('Not able to load weights. File does not exist')

        return dispsegnet

class DispSegNetBasicModel:
    def __init__(self, shape: Tuple[int, int, int], num_classes: int, pre_trained_encoder: bool, segnet_indices: bool,
                 weights: Any, load_weights_by_name: bool, kernel_init='he_normal'):
        super().__init__()
        self.input_shape = shape
        self.num_classes = num_classes
        self.pre_trained_encoder = 'imagenet' if pre_trained_encoder else None
        self.segnet_indices = segnet_indices
        self.weights_init = kernel_init
        self.pre_trained_weights = weights  # path to a .hdf5 file with the weights that we want to use.
        # If None the weights are initialized according to "kernel_init"
        self.load_weights_by_name = load_weights_by_name

    # def segnet_model(self, input_tensor, input_shape):
    #     name = 'Image'
    #     encoder = VGG16(
    #         include_top=False,
    #         weights=self.pre_trained_encoder,
    #         input_tensor=input_tensor,
    #         input_shape=input_shape,
    #         pooling="None")  # type: tModel
    #
    #     L = [layer for i, layer in enumerate(encoder.layers)]  # type: List[Layer]
    #     # for layer in L: layer.trainable = False # freeze VGG16
    #     L.reverse()
    #
    #     x = encoder.output
    #
    #     #x = Dropout(0.2)(x)
    #
    #     # Block 5
    #     if self.segnet_indices:
    #         x = DePool2D(L[0], size=L[0].pool_size, input_shape=encoder.output_shape[1:], name='DePool5' + name)(x)
    #     else:
    #         x = UpSampling2D(size=L[0].pool_size, input_shape=encoder.output_shape[1:], name='UpSampling5' + name)(x)
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[1].filters, L[1].kernel_size, padding=L[1].padding, kernel_initializer=self.weights_init,
    #                name='block5_deconv3' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[2].filters, L[2].kernel_size, padding=L[2].padding, kernel_initializer=self.weights_init,
    #                name='block5_deconv5' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[3].filters, L[3].kernel_size, padding=L[3].padding, kernel_initializer=self.weights_init,
    #                name='block5_deconv1' + name)(x)))
    #     #x = Dropout(0.2)(x)
    #     # Block 4
    #     if self.segnet_indices:
    #         x = DePool2D(L[4], size=L[4].pool_size, name='DePool4' + name)(x)
    #     else:
    #         x = UpSampling2D(size=L[4].pool_size, name='UpSampling4' + name)(x)
    #     # x = ZeroPadding2D(padding=(0, 1))(x)
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[5].filters, L[5].kernel_size, padding=L[5].padding, kernel_initializer=self.weights_init,
    #                name='block4_deconv3' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[6].filters, L[6].kernel_size, padding=L[6].padding, kernel_initializer=self.weights_init,
    #                name='block4_deconv2' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[7].filters, L[7].kernel_size, padding=L[7].padding, kernel_initializer=self.weights_init,
    #                name='block4_deconv1' + name)(x)))
    #     #x = Dropout(0.3)(x)
    #     # Block 3
    #     if self.segnet_indices:
    #         x = DePool2D(L[8], size=L[8].pool_size, name='DePool3' + name)(x)
    #     else:
    #         x = UpSampling2D(size=L[8].pool_size, name='UpSampling3' + name)(x)
    #     # x = ZeroPadding2D(padding=(0, 1))(x)
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[10].filters, L[10].kernel_size, padding=L[10].padding, kernel_initializer=self.weights_init,
    #                name='block3_deconv2' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[11].filters, L[11].kernel_size, padding=L[11].padding, kernel_initializer=self.weights_init,
    #                name='block3_deconv1' + name)(x)))
    #     #x = Dropout(0.3)(x)
    #     # Block 2
    #     if self.segnet_indices:
    #         x = DePool2D(L[12], size=L[12].pool_size, name='DePool2' + name)(x)
    #     else:
    #         x = UpSampling2D(size=L[12].pool_size, name='UpSampling2' + name)(x)
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[13].filters, L[13].kernel_size, padding=L[13].padding, kernel_initializer=self.weights_init,
    #                name='block2_deconv2' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[14].filters, L[14].kernel_size, padding=L[14].padding, kernel_initializer=self.weights_init,
    #                name='block2_deconv1' + name)(x)))
    #     # Block 1
    #     if self.segnet_indices:
    #         x = DePool2D(L[15], size=L[15].pool_size, name='DePool1' + name)(x)
    #     else:
    #         x = UpSampling2D(size=L[15].pool_size, name='UpSampling1' + name)(x)
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[16].filters, L[16].kernel_size, padding=L[16].padding, kernel_initializer=self.weights_init,
    #                name='block1_deconv2' + name)(x)))
    #     x = Activation('relu')(BatchNormalization()(
    #         Conv2D(L[17].filters, L[17].kernel_size, padding=L[17].padding, kernel_initializer=self.weights_init,
    #                name='block1_deconv1' + name)(x)))
    #
    #     #x = Dropout(0.3)(x)
    #
    #     return x

    def segnet_basic_model(self, input_tensor):
        name = 'Disp'
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(64, 3, padding='same', kernel_initializer=self.weights_init,
                   name='block1_conv' + name)(input_tensor)))
        #x = Dropout(0.2)(x)
        x = MaxPooling2D((2, 2))(x)

        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(128, 3, padding='same', kernel_initializer=self.weights_init,
                   name='block2_conv' + name)(x)))
        #x = Dropout(0.2)(x)
        x = MaxPooling2D((2, 2))(x)

        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(256, 3, padding='same', kernel_initializer=self.weights_init,
                   name='block3_conv' + name)(x)))
        #x = Dropout(0.2)(x)
        x = MaxPooling2D((2, 2))(x)

        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(512, 3, padding='same', kernel_initializer=self.weights_init,
                   name='block4_conv' + name)(x)))
        #x = Dropout(0.2)(x)

        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(512, 3, padding='same', kernel_initializer=self.weights_init,
                   name='block4_deconv' + name)(x)))
        #x = Dropout(0.2)(x)

        x = UpSampling2D(size=(2, 2), name='block1_UpSampling' + name)(x)
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(256, 3, padding='same', kernel_initializer=self.weights_init,
                   name='block3_deconv' + name)(x)))
        #x = Dropout(0.2)(x)

        x = UpSampling2D(size=(2, 2), name='block2_UpSampling' + name)(x)
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(128, 3, padding='same', kernel_initializer=self.weights_init,
                   name='block2_deconv' + name)(x)))
        #x = Dropout(0.2)(x)

        x = UpSampling2D(size=(2, 2), name='block3_UpSampling' + name)(x)
        x = LeakyReLU(alpha=0.1)(BatchNormalization()(
            Conv2D(64, 3, padding='same', kernel_initializer=self.weights_init,
                   name='block1_deconv' + name)(x)))
        #x = Dropout(0.2)(x)

        return x

    def create_model(self):
        # 1 concatenate disp map in the end, 2 concatenate disp map in input similar to RGB-D files
        h, w, c = self.input_shape
        image_tensor = Input(shape=self.input_shape)
        disp_tensor = Input(shape=(h, w, 1))

        img_encoder = VGG16_encoder(image_tensor, self.weights_init, "Img")
        img_decoder = VGG16_decoder(img_encoder, self.weights_init, "Img")
        disp_decoder = self.segnet_basic_model(disp_tensor)

        x = Concatenate(name="concat")([img_decoder, disp_decoder])
        x = Conv2D(self.num_classes, (1, 1), padding='same', kernel_initializer=self.weights_init,
                   name='block1_deconv_classes' + str(self.num_classes))(x)
        x = Activation('softmax')(x)

        predictions = x

        dispsegnetbasicnet = Model(inputs=[image_tensor, disp_tensor], outputs=predictions)  # type: tModel

        if self.pre_trained_weights is not None:
            if os.path.exists(self.pre_trained_weights):
                dispsegnetbasicnet.load_weights(self.pre_trained_weights, by_name=self.load_weights_by_name)
            else:
                raise AssertionError('Not able to load weights. File does not exist')

        return dispsegnetbasicnet

class PydSegNetModel(object):

    def __init__(self, shape: Tuple[int, int, int], num_classes: int, segnet_indices: bool,
                 weights: Any, load_weights_by_name: bool, kernel_init='he_normal'):
        super().__init__()
        self.input_shape = shape
        self.num_classes = num_classes
        self.segnet_indices = segnet_indices
        self.weights_init = kernel_init
        self.pre_trained_weights = weights  # path to a .hdf5 file with the weights that we want to use.
        # If None the weights are initialized according to "kernel_init"
        self.load_weights_by_name = load_weights_by_name

    def create_model(self):

        h, w, c = self.input_shape
        image_tensor = Input(shape=self.input_shape)
        disp_tensor = Input(shape=(h, w, 1))

        x = Concatenate(name="concat")([image_tensor, disp_tensor])
        x = VGG16_encoder(x, self.weights_init, "imgDisp")
        x = VGG16_decoder(x, self.weights_init, "ImgDisp")
        x = Conv2D(self.num_classes, (1, 1), padding='valid', kernel_initializer=self.weights_init,
                   name='block1_deconv_classes' + str(self.num_classes))(x)
        x = Activation('softmax')(x)
        predictions = x

        model = Model(inputs=[image_tensor, disp_tensor], outputs=predictions)  # type: tModel

        if self.pre_trained_weights is not None:
            if os.path.exists(self.pre_trained_weights):
                model.load_weights(self.pre_trained_weights, by_name=self.load_weights_by_name)
            else:
                raise AssertionError('Not able to load weights. File does not exist')

        return model

class EncFuseModel(object):

    def __init__(self, shape: Tuple[int, int, int], num_classes: int, pre_trained_encoder: bool, segnet_indices: bool,
                 weights: Any, load_weights_by_name: bool, kernel_init='he_normal'):
        super().__init__()
        self.input_shape = shape
        self.num_classes = num_classes
        self.segnet_indices = segnet_indices
        self.weights_init = kernel_init
        self.pre_trained_encoder = 'imagenet' if pre_trained_encoder else None
        self.pre_trained_weights = weights  # path to a .hdf5 file with the weights that we want to use.
        # If None the weights are initialized according to "kernel_init"
        self.load_weights_by_name = load_weights_by_name

    def create_model(self):

        h, w, c = self.input_shape
        image_tensor = Input(shape=self.input_shape)
        disp_tensor = Input(shape=(h, w, 1))

        # img_encoder = VGG16(
        #     include_top=False,
        #     weights=self.pre_trained_encoder,
        #     input_tensor=image_tensor,
        #     input_shape=self.input_shape,
        #     pooling="None")
        #
        # name1 = 'Image'
        # for i, layer in enumerate(img_encoder.layers):
        #     layer.name = 'Layer_' + str(i) + name1
        #
        # disp_encoder = VGG16(
        #     include_top=False,
        #     weights=None,
        #     input_tensor=disp_tensor,
        #     input_shape=(h, w, 1),
        #     pooling="None")
        #
        # name2 = 'Disparity'
        # for i, layer in enumerate(disp_encoder.layers):
        #     layer.name = 'Layer_' + str(i) + name2
        img_encoder = VGG16_encoder(image_tensor, self.weights_init, "Img")
        disp_encoder = VGG16_encoder(disp_tensor, self.weights_init, "Disp")

        x = Concatenate(name="concat")([img_encoder, disp_encoder])
        x = VGG16_decoder(x, self.weights_init, "ImgDisp")
        x = Conv2D(self.num_classes, (1, 1), padding='valid', kernel_initializer=self.weights_init,
                   name='block1_deconv_classes' + str(self.num_classes))(x)
        x = Activation('softmax')(x)
        predictions = x

        model = Model(inputs=[image_tensor, disp_tensor], outputs=predictions)  # type: tModel

        if self.pre_trained_weights is not None:
            if os.path.exists(self.pre_trained_weights):
                model.load_weights(self.pre_trained_weights, by_name=self.load_weights_by_name)
            else:
                raise AssertionError('Not able to load weights. File does not exist')

        return model

def VGG16_encoder(input_tensor, weights_init, tensor_type: str):
    """

    :param input_tensor: An tensor which will be enconded through VGG16 without the top layer
    :return: Encoded tensor
    """
    # Encoder block 1
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(64, 3, padding="same", kernel_initializer=weights_init,
               name='block1_conv1' + tensor_type)(input_tensor)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(64, 3, padding="same", kernel_initializer=weights_init,
               name='block1_conv2' + tensor_type)(x)))
    #x = Dropout(0.5)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name="MaxPool1" + tensor_type)(x)

    # Encoder block 2
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(128, 3, padding="same", kernel_initializer=weights_init,
               name='block2_conv1' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(128, 3, padding="same", kernel_initializer=weights_init,
               name='block2_conv2' + tensor_type)(x)))
    #x = Dropout(0.5)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name="MaxPool2" + tensor_type)(x)

    # Encoder block 3
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(256, 3, padding="same", kernel_initializer=weights_init,
               name='block3_conv1' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(256, 3, padding="same", kernel_initializer=weights_init,
               name='block3_conv2' + tensor_type)(x)))
    #x = Dropout(0.3)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name="MaxPool3" + tensor_type)(x)

    # Encoder block 4
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block4_conv1' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block4_conv2' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block4_conv3' + tensor_type)(x)))
    #x = Dropout(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name="MaxPool4" + tensor_type)(x)

    # Encoder block 5
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block5_conv1' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block5_conv2' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block5_conv3' + tensor_type)(x)))
    #x = Dropout(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name="MaxPool5" + tensor_type)(x)

    return x

def VGG16_decoder(input_tensor, weights_init, tensor_type):
    """

    :param input_tensor: Input tensor which is encoded and will be decoded by VGG16 arch
    :param segnet_indices: Boolean if one want to use presaved pooling indiced from encoder
    :return: Decoded tensor
    """
    x = input_tensor
    # Decoder block 5
    x = UpSampling2D(size=(2,2), name='UpSampling5' + tensor_type)(x)
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block5_deconv1' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block5_deconv2' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block5_deconv3' + tensor_type)(x)))
    #x = Dropout(0.2)(x)
    # Block 4
    x = UpSampling2D(size=(2, 2), name='UpSampling4' + tensor_type)(x)
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block4_deconv1' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block4_deconv2' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(512, 3, padding="same", kernel_initializer=weights_init,
               name='block4_deconv3' + tensor_type)(x)))
    #x = Dropout(0.2)(x)
    # Block 3
    x = UpSampling2D(size=(2, 2), name='UpSampling3' + tensor_type)(x)
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(256, 3, padding="same", kernel_initializer=weights_init,
               name='block3_deconv1' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(256, 3, padding="same", kernel_initializer=weights_init,
               name='block3_deconv2' + tensor_type)(x)))
    #x = Dropout(0.3)(x)
    # Block 2
    x = UpSampling2D(size=(2,2), name='UpSampling2' + tensor_type)(x)
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(128, 3, padding="same", kernel_initializer=weights_init,
               name='block2_deconv1' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(128, 3, padding="same", kernel_initializer=weights_init,
               name='block2_deconv2' + tensor_type)(x)))
    #x = Dropout(0.5)(x)
    # Block 1
    x = UpSampling2D(size=(2,2), name='UpSampling1' + tensor_type)(x)
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(64, 3, padding="same", kernel_initializer=weights_init,
               name='block1_deconv1' + tensor_type)(x)))
    x = LeakyReLU(alpha=0.1)(BatchNormalization()(
        Conv2D(64, 3, padding="same", kernel_initializer=weights_init,
               name='block1_deconv2' + tensor_type)(x)))
    #x = Dropout(0.5)(x)

    return x

def initilize_model(model,
                    INPUT_SHAPE,
                    nbr_classes,
                    pre_trained_encoder,
                    indices,
                    weights,
                    load_weights_by_name):

    if model == 'SegNetModel':
        Model = SegNetModel(shape=INPUT_SHAPE,
                            num_classes=nbr_classes,
                            pre_trained_encoder=pre_trained_encoder,
                            segnet_indices=indices,
                            weights=weights,
                            load_weights_by_name=load_weights_by_name)
    elif model == 'dSegNetModel':
        Model = dSegNetModel(shape=INPUT_SHAPE,
                             num_classes=nbr_classes,
                             pre_trained_encoder=pre_trained_encoder,
                             segnet_indices=indices,
                             weights=weights,
                             load_weights_by_name=load_weights_by_name)
    elif model == 'DispSegNetModel':
        Model = DispSegNetModel(shape=INPUT_SHAPE,
                                num_classes=nbr_classes,
                                pre_trained_encoder=pre_trained_encoder,
                                segnet_indices=indices,
                                weights=weights,
                                load_weights_by_name=load_weights_by_name)
    elif model == 'DispSegNetBasicModel':
        Model = DispSegNetBasicModel(shape=INPUT_SHAPE,
                                     num_classes=nbr_classes,
                                     pre_trained_encoder=pre_trained_encoder,
                                     segnet_indices=indices,
                                     weights=weights,
                                     load_weights_by_name=load_weights_by_name)
    elif model == 'PydSegNetModel':
        Model = PydSegNetModel(shape=INPUT_SHAPE,
                                     num_classes=nbr_classes,
                                     segnet_indices=indices,
                                     weights=weights,
                                     load_weights_by_name=load_weights_by_name)

    elif model == 'EncFuseModel':
        Model = EncFuseModel(shape=INPUT_SHAPE,
                             num_classes=nbr_classes,
                             pre_trained_encoder=pre_trained_encoder,
                             segnet_indices=indices,
                             weights=weights,
                             load_weights_by_name=load_weights_by_name
                             )

    else:
        raise NameError('Input a Model which are defined in hyperparameters.. N00b')

    return Model
