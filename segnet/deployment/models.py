
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
from keras.layers import Input
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

    def __init__(self, shape: Tuple[int, int, int], num_classes: int, pre_trained_encoder: bool, segned_indices: bool,
                 weights: Any, load_weights_by_name: bool, kernel_init='he_normal'):
        super().__init__()
        self.input_shape = shape
        self.num_classes = num_classes
        self.pre_trained_encoder = 'imagenet' if pre_trained_encoder else None
        self.segnet_indices = segned_indices
        self.weights_init = kernel_init
        self.pre_trained_weights = weights  # path to a .hdf5 file with the weights that we want to use.
        # If None the weights are initialized according to "kernel_init"
        self.load_weights_by_name = load_weights_by_name

    def create_model(self):

        input_tensor = Input(shape=self.input_shape)  # type: object

        encoder = VGG16(
            include_top=False,
            weights=self.pre_trained_encoder,
            input_tensor=input_tensor,
            input_shape=self.input_shape,
            pooling="None")  # type: tModel

        L = [layer for i, layer in enumerate(encoder.layers)]  # type: List[Layer]
        # for layer in L: layer.trainable = False # freeze VGG16
        L.reverse()

        x = encoder.output

        x = Dropout(0.2)(x)

        # Block 5
        if self.segnet_indices:
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
        if self.segnet_indices:
            x = DePool2D(L[4], size=L[4].pool_size, name='DePool4')(x)
        else:
            x = UpSampling2D(size=L[4].pool_size, name='UpSampling4')(x)
        # x = ZeroPadding2D(padding=(0, 1))(x)
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
        if self.segnet_indices:
            x = DePool2D(L[8], size=L[8].pool_size, name='DePool3')(x)
        else:
            x = UpSampling2D(size=L[8].pool_size, name='UpSampling3')(x)
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[10].filters, L[10].kernel_size, padding=L[10].padding, kernel_initializer=self.weights_init,
                   name='block3_deconv2')(x)))
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[11].filters, L[11].kernel_size, padding=L[11].padding, kernel_initializer=self.weights_init,
                   name='block3_deconv1')(x)))
        x = Dropout(0.3)(x)
        # Block 2
        if self.segnet_indices:
            x = DePool2D(L[12], size=L[12].pool_size, name='DePool2')(x)
        else:
            x = UpSampling2D(size=L[12].pool_size, name='UpSampling2')(x)
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[13].filters, L[13].kernel_size, padding=L[13].padding, kernel_initializer=self.weights_init,
                   name='block2_deconv2')(x)))
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[14].filters, L[14].kernel_size, padding=L[14].padding, kernel_initializer=self.weights_init,
                   name='block2_deconv1')(x)))
        # Block 1
        if self.segnet_indices:
            x = DePool2D(L[15], size=L[15].pool_size, name='DePool1')(x)
        else:
            x = UpSampling2D(size=L[15].pool_size, name='UpSampling1')(x)
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[16].filters, L[16].kernel_size, padding=L[16].padding, kernel_initializer=self.weights_init,
                   name='block1_deconv2')(x)))
        x = Activation('relu')(BatchNormalization()(
            Conv2D(L[17].filters, L[17].kernel_size, padding=L[17].padding, kernel_initializer=self.weights_init,
                   name='block1_deconv1')(x)))

        x = Conv2D(self.num_classes, (1, 1), padding='valid', kernel_initializer=self.weights_init,
                   name='block1_deconv_classes' + str(self.num_classes))(x)
        x = Activation('softmax')(x)
        predictions = x

        segnet = Model(inputs=encoder.inputs, outputs=predictions)  # type: tModel

        if self.pre_trained_weights is not None:
            if os.path.exists(self.pre_trained_weights):
                segnet.load_weights(self.pre_trained_weights, by_name=self.load_weights_by_name)
            else:
                raise AssertionError('Not able to load weights. File does not exist')

        return segnet


class LMSegmentationModel(object):
    """
    new network inspired by vgg16 and segnet, but intended to be smaller
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, weights: AnyStr, load_weights_by_name: bool):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pre_trained_weights = weights
        self.load_weights_by_name = load_weights_by_name

    def create_model(self):

        # Determine proper input shape
        if self.input_shape is None:
            self.input_shape = (480, 320, 3)

        img_input = Input(shape=self.input_shape)

        # encoder
        # Block 1
        x = Conv2D(10, (7, 7), padding='same', name='LM_block1_conv1')(img_input)
        x = BatchNormalization(axis=3, name='LM_batchnorm_enc1_1')(x)
        x = Activation('relu')(x)
        x = Conv2D(10, (5, 5), padding='same', name='LM_block1_conv2')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_enc1_2')(x)
        x = Activation('relu')(x)
        x = Conv2D(10, (3, 3), padding='same', name='LM_block1_conv3')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_enc1_3')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='LM_block1_pool')(x)

        # Block 2
        x = Conv2D(20, (5, 5), padding='same', name='LM_block2_conv1')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_enc2_1')(x)
        x = Activation('relu')(x)
        x = Conv2D(20, (3, 3), padding='same', name='LM_block2_conv2')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_enc2_2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='LM_block2_pool')(x)

        # Block 3
        x = Conv2D(40, (3, 3), padding='same', name='LM_block3_conv1')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_enc3_1')(x)
        x = Activation('relu')(x)
        x = Conv2D(40, (3, 3), padding='same', name='LM_block3_conv2')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_enc3_2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='LM_block3_pool')(x)

        # Block 4
        x = Conv2D(40, (3, 3), padding='same', name='LM_block4_conv1')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_enc4_1')(x)
        x = Activation('relu')(x)
        x = Conv2D(40, (3, 3), padding='same', name='LM_block4_conv2')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_enc4_2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='LM_block4_pool')(x)

        # decoder
        # mirroring block 4
        x = UpSampling2D(name='LM_block4_depool')(x)
        x = Conv2D(40, (3, 3), padding='same', name='LM_block4_deconv2')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_dec4_2')(x)
        x = Activation('relu')(x)
        x = Conv2D(40, (3, 3), padding='same', name='LM_block4_deconv1')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_dec4_1')(x)
        x = Activation('relu')(x)

        # mirroring block 3
        x = UpSampling2D(name='LM_block3_depool')(x)
        x = Conv2D(40, (3, 3), padding='same', name='LM_block3_deconv2')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_dec3_2')(x)
        x = Activation('relu')(x)
        x = Conv2D(20, (3, 3), padding='same', name='LM_block3_deconv1')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_dec3_1')(x)
        x = Activation('relu')(x)

        #mirroring block 2
        x = UpSampling2D(name='LM_block2_depool')(x)
        x = Conv2D(20, (3, 3), padding='same', name='LM_block2_deconv2')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_dec2_2')(x)
        x = Activation('relu')(x)
        x = Conv2D(10, (5, 5), padding='same', name='LM_block2_deconv1')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_dec2_1')(x)
        x = Activation('relu')(x)

        #mirroring block 1
        x = UpSampling2D(name='LM_block1_depool')(x)
        x = Conv2D(10, (3, 3), padding='same', name='LM_block1_deconv3')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_dec1_3')(x)
        x = Activation('relu')(x)
        x = Conv2D(10, (5, 5), padding='same', name='LM_block1_deconv2')(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_dec1_2')(x)
        x = Activation('relu')(x)
        x = Conv2D(self.num_classes, (7, 7), padding='same', name='LM_block1_classes' + str(self.num_classes))(x)
        x = BatchNormalization(axis=3, name='LM_batchnorm_classes' + str(self.num_classes))(x)
        x = Activation('softmax')(x)
        prediction = x

        # Create model.
        LMmodel = Model(img_input, prediction, name='LMseg')

        if self.pre_trained_weights is not None:
            if os.path.exists(self.pre_trained_weights):
                LMmodel.load_weights(self.pre_trained_weights, by_name=self.load_weights_by_name)
            else:
                raise AssertionError('Not able to load weights. File does not exist')

        return LMmodel




# based on the structure of resnet50 blocks I create the "inverse" of these blocks to map back to input dimension

def inv_identity_block(input_tensor, kernel_size, filters, stage, block):

    filters2, filters1, filters_inp = filters
    bn_axis = 3
    conv_name_base = 'inv_res' + str(stage) + block + '_branch'
    bn_name_base = 'inv_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters2, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters1, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters_inp, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def inv_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    filters2, filters1, filters_inp = filters
    bn_axis = 3
    conv_name_base = 'inv_res' + str(stage) + block + '_branch'
    bn_name_base = 'inv_bn' + str(stage) + block + '_branch'
    # CONVOLUTION WITH STRIDES (2,2) IS REDUCING DIMENSIONS. INVERSE OPERATION SHOULD UPSAMPLE AGAIN

    x = Conv2D(filters2, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters1, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=strides)(x)
    x = Conv2D(filters_inp, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = UpSampling2D(size=strides)(input_tensor)
    shortcut = Conv2D(filters_inp, (1, 1), name=conv_name_base + '1')(shortcut)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


class SegResnet50Model(object):
    """
    Segnet version out of Resnet50
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, pre_trained_encoder: bool,
                 weights: Any, load_weights_by_name: bool):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pre_trained_encoder = 'imagenet' if pre_trained_encoder else None
        self.pre_trained_weights = weights
        self.load_weights_by_name = load_weights_by_name

    def create_model(self):

        input_tensor = Input(shape=self.input_shape)  # type: object
        encoder = ResNet50(include_top=False,
                           weights=self.pre_trained_encoder,
                           input_tensor=input_tensor,
                           input_shape=self.input_shape,
                           pooling="None")

        x = encoder.output

        # decoder
        x = UpSampling2D(size=(7, 7), name='inv_average_pool')(x)

        x = inv_identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        x = inv_identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = inv_conv_block(x, 3, [512, 512, 1024], stage=5, block='a', strides=(1, 1))

        x = inv_identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
        x = inv_identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = inv_identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = inv_identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = inv_identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = inv_conv_block(x, 3, [256, 256, 512], stage=4, block='a')

        x = inv_identity_block(x, 3, [128, 128, 512], stage=3, block='d')
        x = inv_identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = inv_identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = inv_conv_block(x, 3, [128, 128, 256], stage=3, block='a')

        x = inv_identity_block(x, 3, [64, 64, 256], stage=2, block='c')
        x = inv_identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = inv_conv_block(x, 3, [64, 64, 64], stage=2, block='a')

        x = UpSampling2D(size=(2, 2), name='inv_max_pooling_stride')(x)
        x = UpSampling2D(size=(2, 2), name='inv_stride')(x)

        x = Conv2D(self.num_classes, (7, 7), padding='same', name='inv_conv_classes' + str(self.num_classes))(x)
        x = BatchNormalization(axis=3, name='inv_bn_conv_classes' + str(self.num_classes))(x)
        predictions = Activation('softmax')(x)

        SegResnet50 = Model(inputs=input_tensor, outputs=predictions)

        if self.pre_trained_weights is not None:
            if os.path.exists(self.pre_trained_weights):
                SegResnet50.load_weights(self.pre_trained_weights, by_name=self.load_weights_by_name)
            else:
                raise AssertionError('Not able to load weights. File does not exist')

        return SegResnet50

