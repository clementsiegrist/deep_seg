from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import os
import keras
from keras.layers import BatchNormalization, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Dense, Input, Activation, concatenate
#from Models.losses import weighted_bce_dice_loss, weighted_dice_loss
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.utils import class_weight
from keras import layers as L
from keras import backend as K
from keras.models import Model
import numpy as np
import random
seed = 42
random.seed = seed
np.random.seed(seed=seed)


def unet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=False, conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder"""
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def CorneaNet(pretrained_weights=None, input_size=(256, 256, 3)):

    inputs = Input(input_size)
    #weight_ip = L.Input(shape=(256, 512, 1))

    unet_layer1 = unet_layer(inputs, num_filters=16)
    unet_layer1 = unet_layer(unet_layer1, num_filters=16)
    pool1 = MaxPooling2D(pool_size=(2, 2))(unet_layer1)

    unet_layer2 = unet_layer(pool1, num_filters=32)
    unet_layer2 = unet_layer(unet_layer2, num_filters=32)
    pool2 = MaxPooling2D(pool_size=(2, 2))(unet_layer2)

    unet_layer3 = unet_layer(pool2, num_filters=64)
    unet_layer3 = unet_layer(unet_layer3, num_filters=64)
    pool3 = MaxPooling2D(pool_size=(2, 2))(unet_layer3)

    unet_layer4 = unet_layer(pool3, num_filters=128)
    unet_layer4 = unet_layer(unet_layer4, num_filters=128)
    pool4 = MaxPooling2D(pool_size=(2, 2))(unet_layer4)

    unet_layer5 = unet_layer(pool4, num_filters=256)
    unet_layer5 = unet_layer(unet_layer5, num_filters=256)
    drop5 = Dropout(0.5, trainable=True)(unet_layer5)

    up6 = UpSampling2D(size=(2, 2))(drop5)
    unet_layer_after_up6 = unet_layer(up6, num_filters=128)
    merge6 = concatenate([unet_layer4, unet_layer_after_up6])
    unet_layer6 = unet_layer(merge6, num_filters=128)
    unet_layer6 = unet_layer(unet_layer6, num_filters=128)

    up7 = UpSampling2D(size=(2, 2))(unet_layer6)
    unet_layer_after_up7 = unet_layer(up7, num_filters=64)
    merge7 = concatenate([unet_layer3, unet_layer_after_up7])
    unet_layer7 = unet_layer(merge7, num_filters=64)
    unet_layer7 = unet_layer(unet_layer7, num_filters=64)

    up8 = UpSampling2D(size=(2, 2))(unet_layer7)
    unet_layer_after_up8 = unet_layer(up8, num_filters=32)
    merge8 = concatenate([unet_layer2, unet_layer_after_up8])
    unet_layer8 = unet_layer(merge8, num_filters=32)
    unet_layer8 = unet_layer(unet_layer8, num_filters=32)

    up9 = UpSampling2D(size=(2, 2))(unet_layer8)
    unet_layer_after_up9 = unet_layer(up9, num_filters=16)
    merge9 = concatenate([unet_layer1, unet_layer_after_up9])
    unet_layer9 = unet_layer(merge9, num_filters=16)
    unet_layer9 = unet_layer(unet_layer9, num_filters=16)
    unet_layer_last9 = unet_layer(unet_layer9, num_filters=16)
    #dense9 = Conv2D(2, 1, activation='softmax')(unet_layer_last9)
    dense9 = Dense(2, activation='sigmoid')(unet_layer_last9)
    model = Model(inputs, dense9)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model