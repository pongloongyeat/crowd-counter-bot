from keras.layers import Conv2D, concatenate, UpSampling2D, Conv2DTranspose, BatchNormalization
from keras.models import Model
from keras.initializers import RandomNormal, Constant
from .inception_v1 import InceptionV1


def IUML(input_shape=(None, None, 3)):
    googlenet = InceptionV1(include_top=False, weights='imagenet', input_shape=input_shape)

    # Remove some layers (intercept output of inception 4f)
    base_model = Model(inputs=googlenet.input, outputs=googlenet.get_layer('Mixed_5a_Concatenated').output)

    # Freeze model```
    base_model.trainable = False

    kernel_initializer  = RandomNormal(stddev=0.01)
    bias_initializer    = Constant(0.001)

    # Decoder
    deconv1             = Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding='same', activation='relu', name='deconv1', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(base_model.layers[-1].output)
    bn_de1              = BatchNormalization(name='bn_de1')(deconv1)
    concat_de1          = concatenate([bn_de1, base_model.get_layer('Mixed_4c_Concatenated').output])

    conv1_de1           = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name='conv1_de1', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(concat_de1)
    conv2_de1           = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name='conv2_de1', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(conv1_de1)

    deconv2             = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding='same', activation='relu', name='deconv2', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(conv2_de1)
    bn_de2              = BatchNormalization(name='bn_de2')(deconv2)
    concat_de2          = concatenate([bn_de2, base_model.get_layer('Mixed_3b_Concatenated').output])

    conv1_de2           = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name='conv1_de2', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(concat_de2)
    conv2_de2           = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name='conv2_de2', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(conv1_de2)

    deconv3             = Conv2DTranspose(filters=192, kernel_size=2, strides=2, padding='same', activation='relu', name='deconv3', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(conv2_de2)
    bn_de3              = BatchNormalization(name='bn_de3')(deconv3)
    concat_de3          = concatenate([bn_de3, base_model.get_layer('Conv2d_2c_3x3_act').output])

    conv1_de3           = Conv2D(filters=192, kernel_size=3, padding='same', activation='relu', name='conv1_de3', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(concat_de3)
    conv2_de3           = Conv2D(filters=192, kernel_size=3, padding='same', activation='relu', name='conv2_de3', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(conv1_de3)

    deconv4             = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding='same', activation='relu', name='deconv4', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(conv2_de3)
    bn_de4              = BatchNormalization(name='bn_de4')(deconv4)
    concat_de4          = concatenate([bn_de4, base_model.get_layer('Conv2d_1a_7x7_act').output])

    conv1_de4           = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv1_de4', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(concat_de4)
    conv2_de4           = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', name='conv2_de4', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(conv1_de4)

    # Let's upsample the image so size(img) == size(output)
    upsample            = UpSampling2D(size=2, interpolation='bilinear', name='upsample')(conv2_de4)

    # Output
    regression_1x1      = Conv2D(filters=1, kernel_size=1, padding='same', activation='relu', name='regression_1x1', kernel_initializer=RandomNormal(stddev=0.01), bias_initializer=Constant(0.001))(upsample)

    model = Model(inputs=base_model.input, outputs=regression_1x1)

    return model
