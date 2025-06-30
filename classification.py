from sklearn import metrics
import numpy as np
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, DepthwiseConv2D, BatchNormalization, GlobalMaxPooling2D


def error(actual, pred):
    #'MSE', 'MAE', 'NMSE', 'RMSE', 'MAPE'
    err=np.empty(5)
    err[0] = metrics.mean_squared_error(actual, pred)
    err[1] = metrics.mean_absolute_error(actual, pred)
    err[2] = metrics.mean_squared_log_error(actual, pred)
    rms = metrics.mean_squared_error(actual, pred)
    err[3] = sqrt(rms)
    err[4] = metrics.mean_absolute_percentage_error(actual, pred)
    return err


# vgg 19
def vgg19(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

    # create VGG19 model
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same', input_shape=x_train[1].shape))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Block 2
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Block 3
    model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Block 4
    model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Block 5
    model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
    model.add(Conv2D(512, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Fully Co layers
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, steps_per_epoch=10)
    pred = np.argmax(model.predict(x_test), axis=1)
    met = error(y_test, pred)
    return pred, met


# EfficientNet
def EfficientNet(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

    model = Sequential()
    model.add(Conv2D(64, (1, 1), activation='relu', padding='same', input_shape=x_train[1].shape))

    # Block 1
    model.add(Conv2D(64, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # Block 2
    model.add(Conv2D(128, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # Block 3
    model.add(Conv2D(256, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # Block 4
    model.add(Conv2D(512, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # Block 5
    model.add(Conv2D(256, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # Block 6
    model.add(Conv2D(128, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # Block 7
    model.add(Conv2D(64, (1, 1), activation='relu', padding='same'))
    model.add(DepthwiseConv2D((1, 1), (1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(GlobalMaxPooling2D())
    model.add(Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, steps_per_epoch=10)
    pred = np.argmax(model.predict(x_test), axis=1)
    met = error(y_test, pred)
    return pred, met


# DenseNet - densely-connected-convolutional networks
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, Activation, AveragePooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name):
    x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(tf.keras.backend.int_shape(x)[-1] * reduction), (1, 1), padding='same', name=name + '_conv')(x)
    x = AveragePooling2D((1, 1), strides=(1, 1), name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name):
    x1 = BatchNormalization(name=name + '_bn1')(x)
    x1 = Activation('relu', name=name + '_relu1')(x1)
    x1 = Conv2D(4 * growth_rate, (1, 1), padding='same', name=name + '_conv1')(x1)

    x1 = BatchNormalization(name=name + '_bn2')(x1)
    x1 = Activation('relu', name=name + '_relu2')(x1)
    x1 = Conv2D(growth_rate, (3, 3), padding='same', name=name + '_conv2')(x1)

    x = tf.keras.layers.concatenate([x, x1], axis=-1, name=name + '_concat')
    return x

def densenet_121(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

    input_tensor = Input(shape=x_train[1].shape, name='input')

    x = Conv2D(64, (7, 7), strides=(1, 1), padding='same', name='conv1')(input_tensor)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu', name='relu_conv1')(x)
    x = AveragePooling2D((1, 1), strides=(1, 1), padding='same', name='pool1')(x)

    x = dense_block(x, 6, name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, 12, name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, 24, name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, 16, name='conv5')

    x = BatchNormalization(name='bn')(x)
    x = Activation('relu', name='relu')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(2, activation='sigmoid', name='fc')(x)

    model = Model(inputs=input_tensor, outputs=x, name='densenet_121_custom')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, steps_per_epoch=10)
    pred = np.argmax(model.predict(x_test), axis=1)
    met = error(y_test, pred)
    return pred, met


# ResNet - Residual neural network
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.models import Model

def identity_block(x, filters, kernel_size=3, strides=1):
    shortcut = x

    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def resnet_conv_block(x, filters, kernel_size=3, strides=2):
    shortcut = x

    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def resnet50(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)
    inputs = Input(shape=x_train[1].shape)

    # Initial convolution
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Max pooling
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual blocks
    x = resnet_conv_block(x, 64)
    x = identity_block(x, 64)
    x = identity_block(x, 64)

    x = resnet_conv_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)

    x = resnet_conv_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)

    x = resnet_conv_block(x, 512)
    x = identity_block(x, 512)
    x = identity_block(x, 512)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='resnet50_non_image')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, steps_per_epoch=10)
    pred = np.argmax(model.predict(x_test), axis=1)
    met = error(y_test, pred)
    return pred, met


def FusionNet_4(x_train, y_train, x_test, y_test):
    vgg19_pred, met = vgg19(x_train, y_train, x_test, y_test)
    efficientnet_pred , met = EfficientNet(x_train, y_train, x_test, y_test)
    densenet121_pred, met = densenet_121(x_train, y_train, x_test, y_test)
    resnet50_pred, met = resnet50(x_train, y_train, x_test, y_test)

    pred = (vgg19_pred + efficientnet_pred + densenet121_pred + resnet50_pred) / 4
    pred = np.round(pred).astype('int32')
    met = error(y_test, pred)
    return pred, met


