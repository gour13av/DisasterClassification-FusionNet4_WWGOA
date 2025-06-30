import tensorflow as tf
from tensorflow.keras import layers, models


def dilated_unet(input_shape, dilation):
    dilation = int(dilation[0])
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.AveragePooling2D(pool_size=(2, 2))(conv1)

    # Add dilated convolution layers
    dilated_conv1 = layers.Conv2D(128, 3, activation='relu', padding='same', dilation_rate=dilation)(pool1)
    dilated_conv1 = layers.Conv2D(128, 3, activation='relu', padding='same', dilation_rate=dilation)(dilated_conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(dilated_conv1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)

    pool2 = layers.AveragePooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)

    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)


    # Decoder
    up_conv3 = layers.Conv2DTranspose(128, 2, strides=(4, 4), padding='same')(pool3)
    up_conv3 = layers.concatenate([up_conv3, conv2], axis=3)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(up_conv3)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up_conv2 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv4)
    up_conv2 = layers.concatenate([up_conv2, conv1], axis=3)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(up_conv2)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # Output layer
    outputs = layers.Conv2D(3, 1, activation='sigmoid')(conv5)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


