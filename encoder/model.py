import tensorflow as tf
import numpy as np

NET_SIZE = 128


def MobileResBlock(x, t=6):
    Cin = int(x.shape[-1])

    x = tf.keras.layers.Conv2D(Cin * t, (1, 1))(x)
    x = tf.keras.layers.Activation('tanh')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same')(x)
    x = tf.keras.layers.Activation('tanh')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(Cin, (1, 1))(x)
    x = tf.keras.layers.Activation('tanh')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x


def downsample(x):
    # x = x + MobileResBlock(x)
    # x = x + MobileResBlock(x)
    x = x + MobileResBlock(x)

    x = tf.keras.layers.Conv2D(int(x.shape[-1] * 2), (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Activation('tanh')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x


def Encoder():
    inputs = tf.keras.layers.Input((NET_SIZE, NET_SIZE, 3))

    x = inputs

    for i in range(int(np.log2(NET_SIZE))):
        x = downsample(x)
        print(x.shape)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation='tanh')(x)

    x = tf.keras.layers.Dense(16, activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)