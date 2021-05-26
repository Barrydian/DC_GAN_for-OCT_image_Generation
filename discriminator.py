from constantes_gan import IMG_WIDTH, IMG_HEIGHT,IMG_CHANNEL,w_init
from tensorflow.keras.layers import *
import cv2
from tensorflow.keras.models import Model



def conv_block(inputs, num_filters, kernel_size, padding="same", strides=2, activation=True):
    x = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=w_init,
        padding=padding,
        strides=strides,
    )(inputs)

    if activation:
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
    return x

def build_discriminator():
    f = [2**i for i in range(4)]
    image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
    x = image_input
    filters = 64
    output_strides = 16
    h_output = IMG_HEIGHT // output_strides
    w_output = IMG_WIDTH // output_strides

    for i in range(0, 4):
        x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, strides=2)

    x = Flatten()(x)
    x = Dense(1)(x)

    return Model(image_input, x, name="discriminator")
