
# Images dimension
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNEL = 3 ## Change 3 to 1 for grayscale.


import tensorflow as tf
w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


## Hyperparameters
batch_size = 5
latent_dim = 128
num_epochs = 30
n_samples = 25     ## n should always be a square of an integer.


# contantes -1 < img <1
IMG_CONST = 127.5