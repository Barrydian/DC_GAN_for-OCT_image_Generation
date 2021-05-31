
# Images dimension
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNEL = 1 ## Change 3 to 1 for grayscale.


import tensorflow as tf
w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


## Hyperparameters
batch_size = 128
latent_dim = 128
num_epochs = 2000
n_samples = 25     ## n should always be a square of an integer.


# contantes -1 < img <1
IMG_CONST = 127.5