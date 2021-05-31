import os
import numpy as np
import tensorflow as tf
from glob import glob
import sys
#import constantes using gan 
from constantes_gan import IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, batch_size,latent_dim,num_epochs,n_samples

# import all function gan from main 
from load_data import tf_dataset
from discriminator import build_discriminator
from generator import build_generator
from gan import GAN, save_plot
from resize_img import resize_img

if __name__ == "__main__":
    if len (sys.argv) != 3 :
        print(" Args not found : global path for arg 1 and image subdirectory for arg 2.")
        sys.exit(1)

    if not os.path.exists(sys.argv[1] + "/" + sys.argv[2] + "_resized"):
        os.mkdir(sys.argv[1] + "/" + sys.argv[2] + "_resized")
    if not os.path.exists(sys.argv[1] + "/" + sys.argv[2] + "_saved_models"):
        os.mkdir(sys.argv[1] + "/" + sys.argv[2] + "_saved_models")
    if not os.path.exists(sys.argv[1] + "/" + sys.argv[2] + "_samples"):
        os.mkdir(sys.argv[1] + "/" + sys.argv[2] + "_samples")
    resize_img(sys.argv[1] + "/" + sys.argv[2], sys.argv[1] + "/" + sys.argv[2] + "_resized", IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)
   
    images_path = glob(sys.argv[1] + "/" + sys.argv[2] + "_resized/*")

    d_model = build_discriminator()
    g_model = build_generator(latent_dim)

    d_model.summary()
    g_model.summary()   

    gan = GAN(d_model, g_model, latent_dim)

    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(d_optimizer, g_optimizer, bce_loss_fn)

    images_dataset = tf_dataset(images_path, batch_size)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        gan.fit(images_dataset, epochs=1)
        g_model.save( sys.argv[1] + "/" + sys.argv[2] + "_saved_models/g_model.h5")
        d_model.save( sys.argv[1] + "/" + sys.argv[2] + "_saved_models/d_model.h5")

        noise = np.random.normal(size=(n_samples, latent_dim))
        examples = g_model.predict(noise)
        save_plot(sys.argv[1] + "/" + sys.argv[2] + "_samples", examples, epoch, int(np.sqrt(n_samples)), IMG_CHANNEL)
