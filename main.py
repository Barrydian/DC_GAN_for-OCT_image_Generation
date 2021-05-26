
import numpy as np
import tensorflow as tf
from glob import glob
import sys
#import constantes using gan 
from constantes_gan import batch_size,latent_dim,num_epochs,n_samples

# import all function gan from main 
from load_data import tf_dataset
from discriminator import build_discriminator
from generator import build_generator
from gan import GAN, save_plot
from resize_img import resize_img

if __name__ == "__main__":

    if (len(sys.argv)!= 1 ):
        exit(1)

    resize_img(sys.argv[1],"database")
   
    images_path = glob("database/*")

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
        gan.fit(images_dataset, epochs=1)
        g_model.save("saved_model/g_model.h5")
        d_model.save("saved_model/d_model.h5")

        noise = np.random.normal(size=(n_samples, latent_dim))
        examples = g_model.predict(noise)
        save_plot(examples, epoch, int(np.sqrt(n_samples)))
