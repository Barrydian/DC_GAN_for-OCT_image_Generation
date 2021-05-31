
import numpy as np
import sys
from tensorflow.keras.models import load_model
from matplotlib.image import imsave
from skimage import exposure
from constantes_gan import IMG_CHANNEL, latent_dim, n_samples


def save_imgs(path, imgs, num_channels):
    for i in range(imgs.shape[0]):
        img = np.squeeze(examples[i], axis=2)
        if num_channels==1:
            img = np.stack((img,)*3, axis=-1)
        img = exposure.rescale_intensity(img, out_range=(0,1))
        imsave(path + "/img_" + str(i) + ".png", img)
        
if __name__ == "__main__":
    if len (sys.argv) != 2 :
        print(" Args not found : path to weights file for arg 1.")
        sys.exit(1)

    model = load_model(sys.argv[1] + "/g_model.h5")

    latent_points = np.random.normal(size=(n_samples, latent_dim))
    examples = model.predict(latent_points)
    save_imgs(sys.argv[1], examples, IMG_CHANNEL)
