# gan.py
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((28, 28))
    image = np.array(image)
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image

def build_generator(latent_dim, img_shape):
    H, W = img_shape
    i = Input(shape=(latent_dim,))
    x = Dense(128)(i)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(np.prod((H, W)), activation='tanh')(x)
    x = Reshape((H, W))(x)
    model = Model(i, x)
    return model

def build_discriminator(img_shape):
    H, W = img_shape
    i = Input(shape=(H, W))
    x = Flatten()(i)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(i, x)
    return model

def sample_images(generator, latent_dim, epoch, save_dir):
    rows, cols = 5, 5
    noise = np.random.randn(rows * cols, latent_dim)
    imgs = generator.predict(noise)
    imgs = 0.5 * imgs + 0.5
    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(imgs[idx], cmap='gray')
            axs[i, j].axis('off')
            idx += 1
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(f"{save_dir}/{epoch}.png")
    plt.close()

def train_gan(image_path, epochs, latent_dim=100, batch_size=32):
    x_custom = load_image(image_path)
    x_custom = np.expand_dims(x_custom, axis=0)
    x_custom = x_custom / 255.0 * 2 - 1
    H, W = x_custom.shape[1:]

    generator = build_generator(latent_dim, (H, W))
    discriminator = build_discriminator((H, W))
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

    z = Input(shape=(latent_dim,))
    img = generator(z)
    discriminator.trainable = False
    validity = discriminator(img)
    combined_model = Model(z, validity)
    combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    ones = np.ones((batch_size, 1))
    zeros = np.zeros((batch_size, 1))

    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        idx = np.random.randint(0, x_custom.shape[0], batch_size)
        real_imgs = x_custom[idx]
        noise = np.random.randn(batch_size, latent_dim)
        fake_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_imgs, ones)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, zeros)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.randn(batch_size, latent_dim)
        g_loss = combined_model.train_on_batch(noise, ones)

        d_losses.append(d_loss[0])
        g_losses.append(g_loss)

        if epoch % 100 == 0:
            print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss[0]}, d_acc: {d_loss[1]}, g_loss: {g_loss}")

        if epoch % 200 == 0:
            sample_images(generator, latent_dim, epoch, "static/gan_images")

    return d_losses, g_losses
