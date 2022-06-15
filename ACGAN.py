# Course Project for CptS 570, Machine Learning
# Instructor: Dr. Doppa
# Washington State University
#
# Author: Colin Greeley


# Starter code taken from https://github.com/eriklindernoren/Keras-GAN/tree/master/acgan
# The Keras-GAN github repo has great examples for many GAN architectures, all built
# for generating images of hand written digits in the MNIST dataset




from __future__ import print_function, division

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.backend import truncated_normal
from tensorflow.python.keras.engine import input_layer
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add, Conv2DTranspose, UpSampling2D, Conv2D, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras import backend
from tensorflow.keras.utils import plot_model
from sklearn.utils import shuffle
import time
import os
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd


class MinibatchStdev(layers.Layer):
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # calculate the mean standard deviation across each pixel coord
    def call(self, inputs):
        mean = backend.mean(inputs, axis=0, keepdims=True)
        mean_sq_diff = backend.mean(backend.square(inputs - mean), axis=0, keepdims=True) + 1e-8
        mean_pix = backend.mean(backend.sqrt(mean_sq_diff), keepdims=True)
        shape = backend.shape(inputs)
        output = backend.tile(mean_pix, [shape[0], shape[1], shape[2], 1])
        return backend.concatenate([inputs, output], axis=-1)

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)

def get_data(data_dir, data_dir2, size):
    start = time.time()
    print("\n[INFO] Gathering images and converting them to np arrays")
    images = [cv2.resize(cv2.imread(data_dir + im_path, flags=cv2.IMREAD_COLOR), (size, size)) for im_path in os.listdir(data_dir)]
    images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images])
    print(images.shape)
    images = images / 127.5 - 1.
    print("[INFO] Conversion took {} seconds".format(round(time.time() - start, 2)))
    df = pd.read_csv(data_dir2)
    labels =  df["Type1"].to_list()
    label_id = list(set(labels))
    multi_class_imgs = [[] for _ in range(len(label_id))]
    for i, l in zip(images, labels):
        multi_class_imgs[label_id.index(l)].append(i)
    for i in range(len(multi_class_imgs)):
        multi_class_imgs[i] = np.asarray(multi_class_imgs[i])
    return multi_class_imgs

def get_multiclass_data(data_dir, size):
    start = time.time()
    print("\n[INFO] Gathering images and converting them to np arrays")
    image_list = []
    image_dict = {}
    classes = os.listdir(data_dir)
    for i in range(len(classes)):
        #print(classes[i])
        class_dir = data_dir + classes[i] + '/'
        images = [cv2.resize(cv2.imread(class_dir + im_path), (size, size), cv2.INTER_AREA) for im_path in os.listdir(class_dir)]
        images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images])
        images = images / 127.5 - 1.
        image_list.append(images)
        image_dict.update({i: classes[i]})
    print("[INFO] Conversion took {} seconds".format(round(time.time() - start, 2)))
    return image_list, image_dict

def get_csv(data_dir):
    df = pd.read_csv(data_dir)
    return df["Type1"].to_numpy()

def label_smoothing(x, a=0.1):
    x = (1 - a) * x + (a / len(x))
    return x

def mix_images(x, y, p):
    n_select = int(p * y.shape[0])
    flip_ix = np.random.choice(y.shape[0], size=n_select, replace=False)
    swap = x[flip_ix]
    x[flip_ix] = y[flip_ix]
    y[flip_ix] = swap
    return x, y

def flip_labels(x, p):
    for i in range(len(x)):
        if np.random.random() < p:
            x[i] = -x[i] + 1
    return x

def add_noise(images, n, e, t):
    for image in images:
        std = n - (n * (e/t)**(1/2))
        img_noise = np.random.normal(0, std, image.shape)
        image += img_noise
    return images



class ACGAN():
    def __init__(self, image_list, size=128):
        self.image_list = image_list
        self.size = size
        self.img_rows = size
        self.img_cols = size
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = size
        self.dropout = 0.25
        self.instance_noise = 0.1
        self.num_classes = len(self.image_list)
        self.history = []

        self.g_optimizer = Adam(0.0002, 0.5)#, clipvalue=1.0)
        self.d_optimizer = Adam(0.0002, 0.5)#, clipvalue=1.0)
        #self.loss = 'mse'
        self.loss = [BinaryCrossentropy(label_smoothing=0.2, from_logits=True), CategoricalCrossentropy(label_smoothing=0.0, from_logits=True)]
        #self.loss = [Hinge(), CategoricalCrossentropy(label_smoothing=0.0, from_logits=True)]
        #self.loss = self.wasserstein_loss
        #self.loss = Hinge()

        # Build the generator
        self.build_generator()
        plot_model(self.generator, to_file='./Generator.png', show_shapes=True, show_layer_names=True)

        # Build and compile the discriminator
        self.build_discriminator()
        self.discriminator.compile(loss=self.loss, optimizer=self.d_optimizer, metrics=['accuracy'])
        plot_model(self.discriminator, to_file='./Discriminator.png', show_shapes=True, show_layer_names=True, expand_nested=True)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        w = Input(shape=(1,))
        img = self.generator([z, w])
        for layer in self.discriminator.layers:
            layer.trainable = False
        self.discriminator.trainable = False
        v, x = self.discriminator(img)
        self.combined = Model([z, w], [v, x], name="ACGAN")
        self.combined.compile(loss=self.loss, optimizer=self.g_optimizer)
        plot_model(self.combined, to_file='./ACGAN.png', show_shapes=True, show_layer_names=True, expand_nested=True)

    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def build_generator(self):

        def res_block(layer_input, filters):
            up = layers.UpSampling2D()(layer_input)
            v = (layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal'))(up)
            u = (layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'))(up)
            u = layers.BatchNormalization()(u)
            u = LeakyReLU(alpha=0.2)(u)
            u = (layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'))(u)
            u = layers.Add()([u, v])
            u = layers.BatchNormalization()(u)
            u = LeakyReLU(alpha=0.2)(u)
            return u

        def upscale_block(layer_input, filters):
            u = layers.UpSampling2D()(layer_input)
            u = (layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal'))(u)
            u = layers.BatchNormalization()(u)
            u = LeakyReLU(alpha=0.2)(u)
            u = (layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal'))(u)
            u = layers.BatchNormalization()(u)
            u = LeakyReLU(alpha=0.2)(u)
            return u


        input_dim = self.size #128 #
        output_size = 8
        self.g_outs = []
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(layers.Embedding(self.num_classes, self.latent_dim)(label))

        n1 = layers.Multiply()([noise, label_embedding])
        u = Reshape((1, 1, input_dim))(n1)
        u = (layers.Conv2DTranspose(input_dim, 4, padding='valid', kernel_initializer='he_normal'))(u)
        u = LeakyReLU(alpha=0.2)(u)
        u = (layers.Conv2D(input_dim, 3, padding='same', kernel_initializer='he_normal'))(u)
        u = layers.BatchNormalization()(u)
        u = LeakyReLU(alpha=0.2)(u)

        for i in range(5):
            #u = res_block(u, self.size//(2**math.ceil(i/2)))
            u = res_block(u, self.size//(2**i))

        u = Conv2D(self.channels, kernel_size=1, padding='same', activation='tanh', kernel_initializer='he_normal')(u)

        self.generator = Model([noise, label], u, name='Generator')
        self.generator.summary()

    def build_discriminator(self):

        def res_block(layer_input, filters):
            """Discriminator layer"""
            p = (Conv2D(filters, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal'))(layer_input)
            d = (Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'))(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = (Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'))(d)
            d = layers.Add()([d, p])
            d = LeakyReLU(alpha=0.2)(d)
            d = AveragePooling2D()(d)
            return d

        def downscale_block(layer_input, filters):
            """Discriminator layer"""
            d = (Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'))(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = (Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'))(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = AveragePooling2D()(d)
            return d

        input_size = self.size
        #output_size = self.size // 4
        output_size = 8

        d_in = Input(shape=(input_size, input_size, self.channels))
        d = res_block(d_in, output_size)
        for i in range(1, 5):
            #d = res_block(d, output_size * (2**math.ceil(i/2)))
            d = res_block(d, output_size * (2**i))

        d = MinibatchStdev()(d)
        d = (Conv2D(input_size, kernel_size=3, padding='same', kernel_initializer='he_normal'))(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = (Conv2D(input_size, kernel_size=4, padding='valid', kernel_initializer='he_normal'))(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = layers.Flatten()(d)
        d = layers.Dropout(self.dropout)(d)
        validity = Dense(1, activation='linear', name="validity")(d)
        label = Dense(self.num_classes, activation="linear", name="label")(d)
        self.discriminator = Model(d_in, [validity, label], name='Discriminator')
        self.discriminator.summary()

    def train(self, iterations, batch_size=100, save_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        t = time.time()
        for iteration in range(iterations):

            # ---------------------
            #  Train Discriminator
            # ----
            imgs = []
            labels = []
            idx = np.random.randint(0, len(self.image_list), size=batch_size)
            for i in idx:
                imgs.append(self.image_list[i][np.random.randint(0, len(self.image_list[i]))])
                labels.append([i])
            imgs, labels = shuffle(imgs, labels)
            imgs = np.asarray(imgs)
            labels = np.asarray(labels)

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_labels = np.random.randint(0, len(self.image_list), (batch_size, 1))
            gen_imgs = self.generator.predict([noise, gen_labels])
            imgs = add_noise(imgs, self.instance_noise, iteration, iterations)
            gen_imgs = add_noise(gen_imgs, self.instance_noise, iteration, iterations)
            valid = flip_labels(np.ones((batch_size, 1)), 0.05)
            fake = flip_labels(np.zeros((batch_size, 1)), 0.05)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, backend.one_hot(labels[:,0], self.num_classes)])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, backend.one_hot(gen_labels[:,0], self.num_classes)])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_labels = np.random.randint(0, len(self.image_list), (batch_size, 1))
                g_loss = self.combined.train_on_batch([noise, gen_labels], [valid, backend.one_hot(gen_labels[:,0], self.num_classes)])

            # Plot the progress
            if iteration % 100 == 0:
                print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f], time: %.2f" % (iteration, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0], time.time() - t))
                self.history.append([d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]])
                t = time.time()

            # If at save interval => save generated image samples
            if iteration % save_interval == 0:
                self.save_imgs(iteration, self.image_list)
                self.save_real_imgs(iteration, imgs)

    def plot_results(self):
        self.history = np.array(self.history)
        plt.figure()
        plt.title("ACGAN_Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.plot(range(len(self.history)), self.history[:,1], label='validity')
        plt.plot(range(len(self.history)), self.history[:,2], label='label')
        plt.legend()
        plt.savefig("ACGAN_Accuracy.png")
        #plt.show()
        plt.clf()
        plt.figure()
        plt.title("ACGAN_Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.plot(range(len(self.history)), self.history[:,0], label='Discriminator')
        plt.plot(range(len(self.history)), self.history[:,3], label='Generator')
        plt.legend()
        plt.savefig("ACGAN_Loss.png")
        #plt.show()
        plt.clf()

    def save_real_imgs(self, iteration, images):
        r, c = 3, 5
        images = [np.clip(0.5 * img + 0.5, 0, 1) for img in images]
        fig, axs = plt.subplots(r, c, figsize=(20,20))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(images[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/real_%d.png" % iteration)
        plt.close()

    def save_imgs(self, iteration, image_list):
        r, c = 3, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        ls = np.random.randint(0, len(image_list), size=c)
        sampled_labels = np.array([ls for _ in range(r)]).flatten()
        sampled_labels = np.expand_dims(sampled_labels, axis=-1)
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = np.clip(0.5 * gen_imgs + 0.5, 0, 1)
        fig, axs = plt.subplots(r, c, figsize=(20,20))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/generated_%d.png" % iteration)
        plt.close()


if __name__ == '__main__':
    # Load the dataset
    if len(sys.argv) > 2:
        image_list = get_data(sys.argv[1], sys.argv[2], 128)
    else:
        image_list, _ = get_multiclass_data(sys.argv[1], 128)
    dcgan = ACGAN(image_list, 128)
    dcgan.train(iterations=100_000, batch_size=100, save_interval=10_000)
    dcgan.plot_results()
    dcgan.generator.save("ACGAN_Generator.h5")