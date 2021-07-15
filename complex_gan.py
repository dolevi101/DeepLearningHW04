import os
import pickle
import time
from os import path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Concatenate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

TEST_SIZE = 0.3
LOSS = 'binary_crossentropy'
RANDOM_FOREST_PARAMS = {'n_estimators': 200,
                        'min_samples_split': 2,
                        'min_samples_leaf': 4,
                        'max_features': 'sqrt',
                        'max_depth': None,
                        'bootstrap': True}


class ComplexGAN:
    """
    A Simple Gan Implementation
    """

    def __init__(self, batch_size, learning_rate, noise_dim, data_shape, layers_dim):
        self.learning_rate = learning_rate
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.data_x_shape = data_shape[0]
        self.data_y_shape = data_shape[1]
        self.layers_dim = layers_dim

        self.optimizer = Adam(self.learning_rate)

        self.generator = Generator(self.batch_size).build_model(input_shape=(self.noise_dim,),
                                                                dim=self.layers_dim,
                                                                data_dim=self.data_y_shape)
        self.generator.compile(loss=LOSS,
                               optimizer=self.optimizer,
                               metrics=['accuracy'])

        self.discriminator = Discriminator(self.batch_size).build_model(input_shape=(self.data_y_shape,),
                                                                        dim=self.layers_dim)
        self.discriminator.compile(loss=LOSS,
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

    def train(self, data, epochs, save_dir=None):
        """
        Train the GAN
        """
        tic = time.perf_counter()

        loss_df = pd.DataFrame(columns=['epoch', 'disc_loss', 'disc_metric', 'gen_loss', 'gen_metric'])
        self._train_black_box_model(data)

        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):
            batch_data = np.reshape(data.sample(n=self.batch_size).values, (self.batch_size, -1))
            noise = tf.random.normal((self.batch_size, self.noise_dim))
            c_gen = np.random.uniform(0, 1, (self.batch_size, 1))
            fake_c_gen = np.random.uniform(0, 1, (self.batch_size, 1))

            # Generate data
            generated_data = self.generator((noise, c_gen), training=True)
            X_generated_data = generated_data[:, 0:generated_data.shape[1] - 1]

            c_bb = self.black_box_model.predict(X_generated_data)

            # Train discriminator
            dis_loss_real_data = self.discriminator.train_on_batch((batch_data, fake_c_gen, valid), valid)
            dis_loss_fake_data = self.discriminator.train_on_batch((batch_data, c_gen, c_bb), fake)
            dis_loss = np.add(dis_loss_fake_data, dis_loss_real_data) * 0.5

            # Train generator
            noise = tf.random.normal((self.batch_size, self.noise_dim))
            c_gen = np.random.uniform(0, 1, (self.batch_size, 1))
            input_data = [noise, c_gen]
            generator_loss = self.generator.train_on_batch(input_data, valid)

            print(f'Epoch: {epoch}  [Disc loss: {format(dis_loss[0], ".3f")}, acc: {format(dis_loss[1], ".3f")}]  '
                  f'[Gen loss: {format(generator_loss[0], ".3f")}]')

            loss_df = loss_df.append({'epoch': epoch,
                                      'disc_loss': dis_loss[0],
                                      'disc_metric': dis_loss[1],
                                      'gen_loss': generator_loss[0],
                                      'gen_metric': generator_loss[1]}, ignore_index=True)

        toc = time.perf_counter()
        print(f'run time (seconds): {toc-tic}')

        if not path.exists(save_dir):
            os.mkdir(save_dir)

        h5_name = f'./{save_dir}/' + '_{}_model_weights.h5'
        self.generator.save_weights(h5_name.format('generator'))
        self.discriminator.save_weights(h5_name.format('discriminator'))
        self.train_black_box.to_csv(f'./{save_dir}/train_data_black_box.csv')
        self.test_black_box.to_csv(f'./{save_dir}/test_data_black_box.csv')
        loss_df.to_csv(f'./{save_dir}/loss.csv')

        with open(f'./{save_dir}/black_box_RF_model.model', 'wb') as f:
            pickle.dump(self.black_box_model, f)

    def _train_black_box_model(self, data: pd.DataFrame):
        self.black_box_model = RandomForestClassifier(random_state=1, **RANDOM_FOREST_PARAMS)

        self.train_black_box, self.test_black_box = train_test_split(data, test_size=TEST_SIZE, random_state=1)

        self.X_train_black_box = self.train_black_box.drop(data.columns[-1], axis=1)
        self.y_train_black_box = self.train_black_box[data.columns[-1]]

        self.X_test_black_box = self.test_black_box.drop(data.columns[-1], axis=1)
        self.y_test_black_box = self.test_black_box[data.columns[-1]]

        self.black_box_model.fit(self.X_train_black_box, self.y_train_black_box)


class Generator(tf.keras.Model):
    """
    The Generator Model Class
    """

    def __init__(self, data_size=None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_size = data_size

    def build_model(self, input_shape, dim, data_dim):
        input_1 = Input(shape=input_shape, batch_size=self.data_size)
        input_2 = Input(shape=1, batch_size=self.data_size)
        x = Dense(dim, activation='relu')(input_1)
        x = Dropout(0.2)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(dim * 4, activation='relu')(x)
        concat = Concatenate()([x, input_2])
        x = Dense(data_dim)(concat)
        return tf.keras.Model(inputs=[input_1, input_2], outputs=x)


class Discriminator(tf.keras.Model):
    """
    The Discriminator Model Class
    """

    def __init__(self, data_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_size = data_size

    def build_model(self, input_shape, dim):
        input = Input(shape=input_shape, batch_size=self.data_size)
        input_gen_c = Input(shape=1, batch_size=self.data_size)
        input_bb_c = Input(shape=1, batch_size=self.data_size)
        x = Dense(dim * 4, activation='relu')(input)
        x = Dropout(0.2)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(dim, activation='relu')(x)
        concat_layer = Concatenate()([x, input_gen_c, input_bb_c])
        x = Dense(1, activation='sigmoid')(concat_layer)
        return tf.keras.Model(inputs=[input, input_gen_c, input_bb_c], outputs=x)
