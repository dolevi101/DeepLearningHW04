import numpy as np
import pandas as pd
from scipy.io import arff

from simplegan import VanilllaGAN

model = VanilllaGAN

# Read the original data and have it preprocessed

data = arff.loadarff("diabetes.arff")
data = pd.DataFrame(data[0])

data_cols = list(data.columns[data.columns != 'class'])
label_cols = ['class']

# TODO: preprocess the data
processed_data = data.copy()
train_data = data.copy()
fraud_w_classes = data.copy()

# Define the GAN and training parameters
noise_dim = 32
dim = 128
batch_size = 128

log_step = 100
epochs = 200 + 1
learning_rate = 5e-4
beta_1 = 0.5
beta_2 = 0.9
models_dir = './cache'

train_sample = fraud_w_classes.copy().reset_index(drop=True)
train_sample = pd.get_dummies(train_sample, columns=['class'], prefix='class', drop_first=True)
label_cols = [i for i in train_sample.columns if 'class' in i]
data_cols = [i for i in train_sample.columns if i not in label_cols]
train_sample[data_cols] = train_sample[data_cols] / 10  # scale to random noise size, one less thing to learn
train_no_label = train_sample[data_cols]

gan_args = [batch_size, learning_rate, beta_1, beta_2, noise_dim, train_sample.shape[1], dim]
train_args = ['', epochs, log_step]

# Training the GAN model chosen: Vanilla GAN, CGAN, DCGAN, etc.
synthesizer = model(gan_args)
synthesizer.train(train_sample, train_args)

# Setup parameters visualization parameters
seed = 17
test_size = 492  # number of fraud cases
noise_dim = 32

np.random.seed(seed)
z = np.random.normal(size=(test_size, noise_dim))
real = synthesizer.get_data_batch(train=train_sample, batch_size=test_size, seed=seed)
real_samples = pd.DataFrame(real, columns=data_cols + label_cols)
labels = fraud_w_classes['class']

p = synthesizer.discriminator.predict([real_samples])

print("")
