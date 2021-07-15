import numpy as np
import pandas as pd
from scipy.io import arff

from complex_gan import ComplexGAN

model = ComplexGAN

# Read the original data and have it preprocessed

data = arff.loadarff("diabetes.arff")
data = pd.DataFrame(data[0])

data_cols = list(data.columns[data.columns != 'class'])
label_cols = ['class']

# Define the GAN and training parameters
noise_dim = 32
layer_dim = 512
batch_size = 32

log_step = 100
epochs = 1000 + 1
learning_rate = 1e-5
models_dir = 'weight_cache'

train_sample = data.copy().reset_index(drop=True)
train_sample = pd.get_dummies(train_sample, columns=['class'], prefix='class', drop_first=True)
label_cols = [i for i in train_sample.columns if 'class' in i]
data_cols = [i for i in train_sample.columns if i not in label_cols]


# Training the GAN model chosen: Vanilla GAN, CGAN, DCGAN, etc.
synthesizer = ComplexGAN(batch_size=batch_size, learning_rate=learning_rate, noise_dim=noise_dim, data_shape=train_sample.shape, layers_dim=layer_dim)
synthesizer.train(train_sample, epochs)

gen_model = synthesizer.generator
df_generated_data = pd.DataFrame(synthesizer.generator.predict(np.random.normal(size=(1000, noise_dim))), columns=data_cols + label_cols)
predict_generated_data = synthesizer.discriminator.predict([df_generated_data])
print(f"We faked {sum(predict_generated_data > 0.5)} out of 1000")
