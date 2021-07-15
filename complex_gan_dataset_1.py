import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import PowerTransformer

from complex_gan import ComplexGAN

data = arff.loadarff("diabetes.arff")
data = pd.DataFrame(data[0])

data = data.copy().reset_index(drop=True)
data = pd.get_dummies(data)
scaled_data = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit_transform(data)

# Parameters
noise_dim = 32
layer_dim = 512
batch_size = 32
log_step = 100
epochs = 200 + 1
learning_rate = 1e-5
models_dir = 'weight_cache'

# Training the GAN
complex_gan = ComplexGAN(batch_size=batch_size, learning_rate=learning_rate, noise_dim=noise_dim, data_shape=data.shape,
                         layers_dim=layer_dim)
complex_gan.train(data, epochs)

# Predicting
gen_model = complex_gan.generator
noise = np.random.normal(size=(100, noise_dim))
c_gen = np.random.uniform(0, 1, size=(100, 1))

df_generated_data = pd.DataFrame(complex_gan.generator.predict((noise, c_gen)), columns=data.columns)
bb_probabilities = complex_gan.black_box_model.predict_proba(
    df_generated_data.iloc[:, 0:df_generated_data.shape[1] - 1])[:, 1]
predict_generated_data = complex_gan.discriminator.predict([df_generated_data, c_gen, bb_probabilities])
print(f"We faked {sum(predict_generated_data > 0.5)} out of 100")
