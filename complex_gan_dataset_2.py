import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import PowerTransformer

from complex_gan import ComplexGAN

data = arff.loadarff("german_credit.arff")
data = pd.DataFrame(data[0])

scaled_data = data.copy().reset_index(drop=True)
scaled_data = pd.get_dummies(scaled_data, prefix_sep='class_col', drop_first=True)
class_col = [col for col in scaled_data.columns if 'class_col' in col]
num_col = [col for col in scaled_data.columns if col not in class_col]
binary_col = [col for col in num_col if len(scaled_data[col].unique()) == 2]

# Remove all the binary columns
[num_col.remove(col) for col in binary_col]

data_transformer = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
scaled_data[num_col] = data_transformer.fit_transform(scaled_data[num_col])
scaled_data = pd.get_dummies(scaled_data, prefix="binary", columns=binary_col, drop_first=True)

# Parameters
noise_dim = 64
layer_dim = 512
batch_size = 64
epochs = 1 + 200
learning_rate = 1e-5
save_dir = 'weight_cache_complex_dataset2'

# Training the GAN
complex_gan = ComplexGAN(batch_size=batch_size,
                         learning_rate=learning_rate,
                         noise_dim=noise_dim,
                         data_shape=scaled_data.shape,
                         layers_dim=layer_dim)
complex_gan.train(scaled_data, epochs, save_dir)

# Predicting
## A
black_box_test_results = complex_gan.black_box_model.predict_proba(complex_gan.X_test_black_box)[:, 1]
np.save(f'./{save_dir}/black_box_test_results.npy', np.array(black_box_test_results))

## B+C
gen_model = complex_gan.generator
noise = np.random.normal(size=(1000, noise_dim))
c_gen = np.random.uniform(0, 1, size=(1000, 1))
df_generated_data = pd.DataFrame(complex_gan.generator.predict((noise, c_gen)), columns=scaled_data.columns)
bb_probabilities = complex_gan.black_box_model.predict_proba(
    df_generated_data.iloc[:, 0:df_generated_data.shape[1] - 1])[:, 1]

df_generated_data.to_csv(f'./{save_dir}/df_generated_data_for_B-C.csv')
np.save(f'./{save_dir}/black_box_generated_data_results.npy', np.array(bb_probabilities))

## D
predict_generated_data = complex_gan.discriminator.predict([df_generated_data, c_gen, bb_probabilities])
np.save(f'./{save_dir}/discriminator_generated_data_results.npy', np.array(predict_generated_data))

print(f"We faked {sum(predict_generated_data > 0.5)} out of 100")
df_generated_data[num_col] = data_transformer.inverse_transform(df_generated_data[num_col])
print(df_generated_data.head(20))
