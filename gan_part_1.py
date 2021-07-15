import tensorflow as tf
from scipy.io import arff
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


def load_dataset(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    for dataset_path in ['german_credit.arff', 'diabetes.arff']:
        df = load_dataset(dataset_path)
        nRow, nCol = df.shape
        #print(f'There are {nRow} rows and {nCol} columns in the {dataset_path} dataset')
        #print(df.info())
        print(df.nunique())

