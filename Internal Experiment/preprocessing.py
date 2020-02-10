
import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def format_input(input_data_path, columns, target_column, class_labels):
    print('Reading input data from {}'.format(input_data_path))
    df = pd.read_csv(input_data_path)
    df = pd.DataFrame(data=df, columns=columns)[df[target_column].isin(class_labels)]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df[target_column].replace(class_labels, [0, 1], inplace=True)
    return df


def print_shape(df):
    negative_examples, positive_examples = np.bincount(df.machine_status)
    print('Data shape: {}, {} positive examples, {} negative examples'.format(df.shape, positive_examples, negative_examples))

def parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    parser.add_argument('--columns', nargs='+', 
                        default=['sensor_00', 'sensor_01', 'sensor_02', 'sensor_03', 'sensor_05', 'sensor_06',
                                 'sensor_07', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_27',
                                 'sensor_28', 'sensor_29', 'sensor_30', 'sensor_31', 'sensor_34', 'sensor_35',
                                 'sensor_36', 'sensor_37', 'sensor_38', 'sensor_39', 'sensor_40', 'sensor_41',
                                 'sensor_42', 'sensor_43', 'sensor_44', 'sensor_45', 'sensor_46', 'sensor_47',
                                 'sensor_48', 'sensor_49', 'sensor_50', 'sensor_51', 'machine_status'])
    parser.add_argument('--target-column', type=string, default="machine_status")
    parser.add_argument('--class-labels', nargs='+', default=["BROKEN", "NORMAL"])
    parser.add_argument('--data-path', type=string, default='/opt/ml/processing/input')
    parser.add_argument('--data-file', type=string, default='sensor.csv')
    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    return args

def build_scaler(X):
    preprocess = StandardScaler()
    preprocess.fit(X)
    return preprocess

def scale(train, test, scaler):
    train, transform = (scaler.transform(train), scaler.transform(test))    
    print(f'Train data shape after preprocessing: {train.shape}')
    print(f'Test data shape after preprocessing: {test.shape}')
    return(train, transform)

def save_set(X, y, path):
    X_path = os.path.join(path, 'features.csv')
    print('Saving training features to {}'.format(X_path))
    pd.DataFrame(X).to_csv(X_path, header=False, index=False)

    y_path = os.path.join(path, 'labels.csv')
    print('Saving training labels to {}'.format(y_path))
    y.to_csv(y_path, header=False, index=False)
    
def save_params(scaler, path):
    mean_path = os.path.join(path, 'means.csv')
    means = scaler.mean_
    print('Saving means to {}'.format(mean_path))
    means.to_csv(mean_path, header=False, index=False)

    stdev_path = os.path.join(path, 'stdev.csv')
    stdevs = scaler.scale_
    print('Saving standard deviations to {}'.format(mean_path))
    stdevs.to_csv(stdev_path, header=False, index=False)
    
if __name__=='__main__':
    args = parse_args()
    columns = args.columns
    class_labels = args.class_labels
    target_column = args.target-column
    input_data_path = os.path.join(args.data_path, data_file = args)
    
    df = format_input(input_data_path, columns, target_column, class_labels)
    print_shape(df)
    
    split_ratio = args.train_test_split_ratio
    print('Splitting data into train and test sets with ratio {}'.format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_column, axis=1), df[target_column], test_size=split_ratio, random_state=0)

    preprocess = build_scaler(X_train)
    print('Running preprocessing and feature engineering transformations')
    train_features, test_features = scale(X_train, X_test, preprocess)
    
    save_set(X_train, y_train, '/opt/ml/processing/train')
    save_set(X_test, y_test, '/opt/ml/processing/train')
    save_params(preprocess)