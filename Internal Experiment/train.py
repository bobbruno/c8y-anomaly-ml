
import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegressionCV

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--columns', nargs='+', 
                        default=['sensor_00', 'sensor_01', 'sensor_02', 'sensor_03', 'sensor_05', 'sensor_06',
                                 'sensor_07', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_27',
                                 'sensor_28', 'sensor_29', 'sensor_30', 'sensor_31', 'sensor_34', 'sensor_35',
                                 'sensor_36', 'sensor_37', 'sensor_38', 'sensor_39', 'sensor_40', 'sensor_41',
                                 'sensor_42', 'sensor_43', 'sensor_44', 'sensor_45', 'sensor_46', 'sensor_47',
                                 'sensor_48', 'sensor_49', 'sensor_50', 'sensor_51', 'machine_status'])
    parser.add_argument('--target-column', type=str, default="machine_status")
    parser.add_argument('--parallelism', type=int, default=os.environ['SM_NUM_CPUS'])
    parser.add_argument('--class-labels', nargs='+', default=["BROKEN", "NORMAL"])
    parser.add_argument('--data-path', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--data-file', type=str, default='sensor.csv')
    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    return args


def format_input(input_data_path, columns, target_column, class_labels):
    print('Reading input data from {}'.format(input_data_path))
    df = pd.read_csv(input_data_path)
    df = pd.DataFrame(data=df, columns=columns)[df[target_column].isin(class_labels)]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df[target_column].replace(class_labels, [0, 1], inplace=True)
    return df


def preprocess(args):
    columns = args.columns
    class_labels = args.class_labels
    target_column = args.target_column
    input_data_path = os.path.join(args.data_path, args.data_file)
    
    df = format_input(input_data_path, columns, target_column, class_labels)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    negative_examples, positive_examples = np.bincount(df.machine_status)
    print('Data shape: {}, {} positive examples, {} negative examples'.format(df.shape, positive_examples, negative_examples))
    return X, y


def train(X, y, args):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.train_test_split_ratio)
    scaler = StandardScaler()
    forestPipeline = Pipeline(
        [('scaler', StandardScaler()), 
         ('isolationforest', 
          IsolationForest(n_estimators=100, max_samples='auto', contamination='auto',
                          behaviour='new', max_features=1.0, bootstrap=False, 
                          n_jobs=args.parallelism))
        ]
    )
    logisticPipeline = Pipeline(
        [('scaler', StandardScaler()),
         ('logistic', 
          LogisticRegressionCV(Cs=10, fit_intercept=True, cv='warn', dual=False,
                               penalty='l2', scoring=None, solver='saga', 
                               tol=0.0001, max_iter=100, n_jobs=args.parallelism,
                               refit=True))
        ]
    )

    print('Starting the Isolation Forest fit...')
    forestPipeline.fit(X_train)
    print('Isolation forest trained')
    
    y_pred_train = forestPipeline.predict(X_train)
    to_forest_domain = np.vectorize(lambda x: -1 if x == 0 else x)
    train_acc_forest = accuracy_score(to_forest_domain(y_train), y_pred_train)
    y_pred_test = forestPipeline.predict(X_test)
    test_acc_forest = accuracy_score(to_forest_domain(y_test), y_pred_test)
    
    print('Starting the logistic fit...')
    logisticPipeline.fit(X_train, y_train)
    print('Logistic regression trained')
    y_pred_train = logisticPipeline.predict(X_train)
    train_acc_lr = accuracy_score(y_train, y_pred_train)
    y_pred_test = logisticPipeline.predict(X_test)
    test_acc_lr = accuracy_score(y_test, y_pred_test)
    print(f"Isolation Forest test Accuracy:    {test_acc_forest:.3f}\n"
          f"Logistic Regression test Accuracy: {test_acc_lr:.3f}"
         )
    model_output_directory = os.environ['SM_MODEL_DIR']
    print('Saving models to {}'.format(model_output_directory))
    dump(forestPipeline, os.path.join(model_output_directory, 'forest.joblib'))
    dump(logisticPipeline, os.path.join(model_output_directory, 'logistic.joblib'))
        

if __name__=='__main__':
    args = parse_args()
    X, y = preprocess(args)
    train(X, y, args)
    