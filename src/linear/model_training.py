#%%

import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import joblib

from plots import plot_noise_vs_metrics
from plots import plot_learning_curve

from data_utils import load_data
from data_utils import merge_data
from data_utils import save_np_errors_tfs

import random
import matplotlib.pyplot as plt

import scipy.stats

import tensorflow as tf

tf.random.set_seed(1111)
np.random.seed(1111)
random.seed(1111)

""" ------------------------------------------------------- 0 --------------------------------------------------
Script for training and validating models
----------------------------------------------------------- 0 --------------------------------------------------  """

GEN_TEST_SPLIT = True # If a new test split is needed

def main():
    data_path = "data/"
    set_name = "100%triplet_ip1_100%ip5_2592"
    TRAIN = True
    MERGE = True
    algorithm = "ridge"

    # Train on generated data
    # Load data
    metrics, n_samples = [], []
    noises = [1e-2] # np.logspace(-5, -2, num=10)
    
    # In order to make a robust model noise should be at least be 1e-2
    
    for noise in noises:

        if MERGE == True:
            input_data, output_data = merge_data(data_path, noise)

            print(input_data.shape)
        else:
            input_data, output_data = load_data(set_name, noise)

            #plt.hist(output_data[:,-1])
            #plt.hist(output_data[:,-2])
            #plt.hist(output_data[:,-3])
            #plt.hist(output_data[:,-4])

        if TRAIN==True:
            n_splits=1
            input_data = np.array_split(input_data, n_splits, axis=0)
            output_data = np.array_split(output_data, n_splits, axis=0)

            for i in range(n_splits):
                results = train_model(np.vstack(input_data[:i+1]), np.vstack(output_data[:i+1]),
                                        algorithm=algorithm, noise=noise)
                n_samples.append(len(np.vstack(input_data[:i+1])))
                metrics.append(results)

            plot_learning_curve(n_samples, metrics, algorithm)

    #plot_noise_vs_metrics(noises, metrics, algorithm)

def train_model(input_data, output_data, algorithm, noise):  

    #Function that loads the data and trains the chosen model with a given noise level
    indices = np.arange(len(input_data))

    # Generating new test split or loading old one
    if GEN_TEST_SPLIT==True:
        train_inputs, test_inputs, train_outputs, test_outputs, indices_train, indices_test = train_test_split(
            input_data, output_data, indices, test_size=0.2, random_state=None)
        np.save("./data_analysis/test_idx.npy", indices_test)
        np.save("./data_analysis/train_idx.npy", indices_train)
    else:
        indices_test = np.load("./data_analysis/test_idx.npy")
        indices_train = np.load("./data_analysis/train_idx.npy")
        train_inputs, test_inputs, train_outputs, test_outputs = input_data[indices_train], input_data[indices_test],\
                                                                output_data[indices_train], output_data[indices_test]
    
    # create and fit a regression model
    if algorithm == "ridge":
        ridge = linear_model.Ridge(tol=1e-50, alpha=5e-4) #normalize=false
        estimator = BaggingRegressor(estimator=ridge, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=1, verbose=0)
        estimator.fit(train_inputs, train_outputs)

    elif algorithm == "linear":
        linear = linear_model.LinearRegression()
        estimator = BaggingRegressor(estimator=linear, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)    
        
    elif algorithm == "tree":
        tree = DecisionTreeRegressor(criterion="squared_error", max_depth=10) #5
        estimator = tree
        estimator.fit(train_inputs, train_outputs)

    # Optionally: save fitted model or load already trained model        
    joblib.dump(estimator, f'./estimators/triplet_phases_only_028_{algorithm}_{noise}.pkl')          

    # Check scores: explained variance and MAE

    y_true_train, y_pred_train = train_outputs, estimator.predict(train_inputs) 
    y_true_test, y_pred_test = test_outputs, estimator.predict(test_inputs) 

    r2_train = r2_score(y_true_train, y_pred_train)
    r2_test = r2_score(y_true_test, y_pred_test)

    mae_train = mean_absolute_error(y_true_train, y_pred_train)
    mae_test = mean_absolute_error(y_true_test, y_pred_test)

    r2_train_triplet = r2_score(y_true_train[:,:32], y_pred_train[:,:32])
    r2_test_triplet = r2_score(y_true_test[:,:32], y_pred_test[:,:32])

    mae_train_triplet = mean_absolute_error(y_true_train[:,:32], y_pred_train[:,:32])
    mae_test_triplet = mean_absolute_error(y_true_test[:,:32], y_pred_test[:,:32])

    print("Train Triplet: R2 = {0}, MAE = {1}".format(r2_train_triplet, mae_train_triplet))
    print("Test Triplet: R2 = {0}, MAE = {1}".format(r2_test_triplet, mae_test_triplet))

    print("Train: R2 = {0}, MAE = {1}".format(r2_train, mae_train))
    print("Test: R2 = {0}, MAE = {1}".format(r2_test, mae_test))

    return r2_train, mae_train, r2_test, mae_test, r2_test_triplet, mae_test_triplet
                                                                                                        

def r2_score(y_true, y_pred):
    # Custom R2 score, needed custom function to test some nn
    total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    r2 = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
    return r2

if __name__ == "__main__":
    main()

# %%                         
