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

# example of reading the data, training ML model and validate results

GEN_TEST_SPLIT = True # If a new test split is needed

def main():
    set_name = "data"
    TRAIN = True
    MERGE = True
    algorithm = "tree"

    # Train on generated data
    # Load data

    metrics, n_samples = [], []
    noises = [1E-3] #np.logspace(-5, -2, num=10)
    for noise in noises:

        if MERGE == True:
            input_data, output_data = merge_data(set_name, noise)
        else:
            input_data, output_data = load_data(set_name, noise)

        if TRAIN==True:

            n_splits=1
            input_data = np.array_split(input_data, n_splits, axis=0)
            output_data = np.array_split(output_data, n_splits, axis=0)

            for i in range(n_splits):
                results = train_model(np.vstack(input_data[:i+1]), np.vstack(output_data[:i+1]),
                                        algorithm=algorithm, noise=noise)
                n_samples.append(len(np.vstack(input_data[:i+1])))
                metrics.append(results)


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
        ridge = linear_model.Ridge(tol=1e-50, alpha=1e-3) #normalize=false
        estimator = BaggingRegressor(estimator=ridge, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)

    elif algorithm == "linear":
        linear = linear_model.LinearRegression()
        estimator = BaggingRegressor(estimator=linear, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)    
        
    elif algorithm == "tree":
        tree = DecisionTreeRegressor(criterion="squared_error", max_depth=100)
        estimator = tree
        estimator.fit(train_inputs, train_outputs)

    elif algorithm.split('-')[0] == "nn":
        train_inputs =scipy.stats.zscore(train_inputs[:10], axis=0, ddof=0)
        train_outputs =train_outputs[:10]
        test_inputs =scipy.stats.zscore(test_inputs, axis=0, ddof=0)
        #test_outputs =scipy.stats.zscore(test_outputs, axis=1)

        with tf.device('/GPU:0'):
            print("Available Devices: ", tf.config.list_physical_devices())
            input_shape = tf.convert_to_tensor(tuple([len(train_inputs[0]), 1]))
            output_dim = len(train_outputs[0])

            if algorithm.split('-')[1] == "lstm":
                estimator = create_compile_lstm_model(input_shape, output_dim)
            if algorithm.split('-')[1] == "cnn":
                estimator = create_compile_cnn_model(input_shape, output_dim)

            history = estimator.fit(x=train_inputs, y=train_outputs,
                        validation_data=(test_inputs, test_outputs),
                                    epochs=1500, batch_size=16)

            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            plt.plot(train_loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            plt.savefig(f"./figures/lc_nn.pdf")

        return 0

    # Optionally: save fitted model or load already trained model                                                                       
    joblib.dump(estimator, f'./estimators/estimator_{algorithm}_{noise}.pkl')


    # Check scores: explained variance and MAE
    r2_train = estimator.score(train_inputs, train_outputs)
    r2_test = estimator.score(test_inputs, test_outputs)

    r2_train_triplet = estimator.score(train_inputs[:32], train_outputs[:32])
    r2_test_triplet = estimator.score(test_inputs[:32], test_outputs[:32])
    
    prediction_train = estimator.predict(train_inputs)
    mae_train = mean_absolute_error(train_outputs, prediction_train)

    prediction_test = estimator.predict(test_inputs)
    mae_test = mean_absolute_error(test_outputs, prediction_test)

    prediction_train = estimator.predict(train_inputs[:32])
    mae_train_triplet = mean_absolute_error(train_outputs[:32], prediction_train)

    prediction_test = estimator.predict(test_inputs[:32])
    mae_test_triplet = mean_absolute_error(test_outputs[:32], prediction_test)

    print("Train Triplet: R2 = {0}, MAE = {1}".format(r2_train_triplet, mae_train_triplet))
    print("Test Triplet: R2 = {0}, MAE = {1}".format(r2_test_triplet, mae_test_triplet))

        
    print("Train: R2 = {0}, MAE = {1}".format(r2_train, mae_train))
    print("Test: R2 = {0}, MAE = {1}".format(r2_test, mae_test))

    return r2_train, mae_train, r2_test, mae_test, r2_test_triplet, mae_test_triplet

def create_compile_lstm_model(input_shape, output_dim):
    estimator = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=128, activation='tanh',
                             input_shape=input_shape,
                             kernel_initializer='glorot_normal',
                             recurrent_initializer='glorot_normal',
                             bias_initializer='zeros'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=output_dim,
                              kernel_initializer='glorot_normal')
    ])
    initial_learning_rate = 1E-4
    final_learning_rate = 1E-5
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/100)
    steps_per_epoch = 3500#87                                                                                                            

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,                # Base learning rate.                                               
        decay_steps=steps_per_epoch,          # Decay step.
        decay_rate=learning_rate_decay_factor,         # Decay rate.                                                                    
        staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    estimator.compile(optimizer=opt, loss='mse', metrics=['mae', r2_score])

    return estimator

def create_compile_cnn_model(input_shape, output_dim):
    estimator =tf.keras.models.Sequential([tf.keras.layers.Reshape((-1, 2, 3), input_shape=(3342,)),
                                           tf.keras.layers.Resizing(60,60),
                                           tf.keras.applications.ResNet50(
                                               weights='imagenet',
                                               include_top=False,
                                               input_shape=(60, 60, 3),
                                               pooling=None),
                                           tf.keras.layers.Flatten(),
                                           tf.keras.layers.Dense(
                                               units=output_dim,
                                               activation=None,
                                               use_bias=True,
                                               kernel_initializer='glorot_uniform',bias_initializer='zeros')
                                           ])

    initial_learning_rate = 1E-4
    final_learning_rate = 1E-6
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/100)
    steps_per_epoch = 3500#87                                                                                                            

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,                # Base learning rate.                                               
        decay_steps=steps_per_epoch,          # Decay step.                                                                             
        decay_rate=learning_rate_decay_factor,         # Decay rate.                                                                    
        staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    estimator.compile(optimizer=opt, loss='mae', metrics=['mae', r2_score])

    return estimator

def r2_score(y_true, y_pred):
    total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    r2 = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
    return r2

if __name__ == "__main__":
    main()

# %%                         
