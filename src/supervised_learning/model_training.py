#%%
from create_dataset import load_dataset
from plots import plot_learning_curve

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import joblib

import scipy

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

def main():
    GEN_TEST_SPLIT = True
    dataset_path = "./datasets/xing_dataset"
    np_dataset_name = "np_dataset_no_missing_ip_bpms.npy"

    #dataset = load_dataset(dataset_path, np_dataset_name)
    dataset = np.load(dataset_path +"/" + np_dataset_name, allow_pickle=True)   

    algorithm = "ridge"
    model_name = "4ksmpl_no_miss_ip_xing"

    metrics, n_samples = [], []
    n_splits = 1

    print(len(dataset[1][1]))
    
    X = np.array([sample[0] for i,sample in enumerate(dataset)])
    y = np.array([sample[1] for i,sample in enumerate(dataset)])

    #X = np.array([sample for i,sample in enumerate(dataset) if i % 2 == 0])
    #y = np.array([sample for i,sample in enumerate(dataset) if i % 2 == 0])
    #print(X[:2])
    #print(len(X))
    #print(X)
    #print(X[0])

    ceros = []
    for i, x in enumerate(X):
        #if len(x)!=6016:
        if len(x)!=8992:
            print(len(x))
            print(i)
            ceros.append(i)

    dataset = np.delete(dataset, ceros, axis=0)
    print(dataset.shape)

    X = np.array([sample[0] for i,sample in enumerate(dataset)])
    y = np.array([sample[1] for i,sample in enumerate(dataset)])
    #np.save("./htcondor_dataset/np_dataset_10.npy", dataset)

    for i in range(n_splits):
        split_n_samples = int((i+1)*len(X)/n_splits)
        results = train_model(np.vstack(X[:split_n_samples]), np.vstack(y[:split_n_samples]), algorithm=algorithm, GEN_TEST_SPLIT=GEN_TEST_SPLIT, model_name=model_name)

        n_samples.append(split_n_samples)
        metrics.append(results)
            
    #print(metrics)
    plot_learning_curve(n_samples, metrics, algorithm)

def train_model(input_data, output_data, algorithm, GEN_TEST_SPLIT, model_name):
    
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
    
    if algorithm == "ridge":
        ridge = linear_model.Ridge(tol=1e-50, alpha=1e-3) #normalize=false
        estimator = BaggingRegressor(estimator=ridge, n_estimators=10, \

            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)

    elif algorithm == "linear":
        estimator = linear_model.LinearRegression()
        #estimator = BaggingRegressor(estimator=linear, n_estimators=10, \
        #    max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)    
        
    elif algorithm == "tree":
        tree = DecisionTreeRegressor(criterion="squared_error")
        estimator = BaggingRegressor(estimator=tree, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)
        
    elif algorithm.split('-')[0] == "nn":
        train_inputs =scipy.stats.zscore(train_inputs[:100], axis=0, ddof=0)
        train_outputs =train_outputs[:100]
        test_inputs =scipy.stats.zscore(test_inputs, axis=0, ddof=0)
        #test_outputs =scipy.stats.zscore(test_outputs, axis=1)

        with tf.device('/GPU:0'):
            print("Available Devices: ", tf.config.list_physical_devices())
            input_shape = tf.convert_to_tensor(tuple([len(train_inputs[0]), 1]))
            output_dim = len(train_outputs[0])

            estimator = create_compile_lstm_model(input_shape, output_dim)

            history = estimator.fit(x=train_inputs, y=train_outputs,
                        validation_data=(test_inputs, test_outputs),
                                    epochs=500, batch_size=16)

            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            plt.plot(train_loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            plt.savefig(f"./figures/lc_nn.pdf")

        return
    
    joblib.dump(estimator, f"./models/{model_name}.joblib")

    y_true_train, y_pred_train = train_outputs, estimator.predict(train_inputs) 
    y_true_test, y_pred_test = test_outputs, estimator.predict(test_inputs) 

    for error_order in ["K2L", "K2SL", "K3L", "K3SL", "ALL"]:
        r2_train, mae_train, r2_test, mae_test = calculate_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test, error_order)
        print(f"\n  {error_order}")
        print("Train: R2 = {0}, MAE = {1}".format(r2_train, mae_train))
        print("Test: R2 = {0}, MAE = {1}".format(r2_test, mae_test))

    return r2_train, mae_train, r2_test, mae_test

def calculate_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test, magnet_type):
    if magnet_type=="ALL":
        r2_train = r2_score(y_true_train, y_pred_train)
        r2_test = r2_score(y_true_test, y_pred_test)

        mae_train = mean_absolute_error(y_true_train, y_pred_train)
        mae_test = mean_absolute_error(y_true_test, y_pred_test)

    else:
        if magnet_type=="K2L":
            ini = 0
            fin = 16
        elif magnet_type=="K2SL":
            ini = 16
            fin = 32
        elif magnet_type=="K3L":
            ini = 32
            fin = 48
        elif magnet_type=="K3SL":
            ini = 48
            fin = 64

        r2_train = r2_score(y_true_train[:,ini:fin], y_pred_train[:,ini:fin])
        r2_test = r2_score(y_true_test[:,ini:fin], y_pred_test[:,ini:fin])

        mae_train = mean_absolute_error(y_true_train[:,ini:fin], y_pred_train[:,ini:fin])
        mae_test = mean_absolute_error(y_true_test[:,ini:fin], y_pred_test[:,ini:fin])

    return r2_train, mae_train, r2_test, mae_test

def create_compile_nn_model(input_shape, output_dim):
    print(input_shape)
    print(output_dim)
    estimator =tf.keras.models.Sequential([tf.keras.layers.Reshape((-1, 1126, 2), input_shape=(2252,)),
                                           tf.keras.layers.Resizing(60,60),
                                           tf.keras.applications.ResNet50(
                                               #weights='imagenet',
                                               include_top=False,
                                               input_shape=(60, 60, 3),
                                               pooling=None),
                                           tf.keras.layers.Flatten(),
                                           tf.keras.layers.Dense(
                                               units=output_dim,
                                               activation=None,
                                               use_bias=True,
                                               kernel_initializer='glorot_uniform', bias_initializer='zeros')
                                           ])

    initial_learning_rate = 1E-4
    final_learning_rate = 1E-6
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/100)
    steps_per_epoch = 3500#87                                                                                                            

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,                # Base learning rate.                                               
        decay_steps=steps_per_epoch,                                # Decay step.                                                                             
        decay_rate=learning_rate_decay_factor,                      # Decay rate.                                                                    
        staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    estimator.compile(optimizer=opt, loss='mae', metrics=['mae', r2_score])

    return estimator

def create_compile_lstm_model(input_shape, output_dim):
    estimator = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=128, activation='tanh',
                             input_shape=input_shape,
                             kernel_initializer='glorot_normal',
                             recurrent_initializer='glorot_normal',
                             bias_initializer='zeros'),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=output_dim,
                              kernel_initializer='glorot_normal')
    ])
    """
    initial_learning_rate = 1E-4
    final_learning_rate = 1E-5
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/100)
    steps_per_epoch = 3500#87                                                                                                            

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,                # Base learning rate.                                               
        decay_steps=steps_per_epoch,          # Decay step.
        decay_rate=learning_rate_decay_factor,         # Decay rate.                                                                    
        staircase=True)
    """
    learning_rate=1e-4

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    estimator.compile(optimizer=opt, loss='mse', metrics=['mae', r2_score])

    return estimator

def r2_score(y_true, y_pred):
    total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    r2 = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
    return r2

if __name__ == "__main__":
    main()
# %%
