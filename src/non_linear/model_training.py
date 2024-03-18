#%%
from create_dataset import load_dataset

from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import joblib
import itertools


#import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

""" ------------------------------------------------------- 0 --------------------------------------------------
Script for training models. Model selection, and validation.

----------------------------------------------------------- 0 --------------------------------------------------  """


def main():
    # Dataset parameters
    GEN_TEST_SPLIT = True
    PREPROCESSING = False # If preprocessing is false it loads the .npy dataset
    XING = False
    noise = 0
    n_samples = 30000 # Number of samples loaded in the .npy file
    dataset_path = "./datasets/tests/dataset_test"

    if XING==True:
        xing_name="_xing"
    elif XING==False:
        xing_name=""
        
    np_dataset_name = f"{str(n_samples)[:2]}K_1030{xing_name}_{noise}"

    # Model hyperparameters
    algorithm = "poly_ridge"
    model_name = f"{algorithm}_{str(n_samples)[:2]}K_1030_n10{xing_name}_{str(noise)}"

    n_splits = 3 # Number of splits to train for the dataset size vs performance
    alphas = [1e-5] # Ridge regularization parameter
    n_bpms = 376 # Number of BPMs used 376 for all arcs

    # Loading data
    if PREPROCESSING == True:
        # List of all saved RDTS
        # Horizontal
        hor_rdt_list = ["RE_300000",
                    "IM_300000",
                    "RE_400000",
                    "IM_400000",
                    "RE_201000",
                    "IM_201000",
                    "RE_103000",
                    "IM_103000"]

        # Vertical
        vert_rdt_list = ["RE_102000",
                    "IM_102000",
                    "RE_013000",
                    "IM_013000",  
                    "RE_003000",
                    "IM_003000",
                    "RE_022000",
                    "IM_022000"] 
        
        # Load new dataset, with noise, and rdts choosen
        dataset = load_dataset(dataset_path,
                               hor_rdt_list=hor_rdt_list,
                               vert_rdt_list=vert_rdt_list, 
                               noise_level=noise, 
                               np_dataset_name=np_dataset_name,
                               n_samples=n_samples)
    
    elif PREPROCESSING == False:
        dataset = np.load(dataset_path + "/" + np_dataset_name + ".npy", allow_pickle=True)   

    X = np.array([sample[0] for i, sample in enumerate(dataset)])
    y = np.array([sample[1] for i, sample in enumerate(dataset)])

    print("Number of samples: ", dataset.shape[0])

    # Normalizing inputs and labels since each error has its own distribution
    X, y = normalize_data(X, y)

    # Generate second-order polynomial combinations
    if algorithm == "poly_ridge":
        X = generate_second_order_polynomials(X, n_bpms)

    # Train for different subsets of data
    metrics, n_samples = [], []

    for alpha in alphas: # Parameter scan for regularization parameter
        X_aux = X
        for i in range(n_splits)[2:]:    

            split_n_samples = int((i+1)*len(X_aux)/n_splits)
            results = train_model(np.vstack(X_aux[:split_n_samples]), np.vstack(y[:split_n_samples]),
                                algorithm=algorithm, 
                                GEN_TEST_SPLIT=GEN_TEST_SPLIT, 
                                model_name=model_name,
                                alpha=alpha)

            n_samples.append(split_n_samples)
            metrics.append(results)

    # Plotting dataset size VS performance
    plt.plot(n_samples, np.array(metrics)[:,0], label='Train', marker='o', linestyle='-')
    plt.plot(n_samples, np.array(metrics)[:,2], label='Test', marker='o', linestyle='-')

    plt.xlabel(r'Number of samples: $n$', fontsize=14)
    plt.ylabel(r'$R^2$', fontsize=14)
    plt.title('Performance VS Dataset Size', fontsize=16)
    plt.legend()
    
    plt.savefig(f"./figures/lc_r2.pdf")

def train_model(input_data, output_data, algorithm, GEN_TEST_SPLIT, model_name, alpha):
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
        ridge = linear_model.Ridge(tol=1e-50, alpha=alpha)
        estimator = BaggingRegressor(estimator=ridge, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)
    
    elif algorithm == "poly_ridge":
        estimator = linear_model.Ridge(tol=1e-50, alpha=alpha)
        estimator = BaggingRegressor(estimator=estimator, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=1, verbose=0)
        # Reducing number of jobs since it loads them in memory and demands too much
        # because of polynomial features

        estimator.fit(train_inputs, train_outputs)

    elif algorithm == "krr":
        krr = kernel_ridge.KernelRidge(alpha=alpha, kernel='polynomial', degree=2)
        estimator = BaggingRegressor(estimator=krr, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)

    elif algorithm == "linear":
        estimator = linear_model.LinearRegression()
        #estimator = BaggingRegressor(estimator=linear, n_estimators=10, \
        #    max_samples = 0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)    
        
    elif algorithm == "tree":
        tree = DecisionTreeRegressor(criterion="squared_error", max_depth=15)
        estimator = BaggingRegressor(estimator=tree, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)
    
    joblib.dump(estimator, f"./models/{model_name}.joblib") 
    
    y_true_train, y_pred_train = train_outputs, estimator.predict(train_inputs) 
    y_true_test, y_pred_test = test_outputs, estimator.predict(test_inputs)

    performance_metrics = []

    for error_order in ["K2L", "K2SL", "K3L", "K3SL", "ALL"]:
        r2_train, r2_test, mae_train, mae_test, adjusted_r2_train, adjusted_r2_test = calculate_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test, error_order)
        
        performance_metrics.append([r2_train, r2_test])

        print(f"\n  {error_order}")
        print(f"Train: R2 = {r2_train:.6g}, MAE = {mae_train:.6g}, CORR R2 = {adjusted_r2_train:.6g}")
        print(f"Test: R2 = {r2_test:.6g}, MAE = {mae_test:.6g}, CORR R2 = {adjusted_r2_test:.6g}")

    return r2_train, mae_train, r2_test, mae_test

def calculate_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test, magnet_type):
    if magnet_type=="ALL":
        ini = 0
        fin = -1

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

    adjusted_r2_train = adjusted_r2_score(y_true_train[:,ini:fin], y_pred_train[:,ini:fin])
    adjusted_r2_test = adjusted_r2_score(y_true_test[:,ini:fin], y_pred_test[:,ini:fin])

    return r2_train, r2_test, mae_train, mae_test, adjusted_r2_train, adjusted_r2_test

'''
def r2_score(y_true, y_pred):
    # Done in TF so that it can be used while training a nn

    total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    r2 = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
    return r2
'''


def adjusted_r2_score(y_true, y_pred):
    # Take the 4 different magnets left and right of the IPs and group them, calculate the
    # correction with those and calculate the R2 with this measurement

    true_corrections, pred_corrections = [], []

    for true_data_point, pred_data_point in zip(y_true, y_pred):
        true_avg, pred_avg = [], []
        for i in range(0, len(true_data_point), 4):        
            subset = true_data_point[i:i + 4]
            true_avg.append(sum(subset))

            subset = pred_data_point[i:i + 4]
            pred_avg.append(sum(subset))

        true_corrections.append(true_avg)
        pred_corrections.append(pred_avg)
    '''    
    total_error = tf.reduce_sum(tf.square(tf.subtract(true_corrections, tf.reduce_mean(true_corrections))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(true_corrections, pred_corrections)))
    r2 = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
    '''
    return r2_score(true_corrections, pred_corrections)

def normalize_data(X, y):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    np.save('./data_analysis/rdt_mean.npy', mean)
    np.save('./data_analysis/rdt_std.npy', std)
    X = (X - mean) / std

    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)
    np.save('./data_analysis/err_mean.npy', mean)
    np.save('./data_analysis/err_std.npy', std)
    y = (y - mean) / std

    return X, y

def generate_second_order_polynomials(data, section_length):
    # Generating the cross terms that have physical meaning, it can take a lot of RAM ideally reducing RDTs
    num_features = int(data.shape[1]/2) # Num features per beam
    num_sections = num_features // section_length

    # Generate all possible combinations of indices for second-order polynomials
    polynomial_indices =  list(set(tuple(sorted(comb)) for comb in itertools.product(range(num_sections), repeat=2)))
    #polynomial_indices = [comb for comb in polynomial_indices if len(set(comb)) == 1] # To take out cross terms

    aux_data = []

    for i, sample in enumerate(data):            
        # Initialize an auxiliary array to store polynomial combinations
        polynomial_features = [[], []]

        for n, bn in enumerate(('b1', 'b2')):

            # Taking one of the beams and generating polynomial features
            sample_bn = sample.reshape(-1, int(len(sample)/2))[n]

            sliced_sample = sample_bn.reshape(-1, section_length)

            # Calculate and append second-order polynomial combinations
            for index1, index2 in polynomial_indices:
                polynomial_feature = sliced_sample[index1] * sliced_sample[index2]
                polynomial_features[n].extend(polynomial_feature)
            
        # Adding the two beams polynomial features
        aux_data.append(np.concatenate((sample, polynomial_features[0], polynomial_features[1])))

    return np.array(aux_data)

if __name__ == "__main__":
    main()
# %%
