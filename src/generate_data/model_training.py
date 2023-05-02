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


# example of reading the data, training ML model and validate results

GEN_TEST_SPLIT = True # If a new test split is needed

def main():
    set_name = "data"
    TRAIN = True
    MERGE = True
    algorithm = "ridge"

    # Train on generated data
    # Load data

    metrics, n_samples = [], []
    noises = np.logspace(-5, -2, num=10)
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

    plot_noise_vs_metrics(noises, metrics, algorithm)

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
        tree = DecisionTreeRegressor(criterion="absolute_error")
        estimator = BaggingRegressor(estimator=tree, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)

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

    
if __name__ == "__main__":
    main()

# %%
