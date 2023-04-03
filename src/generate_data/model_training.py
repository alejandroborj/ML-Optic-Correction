#%%
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

import joblib
from pathlib import Path
import matplotlib.pyplot as plt


# example of reading the data, training ML model and validate results

def main():
    set_name = "data"
    TRAIN = True
    MERGE = True
    algorithm = "ridge"
    # Train on generated data
    if TRAIN==True:
        metrics, n_samples = [], []
        # Load data
        if MERGE == True:
            input_data, output_data = merge_data(set_name)
        else:
            input_data, output_data = load_data(set_name)

        n_splits=1
        input_data = np.array_split(input_data, n_splits, axis=0)
        output_data = np.array_split(output_data, n_splits, axis=0)

        for i in range(n_splits):
            results = train_model(np.vstack(input_data[:i+1]), np.vstack(output_data[:i+1]), algorithm=algorithm)
            n_samples.append(len(np.vstack(input_data[:i+1])))
            metrics.append(results)
        
        metrics = np.array(metrics, dtype=object)
    
        #print(metrics)
        #print(n_samples)

        #MAE
        plt.plot(n_samples, metrics[:,1], label="Train")
        plt.plot(n_samples, metrics[:,3], label="Test")
        plt.show()

        #R2
        plt.clf()
        plt.plot(n_samples, metrics[:,0])
        plt.plot(n_samples, metrics[:,2])


def train_model(input_data, output_data, algorithm):
    # split into train and test
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        input_data, output_data, test_size=0.2, random_state=None)
    
    # create and fit a regression model
    if algorithm == "ridge":
        ridge = linear_model.Ridge(tol=1e-50, alpha=1e-03) #normalize=false
        estimator = BaggingRegressor(estimator=ridge, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)
    elif algorithm == "tree":
        tree = DecisionTreeRegressor()
        estimator = BaggingRegressor(estimator=tree, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)

    # Optionally: save fitted model or load already trained model
    joblib.dump(estimator, 'estimator.pkl') 
    #estimator = joblib.load('estimator.pkl')

    # Check scores: explained variance and MAE
    r2_train = estimator.score(train_inputs, train_outputs)
    r2_test = estimator.score(test_inputs, test_outputs)
    prediction_train = estimator.predict(train_inputs)
    mae_train = mean_absolute_error(train_outputs, prediction_train)
    prediction_test = estimator.predict(test_inputs)
    mae_test = mean_absolute_error(test_outputs, prediction_test)

    print("Training: R2 = {0}, MAE = {1}".format(r2_train, mae_train))
    print("Test: R2 = {0}, MAE = {1}".format(r2_test, mae_test))

    return r2_train, mae_train, r2_test, mae_test


def load_data(set_name):
    #Function that inputs the .npy file and returns the data in a readable format for the algoritms
    all_samples = np.load('./{}.npy'.format(set_name), allow_pickle=True)
    all_samples = np.delete(all_samples, 58, axis=0) # Delete elements where mqt errors failed
 
    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2, n_disp_b1, n_disp_b2, \
                triplet_errors, arc_errors_b1, arc_errors_b2, \
                mqt_errors_b1, mqt_errors_b2 = all_samples.T
    
    # select features for input
    # Optionally: add noise to simulated optics functions
    
    """    print(triplet_errors[0].shape, arc_errors_b1[0].shape, 
          arc_errors_b2[0].shape, mqt_errors_b1[0].shape, 
          mqt_errors_b2[0].shape)
    
    print([len(mqt_errors_b1[i]) for i,_ in enumerate(mqt_errors_b1)])
    print([len(mqt_errors_b2[i]) for i,_ in enumerate(mqt_errors_b2)])
    print(set([len(triplet_errors[i]) for i,_ in enumerate(triplet_errors)]))
    print([len(mqt_errors_b2[i]) for i,_ in enumerate(mqt_errors_b2)].index(1))"""

    input_data = np.concatenate(( \
        np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
        np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
        np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
        np.vstack(delta_mux_b2), np.vstack(delta_muy_b2), \
        np.vstack(n_disp_b1), np.vstack(n_disp_b2)
        ), axis=1)
    # select targets for output
    
    output_data = np.concatenate((np.vstack(triplet_errors), np.vstack(arc_errors_b1),\
                                   np.vstack(arc_errors_b2), np.vstack(mqt_errors_b1), \
                                    np.vstack(mqt_errors_b2)), axis=1)
    
    return input_data, output_data

def merge_data(data_path):
    #Takes folder path for all different data files and merges them
    input_data, output_data = [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist]

    for file_name in file_names:
        aux_input, aux_output = load_data(file_name)
        input_data.append(aux_input)
        output_data.append(aux_output)

    return np.concatenate(input_data), np.concatenate(output_data)


if __name__ == "__main__":
    main()

# %%
