#%%

import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import joblib
from pathlib import Path
import matplotlib.pyplot as plt

import tfs
import pandas as pd


# example of reading the data, training ML model and validate results

GEN_TEST_SPLIT = True # If a new test split is needed

def main():
    set_name = "data_45cm"
    TRAIN = True
    MERGE = True
    algorithm = "linear"
    # Train on generated data
    # Load data
    if MERGE == True:
        input_data, output_data = merge_data(set_name)
    else:
        input_data, output_data = load_data(set_name)

    if TRAIN==True:
        metrics, n_samples = [], []

        n_splits=1
        input_data = np.array_split(input_data, n_splits, axis=0)
        output_data = np.array_split(output_data, n_splits, axis=0)

        for i in range(n_splits):
            print(input_data[0].shape)
            results = train_model(np.vstack(input_data[:i+1]), np.vstack(output_data[:i+1]), algorithm=algorithm)
            n_samples.append(len(np.vstack(input_data[:i+1])))
            metrics.append(results)
        
        metrics = np.array(metrics, dtype=object)
    
        #print(metrics)
        #print(n_samples)

        #MAE                                                                                                                            
        plt.title("Mean Average Error")
        plt.xlabel("Number of samples")
        plt.ylabel("MAE")
        plt.plot(n_samples, metrics[:,1], label="Train", marker='o')
        plt.plot(n_samples, metrics[:,3], label="Test", marker='o')
        plt.show()
        plt.savefig(f"./figures/mae.pdf")

        #R2                                                                                                                             
        plt.clf()        
        plt.title("Correlation Coefficient")
        plt.xlabel("Number of samples")
        plt.ylabel(r"$R^2$")
        plt.plot(n_samples, metrics[:,0], label="Train", marker='o')
        plt.plot(n_samples, metrics[:,2], label="Test", marker='o')
        plt.show()
        plt.savefig(f"./figures/r2.pdf")


def train_model(input_data, output_data, algorithm):
    indices = np.arange(len(input_data))

    # Generating new test split or loading old one
    if GEN_TEST_SPLIT==True:
        train_inputs, test_inputs, train_outputs, test_outputs, indices_train, indices_test = train_test_split(
            input_data, output_data, indices, test_size=0.2, random_state=None)
        np.save("test_45_idx.npy", indices_test)
        np.save("train_45_idx.npy", indices_train)
    else:
        indices_test = np.load("test_idx.npy")
        indices_train = np.load("train_idx.npy")
        train_inputs, test_inputs, train_outputs, test_outputs = input_data[indices_train], input_data[indices_test],\
                                                                output_data[indices_train], output_data[indices_test]
    
    # create and fit a regression model
    if algorithm == "ridge":
        ridge = linear_model.Ridge(tol=1e-50, alpha=1e-03) #normalize=false
        estimator = BaggingRegressor(estimator=ridge, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)

    elif algorithm == "linear":
        linear = linear_model.LinearRegression()
        estimator = BaggingRegressor(estimator=linear, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)    
        
    elif algorithm == "tree":
        tree = DecisionTreeRegressor()
        estimator = BaggingRegressor(estimator=tree, n_estimators=10, \
            max_samples=0.9, max_features=1.0, n_jobs=16, verbose=0)
        estimator.fit(train_inputs, train_outputs)

    # Optionally: save fitted model or load already trained model
    joblib.dump(estimator, 'estimator.pkl') 

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

    print("Training Triplet: R2 = {0}, MAE = {1}".format(r2_train_triplet, mae_train_triplet))
    print("Test Triplet: R2 = {0}, MAE = {1}".format(r2_test_triplet, mae_test_triplet))

        
    print("Training: R2 = {0}, MAE = {1}".format(r2_train, mae_train))
    print("Test: R2 = {0}, MAE = {1}".format(r2_test, mae_test))

    return r2_train, mae_train, r2_test, mae_test
    # split into train and test


def load_data(set_name):
    #Function that inputs the .npy file and returns the data in a readable format for the algoritms
    all_samples = np.load('./data_45cm/{}.npy'.format(set_name), allow_pickle=True)
    '''
    idxs = []
    for idx, sample in enumerate(all_samples):
        mqt_err_b1, mqt_err_b2 = sample[-2], sample[-1]
        if len(mqt_err_b1) != 2 or len(mqt_err_b2) != 2:
            idxs.append(idx)
    print(idxs)
    all_samples = np.delete(all_samples, idxs, axis=0) # Delete elements where mqt errors failed
    np.save('./data/{}.npy'.format(set_name), arr=all_samples , allow_pickle=True) # Save with correct dim
    '''
    
    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2, n_disp_b1, n_disp_b2, \
                triplet_errors, arc_errors_b1, arc_errors_b2, \
                mqt_errors_b1, mqt_errors_b2 = all_samples.T
    
    # select features for input
    # Optionally: add noise to simulated optics functions

    #print(len(delta_beta_star_x_b1[0]))
    #print(len(delta_beta_star_x_b2[0]))
    #print(len(delta_beta_star_y_b1[0]))
    #print(len(delta_beta_star_y_b2[0]))
    #print(len(delta_mux_b1[0]))
    #print(len(delta_mux_b2[0]))
    #print(len(delta_muy_b1[0]))
    #print(len(delta_mux_b2[0]))
    #print(len(n_disp_b1[0]))
    #print(len(n_disp_b2[0]))
    
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
    print(input_data.shape)
    
    return input_data, output_data


def merge_data(data_path):
    #Takes folder path for all different data files and merges them
    input_data, output_data = [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist][:5]

    for file_name in file_names:
        aux_input, aux_output = load_data(file_name)
        input_data.append(aux_input)
        output_data.append(aux_output)

    return np.concatenate(input_data), np.concatenate(output_data)


def obtain_errors(input_data, output_data, estimator, NORMALIZE=False):
    pred_data = estimator.predict(input_data)
    if NORMALIZE==True:
        pred_data = normalize_errors(pred_data)
        output_data = normalize_errors(output_data)

    pred_triplet = np.hstack(pred_data[:,:32])
    true_triplet = np.hstack(output_data[:,:32])

    pred_arc = np.hstack(pred_data[:,32:1248])
    true_arc = np.hstack(output_data[:,32:1248])

    pred_mqt = np.hstack(pred_data[:,-4:])
    true_mqt = np.hstack(output_data[:,-4:])

    return pred_triplet, true_triplet, pred_arc, true_arc, pred_mqt, true_mqt

def normalize_errors(data):
    with open("mq_names.txt", "r") as f:
        lines = f.readlines()
        names = [name.replace("\n", "").upper().split(':')[0] for name in lines]

    B1_ELEMENTS_MDL_TFS = tfs.read_tfs("./b1_nominal_elements_30.dat").set_index("NAME")
    B2_ELEMENTS_MDL_TFS = tfs.read_tfs("./b2_nominal_elements_30.dat").set_index("NAME")
    ELEMENTS_MDL_TFS = tfs.frame.concat([B1_ELEMENTS_MDL_TFS, B2_ELEMENTS_MDL_TFS])
    TRIPLET_NOM_K1 = ELEMENTS_MDL_TFS.loc[names, "K1L"][:64]
    TRIPLET_NOM_K1 = TRIPLET_NOM_K1[TRIPLET_NOM_K1.index.duplicated(keep="first")]

    ARC_NOM_K1 = ELEMENTS_MDL_TFS.loc[names, "K1L"][64:]

    nom_k1 = tfs.frame.concat([TRIPLET_NOM_K1, ARC_NOM_K1])
    nom_k1 = np.append(nom_k1, [1,1,1,1])
    
    for i, sample in enumerate(data):
        data[i] = (sample)/nom_k1
    return data

def save_np_errors_tfs(np_errors):
    with open("mq_names.txt", "r") as f:
        lines = f.readlines()
        names = [name.replace("\n", "") for name in lines]

    df = pd.DataFrame(columns=["names","k1l"])
    df.k1l = np_errors
    df.names = names
    df = tfs.frame.TfsDataFrame(df)
    tfs.writer.write_tfs(tfs_file_path="mag_errors.tfs", data_frame=df)

def obtain_twiss(input_data):
    print(input_data.shape)

    
if __name__ == "__main__":
    main()

# %%
