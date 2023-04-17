#%%

import numpy as np
import tfs

import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from model_training import load_data
from model_training import merge_data
from model_training import obtain_errors
from model_training import obtain_twiss


def main():    
    B1_MONITORS_MDL_TFS = tfs.read_tfs("./b1_nominal_monitors.dat").set_index("NAME")
    B2_MONITORS_MDL_TFS = tfs.read_tfs("./b2_nominal_monitors.dat").set_index("NAME")   
    estimator = joblib.load('estimator.pkl')
    set_name = "data"
    MERGE = True

    # Train on generated data
    # Load data
    if MERGE == True:
        input_data, output_data = merge_data(set_name)
    else:
        input_data, output_data = load_data(set_name)

    obtain_twiss(input_data)
    plot_example_errors(input_data, output_data, estimator)
    


def plot_example_errors(input_data, output_data, estimator):
    #test_idx = np.load("test_idx.npy")
    test_idx = [1]
    pred_triplet, true_triplet, pred_arc,\
    true_arc, pred_mqt, true_mqt = obtain_errors([input_data[test_idx[:5]]], 
                                                    output_data[test_idx[:5]], 
                                                    estimator)
    
    errors = (("Triplet Errors: ", pred_triplet, true_triplet), 
            ("Arc Errors: ", pred_arc, true_arc), 
            ("MQT Knob: ", pred_mqt, true_mqt))

    for idx, (name, pred_error, true_error) in enumerate(errors):
        x = [idx for idx, error in enumerate(true_error)]
        print(name)
        plt.bar(x, true_error, label="True")
        plt.bar(x, pred_error, label="Pred")

        plt.title(f"{name}")
        plt.xlabel(r"MQ")
        plt.legend()
        plt.show()
        plt.savefig(f"./figures/{name}.pdf")
        plt.clf()

def plot_example_betabeat(input_data):
    input_data

if __name__ == "__main__":
    main()

# %%
