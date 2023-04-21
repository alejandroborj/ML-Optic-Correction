#%%

import numpy as np
import pandas as pd
import tfs

import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from model_training import load_data
from model_training import merge_data
from model_training import obtain_errors



def main():    
    estimator = joblib.load('estimator_linear_45cm.pkl')
    set_name = "data_45cm"
    MERGE = True

    # Train on generated data
    # Load data
    if MERGE == True:
        input_data, output_data = merge_data(set_name)
    else:
        input_data, output_data = load_data(set_name)

    plot_example_errors(input_data, output_data, estimator)
    
    plot_example_hist(input_data, output_data, estimator)
    
def plot_example_errors(input_data, output_data, estimator):
    test_idx = sorted(np.load("test_idx.npy"))[18:19]
    pred_triplet, true_triplet, pred_arc,\
    true_arc, pred_mqt, true_mqt = obtain_errors(input_data[test_idx], 
                                                    output_data[test_idx], 
                                                    estimator)
    
    errors = (("Triplet Errors: ", pred_triplet, true_triplet), 
            ("Arc Errors: ", pred_arc, true_arc), 
            ("MQT Knob: ", pred_mqt, true_mqt))

    for idx, (name, pred_error, true_error) in enumerate(errors):
        x = [idx for idx, error in enumerate(true_error)]
        plt.bar(x, true_error, label="True")
        plt.bar(x, pred_error, label="Pred")
        plt.bar(x, pred_error-true_error, label="Res")

        plt.title(f"{name}")
        plt.xlabel(r"MQ")
        plt.legend()
        plt.savefig(f"./figures/error_bars_{name[:-2]}.pdf")
        plt.show()
        plt.clf()

def plot_example_betabeat(tw_nominal, tw_errors, beam):
    print("ERROR PANDAS")
    print(tw_errors)
    print(tw_nominal)
    bbeat_x = 100*(np.array(tw_errors.BETX - tw_nominal.BETX))/tw_nominal.BETX
    bbeat_y = 100*(np.array(tw_errors.BETY - tw_nominal.BETY))/tw_nominal.BETY

    fig, axs = plt.subplots(2)
    axs[0].plot(tw_errors.S, bbeat_x)
    axs[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[0].set_ylabel(r"$\Delta \beta _x / \beta _x [\%]$")
    axs[0].set_xticklabels(labels=['IP2', 'IP3', 'IP4', 'IP5', 'IP6', 'IP7', 'IP8', 'IP1'])
    axs[0].set_xticks([i for i in np.linspace(0, int(tw_errors.S[-1]), num=8)])
    #print("IP TICKS", ip_ticks)
    #axs[0].set_xticklabels(ip_ticks)

    axs[1].plot(tw_errors.S, bbeat_y)
    axs[1].set_ylabel(r"$\Delta \beta _y / \beta _y [\%]$")
    axs[1].set_xlabel(r"Longitudinal location $[m]$")
    fig.suptitle(f"Beam {beam}")
    fig.savefig(f"./figures/example_twiss_beam{beam}.pdf")
    fig.show()


def plot_example_hist(input_data, output_data, estimator):
    test_idx = sorted(np.load("test_idx.npy"))[:200]

    pred_triplet, true_triplet, pred_arc,\
    true_arc, pred_mqt, true_mqt = obtain_errors(input_data[test_idx], 
                                                    output_data[test_idx], 
                                                    estimator, NORMALIZE=True)
    
    errors = (("Triplet Errors: ", pred_triplet, true_triplet), 
            ("Arc Errors: ", pred_arc, true_arc), 
            ("MQT Knob: ", pred_mqt, true_mqt))

    for idx, (name, pred_error, true_error) in enumerate(errors):
        #ax = axs[idx]
        fig, ax = plt.subplots()

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        _, bins, _ = ax.hist(true_error, bins=50, alpha = 0.5,label="True")
        ax.hist(pred_error, bins=bins, alpha=0.5, label="Predicted")
        ax.hist(pred_error-true_error, bins=bins, alpha=0.5, label="Residuals")

        #ax.set_xlim(-0.02, 0.02)
        if name == "MQT Knob: ":
            ax.set_xlim(-0.0005, 0.0005)
        if name == "Triplet Errors: ":
            print("Tripl")
            #ax.set_xlim(-0.002, 0.002)
        ax.set_title(f"{name}")
        ax.set_xlabel(r"Relative Errors $\Delta k$")

        fig.legend()
        fig.savefig(f"./figures/hist_{name}.pdf")
        fig.show()
        

if __name__ == "__main__":
    main()

# %%
