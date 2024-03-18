#%%

import numpy as np
import pandas as pd
import tfs

import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from data_utils import load_data
from data_utils import merge_data
from data_utils import obtain_errors


def main():    
    estimator = joblib.load('./estimators/estimator_linear.pkl')
    set_name = "data"
    MERGE = True

    # Train on generated data
    # Load data
    if MERGE == True:
        input_data, output_data = merge_data(set_name, 0)
    else:
        input_data, output_data = load_data(set_name, 0)

    #plot_example_errors(input_data, output_data, estimator)
    
    #plot_example_hist(input_data, output_data, estimator

    plot_bbeat_hist()
    
def plot_example_errors(input_data, output_data, estimator):
    test_idx = sorted(np.load("./data_analysis/test_idx.npy"))[0:1]
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
        plt.xlabel(r"MQ [#]")
        plt.ylabel(r"Absolute Error: $\Delta k$")
        plt.legend()
        plt.savefig(f"./figures/error_bars_{name[:-2]}.pdf")
        plt.show()
        plt.clf()

def plot_example_betabeat(tw_nominal, tw_errors, beam):

    '''Takes the nominal and perturbed twiss tfs-pandas data and gives the betabeating'''

    print(len(tw_errors.BETX - tw_nominal.BETX))
    #bbeat_x = 100*(np.array(tw_errors.BETX) - np.array(tw_nominal.BETX))/tw_nominal.BETX
    bbeat_x = 100*(np.array(tw_errors.BETX - tw_nominal.BETX))/tw_nominal.BETX
    bbeat_y = 100*(np.array(tw_errors.BETY - tw_nominal.BETY))/tw_nominal.BETY

    fig, axs = plt.subplots(2)
    axs[0].plot(tw_errors.S, bbeat_x)
    axs[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[0].set_ylabel(r"$\Delta \beta _x / \beta _x [\%]$")
    axs[0].set_xticklabels(labels=['IP2', 'IP3', 'IP4', 'IP5', 'IP6', 'IP7', 'IP8', 'IP1'])
    axs[0].set_xticks([i for i in np.linspace(0, int(tw_errors.S[-1]), num=8)])

    axs[1].plot(tw_errors.S, bbeat_y)
    axs[1].set_ylabel(r"$\Delta \beta _y / \beta _y [\%]$")
    axs[1].set_xlabel(r"Longitudinal location $[m]$")
    fig.suptitle(f"Beam {beam}")
    fig.savefig(f"./figures/example_twiss_beam{beam}.pdf")
    fig.show()


def plot_example_hist(input_data, output_data, estimator):
    test_idx = sorted(np.load("./data_analysis/test_idx.npy"))[:200]

    pred_triplet, true_triplet, pred_arc,\
    true_arc, pred_mqt, true_mqt = obtain_errors(input_data[test_idx], 
                                                    output_data[test_idx], 
                                                    estimator, NORMALIZE=True)
    
    errors = (("Triplet Errors: ", pred_triplet, true_triplet), 
            ("Arc Errors: ", pred_arc, true_arc), 
            ("MQT Knob: ", pred_mqt, true_mqt))
    
    for idx, (name, pred_error, true_error) in enumerate(errors):
        fig, ax = plt.subplots()

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        _, bins, _ = ax.hist(true_error, bins=200, alpha = 0.5, label="True")
        ax.hist(pred_error, bins=bins, alpha=0.5, label="Predicted")
        ax.hist(pred_error-true_error, bins=bins, alpha=0.5, label="Residuals")

        if name == "MQT Knob: ":
            ax.set_xlim(-0.0005, 0.0005)
        if name == "Triplet Errors: ":
            ax.set_xlim(-0.002, 0.002)
        if name == "Arc Errors: ":
            ax.set_xlim(-0.0075, 0.0075)
        ax.set_title(f"{name}")
        ax.set_xlabel(r"Relative Errors $\Delta k$")

        fig.legend()
        fig.savefig(f"./figures/hist_{name}.pdf")
        fig.show()
        
def plot_bbeat_hist():
    for beam in (1,2):
        bbeat_hist = pd.read_csv(f"./data_analysis/bbeat{beam}.csv")

        fig, axs = plt.subplots(2, figsize=(7,10))
        
        axs[0].set_xlabel(r"Mean beta beating $\Delta \beta / \beta [\%]$")
        axs[0].hist(np.array(bbeat_hist["mean_x"], dtype=float), bins=50, alpha = 0.5, label="Horizontal")
        axs[0].hist(np.array(bbeat_hist["mean_y"], dtype=float), bins=50, alpha = 0.5, label="Vertical")
        axs[0].legend()

        axs[1].set_xlabel(r"Max beta beating $\Delta \beta / \beta [\%]$")
        axs[1].hist(np.array(bbeat_hist["max_x"], dtype=float), bins=50, alpha = 0.5, label="Horizontal")
        axs[1].hist(np.array(bbeat_hist["max_y"], dtype=float), bins=50, alpha = 0.5, label="Vertical")
        axs[1].legend()

        fig.savefig(f"./figures/bbeat_hist{beam}.pdf")

def plot_learning_curve(samples, metrics, algorithm):
    metrics = np.array(metrics, dtype=object)
    #MAE
    print(samples, metrics[:,1])                                                                                                                      
    plt.title("Mean Average Error")
    plt.xlabel("N Samples")
    plt.ylabel("MAE")
    plt.plot(samples, metrics[:,1], label="Train", marker='o')
    plt.plot(samples, metrics[:,3], label="Test", marker='o')
    plt.legend()
    plt.show()
    plt.savefig(f"./figures/mae_{algorithm}.pdf")

    #R2                                                                                                                             
    plt.clf()
    plt.title("Correlation Coefficient")
    plt.xlabel("N Samples")
    plt.ylabel(r"$R^2$")
    plt.plot(samples, metrics[:,0], label="Train", marker='o')
    plt.plot(samples, metrics[:,2], label="Test", marker='o')
    plt.legend()
    plt.show()
    plt.savefig(f"./figures/r2_{algorithm}.pdf")


def plot_noise_vs_metrics(noises, metrics, algorithm):
    metrics = np.array(metrics, dtype=object)
    #MAE                                                                                                                            
    plt.title("Mean Average Error")
    plt.xlabel("Noise")
    plt.ylabel("MAE")
    plt.xscale('log')
    plt.plot(noises, metrics[:,1], label="Train", marker='o')
    plt.plot(noises, metrics[:,3], label="Test", marker='o')
    plt.plot(noises, metrics[:,5], label="Test Triplet", marker='o')
    plt.legend()
    plt.show()
    plt.savefig(f"./figures/mae_noise_{algorithm}.pdf")

    #R2                                                                                                                             
    plt.clf()
    plt.title("Correlation Coefficient")
    plt.xlabel("Noise")
    plt.ylabel(r"$R^2$")
    plt.xscale('log')
    plt.plot(noises, metrics[:,0], label="Train", marker='o')
    plt.plot(noises, metrics[:,2], label="Test", marker='o')
    plt.plot(noises, metrics[:,4], label="Test Triplet", marker='o')
    plt.legend()
    plt.show()
    plt.savefig(f"./figures/r2_noise_{algorithm}.pdf")

if __name__ == "__main__":
    main()

# %%
