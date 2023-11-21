#%%

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from create_response_matrix import check_distance_from_ip


def main(): 
    print("")
    

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


def plot_sample_rdt(sample_path, corrected_sample_path, jklm, plot_title, XING):
    # Plots the RDTs with 
    order_list = []
    rdt_df = pd.read_csv(sample_path, sep="\t")
    rdt_corrected_df = pd.read_csv(corrected_sample_path, sep="\t")

    if XING==True:
        rdt_nom_df = pd.read_csv('./RDT_BPMS_NOMINAL_B1_XING.csv', sep="\t")
    else:
        rdt_nom_df = pd.read_csv('./RDT_BPMS_NOMINAL_B1.csv', sep="\t")
    
    rdt_df["PART_OF_MEAS"] = rdt_df["NAME"].apply(check_distance_from_ip)
    rdt_corrected_df["PART_OF_MEAS"] = rdt_corrected_df["NAME"].apply(check_distance_from_ip)
    rdt_nom_df["PART_OF_MEAS"] = rdt_nom_df["NAME"].apply(check_distance_from_ip)


    rdt_df.loc[rdt_df["PART_OF_MEAS"] == False, "RE_" + jklm] = np.nan
    rdt_df.loc[rdt_df["PART_OF_MEAS"] == False, "IM_" + jklm] = np.nan
    rdt_nom_df.loc[rdt_df["PART_OF_MEAS"] == False, "RE_" + jklm] = np.nan
    rdt_nom_df.loc[rdt_df["PART_OF_MEAS"] == False, "IM_" + jklm] = np.nan
    rdt_corrected_df.loc[rdt_df["PART_OF_MEAS"] == False, "RE_" + jklm] = np.nan
    rdt_corrected_df.loc[rdt_df["PART_OF_MEAS"] == False, "IM_" + jklm] = np.nan


    plt.title(plot_title + " Real")
    plt.xlabel("BPM #")
    plt.ylabel(f"$RE(f_{{{jklm}}})$")
    plt.plot(range(len(rdt_df["RE_" + jklm])), rdt_df["RE_" + jklm], label="Sim", alpha=0.7)
    plt.plot(range(len(rdt_corrected_df["RE_" + jklm])), rdt_corrected_df["RE_" + jklm], label="Corr", alpha=0.5)
    plt.plot(range(len(rdt_nom_df["RE_" + jklm])), rdt_nom_df["RE_" + jklm], label="Nom", alpha=0.5)
    plt.legend()
    plt.show()

    plt.title(plot_title + " Imaginary part")
    plt.xlabel("BPM #")
    plt.ylabel(f"$IM(f_{{{jklm}}})$")
    plt.plot(range(len(rdt_df["IM_" + jklm])), rdt_df["IM_" + jklm], label="Sim", alpha=0.5)
    plt.plot(range(len(rdt_corrected_df["IM_" + jklm])), rdt_corrected_df["IM_" + jklm], label="Corr", alpha=0.5)
    plt.plot(range(len(rdt_nom_df["IM_" + jklm])), rdt_nom_df["IM_" + jklm], label="Nom", alpha=0.5)
    plt.legend()
    plt.show()

    plt.title(plot_title)
    plt.xlabel("BPM #")
    plt.ylabel(f"$|f_{{{jklm}}}|$")
    plt.plot(range(len(rdt_df["RE_" + jklm])), np.sqrt(rdt_df["RE_" + jklm]**2 + rdt_df["RE_" + jklm]**2), label="Sim", alpha=0.5)
    plt.plot(range(len(rdt_corrected_df["RE_" + jklm])), np.sqrt(rdt_corrected_df["RE_" + jklm]**2 + rdt_corrected_df["RE_" + jklm]**2), label="Corr", alpha=0.5)
    plt.plot(range(len(rdt_nom_df["RE_" + jklm])), np.sqrt(rdt_nom_df["RE_" + jklm]**2 + rdt_nom_df["RE_" + jklm]**2), label="Nom", alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

# %%
