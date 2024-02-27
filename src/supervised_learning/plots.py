#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from create_dataset import check_distance_from_ip, load_sample

from create_response_matrix import abs_to_rel, create_sample, np_to_df_errors

from model_training import generate_second_order_polynomials

import pickle

def main():
    """with open(r"./figures/hist_noxing/coor_dict_RM_b1.pkl", "rb") as input_file:
        corr_rm_dict = pickle.load(input_file)

    with open(r"./figures/hist_noxing/coor_dict_ML_b1.pkl", "rb") as input_file:
        corr_ml_dict = pickle.load(input_file)

    with open(r"./figures/hist_noxing/sample_dict_ML_b1.pkl", "rb") as input_file:
        sample_dict = pickle.load(input_file)

    plot_rms_hist(sample_dict, corr_ml_dict, corr_rm_dict)"""

    with open(r"./figures/hist_xing/coor_dict_RM_b1_xing.pkl", "rb") as input_file:
        corr_rm_dict = pickle.load(input_file)

    with open(r"./figures/hist_xing/coor_dict_ML_b1_xing.pkl", "rb") as input_file:
        corr_ml_dict = pickle.load(input_file)

    with open(r"./figures/hist_xing/sample_dict_ML_b1_xing.pkl", "rb") as input_file:
        sample_dict = pickle.load(input_file)

    plot_rms_hist(sample_dict, corr_ml_dict, corr_rm_dict)
    

def plot_learning_curve(samples, metrics, algorithm):
    plt.figure(figsize=(6.2, 4))
    plt. tight_layout()

    metrics = np.array(metrics, dtype=object)
    #MAE
    print(samples, metrics[:,1])                                                                                                                      
    plt.title("Mean Average Error")
    plt.xlabel("N Samples")
    plt.ylabel("MAE", fontsize=14)
    plt.plot(samples, metrics[:,1], label="Train", marker='o')
    plt.plot(samples, metrics[:,3], label="Test", marker='o')
    plt.legend()
    plt.show()
    plt.savefig(f"./figures/mae_{algorithm}.pdf")

    #R2                                                                                                                             
    plt.clf()
    plt.title("Correlation Coefficient")
    plt.xlabel("N Samples")
    plt.ylabel(r"$R^2$", fontsize=14)
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


def plot_sample_rdt(sample_path, corr_sample_path, jklm, method, XING, BEATING, noise_level, hor_rdt_list, vert_rdt_list):
    # Plots the RDTs with 
    rdt_df = pd.read_csv(sample_path, sep="\t")
    rdt_corr_df = pd.read_csv(corr_sample_path, sep="\t")

    bn = sample_path.split('.')[-2][-2:]

    if XING == True:
      if bn == "b1":
        rdt_nom_df = pd.read_csv('./lhc_data/RDT_BPMS_NOMINAL_B1_XING.csv', sep="\t")
      if bn == "b2":
        rdt_nom_df = pd.read_csv('./lhc_data/RDT_BPMS_NOMINAL_B2_XING.csv', sep="\t")

    elif XING == False:
      if bn == "b1":
        rdt_nom_df = pd.read_csv('./lhc_data/RDT_BPMS_NOMINAL_B1.csv', sep="\t")
      if bn == "b2":
        rdt_nom_df = pd.read_csv('./lhc_data/RDT_BPMS_NOMINAL_B2.csv', sep="\t")
    
    rdt_df["PART_OF_MEAS"] = rdt_df["NAME"].apply(check_distance_from_ip)
    rdt_corr_df["PART_OF_MEAS"] = rdt_corr_df["NAME"].apply(check_distance_from_ip)
    rdt_nom_df["PART_OF_MEAS"] = rdt_nom_df["NAME"].apply(check_distance_from_ip)

    rdt_df.loc[rdt_df["PART_OF_MEAS"] == False, "RE_" + jklm] = np.nan
    rdt_df.loc[rdt_df["PART_OF_MEAS"] == False, "IM_" + jklm] = np.nan
    rdt_nom_df.loc[rdt_df["PART_OF_MEAS"] == False, "RE_" + jklm] = np.nan
    rdt_nom_df.loc[rdt_df["PART_OF_MEAS"] == False, "IM_" + jklm] = np.nan
    rdt_corr_df.loc[rdt_df["PART_OF_MEAS"] == False, "RE_" + jklm] = np.nan
    rdt_corr_df.loc[rdt_df["PART_OF_MEAS"] == False, "IM_" + jklm] = np.nan

    j_ = int(jklm[0])
    l_ = int(jklm[2])
    
    if "RE_"+jklm in hor_rdt_list:
        noise = noise_level/2*j_ # This is the relative error wrt the spectral line strength

    elif "RE_"+jklm in vert_rdt_list:
        noise = noise_level/2*l_

    rdt_df["RE_" + jklm] = rdt_df["RE_" + jklm]*(1+np.random.normal(0, noise, len(rdt_df["RE_" + jklm]))) # Adding noise
    rdt_df["IM_" + jklm] = rdt_df["IM_" + jklm]*(1+np.random.normal(0, noise, len(rdt_df["IM_" + jklm]))) # Adding noise

    rdt_corr_df["RE_" + jklm] = rdt_corr_df["RE_" + jklm]*(1+np.random.normal(0, noise, len(rdt_corr_df["RE_" + jklm]))) # Adding noise
    rdt_corr_df["IM_" + jklm] = rdt_corr_df["IM_" + jklm]*(1+np.random.normal(0, noise, len(rdt_corr_df["IM_" + jklm]))) # Adding noise

    nom_rdt_amp = np.sqrt(rdt_nom_df["RE_" + jklm]**2 + rdt_nom_df["IM_" + jklm]**2)
    sample_rdt_amp = np.sqrt(rdt_df["RE_" + jklm]**2 + rdt_df["IM_" + jklm]**2)
    corr_rdt_amp = np.sqrt(rdt_corr_df["RE_" + jklm]**2 + rdt_corr_df["IM_" + jklm]**2)


    if BEATING == False:
        plt.figure(figsize=(4.2, 2.5))
        plt.title(f"{method} Correction {bn}")
        plt.xlabel("BPM #")
        plt.ylabel(f"$|f_{{{jklm}}}|$", fontsize=13)
        plt.plot(range(len(nom_rdt_amp)), sample_rdt_amp, label="Sim", alpha=0.5)
        plt.plot(range(len(nom_rdt_amp)), corr_rdt_amp, label="Corr", alpha=0.5)
        plt.plot(range(len(nom_rdt_amp)), nom_rdt_amp, label="Nom", alpha=0.5)
        plt.legend()
        plt.show()
        
        plt.tight_layout()
        plt.savefig(f"./figures/rdts_test/|f_{{{jklm}}}_{bn}|_{method}.png")

    elif BEATING == True:
        plt.figure(figsize=(4.2, 2.5))
        plt.title(f"{method} Correction {bn}")
        plt.xlabel("BPM #")
        plt.ylabel(f"$\Delta|f_{{{jklm}}}|$")#/|f_{{{jklm}}}|_{{NOM}}
        plt.plot(range(len(rdt_df["RE_" + jklm])), (sample_rdt_amp-nom_rdt_amp), label="Sim", alpha=0.5)#/nom_rdt_amp
        plt.plot(range(len(rdt_corr_df["RE_" + jklm])), (corr_rdt_amp-nom_rdt_amp), label="Corr", alpha=0.5)#/nom_rdt_amp
        plt.legend()
        plt.show()

        plt.tight_layout()
        plt.savefig(f"./figures/rdts_test/|f_{{{jklm}}}_{bn}|_{method}.png")


def plot_rms_hist(sample_rms_rdt_dict, corr_ml_rms_rdt_dict, corr_rm_rms_rdt_dict):
    y_ranges = [300, 100, 500,  300, 300, 200, 200, 200]
    y_peaks = [1000,1000, 1000, 1000, 1000, 1000, 850, 1000]

    y_ranges = [25, 10, 50,  30, 30, 20, 20, 20]
    y_peaks = [50,70, 50, 50, 50, 90, 70, 50]
  
    rm_filter = [100, 4e4, 2e2, 2e2, 2e2, 3e6, 1e5, 1e5]

    for i, (key, values) in enumerate(sample_rms_rdt_dict.items()):
        print(key)
        if key not in corr_rm_rms_rdt_dict.keys():
           print(f"RDT ({key}) not in RM data")
           continue

        plt.clf()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,4]}, figsize=(4.2, 2.5), sharex=True)
        fig.subplots_adjust(hspace=0.05)  # adjust space between axes

        ax2.set_ylim(0, y_ranges[i])
        ax1.set_ylim(y_peaks[i]-5, y_peaks[i]+5)

        locs = [int(i) for i in list(ax1.get_yticks())]
        locs[0] = ""
        # remove the first and the last labels
        # set these new labels
        ax1.set_yticklabels(locs)
        
        # Taking outliers
        corr_rm_rms_rdt_dict[key] = [corr for corr in corr_rm_rms_rdt_dict[key] if corr <= rm_filter[i]]
        sample_rms_rdt_dict[key] = [corr for corr in sample_rms_rdt_dict[key] if corr <= rm_filter[i]]

        ax1.set_title('RMS $\Delta RDT$  before and after correction')
        plt.ylabel("Counts")
        plt.xlabel(f"RDT Error: RMS($\Delta|f_{{{key}}}|$)", fontsize=13)
        _, bins, _ = ax2.hist(sample_rms_rdt_dict[key], bins=40, alpha=0.5, label="Error", color='tab:blue')
        binwidth = (bins[1]-bins[0])

        minimum = min(corr_ml_rms_rdt_dict[key] + corr_rm_rms_rdt_dict[key]) 
        maximum = max(corr_ml_rms_rdt_dict[key] + corr_rm_rms_rdt_dict[key])

        bins = np.arange(minimum, maximum + binwidth, binwidth)

        ax1.hist(corr_ml_rms_rdt_dict[key], bins=bins, alpha=0.5, label="ML Corr", color='tab:orange')
        ax1.hist(corr_rm_rms_rdt_dict[key], bins=bins, alpha=0.5, label="RM Corr", color='tab:gray')
    
        ax2.hist(corr_ml_rms_rdt_dict[key], bins=bins, alpha=0.5, label="ML Corr", color='tab:orange')
        ax2.hist(corr_rm_rms_rdt_dict[key], bins=bins, alpha=0.5, label="RM Corr", color='tab:gray')

        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        d = .015  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        plt.legend()
        ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        fig.show()
        fig.savefig(f"./figures/hist_{key}_xing.png", bbox_inches='tight')
    

def plot_svd_cutoff(R):
    U, S, Vt = np.linalg.svd(R)
    #print(U)
    #print(Vt)

    # Plot the singular values on a scree plot
    plt.plot(S, marker='o', linestyle='-', color='b')
    plt.yscale('log')  # Use a logarithmic scale for better visibility
    plt.title('Scree Plot of Singular Values')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

# %%
