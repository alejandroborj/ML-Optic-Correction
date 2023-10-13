#%%

import numpy as np

import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    main()

# %%
