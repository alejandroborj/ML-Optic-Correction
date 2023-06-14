#%%
from measurement_to_input import get_delta_and_meas_phase
from measurement_to_input import get_beta_beating_from_measurement
from measurement_to_input import get_phase_diff_from_twiss
import numpy as np
import tfs
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# measurement_path => Results folder
# model_path => Nominal tfs
# twiss_perturbet => Simulated error twiss

QX = 62.31
QY = 60.32

def main():
    measurement_name = "test_pred_q4"
    measurement_path_b2 = "./measurements/19-06-04_import_b2_30cm_nocorrinarcs/"
    twiss_b2 = tfs.read_tfs("./data_analysis/b2_nominal_monitors.dat").set_index("NAME")

    phasecut_x = 0.4
    phasecut_y = 0.4
    bbeat_cut = 150 #No bbeat_cut

    # Checking the measured Beta beat
    betabeatx_b2, betabeaty_b2 = get_beta_beating_from_measurement(measurement_path_b2)

    plt.title("BEAM 2")
    plt.plot(range(len(betabeatx_b2)),betabeatx_b2, label="bbeatx")
    plt.plot(range(len(betabeaty_b2)),betabeaty_b2, label="bbeaty")
    plt.legend()
    plt.show()

    # Only interested on the B2 input
    delta_meas_mux_b2, delta_meas_muy_b2, meas_mux_b2, meas_muy_b2 = get_delta_and_meas_phase(measurement_path_b2, twiss_b2, 
                                                              phasecut_x, phasecut_y, bbeat_cut)

    sample = delta_meas_mux_b2, delta_meas_muy_b2

    np.save(f'./measurements/meas_input_{measurement_name}.npy', np.array(sample, dtype=object))
    delta_meas_mux_b2, delta_meas_muy_b2 = np.load(f'./measurements/meas_input_{measurement_name}.npy', allow_pickle=True)
    meas_input = np.concatenate([delta_meas_mux_b2, delta_meas_muy_b2])
 
    estimator = joblib.load("./md_models/b2_arcs_phases_virgin_e4.pkl")
    
    pred_output = estimator.predict([meas_input])[0]

    plot_example_errors(pred_output)
    error_tfs = save_np_errors_tfs(pred_output, "pred_q4_err.tfs")
    tfs_to_corr(error_tfs, "./corrections/q4_corrections.madx")


def plot_example_errors(pred_output):
    pred_arc, pred_mqt = np.hstack(pred_output[:-2]), np.hstack(pred_output[-2:])
    
    errors = (("Arc Errors", pred_arc), 
            ("MQT Knob", pred_mqt))

    for idx, (name, pred_error) in enumerate(errors):
        x = [idx for idx, error in enumerate(pred_error)]

        plt.bar(x, pred_error, label="Pred")

        plt.title(f"{name}")
        plt.xlabel(r"MQ [#]")
        plt.ylabel(r"Absolute Error: $\Delta k$")
        plt.legend()
        plt.show()
        plt.clf()


def save_np_errors_tfs(np_errors, filename):
    #This is the tfs format that can be read, this model of file is then copied and filled
    error_tfs_model_b2 = tfs.read_tfs("./data_analysis/errors_b2.tfs")

    #Function that takes np errors and outputs .tfs file with all error values
    with open("./data_analysis/mq_names.txt", "r") as f:
        lines = f.readlines()
        names = [name.replace("\n", "") for name in lines]

    # Recons_df is a dataframe with the correct names and errors but not format
    recons_df = pd.DataFrame(columns=["NAME","K1L"])
    recons_df.K1L = np_errors[:-2] # No mqt knob
    recons_df.NAME = [name for name in names if '.B2' in name and 'MQX' not in name]
    
    for beam, error_tfs_model in enumerate([error_tfs_model_b2]):
        for i in range(len(error_tfs_model)):
            # check if the name is in recons_df
            if error_tfs_model.loc[i, 'NAME'] in list(recons_df['NAME']):
                error_tfs_model.loc[i, 'K1L'] = recons_df.loc[recons_df['NAME'] == error_tfs_model.loc[i, 'NAME']].values[0][1]
    
    error_tfs_model_b2 = error_tfs_model_b2.loc[error_tfs_model_b2['K1L']!=0]
    tfs.writer.write_tfs(tfs_file_path=f"./measurements/b2_{filename}", data_frame=error_tfs_model_b2)

    return error_tfs_model_b2

def tfs_to_corr(tfs_error, corr_filename):
    with open(corr_filename, 'w') as r:
        for error in tfs_error.iterrows():
            name = error[1]["NAME"]
            k1l = error[1]["K1L"]
            if "MQY.4" in name: # Only tanking mq4
                r.write(f"{name}->K1 = {name}->K1 + ({-1*k1l});\n") #Changing the sign

if __name__ == "__main__":
    main()
 # %%