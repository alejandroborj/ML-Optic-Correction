#%%
from measurement_to_input import get_delta_and_meas_phase
from measurement_to_input import get_beta_beating_from_measurement
from measurement_to_input import get_phase_diff_from_twiss

from data_utils import load_data

import numpy as np
import tfs
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from string import Template

# measurement_path => Results folder
# model_path => Nominal tfs
# twiss_perturbet => Simulated error twiss

QX = 62.28
QY = 60.31

def main():
    measurement_name = "pred_triplet"
    
    #measurement_path_b1 = "./measurements/B1_30cm_without_guessed_IP1_local_corrections"
    measurement_path_b1 = "./measurements/B1_30cm_without_IP1_localcor_WITH_energy_trim"
    measurement_path_b2 = "./measurements/B2_30cm_without_guessed_IP1_local_corrections"
    
    #estimator = joblib.load("./md_models/local_028_1e-2_45k.pkl")
    #estimator = joblib.load("./md_models/local_028_5e-3.pkl")
    estimator = joblib.load("./md_models/triplet_phases_only_028_ridge_0.01.pkl")

    twiss_b2 = tfs.read_tfs("./data_analysis/b2_nominal_monitors.dat").set_index("NAME")
    twiss_b1 = tfs.read_tfs("./data_analysis/b1_nominal_monitors.dat").set_index("NAME")

    phasecut_x = 0.4
    phasecut_y = 0.4
    bbeat_cut = 80

    # Checking the measured Beta beat
    betabeatx_b1, betabeaty_b1 = get_beta_beating_from_measurement(measurement_path_b1)
    betabeatx_b2, betabeaty_b2 = get_beta_beating_from_measurement(measurement_path_b2)

    plt.title("BEAM 1")
    plt.plot(range(len(betabeatx_b1)),betabeatx_b1, label="bbeatx")
    plt.plot(range(len(betabeaty_b1)),betabeaty_b1, label="bbeaty")
    plt.legend()
    plt.savefig("./figures/bbeat_b1.pdf")
    plt.show()

    plt.title("BEAM 2")
    plt.plot(range(len(betabeatx_b2)),betabeatx_b2, label="bbeatx")
    plt.plot(range(len(betabeaty_b2)),betabeaty_b2, label="bbeaty")
    plt.legend()
    plt.savefig("./figures/bbeat_b2.pdf")
    plt.show()

    # Taking both inputs
    delta_meas_mux_b1, delta_meas_muy_b1, meas_mux_b1, meas_muy_b1 = get_delta_and_meas_phase(measurement_path_b1, twiss_b1, 
                                                              phasecut_x, phasecut_y, bbeat_cut)
    delta_meas_mux_b2, delta_meas_muy_b2, meas_mux_b2, meas_muy_b2 = get_delta_and_meas_phase(measurement_path_b2, twiss_b2, 
                                                              phasecut_x, phasecut_y, bbeat_cut)

    plt.title("BEAM 1")
    plt.plot(range(len(delta_meas_mux_b1)),delta_meas_mux_b1, label="mux")
    plt.plot(range(len(delta_meas_muy_b1)),delta_meas_muy_b1, label="muy")
    plt.legend()
    plt.savefig("./figures/mu_b1.pdf")
    plt.show()

    plt.title("BEAM 2")
    plt.plot(range(len(delta_meas_mux_b2)),delta_meas_mux_b2, label="mux")
    plt.plot(range(len(delta_meas_muy_b2)),delta_meas_muy_b2, label="muy")
    plt.legend()
    plt.savefig("./figures/mu_b2.pdf")
    plt.show()

    sample = delta_meas_mux_b1, delta_meas_muy_b1, delta_meas_mux_b2, delta_meas_muy_b2

    np.save(f'./measurements/meas_input_{measurement_name}.npy', np.array(sample, dtype=object))
    delta_meas_mux_b1, delta_meas_muy_b1, delta_meas_mux_b2, delta_meas_muy_b2 = np.load(f'./measurements/meas_input_{measurement_name}.npy', allow_pickle=True)
    meas_input = np.concatenate([delta_meas_mux_b1, delta_meas_muy_b1, delta_meas_mux_b2, delta_meas_muy_b2])

    set_name = "./measurements/100%triplet_ip1_100%ip5_2592.npy"
    noise = 1e-3

    input_data, output_data = load_data(set_name, noise)

    example_input = estimator.predict(input_data)[1]
    example_errors = output_data[1]

    plot_example_test_errors(example_input, example_errors)
    
    pred_output = estimator.predict([meas_input])[0]

    plot_example_errors(pred_output)

    print(input_data[0])
    print(meas_input)

    pred_output = -1*pred_output # For corrections
    
    error_tfs = save_np_errors_tfs(pred_output, "pred_triplet_inv_err.tfs")

    tfs_to_local_corr(error_tfs)
    q2_k1l_values, q1_left_k1l, q1_right_k1l, q3_left_k1l, q3_right_k1l = compute_corrections(error_tfs.set_index("NAME"))
    print("KNOB VALUES: ", q2_k1l_values, q1_left_k1l, q1_right_k1l, q3_left_k1l, q3_right_k1l)


def plot_example_errors(pred_output):

    x = [idx for idx, error in enumerate(pred_output)]

    plt.bar(x, 1e4*pred_output, label="Pred")

    plt.title(f"Triplet Errors Predictions For all IPs")
    plt.xlabel(r"MQ [#]")
    plt.ylabel(r"Absolute Error: $\Delta knl$ [1E-4]")
    plt.legend()
    plt.savefig("./figures/pred_errors.pdf")
    plt.show()
    plt.clf()

def plot_example_test_errors(pred_output, simulated_errors):

    x = [idx for idx, error in enumerate(pred_output)]

    plt.bar(x, 1e4*pred_output, label="Pred", alpha=0.5)
    plt.bar(x, 1e4*simulated_errors, label="Simu", alpha=0.5)

    plt.title(f"Triplet Errors Predictions For all IPs")
    plt.xlabel(r"MQ [#]")
    plt.ylabel(r"Absolute Error: $\Delta knl$ [1E-4]")
    plt.legend()
    plt.savefig("./figures/pred_errors.pdf")
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
    recons_df.K1L = np_errors #[:-4] # No mqt knob
    recons_df.NAME = names[:32] #Only triplets
    
    for beam, error_tfs_model in enumerate([error_tfs_model_b2]):
        for i in range(len(error_tfs_model)):
            # check if the name is in recons_df
            if error_tfs_model.loc[i, 'NAME'] in list(recons_df['NAME']):
                error_tfs_model.loc[i, 'K1L'] = recons_df.loc[recons_df['NAME'] == error_tfs_model.loc[i, 'NAME']].values[0][1]
    
    error_tfs_model_b2 = error_tfs_model_b2.loc[error_tfs_model_b2['K1L']!=0]
    tfs.writer.write_tfs(tfs_file_path=f"./corrections/{filename}", data_frame=error_tfs_model_b2)

    return error_tfs_model_b2

def tfs_to_local_corr(tfs_errors):
    tfs_errors.index = tfs_errors.NAME

    Q2_LEN = 5.5
    Q1Q3_LEN = 6.37

    for IP_n in np.array([1,2,5,8]):
        with open("./corrections/IP_corrections_template.txt".format(IP_n), 'r') as template:
            template_str = template.read()
        t = Template(template_str)
        q2a_q2b_left = '{:0.2e}'.format(2.6*(tfs_errors.loc["MQXB.A2L{}".format(IP_n),"K1L"] + tfs_errors.loc["MQXB.B2L{}".format(IP_n),"K1L"])/2/Q2_LEN)
        q2a_q2b_right = '{:0.2e}'.format(2.6*(tfs_errors.loc["MQXB.A2R{}".format(IP_n),"K1L"] + tfs_errors.loc["MQXB.B2R{}".format(IP_n),"K1L"])/2/Q2_LEN)
        Q1L = '{:0.2e}'.format(2.6*tfs_errors.loc["MQXA.1L{}".format(IP_n),"K1L"]/Q1Q3_LEN)
        Q1R = '{:0.2e}'.format(2.6*tfs_errors.loc["MQXA.1R{}".format(IP_n),"K1L"]/Q1Q3_LEN)
        Q3L = '{:0.2e}'.format(2.6*tfs_errors.loc["MQXA.3L{}".format(IP_n),"K1L"]/Q1Q3_LEN)
        Q3R = '{:0.2e}'.format(2.6*tfs_errors.loc["MQXA.3R{}".format(IP_n),"K1L"]/Q1Q3_LEN)
        content = t.substitute(IPn = str(IP_n), Q1L=Q1L, Q1R=Q1R,
                                      Q3L=Q3L, Q3R=Q3R, 
                                      Q2L= q2a_q2b_left,
                                      Q2R=q2a_q2b_right)
        with open("./corrections/v1_RF_IP{}_corrections_predicted.madx".format(IP_n), "w") as f:
            f.write(content)


def compute_corrections(tfs_file):
    Q2_GROUPS = [["MQXB.B2L2", "MQXB.A2L2"], ["MQXB.A2R2","MQXB.B2R2"],["MQXB.B2L5","MQXB.A2L5"], ["MQXB.A2R5","MQXB.B2R5"], ["MQXB.B2L8","MQXB.A2L8"], ["MQXB.A2R8", "MQXB.B2R8"], ["MQXB.B2L1", "MQXB.A2L1"], ["MQXB.A2R1", "MQXB.B2R1"]]
    Q1_LEFT = ["MQXA.1L2", "MQXA.1L5", "MQXA.1L8", "MQXA.1L1"]
    Q1_RIGHT = ["MQXA.1R2", "MQXA.1R5", "MQXA.1R8", "MQXA.1R1"]
    Q3_LEFT = ["MQXA.3L2", "MQXA.3L5", "MQXA.3L8", "MQXA.3L1"]
    Q3_RIGHT = ["MQXA.3R2", "MQXA.3R5", "MQXA.3R8", "MQXA.3R1"]

    q2_group_values = []
    q1_left_values = []
    q1_right_values = []
    q3_left_values = []
    q3_right_values = []
    
    for pair in Q2_GROUPS:
        q2_group_values.append((tfs_file.loc[pair[0], "K1L"] + tfs_file.loc[pair[1], "K1L"]) / 2)
    for name in Q1_LEFT:
        q1_left_values.append(tfs_file.loc[name, "K1L"])
    for name in Q1_RIGHT:
        q1_right_values.append(tfs_file.loc[name, "K1L"])
    for name in Q3_LEFT:
        q3_left_values.append(tfs_file.loc[name, "K1L"])
    for name in Q3_RIGHT:
        q3_right_values.append(tfs_file.loc[name, "K1L"])

    return np.array(q2_group_values), np.array(q1_left_values), np.array(q1_right_values), np.array(q3_left_values), np.array(q3_right_values)

if __name__ == "__main__":
    main()
 # %%