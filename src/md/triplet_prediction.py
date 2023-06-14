#%%
from measurement_to_input import get_delta_and_meas_phase
from measurement_to_input import get_beta_beating_from_measurement
from measurement_to_input import get_phase_diff_from_twiss
import numpy as np
import tfs
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from string import Template

# measurement_path => Results folder
# model_path => Nominal tfs
# twiss_perturbet => Simulated error twiss

QX = 62.31
QY = 60.32

def main():
    measurement_name = "test_pred_triplet"
    measurement_path_b1 = "./measurements/02-38-46_import_4_files_beam1/"
    measurement_path_b2 = "./measurements/02-38-46_import_b2_3files_12percent_30cm_beforecorrection/"
    twiss_b2 = tfs.read_tfs("./data_analysis/b2_nominal_monitors.dat").set_index("NAME")
    twiss_b1 = tfs.read_tfs("./data_analysis/b1_nominal_monitors.dat").set_index("NAME")

    phasecut_x = 0.4
    phasecut_y = 0.4
    bbeat_cut = 150 #No bbeat_cut

    # Checking the measured Beta beat
    betabeatx_b1, betabeaty_b1 = get_beta_beating_from_measurement(measurement_path_b1)
    betabeatx_b2, betabeaty_b2 = get_beta_beating_from_measurement(measurement_path_b2)

    plt.title("BEAM 1")
    plt.plot(range(len(betabeatx_b1)),betabeatx_b1, label="bbeatx")
    plt.plot(range(len(betabeaty_b1)),betabeaty_b1, label="bbeaty")
    plt.legend()
    plt.show()

    plt.title("BEAM 2")
    plt.plot(range(len(betabeatx_b2)),betabeatx_b2, label="bbeatx")
    plt.plot(range(len(betabeaty_b2)),betabeaty_b2, label="bbeaty")
    plt.legend()
    plt.show()

    # Taking both inputs
    delta_meas_mux_b1, delta_meas_muy_b1, meas_mux_b1, meas_muy_b1 = get_delta_and_meas_phase(measurement_path_b1, twiss_b1, 
                                                              phasecut_x, phasecut_y, bbeat_cut)
    delta_meas_mux_b2, delta_meas_muy_b2, meas_mux_b2, meas_muy_b2 = get_delta_and_meas_phase(measurement_path_b2, twiss_b2, 
                                                              phasecut_x, phasecut_y, bbeat_cut)

    sample = delta_meas_mux_b1, delta_meas_muy_b1, delta_meas_mux_b2, delta_meas_muy_b2

    np.save(f'./measurements/meas_input_{measurement_name}.npy', np.array(sample, dtype=object))
    delta_meas_mux_b1, delta_meas_muy_b1, delta_meas_mux_b2, delta_meas_muy_b2 = np.load(f'./measurements/meas_input_{measurement_name}.npy', allow_pickle=True)
    meas_input = np.concatenate([delta_meas_mux_b1, delta_meas_muy_b1, delta_meas_mux_b2, delta_meas_muy_b2])
 
    estimator = joblib.load("./md_models/triplets_phases_b1_corr_b2_corr_0.0001.pkl")
    
    pred_output = estimator.predict([meas_input])[0]

    plot_example_errors(pred_output)
    error_tfs = save_np_errors_tfs(pred_output, "pred_triplet_err.tfs")
    tfs_to_local_corr(error_tfs)


def plot_example_errors(pred_output):
    pred_arc, pred_mqt = np.hstack(pred_output[:-4]), np.hstack(pred_output[-4:])
    
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
    recons_df.K1L = np_errors[:-4] # No mqt knob
    recons_df.NAME = names[:32] #Only triplets
    
    for beam, error_tfs_model in enumerate([error_tfs_model_b2]):
        for i in range(len(error_tfs_model)):
            # check if the name is in recons_df
            if error_tfs_model.loc[i, 'NAME'] in list(recons_df['NAME']):
                error_tfs_model.loc[i, 'K1L'] = recons_df.loc[recons_df['NAME'] == error_tfs_model.loc[i, 'NAME']].values[0][1]
    
    error_tfs_model_b2 = error_tfs_model_b2.loc[error_tfs_model_b2['K1L']!=0]
    tfs.writer.write_tfs(tfs_file_path=f"./measurements/{filename}", data_frame=error_tfs_model_b2)

    return error_tfs_model_b2

def tfs_to_local_corr(tfs_errors):
    tfs_errors.index = tfs_errors.NAME

    Q2_LEN = 5.5
    Q1Q3_LEN = 6.37

    for IP_n in np.array([1,2,5,8]):
        with open("./corrections/IP_corrections_template.txt".format(IP_n), 'r') as template:
            template_str = template.read()
        t = Template(template_str)
        q2a_q2b_left = (tfs_errors.loc["MQXB.A2L{}".format(IP_n),"K1L"] + tfs_errors.loc["MQXB.B2L{}".format(IP_n),"K1L"])/2
        q2a_q2b_right = (tfs_errors.loc["MQXB.A2R{}".format(IP_n),"K1L"] + tfs_errors.loc["MQXB.B2R{}".format(IP_n),"K1L"])/2
        content = t.substitute(IPn = str(IP_n), Q1L= str(tfs_errors.loc["MQXA.1L{}".format(IP_n),"K1L"]/Q1Q3_LEN), Q1R= str(tfs_errors.loc["MQXA.1R{}".format(IP_n),"K1L"]/Q1Q3_LEN),
                                      Q3L= str(tfs_errors.loc["MQXA.3L{}".format(IP_n),"K1L"]/Q1Q3_LEN), Q3R= str(tfs_errors.loc["MQXA.3R{}".format(IP_n),"K1L"]/Q1Q3_LEN), 
                                      Q2L= str(q2a_q2b_left/Q2_LEN), Q2R= str(q2a_q2b_right/Q2_LEN))
        with open("./corrections/v1_RF_IP{}_corrections_predicted.madx".format(IP_n), "w") as f:
            f.write(content)

if __name__ == "__main__":
    main()
 # %%