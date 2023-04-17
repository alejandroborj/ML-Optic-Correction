
from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
# Beta-Beat.src to import tfs_pandas
# Using python 3 --> import tfs
sys.path.append('./Beta-Beat.src/')
import tfs
# from tfs_files import tfs_pandas
import tfs_files.tfs_file_writer as tfs_writer
import joblib
import measurement_to_input
from string import Template
from subprocess import call

B1_MONITORS_TRAINING = tfs.read_tfs("./b1_nominal_monitors.dat").set_index("NAME")
B2_MONITORS_TRAINING = tfs.read_tfs("./b2_nominal_monitors.dat").set_index("NAME")


predictor_file = "./onlyPhaseAdv_100k_FOREST_predict_triplets.pkl"

measurement_b1 = "./b1_30cm_beforeKmod"
measurement_b2 = "./b2_3files_12percent_30cm_beforecorrection"

Q2_GROUPS = [["MQXB.B2L2", "MQXB.A2L2"], ["MQXB.A2R2","MQXB.B2R2"],["MQXB.B2L5","MQXB.A2L5"], ["MQXB.A2R5","MQXB.B2R5"], ["MQXB.B2L8","MQXB.A2L8"], ["MQXB.A2R8", "MQXB.B2R8"], ["MQXB.B2L1", "MQXB.A2L1"], ["MQXB.A2R1", "MQXB.B2R1"]]
Q1_LEFT = ["MQXA.1L2", "MQXA.1L5", "MQXA.1L8", "MQXA.1L1"]
Q1_RIGHT = ["MQXA.1R2", "MQXA.1R5", "MQXA.1R8", "MQXA.1R1"]
Q3_LEFT = ["MQXA.3L2", "MQXA.3L5", "MQXA.3L8", "MQXA.3L1"]
Q3_RIGHT = ["MQXA.3R2", "MQXA.3R5", "MQXA.3R8", "MQXA.3R1"]

Q2_LEN = 5.5
Q1Q3_LEN = 6.37


def main():
    b1_mdl_twiss = tfs.read("./b1_30cm_beforeKmod/twiss.dat").set_index("NAME")
    b2_mdl_twiss = tfs.read("./b2_3files_12percent_30cm_beforecorrection/twiss.dat").set_index("NAME")

    # b1_mdl_twiss = B1_MONITORS_TRAINING
    # b2_mdl_twiss = B2_MONITORS_TRAINING

    model = joblib.load(predictor_file)

    b1_delta_mux_interpolated, b1_delta_muy_interpolated, b1_meas_mux, b1_meas_muy= measurement_to_input.get_delta_and_meas_phase(
                        measurement_b1, 
                        b1_mdl_twiss, 
                        phasecut_x=0.3, phasecut_y=0.4, bbeat_cut=150)
    b2_delta_mux_interpolated, b2_delta_muy_interpolated, b2_meas_mux, b2_meas_muy = measurement_to_input.get_delta_and_meas_phase(
                        measurement_b2, 
                        b2_mdl_twiss, 
                        phasecut_x=0.4, phasecut_y=0.4, bbeat_cut=150)
    input_from_measurement = np.concatenate((np.array(b1_delta_mux_interpolated), np.array(b1_delta_muy_interpolated), \
                np.array(b2_delta_mux_interpolated), np.array(b2_delta_muy_interpolated)), axis=0)

    predicted_triplets = model.predict(input_from_measurement.reshape(1, -1))
    predicted_triplets = predicted_triplets[0] 

    #write error file
    triplets_tfs = tfs.read("./predicted_triplets_RF.tfs").set_index("NAME")
    predicted_errtable_triplets = triplets_tfs.copy()
    predicted_errtable_triplets.loc[:,"K1L"] = predicted_triplets

    #save predicted triplet errors as tfs error table
    current_prediction_fn = 'prediction_from_uncorrected_meas'
    tfs.write(os.path.join("./", '{}.tfs'.format(current_prediction_fn)), predicted_errtable_triplets, save_index=True)
    
    # replace values in correctioins template file
    for IP_n in np.array([1,2,5,8]):
        with open("./IP_corrections_template.dat".format(IP_n), 'r') as template:
            template_str = template.read()
        t = Template(template_str)
        q2a_q2b_left = (predicted_errtable_triplets.loc["MQXB.A2L{}".format(IP_n),"K1L"] + predicted_errtable_triplets.loc["MQXB.B2L{}".format(IP_n),"K1L"])/2
        q2a_q2b_right = (predicted_errtable_triplets.loc["MQXB.A2R{}".format(IP_n),"K1L"] + predicted_errtable_triplets.loc["MQXB.B2R{}".format(IP_n),"K1L"])/2
        content = t.substitute(IPn = str(IP_n), Q1L= str(predicted_errtable_triplets.loc["MQXA.1L{}".format(IP_n),"K1L"]/Q1Q3_LEN), 
                                      Q1R= str(predicted_errtable_triplets.loc["MQXA.1R{}".format(IP_n),"K1L"]/Q1Q3_LEN),
                                      Q3L= str(predicted_errtable_triplets.loc["MQXA.3L{}".format(IP_n),"K1L"]/Q1Q3_LEN), 
                                      Q3R= str(predicted_errtable_triplets.loc["MQXA.3R{}".format(IP_n),"K1L"]/Q1Q3_LEN), 
                                      Q2L= str(q2a_q2b_left/Q2_LEN), 
                                      Q2R= str(q2a_q2b_right/Q2_LEN))
        with open("./RF_IP{}_corrections_predicted.madx".format(IP_n), "w") as f:
            f.write(content)

    #run madx (job.prediction_b1/b2.madx), NOTE: replace the paths for the error file
    # compare_meas_vs_reconstruction(b1_mdl_twiss, b2_mdl_twiss) 


# compare measurements and simulation using predicted errors
# replace missing values with 0
def compare_meas_vs_reconstruction(b1_mdl_twiss, b2_mdl_twiss):
    b1_bbeatx, b1_bbeaty = measurement_to_input.get_beta_beating_from_measurement(measurement_b1)
    b2_bbeatx, b2_bbeaty = measurement_to_input.get_beta_beating_from_measurement(measurement_b2)

    b1_bbeatx_meas = b1_bbeatx.reindex(b1_mdl_twiss.index, fill_value=0.0)
    b1_bbeaty_meas = b1_bbeaty.reindex(b1_mdl_twiss.index, fill_value=0.0)
    b2_bbeatx_meas = b2_bbeatx.reindex(b2_mdl_twiss.index, fill_value=0.0)
    b2_bbeaty_meas = b2_bbeaty.reindex(b2_mdl_twiss.index, fill_value=0.0)

    reconstructed_twiss_b1 = tfs.read("./b1_predicted_twiss.tfs").set_index("NAME")
    reconstructed_twiss_b2 = tfs.read("./b2_predicted_twiss.tfs").set_index("NAME")

    b1_bbeatx_reco = (reconstructed_twiss_b1.BETX - b1_mdl_twiss.BETX) / b1_mdl_twiss.BETX *100
    b1_bbeaty_reco = (reconstructed_twiss_b1.BETY - b1_mdl_twiss.BETY) / b1_mdl_twiss.BETY * 100

    b2_bbeatx_reco = (reconstructed_twiss_b2.BETX - b2_mdl_twiss.BETX) / b2_mdl_twiss.BETX *100
    b2_bbeaty_reco = (reconstructed_twiss_b2.BETY - b2_mdl_twiss.BETY) / b2_mdl_twiss.BETY *100

    plot_reconstruction(b1_mdl_twiss.BETX.values, b1_bbeatx_meas, b1_bbeatx_reco, "Beam 1, Horizontal")
    plot_reconstruction(b1_mdl_twiss.BETY.values, b1_bbeaty_meas, b1_bbeaty_reco, "Beam 1, Vertical")
    plot_reconstruction(b2_mdl_twiss.BETX.values, b2_bbeatx_meas, b2_bbeatx_reco, "Beam 2, Horizontal")
    plot_reconstruction(b2_mdl_twiss.BETY.values, b2_bbeaty_meas, b2_bbeaty_reco, "Beam 2, Vertical")



def compute_corrections(tfs_file):
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


def print_corrections_for_lsa(triplet_errors_path):
    q2_k1l_values, q1_left_k1l, q1_right_k1l, q3_left_k1l, q3_right_k1l = compute_corrections(tfs.read_tfs(triplet_errors_path).set_index("NAME"))
    
    corrections_with_opposite_sign_q2 = -1 * q2_k1l_values/Q2_LEN
    corrections_with_opposite_sign_q3 = -1 * q3_right_k1l/Q1Q3_LEN
    
    # K1L is predicted, the predicted values are to be devided by the length
    # All field errors are specified as the integrated value, Kds of the field components along the magnet axis in m-i
    print("Q2: L2, R2, L5, R5, L8, R8, L1, R1")
    print(corrections_with_opposite_sign_q2)
    print("Q3: L2, R2, L5, R5, L8, R8, L1, R1")
    print(corrections_with_opposite_sign_q3)


def plot_reconstruction(betas_mdl, meas, reco, title_str):
    plt.plot(range(len(betas_mdl)), meas, label="Measurement")
    plt.plot(range(len(betas_mdl)), reco, label="Reconstruction")
    plt.title(title_str)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()