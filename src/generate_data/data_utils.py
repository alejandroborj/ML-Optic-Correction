import numpy as np
import pandas as pd
import tfs
from pathlib import Path


def load_data(set_name, noise):
    #Function that inputs the .npy file and returns the data in a readable format for the algoritms
    all_samples = np.load('./data/{}.npy'.format(set_name), allow_pickle=True)
    '''
    idxs = []
    for idx, sample in enumerate(all_samples):
        mqt_err_b1, mqt_err_b2 = sample[-2], sample[-1]
        if len(mqt_err_b1) != 2 or len(mqt_err_b2) != 2:
            idxs.append(idx)
    print(idxs)
    all_samples = np.delete(all_samples, idxs, axis=0) # Delete elements where mqt errors failed
    np.save('./data/{}.npy'.format(set_name), arr=all_samples , allow_pickle=True) # Save with correct dim
    '''
    
    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2, n_disp_b1, n_disp_b2, \
                triplet_errors, arc_errors_b1, arc_errors_b2, \
                mqt_errors_b1, mqt_errors_b2 = all_samples.T
    
    # select features for input
    # Optionally: add noise to simulated optics functions
    #print("Not Noise", n_disp_b1[1])
    #print("Not Noise", delta_mux_b1[1])
    n_disp_b1 = [add_dispersion_noise(n_disp, noise) for n_disp in n_disp_b1]  
    n_disp_b1 = [add_dispersion_noise(n_disp, noise) for n_disp in n_disp_b2]  

    delta_mux_b1 = [add_phase_noise(delta_mu, 25, noise) for delta_mu in delta_mux_b1]
    delta_muy_b1 = [add_phase_noise(delta_mu, 25, noise) for delta_mu in delta_muy_b1]
    delta_mux_b2 = [add_phase_noise(delta_mu, 25, noise) for delta_mu in delta_mux_b2]
    delta_muy_b2 = [add_phase_noise(delta_mu, 25, noise) for delta_mu in delta_muy_b2]

    #print("Noise", n_disp_b1[1])
    #print("Noise", delta_mux_b1[1])

    input_data = np.concatenate((np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
        np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
        np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
        np.vstack(delta_mux_b2), np.vstack(delta_muy_b2), \
        np.vstack(n_disp_b1), np.vstack(n_disp_b2)
        ), axis=1)
    # select targets for output
    
    output_data = np.concatenate((np.vstack(triplet_errors), np.vstack(arc_errors_b1),\
                                   np.vstack(arc_errors_b2), np.vstack(mqt_errors_b1), \
                                    np.vstack(mqt_errors_b2)), axis=1)
    
    return input_data, output_data


def merge_data(data_path, noise):
    #Takes folder path for all different data files and merges them
    input_data, output_data = [], []
    pathlist = Path(data_path).glob('**/*.npy')
    file_names = [str(path).split('/')[-1][:-4] for path in pathlist][:2]

    for file_name in file_names:
        aux_input, aux_output = load_data(file_name, noise)
        input_data.append(aux_input)
        output_data.append(aux_output)

    return np.concatenate(input_data), np.concatenate(output_data)


def obtain_errors(input_data, output_data, estimator, NORMALIZE=False):
    # Function that gives the predicted and real values of errors for a numpy array imputs
    pred_data = estimator.predict(input_data)
    if NORMALIZE==True:
        pred_data = normalize_errors(pred_data)
        output_data = normalize_errors(output_data)

    pred_triplet = np.hstack(pred_data[:,:32])
    true_triplet = np.hstack(output_data[:,:32])

    pred_arc = np.hstack(pred_data[:,32:1248])
    true_arc = np.hstack(output_data[:,32:1248])

    pred_mqt = np.hstack(pred_data[:,-4:])
    true_mqt = np.hstack(output_data[:,-4:])

    return pred_triplet, true_triplet, pred_arc, true_arc, pred_mqt, true_mqt

def normalize_errors(data):
    # Function with error values as input and outputs the results normalized by the nominal value
    with open("./data_analysis/mq_names.txt", "r") as f:
        lines = f.readlines()
        names = [name.replace("\n", "").upper().split(':')[0] for name in lines][:-4]

    B1_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_elements_30.dat").set_index("NAME")
    B2_ELEMENTS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b2_nominal_elements_30.dat").set_index("NAME")
    ELEMENTS_MDL_TFS = tfs.frame.concat([B1_ELEMENTS_MDL_TFS, B2_ELEMENTS_MDL_TFS])
    TRIPLET_NOM_K1 = ELEMENTS_MDL_TFS.loc[names, "K1L"][:64]
    TRIPLET_NOM_K1 = TRIPLET_NOM_K1[TRIPLET_NOM_K1.index.duplicated(keep="first")]

    ARC_NOM_K1 = ELEMENTS_MDL_TFS.loc[names, "K1L"][64:]

    nom_k1 = tfs.frame.concat([TRIPLET_NOM_K1, ARC_NOM_K1])
    nom_k1 = np.append(nom_k1, [1,1,1,1]) # Not the cleanest function ever
    
    for i, sample in enumerate(data):
        data[i] = (sample)/nom_k1
    return data

def save_np_errors_tfs(np_errors):
    #Function that takes np errors and outputs .tfs file with all error values
    with open("./data_analysis/mq_names.txt", "r") as f:
        lines = f.readlines()
        names = [name.replace("\n", "") for name in lines]

    df = pd.DataFrame(columns=["names","k1l"])
    df.k1l = np_errors
    df.names = names
    df = tfs.frame.TfsDataFrame(df)
    tfs.writer.write_tfs(tfs_file_path="./data_analysis/mag_errors.tfs", data_frame=df)

def add_phase_noise(phase_errors, betas, expected_noise):
    #Add noise to generated phase advance deviations as estimated from measurements
    my_phase_errors = np.array(phase_errors)
    noises = np.random.standard_normal(phase_errors.shape)
    betas_fact = (expected_noise * (171**0.5) / (betas**0.5))
    noise_with_beta_fact = np.multiply(noises, betas_fact)
    phase_errors_with_noise = my_phase_errors + noise_with_beta_fact
    #print("a", noise_with_beta_fact[0])
    #print("b ", len(my_phase_errors[0]))
    #print("c ", len(phase_errors_with_noise[0]))
    return phase_errors_with_noise


def add_dispersion_noise(disp_errors, noise):
    # Add noise to generated dispersion deviations as estimated from measurements in 2018
    my_disp_errors = np.array(disp_errors)
    noises = noise * np.random.noncentral_chisquare(4, 0.0035, disp_errors.shape)
    disp_errors_with_noise = my_disp_errors + noises
    
    return disp_errors_with_noise