#%%
# cat generate_data.py | ssh aborjess@cs-ccr-dev3 python - RUNNING FILE IN SERVER

from __future__ import print_function
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
# Beta-Beat.src to import tfs_pandas
# Using python 3 --> import tfs
sys.path.append('/afs/cern.ch/work/e/efol/public/Beta-Beat.src/')
#from Utilities import tfs_pandas
import tfs
#from omc3 import madx_wrapper Old Version
import cpymad.madx

from multiprocessing import Pool
from collections import OrderedDict
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

import pandas as pd
import time

# mad-x scripts and model files
OPTICS_40CM_2016 = './modifiers.madx'
NOMINAL_TWISS_TEMPL = './job.nominal2016.madx'
MAGNETS_TEMPLATE_B1 = './job.magneterrors_b1.madx'
MAGNETS_TEMPLATE_B2 = './job.magneterrors_b2.madx'

B1_MONITORS_MDL_TFS = tfs.read_tfs("./b1_nominal_monitors.dat").set_index("NAME")
B2_MONITORS_MDL_TFS = tfs.read_tfs("./b2_nominal_monitors.dat").set_index("NAME")

# tunes for phase advance computation
QX = 64.28
QY = 59.31

# DIY verbose turn off
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def main():
    #global madx # Global variable
    set_name = "training_set"
    num_sim = 5
    all_samples = []
    valid_samples = []
    GENERATE_DATA = True
    start = time.time()

    print("\nStart creating dataset\n")
    if GENERATE_DATA==True:

        # Run simulations in parallel
        pool = Pool(processes=5) # Max 7 # Max Elena30
        all_samples = pool.map(create_sample, range(num_sim))
        pool.close()
        pool.join()
        
        #all_samples = [create_sample(i) for i in range(num_sim)] # No parallel computing

        # if twiss failed, sample will be None --> filter, check number of samples
        # usually, around ~2% of simulations fail due to generated error distribition
        for sample in all_samples:
            if sample is not None:
                valid_samples.append(sample)
        print("Number of generated samples: {}".format(len(valid_samples)))
        np.save('./{}.npy'.format(set_name), np.array(valid_samples, dtype=object))
        
    stop = time.time()
    print('Execution time (s): ', stop-start)
    # Train on generated data
    train_model(set_name)


# example of reading the data, training ML model and validate results
def train_model(set_name):
    input_data, output_data = load_data(set_name)
    # split into train and test
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        input_data, output_data, test_size=0.2, random_state=None)
    
    # create and fit a regression model
    ridge = linear_model.Ridge(tol=1e-50, alpha=1e-03) #normalize=false
    estimator = BaggingRegressor(base_estimator=ridge, n_estimators=10, \
        max_samples=0.9, max_features=1.0, n_jobs=16, verbose=3)
    estimator.fit(train_inputs, train_outputs)
    # Optionally: save fitted model or load al    enablePrint()ready trained model
    joblib.dump(estimator, 'estimator.pkl')
    #estimator = joblib.load('estimator.pkl')

    # Check scores: explained variance and MAE
    training_score = estimator.score(train_inputs, train_outputs)
    test_score = estimator.score(test_inputs, test_outputs)
    prediction_train = estimator.predict(train_inputs)
    mae_train = mean_absolute_error(train_outputs, prediction_train)
    prediction_test = estimator.predict(test_inputs)
    mae_test = mean_absolute_error(test_outputs, prediction_test)
    print("Training: R2 = {0}, MAE = {1}".format(training_score, mae_train))
    print("Test: R2 = {0}, MAE = {1}".format(test_score, mae_test))


def load_data(set_name):
    #Function that inputs the .npy file and returns the data in a readable format for the algoritms
    all_samples = np.load('./{}.npy'.format(set_name), allow_pickle=True)
    delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, \
        delta_beta_star_y_b2, delta_mux_b1, delta_muy_b1, delta_mux_b2, \
            delta_muy_b2, n_disp_b1, n_disp_b2, \
                triplet_errors = all_samples.T
    madx = cpymad.madx.Madx()
    # select features for input
    # Optionally: add noise to simulated optics functions
    input_data = np.concatenate(( \
        np.vstack(delta_beta_star_x_b1), np.vstack(delta_beta_star_y_b1), \
        np.vstack(delta_beta_star_x_b2), np.vstack(delta_beta_star_y_b2), \
        np.vstack(delta_mux_b1), np.vstack(delta_muy_b1), \
        np.vstack(delta_mux_b2), np.vstack(delta_muy_b2), \
        np.vstack(n_disp_b1), np.vstack(n_disp_b2), \
        ), axis=1)
    # select targets for output
    output_data = np.vstack(triplet_errors)#np.concatenate(( \np.vstack(triplet_errors), np.vstack(arc_errors_b1), \np.vstack(arc_errors_b2), np.vstack(mqt_errors_b1), \np.vstack(mqt_errors_b2),), axis=1)
    return input_data, output_data


# Add noise to generated phase advance deviations as estimated from measurements
def add_phase_noise(phase_errors, betas, expected_noise):
    my_phase_errors = np.array(phase_errors)
    noises = np.random.standard_normal(phase_errors.shape)
    betas_fact = (expected_noise * np.sqrt(171) / np.sqrt(betas))
    noise_with_beta_fact = np.multiply(noises, betas_fact)
    phase_errors_with_noise = my_phase_errors + noise_with_beta_fact
    return phase_errors_with_noise


# Read all generated error tables (as tfs), return k1l absolute for sample output
def get_errors_from_sim(common_errors):#, b1_errors_file_path, b2_errors_file_path, b1_tw_before_match, b1_tw_after_match, b2_tw_before_match, b2_tw_after_match):
    # Triplet errors 
    #triplet_errors_tfs = tfs.read_tfs(common_errors).set_index("NAME")
    triplet_errors = common_errors.k1l

    return np.array(triplet_errors)#, np.array(arc_errors_b1), np.array(arc_errors_b2), np.array(mqt_errors_b1), np.array(mqt_errors_b2)


def create_nominal_twiss():
    madx = cpymad.madx.Madx()
    blockPrint()
    with open(NOMINAL_TWISS_TEMPL, 'r') as template:
        template_str = template.read()
    madx.input(template_str % {"OPTICS": OPTICS_40CM_2016})
    enablePrint()
    madx.quit()


def create_sample(index):
    print("\nDoing index: ", str(index), "\n")

    blockPrint() # Interrupt MADX output messages
    madx = cpymad.madx.Madx()
    sample = None
    np.random.seed(seed=None)
    seed = random.randint(0, 999999999)

    # Run mad-x for b1 and b2
    with open(MAGNETS_TEMPLATE_B1, 'r') as template:
        template_str = template.read() 

    madx.input(template_str % {"INDEX": str(index), \
        "OPTICS": OPTICS_40CM_2016, "SEED": seed})

    #madx.quit()
    #print("Sim1")

    twiss_data_b1 = madx.table.twiss.dframe()
    common_errors = madx.table.cetab.dframe()

    with open(MAGNETS_TEMPLATE_B2, 'r') as template:
        template_str = template.read()
     #Creating another instance of MADX

    madx.input(template_str % {"INDEX": str(index), \
        "OPTICS": OPTICS_40CM_2016, "SEED": seed})

    twiss_data_b2 = madx.table.twiss.dframe()

    #errors_matched_b2 = tfs_to_df(madx.table.errtab)

    delta_beta_star_x_b1, delta_beta_star_y_b1, \
        delta_mux_b1, delta_muy_b1, n_disp_b1 = get_input_for_beam(twiss_data_b1,  B1_MONITORS_MDL_TFS, 1)
    
    delta_beta_star_x_b2, delta_beta_star_y_b2, \
        delta_mux_b2, delta_muy_b2, n_disp_b2 = get_input_for_beam(twiss_data_b2 , B2_MONITORS_MDL_TFS, 2)
    
    triplet_errors= get_errors_from_sim(common_errors)

    # Create a training sample
    sample = delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, delta_beta_star_y_b2, \
        delta_mux_b1, delta_muy_b1, delta_mux_b2, delta_muy_b2, n_disp_b1, n_disp_b2, \
            triplet_errors

    files = glob.glob(f"./magnet_errors/*{index}.tfs")
    for f in files: # Remove files for this simulation
        os.remove(f)

    madx.quit() # Stop all processes and free memory

    enablePrint()

    return sample


# extract input data from generated twiss
def get_input_for_beam(twiss_df, meas_mdl, beam):
    ip_bpms_b1 = ["BPMSW.1L1.B1", "BPMSW.1R1.B1", "BPMSW.1L2.B1", "BPMSW.1R2.B1", "BPMSW.1L5.B1", "BPMSW.1R5.B1", "BPMSW.1L8.B1", "BPMSW.1R8.B1"]
    ip_bpms_b2 = ["BPMSW.1L1.B2", "BPMSW.1R1.B2", "BPMSW.1L2.B2", "BPMSW.1R2.B2", "BPMSW.1L5.B2", "BPMSW.1R5.B2", "BPMSW.1L8.B2", "BPMSW.1R8.B2"]

    #tw_perturbed_elements = tfs.read_tfs(twiss_pert_elements_path).set_index("NAME")
    
    tw_perturbed_elements = twiss_df.set_index("name") 
    # Uppercase and taking the relevant index 
    tw_perturbed_elements.index = [idx.upper()[:-2] for idx in tw_perturbed_elements.index]
    tw_perturbed_elements.columns = [col.upper() for col in tw_perturbed_elements.columns]

    tw_perturbed = tw_perturbed_elements[tw_perturbed_elements.index.isin(meas_mdl.index)]
    ip_bpms = ip_bpms_b1 if beam == 1 else ip_bpms_b2
    
    # phase advance deviations
    phase_adv_x = get_phase_adv(tw_perturbed['MUX'], QX)
    phase_adv_y = get_phase_adv(tw_perturbed['MUY'], QY)
    mdl_ph_adv_x = get_phase_adv(meas_mdl['MUX'], QX)
    mdl_ph_adv_y = get_phase_adv(meas_mdl['MUY'], QY)
    delta_phase_adv_x = phase_adv_x - mdl_ph_adv_x
    delta_phase_adv_y = phase_adv_y - mdl_ph_adv_y


    # beta deviations at bpms around IPs
    delta_beta_star_x = np.array(tw_perturbed.loc[ip_bpms, "BETX"] - meas_mdl.loc[ip_bpms, "BETX"])
    delta_beta_star_y = np.array(tw_perturbed.loc[ip_bpms, "BETY"] - meas_mdl.loc[ip_bpms, "BETY"])

    #normalized dispersion deviation
    n_disp = tw_perturbed['NDX']

    return np.array(delta_beta_star_x), np.array(delta_beta_star_y), np.array(delta_phase_adv_x), np.array(delta_phase_adv_y), np.array(n_disp)


'''def tfs_to_df(madx_table):
    df = pd.DataFrame(np.array([madx_table[col] for col in list(madx_table)]).transpose())
    df.columns = list(madx_table)
    df.transpose
    return df'''


def get_phase_adv(total_phase, tune):
    phase_diff = np.diff(total_phase)
    last_to_first = total_phase[0] - (total_phase[-1] - tune)
    phase_adv = np.append(phase_diff, last_to_first)
    return phase_adv


def get_tot_phase(phase_from_twiss):
    total_phase = (phase_from_twiss - phase_from_twiss[0]) % 1.0
    return np.array(total_phase)


if __name__ == "__main__":
    main()

# %%
