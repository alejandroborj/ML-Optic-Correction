#%%
# UPDATING SCRIPT AND RUNNING IT ON THE SERVER
#--------------------------------------------- 

import numpy as np
import random
import tfs

import time

from madx_jobs import madx_ml_op

from plots import plot_example_betabeat

# mad-x scripts and model files
OPTICS_30CM_2023 = '/afs/cern.ch/user/a/aborjess/work/private/models/LHCB1/B1_30cm_flat/modifiers.madx'
OPTICS_45CM_2023 =  '/afs/cern.ch/eng/acc-models/lhc/2022/operation/optics/R2023a_A45cmC45cmA10mL200cm.madx'

B1_MONITORS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b1_nominal_monitors.dat").set_index("NAME")
B2_MONITORS_MDL_TFS = tfs.read_tfs("./nominal_twiss/b2_nominal_monitors.dat").set_index("NAME")

# tunes for phase advance computation
# Colision optics 2023
QX = 62.31
QY = 60.32

#mdx = madx_ml_op()
#mdx.job_nominal2023()

def main():

    start = time.time()
    set_name = f"10%triplet_100%b2arcs{np.random.randint(0, 99999)}"
    num_sim = 1
    valid_samples = []
    GENERATE_DATA = True

    print("\nStart creating dataset\n")
    if GENERATE_DATA==True:
        all_samples = [create_sample(i) for i in range(num_sim)] # No parallel computing

        # if twiss failed, sample will be None --> filter, check number of samples
        # usually, around ~2% of simulations fail due to generated error distribition
        for sample in all_samples:
            if sample is not None:
                valid_samples.append(sample)
        print("Number of generated samples: {}".format(len(valid_samples)))
        np.save('./data_local_corr_md/{}.npy'.format(set_name), np.array(valid_samples, dtype=object))
    
    stop = time.time()
    print('Execution time (s): ', stop-start)


def create_sample(index):
    sample = None
    print("\nDoing index: ", str(index), "\n")

    np.random.seed(seed=None)
    seed = random.randint(0, 999999999)
    mdx = madx_ml_op()

    # Run mad-x for b1 and b2
    try:
        # BEAM 1
        mdx.job_magneterrors_b1(OPTICS_30CM_2023, str(index), seed)
        b1_tw_before_match = mdx.table.twiss.dframe() # Twiss before match

        mdx.match_tunes_b1()
        b1_tw_after_match = mdx.table.twiss.dframe()# Twiss after match
        common_errors = mdx.table.cetab.dframe() # Errors for both beams, triplet errors
        b1_errors = mdx.table.etabb1.dframe() # Table error for MQ- magnets
        
        #tfs.writer.write_tfs(tfs_file_path=f"b1_correct_example.tfs", data_frame=b1_errors)
        #tfs.writer.write_tfs(tfs_file_path=f"b1_example_twiss.tfs", data_frame=b1_tw_after_match)

        # BEAM 2
        mdx.job_magneterrors_b2(OPTICS_30CM_2023, str(index), seed)

        b2_tw_before_match = mdx.table.twiss.dframe() # Twiss before match

        mdx.match_tunes_b2()

        b2_tw_after_match = mdx.table.twiss.dframe()# Twiss after match
        #twiss_data_b2 = mdx.table.twiss.dframe() # Relevant to training Twiss data
        b2_errors= mdx.table.etabb2.dframe() # Table error for MQX magnets

        delta_beta_star_x_b1, delta_beta_star_y_b1, \
        delta_mux_b1, delta_muy_b1, n_disp_b1, \
            beta_bpm_x_b1, beta_bpm_y_b1 = get_input_for_beam(b1_tw_after_match,  B1_MONITORS_MDL_TFS, 1)
        
        delta_beta_star_x_b2, delta_beta_star_y_b2, \
            delta_mux_b2, delta_muy_b2, n_disp_b2, \
            beta_bpm_x_b2, beta_bpm_y_b2  = get_input_for_beam(b2_tw_after_match , B2_MONITORS_MDL_TFS, 2)

        # Reading errors from MADX tables
        triplet_errors, arc_errors_b1, arc_errors_b2, mqt_errors_b1, mqt_errors_b2 = \
        get_errors_from_sim(common_errors, b1_errors, b2_errors, \
                b1_tw_before_match, b1_tw_after_match, b2_tw_before_match, b2_tw_after_match)
        
        # Create a training sample
        sample = delta_beta_star_x_b1, delta_beta_star_y_b1, delta_beta_star_x_b2, delta_beta_star_y_b2, \
            delta_mux_b1, delta_muy_b1, delta_mux_b2, delta_muy_b2, n_disp_b1, n_disp_b2, \
            beta_bpm_x_b1, beta_bpm_y_b1, beta_bpm_x_b2, beta_bpm_y_b2, \
            triplet_errors, arc_errors_b1, arc_errors_b2, mqt_errors_b1, mqt_errors_b2

        # Sometimes matching fails, this is a half measure        
        if len(mqt_errors_b1) != 2 or len(mqt_errors_b2) != 2:
            sample = None
        
        mdx.quit()

    except:
        sample = None
        print("TWISS Failed")
    #print("Shapes")
    #for data in sample:
    #    print(data.shape)

    return sample

# Read all generated error tables (as tfs), return k1l absolute for sample output
def get_errors_from_sim(common_errors, b1_errors, b2_errors, b1_tw_before_match,\
                            b1_tw_after_match, b2_tw_before_match, b2_tw_after_match):
    # Triplet errors  
    triplet_errors = common_errors.k1l
    tfs_error_file_b1 = b1_errors.set_index("name", drop=False) 
    # replace K1L of MQT in original table (0) with matched - unmatched difference, per knob (2 different values for all MQTs)
    b1_unmatched = b1_tw_before_match.set_index("name", drop=False)
    b1_matched = b1_tw_after_match.set_index("name", drop=False)

    mqt_names_b1 = [name for name in b1_unmatched.index.values if "mqt." in name]

    mqt_errors_b1 = np.unique(np.array(b1_matched.loc[mqt_names_b1, "k1l"].values - \
        b1_unmatched.loc[mqt_names_b1, "k1l"].values, dtype=float).round(decimals=8))
    mqt_errors_b1 = [k for k in mqt_errors_b1 if k != 0]
    
    # The rest of magnets errors
    arc_magnets_names_b1 = [name for name in tfs_error_file_b1.index.values if ("mqt." not in name and "mqx" not in name)]
    arc_errors_b1 = tfs_error_file_b1.loc[arc_magnets_names_b1, "k1l"]
    tfs_error_file_b2 = b2_errors.set_index("name", drop=False)
    b2_unmatched = b2_tw_before_match.set_index("name", drop=False)
    b2_matched = b2_tw_after_match.set_index("name", drop=False)
    mqt_names_b2 = [name for name in b2_unmatched.index.values if "mqt." in name]
    mqt_errors_b2 = np.unique(np.array(b2_matched.loc[mqt_names_b2, "k1l"].values - \
        b2_unmatched.loc[mqt_names_b2, "k1l"].values, dtype=float).round(decimals=8))
    
    mqt_errors_b2 = [k for k in mqt_errors_b2 if k != 0]
    arc_magnets_names_b2 = [name for name in tfs_error_file_b2.index.values if ("mqt." not in name and "mqx" not in name)]
    arc_errors_b2 = tfs_error_file_b2.loc[arc_magnets_names_b2, "k1l"]

    return np.array(triplet_errors), np.array(arc_errors_b1), \
        np.array(arc_errors_b2), np.array(mqt_errors_b1), np.array(mqt_errors_b2)


# Extract input data from generated twiss
def get_input_for_beam(twiss_df, meas_mdl, beam):
    ip_bpms_b1 = ["BPMSW.1L1.B1", "BPMSW.1R1.B1", "BPMSW.1L2.B1", "BPMSW.1R2.B1", "BPMSW.1L5.B1", "BPMSW.1R5.B1", "BPMSW.1L8.B1", "BPMSW.1R8.B1"]
    ip_bpms_b2 = ["BPMSW.1L1.B2", "BPMSW.1R1.B2", "BPMSW.1L2.B2", "BPMSW.1R2.B2", "BPMSW.1L5.B2", "BPMSW.1R5.B2", "BPMSW.1L8.B2", "BPMSW.1R8.B2"]

    #tw_perturbed_elements = tfs.read_tfs(twiss_pert_elements_path).set_index("NAME")
    
    tw_perturbed_elements = twiss_df.set_index("name") 
    # Uppercase and taking the relevant index 
    tw_perturbed_elements.index = [(idx.upper()).split(':')[0] for idx in tw_perturbed_elements.index]
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
    
    # Adding betas at bpms for phase advance measurement in order to compute the noise

    beta_bpms_x = np.array(tw_perturbed["BETX"])
    beta_bpms_y = np.array(tw_perturbed["BETY"])
    
    #print("Beta Beat", meas_mdl, tw_perturbed)
    #plot_example_betabeat(meas_mdl, tw_perturbed, beam)
    
    return np.array(delta_beta_star_x), np.array(delta_beta_star_y), \
        np.array(delta_phase_adv_x), np.array(delta_phase_adv_y), np.array(n_disp),\
        np.array(beta_bpms_x), np.array(beta_bpms_y)


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
