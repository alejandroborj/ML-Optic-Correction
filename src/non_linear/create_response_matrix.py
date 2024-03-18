#%%
from pymadng import MAD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import glob
import pickle

import itertools

from create_dataset import load_sample, create_sample, check_distance_from_ip

from model_training import generate_second_order_polynomials, calculate_metrics

""" ------------------------------------------------------- 0 --------------------------------------------------
Script for creating response matrix simulations and a function to load this simulations into an array, response matrix
needs to load the difference between the nominal and simulation RDT, divided by the an bn value. The response matrix
predicts the an bn value, however, the ML method predicts the knl values. To change from knl to an bn values the function
abs_to_rel does this. 
----------------------------------------------------------- 0 --------------------------------------------------  """

def main():
    magnet_names = ['MQXA.1R1','MQXB.A2R1','MQXB.B2R1','MQXA.3R1',	
    'MQXA.3L5','MQXB.B2L5','MQXB.A2L5','MQXA.1L5',
    'MQXA.1R5','MQXB.A2R5','MQXB.B2R5','MQXA.3R5',
    'MQXA.3L1','MQXB.B2L1','MQXB.A2L1','MQXA.1L1']
    
    # Usual knl values to change, according to Mael
    ks = [[0.1, 0, 0, 0],
        [0, 0.1, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 2]]

    # Create all simulations for response matrix
    for magnet_name in magnet_names:
        for k in ks:
            k2_err, k2s_err, k3_err, k3s_err = k

            # This errors are an, bn!!!
            data = {
            "NAME": [magnet_name],
            "K2L": [k2_err],
            "K2SL": [k2s_err],
            "K3L": [k3_err],
            "K3SL": [k3s_err]
            }

            error_df = pd.DataFrame(data)

            create_sample(error_df, "./datasets/tests/response_matrix_xing", XING=True)

  
def np_to_df_errors(errors_np, magnet_names_list):
  """
  Takes a numpy array representing the errors the same way is formatted in load_dataset and a magnet list
  for the magnets and returns the array in a dataframe with the correct names and shape

  """
  errors_np = errors_np.reshape(4, int(len(magnet_names_list)))
  
  data = {'NAME': magnet_names_list, 
          'K2L': errors_np[0],
          'K2SL': errors_np[1],
          'K3L': errors_np[2],
          'K3SL': errors_np[3]}
  
  error_dataframe = pd.DataFrame(data)

  return error_dataframe

def abs_to_rel(error_dataframe, XING):
  """
  Takes a dataframe of absolute errors and returns the relative a, b error values in a dataframe

  Input:
    - error_dataframe: dataframe with KNL errors, returns an bn errors
    - XING: is it crossing angle setup, should not matter

  Output:
    - rel_error_dataframe: dataframe with an, bn errors

  """
  magnet_names = error_dataframe['NAME']
  k2_errs = error_dataframe['K2L']
  k2s_errs = error_dataframe['K2SL']
  k3_errs = error_dataframe['K3L']
  k3s_errs = error_dataframe['K3SL']
  
  data = {'NAME':magnet_names, 
          'K2L':[0 for i in magnet_names],
          'K2SL':[0 for i in magnet_names],
          'K3L':[0 for i in magnet_names],
          'K3SL':[0 for i in magnet_names]}
  
  rel_error_dataframe = pd.DataFrame(data)
  rel_error_dataframe = rel_error_dataframe.set_index("NAME")

  with MAD(mad_path = r"/afs/cern.ch/user/a/aborjess/work/public/mad-linux-0.9.7", debug=True) as mad:
    if XING == True:
      seq = "./lhc_data/lhcb1_saved_xing"
    else:
      seq = "./lhc_data/lhcb1_saved"
    
    mad.send(f"""
      MADX:load("{seq}.seq", "{seq}.mad") -- convert on need
      lhc = MADX['lhc'..'b1']
      """)
    
    # Error generation for all triplet magnets
    for magnet_name, k2_err, k2s_err, k3_err, k3s_err in zip(magnet_names, k2_errs, k2s_errs, k3_errs, k3s_errs):
      # Inputing errors to madng, taking into consideration conversion
      # between an and bn notation to knl notation

      # Formula for absolute errors from relative MADX manual
      
      mad.send(f"""   
      local function calculate_error (element)
              
        !Calculate the absolute errors given a set of relative errors
        local k_ref = element.k1
        local ks_ref = element.k1
               
        r_r = 0.017 ! For arcs
        r_r = 0.05 ! For triplets
              
        local b3_err = r_r*{1e4*k2_err}/k_ref/element.l/2
        local a3_err = r_r*{1e4*k2s_err}/ks_ref/element.l/2
        local b4_err = (r_r^2)*{1e4*k3_err}/k_ref/element.l/6
        local a4_err = (r_r^2)*{1e4*k3s_err}/ks_ref/element.l/6
        
        py:send({{b3_err, a3_err, b4_err, a4_err}})

        end
      
      act = \e -> calculate_error(e)

      lhc:foreach{{action=act, pattern="{magnet_name}"}}
      """)
      dknl = mad.recv() # Signal pymadng to wait for madng
      rel_error_dataframe.loc[magnet_name,:] = dknl

  return rel_error_dataframe


def load_resp_matrix(matrix_path, XING, orders, hor_rdt_list, vert_rdt_list):
  """
  Takes a path where the response matrix simulations are and returns the response matrix with
  the same shape from the load_sample formula 

  Input:
    - matrix_path: path to the RM folder
    - XING: in order to load the relative sample this changes the nominal
    - orders: error orders to load in the response_matrix
    - hor_rdt_list: rdt inputs to load

  Output:
    - R: numpy array of the response matrix

  """

  paths = glob.glob(matrix_path + "/*")
  paths = [path for path in paths if "b1" in path] # Taking only b1

  magnet_names = ['MQXA.1R1','MQXB.A2R1','MQXB.B2R1','MQXA.3R1',	
    'MQXA.3L5','MQXB.B2L5','MQXB.A2L5','MQXA.1L5',
    'MQXA.1R5','MQXB.A2R5','MQXB.B2R5','MQXA.3R5',
    'MQXA.3L1','MQXB.B2L1','MQXB.A2L1','MQXA.1L1']
  
  all_orders = ['k2l', 'k2sl', 'k3l', 'k3sl']
    
  key_list = ['_'.join(params) for params in list(itertools.product(magnet_names, orders))]
  
  # Make sure to take only the needed orders
  key_list = [key for key in key_list if any(order in key for order in orders)]

  R = pd.DataFrame(columns=key_list)

  for path in tqdm(paths):
    try:
      sim_name = path.split('/')[-1][:-4]
      magnet_name = sim_name.split('_')[0]

      ks = sim_name.split('_')[1:-1]

      for idx, k in enumerate(ks):
         if k!='0.0':
          order = idx

      if all_orders[order] in orders:
        key = magnet_name +'_'+ all_orders[order]

        R[key] = load_sample(path, REL_TO_NOM=True, XING=XING, noise_level=0, hor_rdt_list=hor_rdt_list, vert_rdt_list=vert_rdt_list )

        if 'k2l' in key or 'k2sl' in key:
          R[key] = R[key]/0.1

        if 'k3l' in key or 'k3sl' in key:
          R[key] = R[key]/2

    except pd.errors.EmptyDataError:
      print(f"Error: Empty dataframe{path}")

  return np.array(R) #R [Order][list of rdts]

def load_mult_samples(dataset_df, XING, noise_level, magnet_names, n_samples, rdts_per_order):
    """
    Takes a dataset_dataframe containing locations, as well as the arguments to load a sample and the magnet
    names of the magnets in IP1 and IP5

    Output:
      - multiple samples and errors in a numpy format

    """
    samples, true_errors = [], []
    
    for idx, row in dataset_df[:n_samples].iterrows():
        sample_path = row['File Path']
        error_path = row['Error Path']

        # Load sample:
        example_sample = []
        for hor_list, vert_list in rdts_per_order:
            example_sample.append(np.array(load_sample(path=sample_path, 
                                REL_TO_NOM=True, 
                                XING=XING, 
                                noise_level=noise_level,
                                hor_rdt_list=hor_list,
                                vert_rdt_list=vert_list)).T)

         # Load errors:
        example_error = pd.read_csv(error_path, sep='\t')
        # Take out IP2 and IP8
        example_error = example_error[example_error['NAME'].isin(magnet_names)]
        example_error = abs_to_rel(example_error, XING=XING).values.flatten()
        
        samples.append(example_sample)
        true_errors.append(example_error)

    return samples, true_errors

def rms_rdt_beat_hist(dataset_df, XING, noise_level, magnet_names, n_samples, method, estimator, directory_name, r_matrices, rdts_per_order):
    magnet_names_ip51 = ['MQXA.1R1','MQXB.A2R1','MQXB.B2R1','MQXA.3R1',	
    'MQXA.3L5','MQXB.B2L5','MQXB.A2L5','MQXA.1L5',
    'MQXA.1R5','MQXB.A2R5','MQXB.B2R5','MQXA.3R5',
    'MQXA.3L1','MQXB.B2L1','MQXB.A2L1','MQXA.1L1']

    rdts = ["300000", "400000", "201000", "102000",  "003000", "022000", "013000", "103000"] # Horizontal and vertical
    
    if method == "RM":
        REL_TO_NOM = True
    if method == "ML":
        REL_TO_NOM = False

    sample_rms_rdt_dict = [{key:[] for key in rdts}, {key:[] for key in rdts}]
    corr_rms_rdt_dict = [{key:[] for key in rdts}, {key:[] for key in rdts}]

    for idx, row in dataset_df[:n_samples].iterrows():
        sample_paths = [row['File Path B1'], row['File Path B2']]
        error_path = row['Error Path']

         # Load errors:
        example_error = pd.read_csv(error_path, sep='\t')
        example_error = example_error[~example_error.duplicated(subset='NAME')]

        # Take out IP2 and IP8, old functionality
        example_error = example_error[example_error['NAME'].isin(magnet_names)]
        example_error = abs_to_rel(example_error, XING=XING).values.T
        
        if method == "RM":
            pred_error = predict_errors_rm(sample_paths, rdts_per_order, XING, r_matrices, noise_level, magnet_names_ip51, magnet_names)

        elif method == "ML":
           pred_error = predict_errors_ml(sample_paths, rdts_per_order, XING, estimator, noise_level, magnet_names_ip51, magnet_names)
        
        #print(example_error, pred_error)
        residual_error = example_error - pred_error

        error_df = pd.DataFrame({
        "NAME": magnet_names,
        "K2L": residual_error[0],
        "K2SL": residual_error[1],
        "K3L": residual_error[2],
        "K3SL": residual_error[3]
        })

        create_sample(error_df=error_df,
                    directory_name=directory_name,
                    XING=XING)

        for bn, sample_path_bn in enumerate(sample_paths):
            for rdt in rdts:
                #plot_sample_rdt(sample_path=sample_path_bn, 
                #                corr_sample_path=directory_name + f"/samples/sample_None_seed_None_b{bn+1}.csv", 
                #                jklm=rdt, 
                #                plot_title=f"{method} Correction",
                #                XING=XING,
                #                BEATING=False,
                #                noise_level=0,
                #                hor_rdt_list=hor_list,
                #                vert_rdt_list=vert_list)
                
                corr_rms_rdt, sample_rms_rdt = calculate_rms_rdt_beat(sample_path=sample_path_bn, 
                                    corr_sample_path=directory_name + f"/samples/sample_None_seed_None_b{bn+1}.csv",  
                                    jklm=rdt,
                                    XING=XING)

                print(f"Corrected RMS(RDTBeat) B{bn+1}:{corr_rms_rdt},\n Sample RMS(RDTBeat) B{bn+1}:{sample_rms_rdt} ")

                sample_rms_rdt_dict[bn][rdt].append(sample_rms_rdt)
                corr_rms_rdt_dict[bn][rdt].append(corr_rms_rdt)

    for bn, sample_path_bn in enumerate(sample_paths):
        with open(f'./figures/sample_dict_{method}_b{bn+1}.pkl', 'wb') as f:
            pickle.dump(sample_rms_rdt_dict[bn], f)

        with open(f'./figures/coor_dict_{method}_b{bn+1}.pkl', 'wb') as f:
            pickle.dump(corr_rms_rdt_dict[bn], f)


def predict_errors_rm(sample_paths, rdts_per_order, XING, r_matrices, noise_level, magnet_names_ip51, magnet_names):
    pred_error = []
    # Calculate response matrix prediction
    # Plotting SVD to see which rcond to use
    #pred_error = np.dot(R_pinv, example_sample).reshape((4, 16))
    #rcond = [2E-2, 3e-3, 2e-3, 2e-2] # FOR XING
    rcond = [2E-2, 3e-3, 2e-3, 5e-1] # FOR NO XING
    
    # Load sample:
    example_sample = []
    for hor_list, vert_list in rdts_per_order:
        example_sample.append(np.array(load_sample(path=sample_paths[0], 
                            REL_TO_NOM=True, 
                            XING=XING, 
                            noise_level=noise_level,
                            hor_rdt_list=hor_list,
                            vert_rdt_list=vert_list)).T)
    
    for i, rm in enumerate(r_matrices):
        R_pinv = np.linalg.pinv(rm, rcond=rcond[i])
        pred_err = np.dot(R_pinv, example_sample[i])
        pred_error.append(pred_err)
    
    pred_error = np.array(pred_error).reshape(4, int(len(magnet_names_ip51)))

    # Create error df with ip2 and ip8
    df = pd.DataFrame({'NAME': magnet_names})
    # Add columns for k2l, k2sl, k3l, k3sl with 0 values
    df[['K2L', 'K2SL', 'K3L', 'K3SL']] = 0
    # Update the values for the magnets present in magnet_names_ip5
    df.loc[df['NAME'].isin(magnet_names_ip51), 'K2L'] = pred_error[0]
    df.loc[df['NAME'].isin(magnet_names_ip51), 'K2SL'] = pred_error[1]
    df.loc[df['NAME'].isin(magnet_names_ip51), 'K3L'] = pred_error[2]
    df.loc[df['NAME'].isin(magnet_names_ip51), 'K3SL'] = pred_error[3]

    pred_error = df

    print(pred_error)

    pred_error = np.array(df.values).T[1:] # For RM!

    if np.mean(np.mean(abs(pred_error).flatten()))>20: # Deleting outliers
        print("Error: Outlier")
        return
    
    return pred_error

def predict_errors_ml(sample_paths, rdts, XING, estimator, noise_level, magnet_names_ip51, magnet_names):
    example_sample = []
    hor_list, vert_list = rdts[0], rdts[1]
    # Calculate ML prediction
    for bn, sample_path_bn in enumerate(sample_paths): 
        example_sample.extend(np.array(load_sample(path=sample_path_bn, 
                                                REL_TO_NOM=False, 
                                                XING=XING, 
                                                noise_level=noise_level,
                                                hor_rdt_list=hor_list,
                                                vert_rdt_list=vert_list,
                                                REMOVE_ARCS=False)).T)
    
    # Normalizing the RDT data
    mean = np.load('./data_analysis/rdt_mean.npy') # Important, if this are not the same rdts calculated for training results are much worse
    std = np.load('./data_analysis/rdt_std.npy')
    example_sample = np.array(example_sample)
    example_sample = (example_sample-mean)/std
    example_sample = generate_second_order_polynomials(np.array([example_sample]), 376)[0] # 188 bpms

    pred_error = estimator.predict([example_sample])

    # Denormalize error predictions
    mean = np.load('./data_analysis/err_mean.npy')
    std = np.load('./data_analysis/err_std.npy')
    pred_error = np.array(pred_error)
    pred_error = (pred_error*std) + mean
            
    pred_error = pred_error.reshape(4, int(len(magnet_names_ip51)))

    # Create error df with ip2 and ip8
    df = pd.DataFrame({'NAME': magnet_names})
    # Add columns for k2l, k2sl, k3l, k3sl with 0 values
    df[['K2L', 'K2SL', 'K3L', 'K3SL']] = 0
    # Update the values for the magnets present in magnet_names_ip5
    df.loc[df['NAME'].isin(magnet_names_ip51), 'K2L'] = pred_error[0]
    df.loc[df['NAME'].isin(magnet_names_ip51), 'K2SL'] = pred_error[1]
    df.loc[df['NAME'].isin(magnet_names_ip51), 'K3L'] = pred_error[2]
    df.loc[df['NAME'].isin(magnet_names_ip51), 'K3SL'] = pred_error[3]

    pred_error = df
    
    # Turn the abs errors to an bn notation
    pred_error = abs_to_rel(pred_error, XING=XING).values.T

    return pred_error


def calculate_rms_rdt_beat(sample_path, corr_sample_path, jklm, XING):
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

    nom_rdt_amp = np.sqrt(rdt_nom_df["RE_" + jklm]**2 + rdt_nom_df["IM_" + jklm]**2)
    corr_rdt_amp = np.sqrt(rdt_corr_df["RE_" + jklm]**2 + rdt_corr_df["IM_" + jklm]**2)
    sample_rdt_amp = np.sqrt(rdt_df["RE_" + jklm]**2 + rdt_df["IM_" + jklm]**2)

    corr_rdt_beating = (corr_rdt_amp-nom_rdt_amp)#/nom_rdt_amp
    sample_rdt_beating = (sample_rdt_amp-nom_rdt_amp)#/nom_rdt_amp

    corr_rms_rdt_beat = np.sqrt(np.mean(corr_rdt_beating**2))
    sample_rms_rdt_beat = np.sqrt(np.mean(sample_rdt_beating**2))

    return corr_rms_rdt_beat, sample_rms_rdt_beat


def calculate_rmatrix_metrics(dataset_df, XING, noise_level, magnet_names, n_samples, orders, rdts_per_order):
    # Calculates the R2 and MAE (an bn units) for the response matrix 
    
    y_pred_rm = []
    method = "RM"    
    rcond = [1E-2, 5e-1, 2e-4, 1e-4] # Hyperparameter for RM
    r_matrices = []

    samples, true_errors = load_mult_samples(dataset_df, XING, noise_level, magnet_names, n_samples, rdts_per_order)

    for i, order in enumerate(orders):
        hor_list = rdts_per_order[i][0]
        vert_list = rdts_per_order[i][1]

        if XING==False:
            R = load_resp_matrix("./datasets/response_matrix", XING=XING, orders=order, hor_rdt_list=hor_list, vert_rdt_list=vert_list)   

        else:        
            R = load_resp_matrix("./datasets/response_matrix_xing", XING=XING, orders=order, hor_rdt_list=hor_list, vert_rdt_list=vert_list)  

        r_matrices.append(R)

    for j, sample in enumerate(samples):

        pred_errors_rm = []
        for i, rm in enumerate(r_matrices):
            R_pinv = np.linalg.pinv(rm, rcond=rcond[i])
            pred_errors = np.dot(R_pinv, sample[i])
            pred_errors_rm.extend(pred_errors)

        y_pred_rm.append(pred_errors_rm)

    true_errors, y_pred_rm = np.array(true_errors), np.array(y_pred_rm)

    for error_order in ["K2L", "K2SL", "K3L", "K3SL", "ALL"]:
            _, r2_test, _, mae_test, _, adjusted_r2_test = calculate_metrics(true_errors, y_pred_rm, true_errors, y_pred_rm, error_order)

            print(f"\n  {error_order}")
            print(f"Test: R2 = {r2_test:.6g}, MAE = {mae_test:.6g}, CORR R2 = {adjusted_r2_test:.6g}")


if __name__ == "__main__":
    main()
# %%

