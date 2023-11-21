#%%
from pymadng import MAD


import numpy as np
import pandas as pd
import scipy
import joblib

import matplotlib.pyplot as plt
from tqdm import tqdm

from create_response_matrix import create_sample, abs_to_rel, load_sample, load_resp_matrix, np_to_df_errors
from plots import plot_sample_rdt


def main():
    #Selecting appropiate parameters
       
    model_name = "miss_ip_no_xing"
    model_name = "miss_ip_xing"
    #model_name = "4ksmpl_no_miss_ip_no_xing"
    #model_name = "4ksmpl_no_miss_ip_xing"
    
    XING = True

    if XING==True:
        directory_name = "./datasets/example_dataset"
        R = load_resp_matrix("./datasets/response_matrix", XING=XING)
        R = np.array(R)
    else:        
        directory_name = "./datasets/example_dataset_xing"
        R = load_resp_matrix("./datasets/response_matrix_xing", XING=XING)
        R = np.array(R)

    #example_sample_path = './datasets/example_dataset_xing/samples/sample_0_1.csv'
    #example_sample_path = './datasets/example_dataset_xing/samples/sample_1r1.csv'
    #example_sample_path = './datasets/example_dataset_xing/samples/sample_k2_0.1.csv'
    example_sample_path = './datasets/example_dataset_xing/samples/sample_random_22394_b1.csv'
    #example_sample_path = './datasets/example_dataset/samples/sample_random_2649780_b1.csv'
    #example_sample_path = './datasets/example_dataset/samples/sample_random_4453966_b1.csv'

    #example_error_path = './datasets/example_dataset_xing/errors/error_0_1.csv'
    #example_error_path = './datasets/example_dataset_xing/errors/error_1r1.csv'
    #example_error_path = './datasets/example_dataset_xing/errors/error_k2_0.1.csv'
    example_error_path = './datasets/example_dataset_xing/errors/error_random_22394.csv'
    #example_error_path = './datasets/example_dataset/errors/error_random_2649780.csv'
    #example_error_path = './datasets/example_dataset/errors/error_random_4453966.csv'

    estimator = joblib.load(f"./models/{model_name}.joblib")
    
    magnet_names = ['MQXA.1R1','MQXB.A2R1','MQXB.B2R1','MQXA.3R1',	
    'MQXA.3L5','MQXB.B2L5','MQXB.A2L5','MQXA.1L5',
    'MQXA.1R5','MQXB.A2R5','MQXB.B2R5','MQXA.3R5',
    'MQXA.3L1','MQXB.B2L1','MQXB.A2L1','MQXA.1L1']

    # Load sample
    example_sample_rmatrix = np.array(load_sample(path=example_sample_path, REL_TO_NOM=True, XING=XING)).T
    example_sample_ml = np.array(load_sample(path=example_sample_path, REL_TO_NOM=False, XING=XING)).T

    #Load errors
    example_errors = pd.read_csv(example_error_path, sep='\t')
    #Take out IP2 and IP8
    example_errors = example_errors[example_errors['NAME'].isin(magnet_names)]
    example_errors = abs_to_rel(example_errors, XING=XING).values.T

    #Calculate response matrix prediction
    R_pinv = np.linalg.pinv(R, rcond=5E-5)
    pred_errors_rm = np.dot(R_pinv, example_sample_rmatrix).reshape((4, 16))
    
    # Calculate ML prediction
    pred_errors_ml = estimator.predict([example_sample_ml])
    pred_errors_ml = np_to_df_errors(pred_errors_ml)
    pred_errors_ml = abs_to_rel(pred_errors_ml, XING=XING).values.T

    residual_errors = example_errors - pred_errors_ml

    print("MAE:")
    print("Sim: ", np.mean(abs(example_errors).flatten()))
    print("Correction: ", np.mean(abs(pred_errors_rm).flatten()))
    print("Corrected: ",np.mean(abs(residual_errors).flatten()))

    create_sample(magnet_names,
    residual_errors[0], residual_errors[1], 
    residual_errors[2], residual_errors[3], directory_name, XING)

    plot_sample_rdt(sample_path=example_sample_path, 
                    corrected_sample_path=directory_name+"/mult_magnets________.csv", 
                    jklm="200200", 
                    plot_title="ML Correction",
                    XING=XING)

if __name__ == "__main__":
    main()
#%%