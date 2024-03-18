#%%
from pymadng import MAD


import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
from tqdm import tqdm

from create_response_matrix import load_resp_matrix, load_mult_samples, predict_errors_ml, predict_errors_rm, rms_rdt_beat_hist, abs_to_rel
from create_dataset import create_sample, load_sample

from model_training import calculate_metrics

from plots import plot_sample_rdt



def main():
    #Selecting appropiate parameters
    model_name = "poly_ridge_10K_1030_n10_0"
    #model_name = "ridge_11K_1030_n10_xing_0" # Xing model
    noise_level = 0
    method = "RM"        
    XING = False

    orders = [['k2l'], ['k2sl'], ['k3l'], ['k3sl']]
    rdts_per_order = [[["RE_300000","IM_300000"],["RE_102000","IM_102000"]], 
                      [["RE_201000","IM_201000"],["RE_003000","IM_003000"]], 
                      [["RE_400000","IM_400000"],["RE_022000","IM_022000"]], 
                      [["RE_103000","IM_103000"],["RE_013000","IM_013000"]]]

    rdts = ["300000", "400000", "201000", "102000",  "003000", "022000", "013000", "103000"]#,  "310000""012100""210100",
    #orders = [['k2l', 'k2sl', 'k3l', 'k3sl']]

    # List of all saved RDTS
    # Horizontal
    hor_rdt_list = ["RE_300000",
                "IM_300000",
                "RE_400000",
                "IM_400000",
                "RE_201000",
                "IM_201000",
                "RE_103000",
                "IM_103000"]
                #"RE_210100",
                #"IM_210100"]

    # Vertical
    vert_rdt_list = ["RE_102000",
                "IM_102000",
                "RE_013000",
                "IM_013000",
                "RE_003000",
                "IM_003000",
                "RE_022000",
                "IM_022000"]
                #"RE_012100",
                #"IM_012100",

    magnet_names_ip51 = ['MQXA.1R1','MQXB.A2R1','MQXB.B2R1','MQXA.3R1',	
    'MQXA.3L5','MQXB.B2L5','MQXB.A2L5','MQXA.1L5',
    'MQXA.1R5','MQXB.A2R5','MQXB.B2R5','MQXA.3R5',
    'MQXA.3L1','MQXB.B2L1','MQXB.A2L1','MQXA.1L1']

    magnet_names = ["MQXA.1R1","MQXB.A2R1","MQXB.B2R1","MQXA.3R1",
    "MQXA.3L2","MQXB.B2L2","MQXB.A2L2","MQXA.1L2",
    "MQXA.1R2","MQXB.A2R2","MQXB.B2R2","MQXA.3R2",
    "MQXA.3L5","MQXB.B2L5","MQXB.A2L5","MQXA.1L5",
    "MQXA.1R5","MQXB.A2R5","MQXB.B2R5","MQXA.3R5",
    "MQXA.3L8","MQXB.B2L8","MQXB.A2L8","MQXA.1L8",
    "MQXA.1R8","MQXB.A2R8","MQXB.B2R8","MQXA.3R8",
    "MQXA.3L1","MQXB.B2L1","MQXB.A2L1","MQXA.1L1"]
    
    r_matrices = []

    for i, order in enumerate(orders):
        hor_list=rdts_per_order[i][0]
        vert_list=rdts_per_order[i][1]

        if XING==False:
            directory_name = "./datasets/tests/example_dataset"
            R = load_resp_matrix("./datasets/tests/response_matrix/samples", XING=XING, orders=order, hor_rdt_list=hor_list, vert_rdt_list=vert_list) 
        else:        
            directory_name = "./datasets/tests/example_dataset_xing"
            R = load_resp_matrix("./datasets/tests/response_matrix_xing/samples", XING=XING, orders=order, hor_rdt_list=hor_list, vert_rdt_list=vert_list)
        
        r_matrices.append(R)

    seed = 290028 #4887719
    
    # Not used for training!
    #example_error_path = "./datasets/tests/dataset_test/job_183/errors/error_74_seed_2469719.csv"
    #example_sample_path_b1 = "./datasets/tests/dataset_test/job_183/samples/sample_74_seed_2469719_b1.csv"
    #example_sample_path_b2 = "./datasets/tests/dataset_test/job_183/samples/sample_74_seed_2469719_b2.csv"

    example_error_path = "./datasets/tests/dataset_test/job_183/errors/error_74_seed_9252156.csv"
    example_sample_path_b1 = "./datasets/tests/dataset_test/job_183/samples/sample_74_seed_9252156_b1.csv"
    example_sample_path_b2 = "./datasets/tests/dataset_test/job_183/samples/sample_74_seed_9252156_b2.csv"

    #example_error_path="./datasets/tests/dataset_xing/job_199/errors/error_29_seed_8438406.csv"
    #example_sample_path_b1="./datasets/tests/dataset_xing/job_199/samples/sample_29_seed_8438406_b1.csv"
    #example_sample_path_b2="./datasets/tests/dataset_xing/job_199/samples/sample_29_seed_8438406_b2.csv"

    # Used for training!
    #example_error_path = f'./datasets/tests/example_dataset/errors/error_0_seed_{seed}.csv'
    #example_sample_path_b1 = f'./datasets/tests/example_dataset/samples/sample_0_seed_{seed}_b1.csv'
    #example_sample_path_b2 = f'./datasets/tests/example_dataset/samples/sample_0_seed_{seed}_b2.csv'
    
    sample_paths = [example_sample_path_b1, example_sample_path_b2]

    estimator = joblib.load(f"./models/{model_name}.joblib") 

    # Load errors
    example_errors = pd.read_csv(example_error_path, sep='\t')
    example_errors = example_errors[~example_errors.duplicated(subset='NAME')]
    example_errors = abs_to_rel(example_errors, XING=XING).values.T

    # Load samples and make predictions
    if method == "RM":
        pred_error = predict_errors_rm(sample_paths, rdts_per_order, XING, r_matrices, noise_level, magnet_names_ip51, magnet_names)

    elif method == "ML":
        pred_error = predict_errors_ml(sample_paths, [hor_rdt_list, vert_rdt_list], XING, estimator, noise_level, magnet_names_ip51, magnet_names)

    print(example_errors)
    print(pred_error)

    residual_errors = example_errors - pred_error

    print("MAE:")
    print("Sim: ", np.mean(abs(example_errors).flatten()))
    print("Correction: ", np.mean(abs(pred_error).flatten()))
    print("Corrected: ", np.mean(abs(residual_errors).flatten()),"\n")

    error_df = pd.DataFrame({
    "NAME": magnet_names,
    "K2L": residual_errors[0],
    "K2SL": residual_errors[1],
    "K3L": residual_errors[2],
    "K3SL": residual_errors[3]
    })

    create_sample(error_df=error_df,
                directory_name=directory_name,
                XING=XING)

    for rdt in rdts:
        for bn, sample_path_bn in enumerate(sample_paths): 
            plot_sample_rdt(sample_path=sample_path_bn, 
                            corr_sample_path=directory_name + f"/samples/sample_None_seed_None_b{bn+1}.csv", 
                            jklm=rdt, 
                            method=method,
                            XING=XING,
                            BEATING=False,
                            noise_level=noise_level,
                            hor_rdt_list=hor_rdt_list,
                            vert_rdt_list=vert_rdt_list)
    
    dataset_df = pd.read_csv("./datasets/tests/dataset_xing/dataset.csv")[-1100:] #Taking the last samples, not used for training!

    #calculate_rmatrix_metrics(dataset_df, XING, noise_level, magnet_names, n_samples=500, orders=orders, rdts_per_order=rdts_per_order)

    rms_rdt_beat_hist(dataset_df=dataset_df,
                      XING=XING, 
                      noise_level=noise_level,
                     magnet_names=magnet_names,
                     n_samples=100,
                     method="RM",
                     estimator=estimator,
                     directory_name=directory_name,
                     r_matrices=r_matrices,
                     rdts_per_order=rdts_per_order)
    
    rms_rdt_beat_hist(dataset_df=dataset_df,
                      XING=XING, 
                      noise_level=noise_level,
                     magnet_names=magnet_names,
                     n_samples=100,
                     method="ML",
                     estimator=estimator,
                     directory_name=directory_name,
                     r_matrices=r_matrices,
                     rdts_per_order=[hor_rdt_list, vert_rdt_list])

if __name__ == "__main__":
    main()
# %%
