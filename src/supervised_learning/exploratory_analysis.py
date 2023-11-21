#%%
import numpy as np
import glob
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

def main():
  df_input, df_output = load_mean_dataset("./xing_dataset", 1)

  norm_sex_rdts = ["RE_011100","RE_012000","RE_100200","RE_101100","RE_102000",'IM_102000',"RE_210000",'IM_210000',"RE_300000",'IM_300000']

  skew_sex_rdts = ["RE_001200",'IM_001200',"RE_002100",'IM_002100',"RE_003000",'IM_003000',"RE_021000",'IM_021000',
       "RE_110100",'IM_110100',"RE_111000",'IM_111000',"RE_200100",'IM_200100',"RE_201000",'IM_201000']
  
  norm_oct_rdts = ["RE_001300",'IM_001300',"RE_003100",'IM_003100',"RE_004000",'IM_004000',"RE_021100",'IM_021100',"RE_110200",'IM_110200',"RE_022000",
              'IM_022000',"RE_112000",'IM_112000',"RE_130000",'IM_130000',"RE_200200",'IM_200200',"RE_201100",'IM_201100',"RE_202000",'IM_202000',
              "RE_400000",'IM_400000',"RE_310000",'IM_310000']

  skew_oct_rdts =  ["RE_011200",'IM_011200',"RE_012100",'IM_012100',"RE_031000",'IM_031000',"RE_013000",'IM_013000',"RE_100300",'IM_100300',"RE_101200",
               'IM_101200',"RE_102100",'IM_102100',"RE_103000",'IM_103000',"RE_120100",'IM_120100',"RE_121000",'IM_121000',"RE_210100",'IM_210100',
               "RE_211000",'IM_211000']

  df = pd.concat([df_input, df_output], axis=1)
  df = df.astype(float)
  
  #Correlation matrices
  threshold = 0.0
  corr = df.corr()
  corr = abs(corr)
  corr[corr < threshold] = np.nan

  RDTvRDT = corr.filter(regex="^(RE|IM)")
  RDTvRDT = RDTvRDT.filter(regex="^(RE|IM)", axis=0)
  
  ERRvRDT = corr.filter(regex="^(RE|IM)")
  ERRvRDT = ERRvRDT.filter(regex="R1|L1|5", axis=0)
  #ERRvRDT = ERRvRDT.filter(regex="^M", axis=0)

  print(ERRvRDT.applymap(abs).max().to_string())

  #Variances of average RDT across the ring
  std_df=df_input.std()
  mean_df=np.sqrt((df_input**2).mean())

  f = plt.figure(figsize=(19, 15))
  #Plots
  '''  plt.xticks(range(RDTvRDT.select_dtypes(['number']).shape[1]), RDTvRDT.select_dtypes(['number']).columns, fontsize=14, rotation=45, ha='left')
  plt.bar(std_df.index, std_df)
  plt.title('STD DEV RDT', fontsize=16)
  plt.show()
  
  f = plt.figure(figsize=(19, 15))
  plt.xticks(range(RDTvRDT.select_dtypes(['number']).shape[1]), RDTvRDT.select_dtypes(['number']).columns, fontsize=14, rotation=45, ha='left')
  plt.bar(mean_df.index, mean_df)
  plt.title('RMS RDT', fontsize=16)
  plt.show()

  f = plt.figure(figsize=(19, 15))
  plt.xticks(range(RDTvRDT.select_dtypes(['number']).shape[1]), RDTvRDT.select_dtypes(['number']).columns, fontsize=14, rotation=45, ha='left')
  plt.bar(std_df.index, std_df/mean_df)
  plt.title('STD DEV/RMS', fontsize=16)
  plt.show()''' 

  for rdt in [norm_sex_rdts, norm_oct_rdts, skew_sex_rdts, skew_oct_rdts]:
    mean_df_ = mean_df.loc[rdt]
    f = plt.figure(figsize=(19, 15))
    plt.xticks(range(mean_df_.shape[0]), mean_df_.index, fontsize=14, rotation=45, ha='right')
    plt.bar(mean_df_.index, mean_df_)
    plt.title(r'RMS Espectral line strenght in horizontal: $h_{x,-} \propto j f_{jklm}$', fontsize=16)
    #plt.title(r'RMS Espectral line strenght in vertical: $h_{y,-}l f_{jklm}$', fontsize=16)
    plt.show()

  f = plt.figure(figsize=(19, 15))
  plt.matshow(RDTvRDT, fignum=f.number)
  plt.xticks(range(RDTvRDT.select_dtypes(['number']).shape[1]), RDTvRDT.select_dtypes(['number']).columns, fontsize=14, rotation=45, ha='left')
  plt.yticks(range(RDTvRDT.select_dtypes(['number']).shape[1]), RDTvRDT.select_dtypes(['number']).columns, fontsize=14)
  cb = plt.colorbar()
  cb.ax.tick_params(labelsize=14)
  plt.title('Correlation Matrix MEAN(RDT) VS MEAN(RDT)', fontsize=16)
  plt.show()
  
  f = plt.figure(figsize=(25, 20))
  plt.matshow(ERRvRDT, fignum=f.number)
  plt.xticks(range(ERRvRDT.select_dtypes(['number']).shape[1]), ERRvRDT.select_dtypes(['number']).columns, fontsize=14, rotation=45, ha='left')
  plt.yticks(range(ERRvRDT.select_dtypes(['number']).shape[0]), ERRvRDT.select_dtypes(['number']).index, fontsize=14)
  cb = plt.colorbar()
  cb.ax.tick_params(labelsize=14)
  plt.title('Correlation Matrix MEAN(ERR) VS MEAN(RDT)', fontsize=16)
  plt.show()

def load_mean_dataset(dataset_path, n_samples):
  # To ensure that every sample is loaded with its corresponding errors a 
  # dataframe is created with all sample and error names and locations
  # afterwards everything is loaded and also stored in a .npy file for quick 
  # access, also this way different preprocessing such as noise can be added
  # for the same dataset

  np_dataset = []
  paths = glob.glob(dataset_path+"/**", recursive=True)
  sample_files = [error for error in paths if "sample" in error and ".csv" in error]
  error_files = [sample for sample in paths if "error" in sample and ".csv" in sample]

  sample_names = [sample_file_name.split("/")[-1] for sample_file_name in sample_files]
  error_names = [error_file_name.split("/")[-1] for error_file_name in error_files]

  sample_ids = [sample_file_name.split("_")[1] + sample_file_name.split("_")[-2] for sample_file_name in sample_names]
  error_ids = [error_file_name.split("_")[1] + error_file_name.split("_")[-1][:-4] for error_file_name in error_names]

  #print(list(set(sample_ids)-set(error_ids))+list(set(error_ids)-set(sample_ids)))

  error_df = pd.DataFrame({'ID':error_ids,
                                'Error Name':error_names,
                                'Error Path':error_files})
  sample_df = pd.DataFrame({'ID':sample_ids,
                                'File Name':sample_names,
                                'File Path':sample_files})
  
  dataset_df = pd.merge(error_df, sample_df, on="ID", how='inner')
  dataset_df.to_csv(dataset_path + '/dataset.csv')
  df_input = pd.DataFrame()
  df_output = pd.DataFrame()
  nominal_df = pd.read_csv("RDT_BPMS_NOMINAL_B1.csv", sep="\t")

  for idx, row in tqdm(dataset_df[:n_samples].iterrows()):
    X, y = [], []
    try:
      X_df = pd.read_csv(row['File Path'], sep="\t")
      y_df = pd.read_csv(row['Error Path'], sep="\t")

      #Check for missing values
      if len(X_df["RE_300000"])==563 and len(y_df["K2L"])==32:
        """
        # For spectral line strength
        X_df.index = X_df['NAME']
        X_df = X_df.drop(columns=['NAME', 'Unnamed: 81'])
        X_df = X_df.apply(pd.to_numeric, errors='coerce')
        j_list = np.array([int(rdt_name[3]) for rdt_name in X_df.columns])
        mean_df = j_list*(np.sqrt(((X_df)**2).mean())).T
        df_input = df_input.append(mean_df, ignore_index=True)

        """
        X_df["PART_OF_MEAS"] = X_df["NAME"].apply(check_distance_from_ip)

        X_df["RE_121000"] = np.where(X_df["PART_OF_MEAS"] == False, np.nan, X_df["RE_121000"])
        X_df["IM_121000"] = np.where(X_df["PART_OF_MEAS"] == False, np.nan, X_df["IM_121000"])

        plt.plot(range(len(X_df["RE_121000"])), np.sqrt(X_df["RE_121000"]**2+X_df["IM_121000"]**2))
        plt.plot(range(len(X_df["RE_121000"])), X_df["IM_121000"])
        plt.plot(range(len(X_df["RE_121000"])), X_df["RE_121000"])

        print(np.sqrt(X_df["RE_300000"]**2+X_df["IM_300000"]**2).to_string())

        X_df = X_df.apply(pd.to_numeric, errors='coerce')
        mean_df = (X_df-nominal_df).mean().T
        df_input = df_input.append(mean_df, ignore_index=True)

        # List of all saved errors
        y_df.index = y_df['NAME']
        prev_index = list(y_df.index)*4
        mean_df = pd.melt(y_df)
        mean_df = mean_df.dropna()
        mean_df = mean_df[32:]
        new_index = [index_name+"_"+col_name for index_name, col_name in zip(prev_index, mean_df["variable"])]
        mean_df.index = new_index
        mean_df = mean_df.drop('variable', axis=1).T
        df_output = df_output.append(mean_df, ignore_index=True)

    except pd.errors.EmptyDataError:
      print(f"Error: Empty dataframe{row['File Path']} {row['Error Path'],}")  
    
  return df_input, df_output

def check_distance_from_ip(name):
    if name == "IP1":
      return False
    else:
      position = name.split('.')[1]
      position = position.split('R')[0]
      position = position.split('L')[0]
      distance = ''
      for char in position:
        if char.isdigit():
          distance += char
      if int(distance) <= 10:
        return False
      else:
        return True

if __name__=="__main__":
  main()

  #%%
