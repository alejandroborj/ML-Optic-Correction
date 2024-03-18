#%%
from pymadng import MAD

import time
import glob
import multiprocessing
import os

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from tqdm import tqdm

""" ------------------------------------------------------- 0 --------------------------------------------------
Script for creating dataset, two folders one with the samples (RDTs) and one with the errors, in .csv format

Script also contains functionality to load samples (load_sample), loading datasets and save in a quicker .npy format (load_dataset)

----------------------------------------------------------- 0 --------------------------------------------------  """

def main():
  n_processes = 2 # Number of parallel processes
  dataset_name = "datasets/toy_dataset" # Folder name to save the data
  n_samples = 30 # Number of samples for each process
  XING = True # Whether to use a Xing angle setup or not

  # Multiprocessing speeds up the process of generating data, for each HTCondor job multiple samples can be generated
  # at once and asking for multiple CPUs. MADNG Is CPU hungry, so the n_processes should not be too big

  with multiprocessing.Pool(processes=n_processes) as pool:
    args = [[dataset_name, XING, np.random.randint(0, 1E7), n_samples] for i in range(n_processes)]
    pool.map(create_dataset, args)


def create_dataset(args):
  """
  This is a function used generate samples when using pythons multiprocessing library
  """
  dataset_name, XING, process_seed, n_samples = args

  # To make each process have different seeds, if I dont give each process a specific seed the 
  # sample_seeds generated are the same for the different parallel instances

  np.random.seed(seed=process_seed)

  # Creating all samples
  for sample_id in range(n_samples):
    sample_seed = np.random.randint(0, 1E7)
    create_random_sample(dataset_name, XING, sample_id, sample_seed)


def create_random_sample(dataset_name, XING, sample_id, sample_seed):

  np.random.seed(seed=sample_seed)
  
  # Loading the previously calculated expected error sigmas for the IRs in an and bn format
  mean_err_data_frame = pd.read_csv("./lhc_data/seeds/fidel_rep_mean.csv")
  std_err_data_frame = pd.read_csv("./lhc_data/seeds/fidel_rep_std.csv")

  magnet_names, k2_err, k2s_err, k3_err, k3s_err = [], [], [], [], []

  # Foar each magnet type errors are generated and stored
  for (idx, row_mean), (idx, row_std) in zip(mean_err_data_frame.iterrows(), std_err_data_frame.iterrows()):
      if "MQX" in row_std['NAME']: #Testing only error in triplets
        magnet_names.append(row_std['NAME'].replace(".","%."))

        std_a3 = float(row_std['a3'])
        std_b3 = float(row_std['b3'])
        std_a4 = float(row_std['a4'])
        std_b4 = float(row_std['b4'])
        
        mean_a3 = float(row_mean['a3'])
        mean_b3 = float(row_mean['b3'])
        mean_a4 = float(row_mean['a4'])
        mean_b4 = float(row_mean['b4'])

        # Generating truncated gaussian random errors
        k2_err.append(std_b3*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1) + mean_b3)
        k2s_err.append(std_a3*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1) + mean_a3)
        k3_err.append(std_b4*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1) + mean_b4)
        k3s_err.append(std_a4*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1) + mean_a4)

  data = {
  "NAME": magnet_names,
  "K2L": k2_err,
  "K2SL": k2s_err,
  "K3L": k3_err,
  "K3SL": k3s_err
  }

  error_df = pd.DataFrame(data)

  create_sample(error_df,
                dataset_name,
                XING,
                sample_seed=sample_seed,
                sample_id=sample_id)
    

def create_sample(error_df, directory_name, XING, sample_seed=None, sample_id=None):

  """ Main simulation script, can be used to generate random error or single error simulation,
   the optional arguments are for caracterizing a set of random errors

   Input:
    - error_df: Dataframe with the an, bn errors
    - directory_name: Location of the /errors and /samples folders for the data to be stores
    - seed: optional, for random errors
    - index: optional, for random errors

   Output:
    - Error file
    - Sample files for B1 and B2
   """

  magnet_names, k2_errs, k2s_errs, k3_errs, k3s_errs = error_df["NAME"], error_df["K2L"], error_df["K2SL"], error_df["K3L"], error_df["K3SL"]

  # Create directories to save the data
  if not os.path.exists(directory_name+'/errors'):
    os.makedirs(directory_name+'/errors')
  
  if not os.path.exists(directory_name+'/samples'):
    os.makedirs(directory_name+'/samples')

  with MAD(mad_path=r"/afs/cern.ch/user/a/aborjess/work/public/mad-linux-0.9.7", debug=False) as mad:
    # Loading all relevant libraries
    mad.load("MAD", "damap")
    mad.load("MAD.utility", "tblcat", "printf")
    mad.load("MAD.gmath", "real", "imag", "abs")

    # Inserting python variables into MADNG
    if len(magnet_names)!=1:
      print("Multiple error simulation")
      mad.send(f"""
      error_file_name = "{directory_name}/errors/error_{sample_id}_seed_{sample_seed}.csv"
      sample_file_name = "{directory_name}/samples/sample_{sample_id}_seed_{sample_seed}_"
      """)
    
    # If it is a single error simulation it is usually for RM, naming changes
    else:
      print("Single error simulation")
      mad.send(f"""
      error_file_name = "{directory_name}/errors/error_{magnet_names[0]}.csv"
      sample_file_name = "{directory_name}/samples/{magnet_names[0]}_{k2_errs[0]*1e-4}_{k2s_errs[0]*1e-4}_{k3_errs[0]*1e-4}_{k3s_errs[0]*1e-4}_"
      """)

    if XING == True:
      seq1 = "./lhc_data/lhcb1_saved_xing"
      seq2 = "./lhc_data/lhcb2_saved_xing"
    elif XING == False:
      seq1 = "./lhc_data/lhcb1_saved"
      seq2 = "./lhc_data/lhcb2_saved"
      #seq1 = "./lhc_data/lhcb1_no_sext_saved"
      #seq2 = "./lhc_data/lhcb2_no_sext_saved"
    
    # Loading beams
    mad.send(f"""
    MADX:load("{seq1}.seq", "{seq1}.mad") -- convert on need
    MADX:load("{seq2}.seq", "{seq2}.mad") -- convert on need
             
    MADX:load("lhc_vars0.mad")            -- avoid warnings
    """)
    mad.load("MADX", "lhcb1", "lhcb2")
    mad.send(f"""
             
    MADX.lhcb1.beam = beam {{particle="proton", energy=6800}}   
    MADX.lhcb2.beam = beam {{particle="proton", energy=6800}}  
    
    lhcb2.dir  = -1 
    !-- lhcb2 is reversed    
    """)

    # Creating the error .csv file, there is probably a better built in method
    # but this is straightforward and flexible

    mad.send(rf"""
    error_file = assert(io.open(error_file_name, "w"))
    error_file:write("NAME\tK2L\tK2SL\tK3L\tK3SL\n")
             
    py:send('Creating Error file')
    """)
    create_file = mad.recv()
    print(create_file)
    # Timing the RDT tracking time
    start = time.time()

    # Error generation for all triplet magnets
    for magnet_name, k2_err, k2s_err, k3_err, k3s_err in zip(magnet_names, k2_errs, k2s_errs, k3_errs, k3s_errs):
        
      # Inputing errors to madng, taking into consideration conversion
      # between an, bn notation to knl notation

      # Formula for absolute errors from relative MADX manual 
      # j=i=2, R=0.05 kref=k1 n=1
      # l is to make knl and not kn

      # Manual way to go about it, there is probably an implemented way on MADNG

      mad.send(f"""      
      local function set_error (element)
              
        ! Making absolute errors
        local k_ref = element.k1
        local ks_ref = element.k1
               
        r_r = 0.017 ! For arcs
        r_r = 0.05 ! For triplets

        local k2l_err = 2*{k2_err*1e-4}*k_ref*element.l/r_r
        local k2sl_err = 2*{k2s_err*1e-4}*ks_ref*element.l/r_r
        local k3l_err = 6*{k3_err*1e-4}*k_ref*element.l/(r_r^2)
        local k3sl_err = 6*{k3s_err*1e-4}*ks_ref*element.l/(r_r^2)

        element.dknl={{0, 0, k2l_err, k3l_err}}
        element.dksl={{0, 0, k2sl_err, k3sl_err}}

        error_file:write(string.format("%s\\t%s\\t%s\\t%s\\t%s\\n", element.name, k2l_err, k2sl_err, k3l_err, k3sl_err))        

        !py:send(table.concat(element.knl, ", "))
        !py:send(string.format("%s L:%f K2L:%E", element.name, element.l, k2l_err))

      end

      local act = \e -> set_error(e)
      
      MADX['lhcb1']:foreach{{action=act, pattern="{magnet_name}"}}
      MADX['lhcb2']:foreach{{action=act, pattern="{magnet_name}"}}

      py:send('Finish error generation')
      """)

      dknl= mad.recv() # Signal pymadng to wait for madng
      print(dknl)

    mad.send(rf"""
    error_file:close()
    
    -- RDTs to save, (all of them)
             
    local rdts = {{"f0111","f0120","f1002","f1011","f1020","f2100","f3000","f0013","f0031","f0040","f0211","f1102","f0220","f1120","f1300","f2002","f2011","f2020","f4000","f3100","f0012","f0021","f0030","f0210","f1101","f1110","f2001","f2010","f0112","f0121","f0310","f0130","f1003","f1012","f1021","f1030","f1201","f1210","f2101","f2110"}}
    local gfs = {{"011100","012000","100200","101100","102000","210000","300000","001300","003100","004000","021100","110200","022000","112000","130000","200200","201100","202000","400000","310000","001200","002100","003000","021000","110100","111000","200100","201000","011200","012100","031000","013000","100300","101200","102100","103000","120100","121000","210100","211000"}}
    
    -- Function to save the RDTs in a more memory efficient format
             
    function save_rdts(bn, seq, mtbl)
      file_name = sample_file_name .. bn .. '.csv'
      local f = assert(io.open(file_name, "w"))
      
      ! Making a list of BPMs

      local bpm_names = {{}}
      
      for i,element in pairs(seq.__dat) do
        local name = element["name"]
        local typ = tostring(type(name))
        if typ~="nil" then
          if string.find(name, "BPM")~=nil then
            table.insert(bpm_names, name)
          end
        end
      end

      -- list of BPMs
      local obs_points = bpm_names
      obs_points[1] = 'IP1'

      !Writing column names
             
      f:write("NAME\t")    
      for j, rdt in ipairs(gfs) do
        f:write(string.format("RE_%s\tIM_%s\t", rdt, rdt))
      end
      f:write("\n")

      ! Writting RDT data

      for i, obs_point in ipairs(obs_points) do
        f:write(string.format("%s\t", obs_point))
        for _,rdt in ipairs(rdts) do
          local v = mtbl[obs_point][rdt]
          f:write(string.format("%12.4f\t%12.4f\t", real(v), imag(v)))
        end
        !End of row
          f:write("\n")
      end
      f:close()
    end
    
    -- Twiss to track RDTs      
             
    for i, bn in ipairs({{'b1', 'b2'}}) do
             
      -- Create initial phase-space damap at the 4th order, this will be tracked along the ring 
      local X0 = damap {{nv=6, mo=4}}
             
      local lhc = MADX['lhc'..bn]
             
      local mtbl = twiss{{sequence=lhc, X0=X0, trkrdt=rdts}} !,method=4, 
      
      mtbl = mtbl:select({{pattern="BPM"}})
      mtbl = mtbl:select({{pattern="IP1"}})
             
      !local file_name = string.format("./%s/z_%s_%s.tfs", '{directory_name}', 'eee', bn)
      !mtbl:write(file_name, tblcat({{"name"}}, rdts), nil, true)

      save_rdts(bn, lhc, mtbl)
             
    end

    py:send('')
    """)
  
    finish = mad.recv() # Signal pymadng to wait for madng
    print(finish)

  stop = time.time()
  print('Execution time (s): ', stop-start) 


def load_dataset(dataset_path, noise_level, hor_rdt_list, vert_rdt_list, np_dataset_name, n_samples):
  """
  To ensure that every sample is loaded with its corresponding errors a dataframe is created with all 
  sample and error names and locations afterwards everything is loaded and also stored in a .npy file 
  for quick  access, also this way different preprocessing such as noise can be added for the same dataset
  
  Input:
    - dataset_path: path to the data to be loaded, all samples on this location will be loaded
    - hor_rdt_list: horizontal RDTs, columns taken 
    - np_dataset_name: name of the .npy file to save
    - noise_level: std_dev of the gaussian noise added
    - n_samples: number of samples to take from the dataset, useful to make smaller tests

  Output:
    - np_dataset: .npy dataset saved THIS IS HOW THE DATA IS ORGANIZED: [sample_idx][x,y][rdt or error]
  """

  np_dataset = []
  paths = glob.glob(dataset_path+"/**", recursive=True)

  sample_files_b1 = [sample for sample in paths if "sample" in sample and ".csv" in sample and "b1" in sample]
  sample_files_b2 = [sample for sample in paths if "sample" in sample and ".csv" in sample and "b2" in sample]

  error_files = [error for error in paths if "error" in error and ".csv" in error]

  sample_names_b1 = [sample_file_name.split("/")[-1] for sample_file_name in sample_files_b1]
  sample_names_b2 = [sample_file_name.split("/")[-1] for sample_file_name in sample_files_b2]

  error_names = [error_file_name.split("/")[-1] for error_file_name in error_files]

  sample_ids_b1 = [sample_file_name.split("_")[1] + sample_file_name.split("_")[-2] for sample_file_name in sample_names_b1]
  sample_ids_b2 = [sample_file_name.split("_")[1] + sample_file_name.split("_")[-2] for sample_file_name in sample_names_b2]
  error_ids = [error_file_name.split("_")[1] + error_file_name.split("_")[-1][:-4] for error_file_name in error_names]

  error_df = pd.DataFrame({'ID':error_ids,
                            'Error Name':error_names,
                            'Error Path':error_files})
  
  sample_df_b1 = pd.DataFrame({'ID':sample_ids_b1,
                            'File Name B1':sample_names_b1,
                            'File Path B1':sample_files_b1})
  
  sample_df_b2 = pd.DataFrame({'ID':sample_ids_b2,
                            'File Name B2':sample_names_b2,
                            'File Path B2':sample_files_b2})
  
  sample_df = pd.merge(sample_df_b1, sample_df_b2, on="ID", how='inner')
  dataset_df = pd.merge(error_df, sample_df, on="ID", how='inner')
  dataset_df.to_csv(dataset_path + '/dataset.csv')

  for idx, row in tqdm(dataset_df[:n_samples].iterrows()):
    X, y = [], []
    try:    
      X.extend(load_sample(row['File Path B1'], 
                      noise_level=noise_level,
                      hor_rdt_list=hor_rdt_list,
                      vert_rdt_list=vert_rdt_list,
                      XING=False,
                      REL_TO_NOM=False))
      
      X.extend(load_sample(row['File Path B2'],
                noise_level=noise_level,
                hor_rdt_list=hor_rdt_list,
                vert_rdt_list=vert_rdt_list,
                XING=False,
                REL_TO_NOM=False))
      
      # Xing is always false, since this is loading an absolute sample it does not matter,
      # XING is needed for RM samples, to load the proper nominal model!

      y_df = pd.read_csv(row['Error Path'], sep="\t", index_col="NAME")

      y_df = y_df.filter(regex="R1|L1|5", axis=0)
      y_df = y_df[~y_df.index.duplicated(keep='first')]

      # List of all saved errors
      y.extend(list(y_df["K2L"]))
      y.extend(list(y_df["K2SL"]))
      y.extend(list(y_df["K3L"]))
      y.extend(list(y_df["K3SL"]))
    
    except pd.errors.EmptyDataError:
      print(f"Error: Empty dataframe{row['File Path B1']} {row['Error Path']}")  

    np_dataset.append([X, y])

  # Check all elements have the same length, some times samples fail, improvable
  ceros = []
  for i, x in enumerate(np_dataset[0]):
      if len(x)!=np_dataset[0][0]:
          ceros.append(i)

  np_dataset = np.delete(np_dataset, ceros, axis=0)
  
  # Saving in a .npy format for quicker access afterwards
  np.save(dataset_path + "/" + np_dataset_name, np_dataset)

  return np_dataset #  THIS IS HOW THE DATA IS ORGANIZED: [sample_idx][x,y][error or rdt]


def load_sample(path, noise_level, hor_rdt_list, vert_rdt_list, XING, REL_TO_NOM, REMOVE_ARCS=False): 

    """
    Functionality: Loads a single sample
    Returns: order_list, a list with all the RDTs for all BPMs so that all RDTs for the BPMs are located in sequence
    
    path: Path to the .csv file for the RDT sample
    noise_level: std_dev of the gaussian noise used
    hor_rdt_list: horizontal rdts used
    vert_rdt_list: vertical
    XING: If a RM sample is loaded the measurement is relative to the nominal level, ML uses absolute RDT data (THIS WAS A RANDOM CHOICE)
    REL_TO_NOM: takes the difference from nominal value, or not depending if its for a RM sample or ML sample 
    REMOVE_ARCS: takes out all arcs that are not besided IP5 and IP1, this test was implemented to reduce number of BPMs
    """

    order_list = []
    
    X_df = pd.read_csv(path, sep="\t")

    bn = path.split('.')[-2][-2:]

    if XING == True:
      if bn == "b1":
        X_nom_df = pd.read_csv('./lhc_data/RDT_BPMS_NOMINAL_B1_XING.csv', sep="\t")
      if bn == "b2":
        X_nom_df = pd.read_csv('./lhc_data/RDT_BPMS_NOMINAL_B2_XING.csv', sep="\t")

    elif XING == False:
      if bn == "b1":
        X_nom_df = pd.read_csv('./lhc_data/RDT_BPMS_NOMINAL_B1.csv', sep="\t")
      if bn == "b2":
        X_nom_df = pd.read_csv('./lhc_data/RDT_BPMS_NOMINAL_B2.csv', sep="\t")
    
    X_df["PART_OF_MEAS"] = X_df["NAME"].apply(check_distance_from_ip, args=(REMOVE_ARCS,)) # Choosing if I remove arcs or not
    X_nom_df["PART_OF_MEAS"] = X_nom_df["NAME"].apply(check_distance_from_ip, args=(REMOVE_ARCS,))

    X_df = X_df[X_df["PART_OF_MEAS"]==True]
    X_nom_df = X_nom_df[X_nom_df["PART_OF_MEAS"]==True]

    if REL_TO_NOM == False:
      # If the sample is not measured wrt the nominal model
      X_nom_df.loc[:, :] = 0     

    for hor_rdt, ver_rdt in zip(hor_rdt_list, vert_rdt_list):
      #for hor_rdt in hor_rdt_list:
      j_ = int(hor_rdt[3])
      l_ = int(ver_rdt[5])
      hor_noise = noise_level/2*j_ # This is the relative error wrt the spectral line strength
      ver_noise = noise_level/2*l_
      
      del_hor_rdt = X_df[hor_rdt]-X_nom_df[hor_rdt]
      del_ver_rdt = X_df[ver_rdt]-X_nom_df[ver_rdt]

      order_list.extend(list(del_hor_rdt + X_df[hor_rdt]*np.random.normal(0, hor_noise, len(X_df))))
      order_list.extend(list(del_ver_rdt + X_df[ver_rdt]*np.random.normal(0, ver_noise, len(X_df))))

    return order_list


def check_distance_from_ip(name, REMOVE_ARCS=False):
    """
    Tells if a BPM is further than "n_bpms_removed" of an IP. This is used when loading data and preprocessing it
    """ 
    n_bpms_removed = 10 # Number of BPMs near IP removed
    if name == "IP1":
      return False
    
    else:
      if REMOVE_ARCS== True:
        arcs_to_delete=["L3", "L4", "L7", "L8", "R2", "R3", "R6", "R7"]
        for arc in arcs_to_delete:
          if arc in name:
            return False
 
      position = name.split('.')[1]
      position = position.split('R')[0]
      position = position.split('L')[0]
      distance = ''
      for char in position:
        if char.isdigit():
          distance += char
      if int(distance) <= n_bpms_removed:
        return False
      else:
        return True

if __name__ == "__main__":
   main()


# %%
