#%%
from pymadng import MAD
import tfs

import time
import glob
import multiprocessing

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from tqdm import tqdm

# Script for creating dataset, two folders one with the samples (RDTs) and one with the errors, in .csv format

def main(dataset_name):
  n_processes = 1
  dataset_name = "example_dataset"
  n_samples = 1

  # For parallel computing, however since it is very CPU intensive
  # using HTCondor is a better way of parallelizing
  with multiprocessing.Pool(processes=n_processes) as pool:
    args = [[dataset_name, n_samples, np.random.randint(0, 1E7)] for i in range(n_processes)]
    pool.map(create_dataset, args)

  #dataset = load_dataset_np("./example_dataset")
  #print(dataset)
  

def create_random_sample(sample_id, sample_seed, dataset_name):
  """Creates a simulation with a random set of errors according to the wise table 
  parameters Pymadng script, probably can be optimized, in writting the data, 
  generating the errors and overall readability
  Input: 
    - sample_id: number of the generated sample
    - sample_seed: a random seed provided for each sample
    - dataset_name: location where to save the data in a 
                    .csv format for each sample"""

  np.random.seed(seed=sample_seed)
  
  #with MAD(mad_path = r"/home/alejandro/mad-linux-0.9.7-pre", debug=True) as mad:
  with MAD(mad_path = r"/afs/cern.ch/user/a/aborjess/work/public/mad-linux-0.9.7-pre", debug=True) as mad:
    # Loading all relevant libraries
    mad.load("MAD", "beam", "track", "twiss", "match", "damap", "option")
    mad.load("MAD.gphys", "normal")
    mad.load("MAD.gmath", "real", "imag", "abs")
    mad.load("MAD.utility", "tblcat", "printf")
    mad.load("MAD.element.flags", "observed")

    # Inputs from python
    mad.send(f"""
    sample_id = {sample_id}
    sample_seed = {sample_seed}
    dataset_name = '{dataset_name}'
    """)

    mad.send(r"""
    -- track and twiss columns
    tkcols = {'name','s','l','x','px','y','py'}
    twcols = {'name','s','beta11','beta22','mu1','mu2','dx','dy','x','y'}

    -- flag to run twiss checks
    twiss_check = 0 -- 1 do checks

    -------------------------------------------------------------------------------o
    -- load LHCB1 and LHCB2                                                      --o
    -------------------------------------------------------------------------------o

    MADX:load("./lhc_data/lhcb1_saved_xing.seq", "./lhc_data/lhcb1_saved_xing.mad") -- convert on need
    !MADX:load("./lhc_data/lhcb2_saved.seq", "./lhc_data/lhcb2_saved.mad") -- convert on need
    MADX:load("lhc_vars0.mad")                      -- avoid warnings
    """)

    mad.load("MADX", "lhcb1")

    mad.send(r"""
    !local lhcb1 in MADX !, lhcb2

    !lhcb2.dir  = -1 -- lhcb2 is reversed, i.e. bv_flag = -1

    -------------------------------------------------------------------------------o
    -- preliminaries                                                             --o
    -------------------------------------------------------------------------------o

    -- need to create a "new" proton for MAD-X compatibility (old pmass?)
    lhc_beam = beam {particle="xproton", charge=1, mass=0.938272046, energy=450}

    for _,lhc in ipairs{lhcb1} do!,lhcb2
    -- attach beam to sequence
    lhc.beam = lhc_beam

    -- select observed elements for twiss
    lhc:deselect(observed)
    lhc:  select(observed, {pattern="BPM"})
    lhc:  select(observed, {pattern="IP" })
    end

    -------------------------------------------------------------------------------o
    -- twiss checks (optional)                                                   --o
    -------------------------------------------------------------------------------o

    function prt_qs (seq, tw)
    printf("% 5s:  q1 = % -.6f,  q2 = % -.6f\n", seq.name, tw. q1, tw. q2)
    printf("      dq1 = % -.6f, dq2 = % -.6f\n",           tw.dq1, tw.dq2)
    end

    if twiss_check ~= 0 then

    tw1 = twiss {sequence=lhcb1, method=4, observe=1, chrom=true}
    !local tw2 = twiss {sequence=lhcb2, method=4, observe=1, chrom=true}


    prt_qs(lhcb1, tw1) ; tw1:write("twiss_b1_n.tfs", twcols)
    !prt_qs(lhcb2, tw2) ; tw2:write("twiss_b2_n.tfs", twcols)

    -- if twiss_check < 0 then os.exit() end
    end
    py:send('Error Generation')
    """)
    error_generation = mad.recv() # Signal pymadng to wait for madng
    print(error_generation)

    # Creating the error .csv file, there is probably a better built in method
    # but this is straightforward and flexible

    mad.send(f"""
      
      lhc = MADX['lhc'..'b1']
             
      error_file = assert(io.open(string.format("./%s/errors/error_%d_seed_%d.csv", dataset_name, sample_id, sample_seed), "w"))
      error_file:write("NAME\\tK2L\\tK2SL\\tK3L\\tK3SL\\n")
      """)
    
    # Loading the previously calculated expected error sigmas for the IRs in an and bn format
    wise_data_frame = pd.read_csv("./seeds/std_optics_23.csv")

    # Error generation for all triplet magnets
    for idx, magnet_row in wise_data_frame[:1].iterrows():
      #print("MAGNET NAME", magnet_row['NAME'])
      if "MQX" in magnet_row['NAME']: #Testing only error in triplets
        magnet_name = magnet_row['NAME'].replace(".","%.")

        std_a3 = float(magnet_row['a3'])
        std_b3 = float(magnet_row['b3'])
        std_a4 = float(magnet_row['a4'])
        std_b4 = float(magnet_row['b4'])

        # Generating truncated gaussian random errors
        k2_err = 1E-4*std_b3*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
        k2s_err = 1E-4*std_a3*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
        k3_err = 1E-4*std_b4*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
        k3s_err = 1E-4*std_a4*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
        
        k2_err = 1E-5
        k2s_err = 0 
        k3_err = 0 
        k3s_err = 0 
        print(k2_err,k2s_err, k3_err, k3s_err)
        
        # Inputing errors to madng, taking into consideration conversion
        # between an and bn notation to knl notation

        # Formula for absolute errors from relative MADX manual 
        # j=i=2, R=0.017 kref=k1 n=1
        # l is to make knl and not kn

        print(magnet_name)
        mad.send(f"""   
        local function set_error (element)
                
          ! Making absolute errors
          local k_ref = element.k1
          local ks_ref = element.k1
                
          local k2l_err = 2*{k2_err}*k_ref*element.l/0.017
          local k2sl_err = 2*{k2s_err}*ks_ref*element.l/0.017
          local k3l_err = 6*{k3_err}*k_ref*element.l/(0.017^2)
          local k3sl_err = 6*{k3s_err}*ks_ref*element.l/(0.017^2)

          element.dknl={{0, 0, k2l_err, k3l_err}}
          element.dksl={{0, 0, k2sl_err, k3sl_err}}

          error_file:write(string.format("%s\\t%s\\t%s\\t%s\\t%s\\n", element.name, k2l_err, k2sl_err, k3l_err, k3sl_err))

          !py:send(table.concat(element.knl, ", "))
          py:send(string.format("%s L:%f K2L:%E", element.name, element.l, k2l_err))
        end

        local act = \e -> set_error(e)
        
        lhc:foreach{{action=act, pattern="{magnet_name}"}}

        """)

        dknl= mad.recv() # Signal pymadng to wait for madng
        print(dknl)

    # Timing the RDT tracking time
    start = time.time()
    mad.send(r"""
             
    error_file:close()
    -------------------------------------------------------------------------------o
    -- twiss checks (optional)                                                   --o
    -------------------------------------------------------------------------------o

    if twiss_check ~= 0 then

    tw1 = twiss {sequence=lhcb1, method=4, observe=1, chrom=true}
    !local tw2 = twiss {sequence=lhcb2, method=4, observe=1, chrom=true}

    prt_qs(lhcb1, tw1) ; tw1:write("twiss_phase_b1_n.tfs", twcols)
    !prt_qs(lhcb2, tw2) ; tw2:write("twiss_phase_b2_n.tfs", twcols)

    if twiss_check < 0 then os.exit() end
    end

    -- knobs families and sectors names
    local knam = {'all_lhc'}
    local snam = {'error_triplets'}    
    local gfs = {"011100","012000","100200","101100","102000","210000","300000","001300","003100","004000","021100","110200","022000","112000","130000","200200","201100","202000","400000","310000","001200","002100","003000","021000","110100","111000","200100","201000","011200","012100","031000","013000","100300","101200","102100","103000","120100","121000","210100","211000"}

    -- index of (IP,GNF)
    local idx_n = #gfs
    local idx_f = \i,j -> 2*((i-1)*idx_n+(j-1))

    -- loop over lhcb1 and lhcb2 ----------------
    for _,bn in ipairs{'b1'} do !,'b2'} do --

    !local lhc = MADX['lhc'..bn] -- current sequence

    io.write("*** Running ", lhc.name, " ***\n")

    -- BPM Names

    local bpm_names = {}

    for i,element in pairs(lhc.__dat) do !ipairs(tw1:getcol'name') do
    local name = element["name"]
    !print(type(name)==string)
    local typ = tostring(type(name))
    if typ~="nil" then
      if string.find(name, "BPM")~=nil then
        table.insert(bpm_names, name)
      end
    end
    end

    -- list of IPs and GNFs
    local obs_points = bpm_names
    obs_points[1] = 'IP1'

    ! First we need to cycle, for that we need a marker such as IP1, afterwards we track

    -- list of all knobs
    local kn = {}
    for _,ks in ipairs(knam) do
    for _,ss in ipairs(snam) do
    kn[#kn+1] = ks .. '_' .. ss .. bn -- knob names
    end end

    -- create phase space damap
    local X0 = damap{nv=6, np=#kn, mo=5, po=1,
                    vn=tblcat({'x','px','y','py','t','pt'}, kn)}


    -- compute RDTs
    local mthd = "trkrdt" -- trkrdt needs new release of MAD-NG
    local function get_nf(mthd)
    local nf, mth = {}, mthd or "trkrdt"

    if mth == "cycle" then       -- 1st method
      for i,obs_point in ipairs(obs_points) do
        io.write("** Tracking ", obs_point, "\n")
        lhc:cycle('$start')
        ! Cycling can only be done in marker points, not elements, such as IPs,
        ! end or arc markers (CHECK WHERE IS SHOULD CYCLE AND IF IT AFFECTS THE WHOLE LHC)
        local _, mflw = track{sequence=lhc, method=4, save=false, X0=X0}
        nf[i] = normal(mflw[1]):analyse();
        !nf[i].a:write("A_"..obs_point.."_cycle")
      end

    elseif mth == "trkrdt" then  -- 2nd method
      io.write("** Tracking ", obs_points[1], "\n") !obs_points[1] must be a marker
      lhc:cycle(obs_points[1])

      local _, mflw = track{sequence=lhc, method=4, save=false, X0=X0}
      local nf1 = normal(mflw[1]):analyse()

      io.write("** Tracking RDT\n")
      local X1 = nf1.a:real():set0(nf1.x0) ; X1.status = 'Aset'
      local mtbl, mflw = track{sequence=lhc, method=4, savemap=true, X0=X1,
                              range=obs_points[1].."/"..obs_points[#obs_points]}
      for i,obs_point in ipairs(obs_points) do
        nf[i] = nf1:analyse('gnf', mtbl[obs_point].__map)
        !nf[i].a:write("A_"..ip.."_trkrdt")
      end
    end

    return nf
    end

    -- run once for reference values
    local nf = get_nf(mthd)

    -- monomial strings for all knobs
    local ks, ki = {}, #kn
             
    for i=0,ki do ks[i] = nf[1]:getks(i) end

    local f = assert(io.open(string.format("./%s/samples/sample_%d_seed_%d_%s.csv",dataset_name, sample_id, sample_seed, bn), "w")) !Different csv for each beam
    
    !Writing column names
    f:write("NAME\t")    
    
    for j, rdt in ipairs(gfs) do
      f:write(string.format("RE_%s\tIM_%s\t", rdt, rdt))
    end
    f:write("\n")

    for i, obs_point in ipairs(obs_points) do
      f:write(string.format("%s\t", obs_point))
      for _,gf in ipairs(gfs) do
        local v = nf[i].gnf[gf..ks[0]]
        f:write(string.format("%12.4f\t%12.4f\t", real(v), imag(v)))
      end
      !End of row
      f:write("\n")
    end
    f:close()

    end -- loop over lhcb1 and lhcb2
             
    py:send('Finish')
    """)
  
    finish = mad.recv() # Signal pymadng to wait for madng
    print(finish)

  stop = time.time()
  print('Execution time (s): ', stop-start)  
  
def create_dataset(args):
  dataset_name, n_samples, process_seed = args
  #To make each process have different seeds
  np.random.seed(seed=process_seed)

  # Creating all samples
  for sample_id in range(n_samples):
    sample_seed = np.random.randint(0, 1E7)
    create_random_sample(sample_id, sample_seed, dataset_name)

def load_dataset(dataset_path, np_dataset_name):
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

  for idx, row in tqdm(dataset_df.iterrows()):
    X, y = [], []
    try:
      X_df = pd.read_csv(row['File Path'], sep="\t")
      
      X_df["PART_OF_MEAS"] = X_df["NAME"].apply(check_distance_from_ip)

      X_df = X_df[X_df["PART_OF_MEAS"]==True]

      y_df = pd.read_csv(row['Error Path'], sep="\t", index_col="NAME")

      y_df = y_df.filter(regex="R1|L1|5", axis=0)

      # List of all saved RDTS
      #Horizontal
      X.extend(list(X_df["RE_300000"]))
      X.extend(list(X_df["IM_300000"]))
      X.extend(list(X_df["RE_200200"]))
      X.extend(list(X_df["IM_200200"]))
      X.extend(list(X_df["RE_201000"]))
      X.extend(list(X_df["IM_201000"]))
      X.extend(list(X_df["RE_210100"]))
      X.extend(list(X_df["IM_210100"]))

      # Vertical
      X.extend(list(X_df["RE_102000"]))
      X.extend(list(X_df["IM_102000"]))
      X.extend(list(X_df["RE_012100"]))
      X.extend(list(X_df["IM_012100"]))
      X.extend(list(X_df["RE_003000"]))
      X.extend(list(X_df["IM_003000"]))
      X.extend(list(X_df["RE_022000"]))
      X.extend(list(X_df["IM_022000"]))    

      # List of all saved errors
      y.extend(list(y_df["K2L"]))
      y.extend(list(y_df["K2SL"]))
      y.extend(list(y_df["K3L"]))
      y.extend(list(y_df["K3SL"]))
    
    except pd.errors.EmptyDataError:
      print(f"Error: Empty dataframe{row['File Path']} {row['Error Path']}")  

    np_dataset.append([X,y])
  
  # Saving in a .npy format for quicker access afterwards
  np.save(dataset_path + "/" + np_dataset_name, np_dataset)

  return np_dataset #  np_dataset [sample_idx][x,y][error or rdt]

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

if __name__ == "__main__":
   dataset_name="xing_dataset"
   main(dataset_name)


# %%
