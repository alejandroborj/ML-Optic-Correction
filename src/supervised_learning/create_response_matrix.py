#%%
from pymadng import MAD

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import glob

import itertools

from create_dataset import check_distance_from_ip

def main():

    magnet_names = ['MQXA.1R1','MQXB.A2R1','MQXB.B2R1','MQXA.3R1',	
    'MQXA.3L5','MQXB.B2L5','MQXB.A2L5','MQXA.1L5',
    'MQXA.1R5','MQXB.A2R5','MQXB.B2R5','MQXA.3R5',
    'MQXA.3L1','MQXB.B2L1','MQXB.A2L1','MQXA.1L1']
    
    #Usual values to get change, according to Mael
    ks = [[0.1,0,0,0],
        [0,0.1,0,0],
        [0,0,2,0],
        [0,0,0,2]]

    for magnet_name in magnet_names:
        for k in ks:
            k2_err, k2s_err, k3_err, k3s_err = k
            #print(k)
            #create_sample([magnet_name], [k2_err], [k2s_err], [k3_err], [k3s_err], "./datasets/response_matrix")

    # Testing the multiple errors
    errors = [0.1 for i in range(len(magnet_names))]
    create_sample(magnet_names, errors, errors, errors, errors, "./datasets/example_dataset", XING=False)    
    #abs_err_to_rel(['MQXA.1R1','MQXB.A2R1'], [1, 0], [0, 0], [0, 0], [0, 0])


def create_sample(magnet_names, k2_errs, k2s_errs, k3_errs, k3s_errs, directory_name, XING):
  """Creates a simulation generating a single error for the given magnet or a list of error for different magnets
  and kn values, input is given in an, bn units with the 1e-4 

  magnet_names, k2_errs, k2s_errs, k3_errs, k3s_errs => lists of values for all magnets """
  
  #with MAD(mad_path = r"/home/alejandro/mad-linux-0.9.7-pre", debug=True) as mad:
  with MAD(mad_path = r"/afs/cern.ch/user/a/aborjess/work/public/mad-linux-0.9.7-pre", debug=True) as mad:
    # Loading all relevant libraries
    mad.load("MAD", "beam", "track", "twiss", "match", "damap", "option")
    mad.load("MAD.gphys", "normal")
    mad.load("MAD.gmath", "real", "imag", "abs")
    mad.load("MAD.utility", "tblcat", "printf")
    mad.load("MAD.element.flags", "observed")
    
    # This values are for naming the files
    if len(magnet_names)!=1:
      print("Multiple error simulation")
      mad.send(f"""
      k2_err = '{'_'}'
      k2s_err = '{'_'}'
      k3_err = '{'_'}'
      k3s_err = '{'_'}'
      magnet_name = '{'mult_magnets'}'
      directory_name = '{directory_name}'
      """)
    
    else:
      print("Single error simulation")
      mad.send(f"""
      k2_err = {k2_errs[0]*1e-4}
      k2s_err = {k2s_errs[0]*1e-4}
      k3_err = {k3_errs[0]*1e-4}
      k3s_err = {k3s_errs[0]*1e-4}
      magnet_name = '{magnet_names[0]}'
      directory_name = '{directory_name}'
      """)

    if XING == True:
      seq = "./lhc_data/lhcb1_saved_xing"
    else:
      seq = "./lhc_data/lhcb1_saved"
    
    mad.send(f"""
    -- track and twiss columns
    tkcols = {{'name','s','l','x','px','y','py'}}
    twcols = {{'name','s','beta11','beta22','mu1','mu2','dx','dy','x','y'}}

    -- flag to run twiss checks
    twiss_check = 0 -- 1 do checks

    -------------------------------------------------------------------------------o
    -- load LHCB1 and LHCB2                                                      --o
    -------------------------------------------------------------------------------o

    MADX:load("{seq}.seq", "{seq}.mad") -- convert on need
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
      error_file = io.open(string.format("%s/abs_error_%s.csv", directory_name, magnet_name), "w")        
      error_file:write("NAME\\tK2L\\tK2SL\\tK3L\\tK3SL\\n")
      """)

    # Timing the RDT tracking time
    start = time.time()

    # Error generation for all triplet magnets
    for magnet_name, k2_err, k2s_err, k3_err, k3s_err in zip(magnet_names, k2_errs, k2s_errs, k3_errs, k3s_errs):
        
      # Inputing errors to madng, taking into consideration conversion
      # between an and bn notation to knl notation

      # Formula for absolute errors from relative MADX manual 
      # j=i=2, R=0.017 kref=k1 n=1
      # l is to make knl and not kn

      mad.send(f"""      
      local function set_error (element)
              
        ! Making absolute errors
        local k_ref = element.k1
        local ks_ref = element.k1
              
        local k2l_err = 2*{k2_err*1e-4}*k_ref*element.l/0.017
        local k2sl_err = 2*{k2s_err*1e-4}*ks_ref*element.l/0.017
        local k3l_err = 6*{k3_err*1e-4}*k_ref*element.l/(0.017^2)
        local k3sl_err = 6*{k3s_err*1e-4}*ks_ref*element.l/(0.017^2)

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

    mad.send(r"""
             
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

    local f = assert(io.open(string.format("%s/%s_%s_%s_%s_%s.csv", directory_name, magnet_name, k2_err, k2s_err, k3_err, k3s_err), "w"))
    !Different csv for each beam
    
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

def np_to_df_errors(errors_np):  
  errors_np = errors_np.reshape(4, 16)

  magnet_names_list = ['MQXA.1R1','MQXB.A2R1','MQXB.B2R1','MQXA.3R1',	
  'MQXA.3L5','MQXB.B2L5','MQXB.A2L5','MQXA.1L5',
  'MQXA.1R5','MQXB.A2R5','MQXB.B2R5','MQXA.3R5',
  'MQXA.3L1','MQXB.B2L1','MQXB.A2L1','MQXA.1L1']
  
  data = {'NAME':magnet_names_list, 
          'K2L': errors_np[0],
          'K2SL':errors_np[1],
          'K3L':errors_np[2],
          'K3SL':errors_np[3]}
  
  error_dataframe = pd.DataFrame(data)

  return error_dataframe

def abs_to_rel(error_dataframe, XING):
  # Takes a dataframe of absolute errors and returns the relative a,b error values in a dataframe
  magnet_names = error_dataframe['NAME']
  k2_errs = error_dataframe['K2L']
  k2s_errs = error_dataframe['K2SL']
  k3_errs = error_dataframe['K3L']
  k3s_errs = error_dataframe['K3SL']

  magnet_names_list = ['MQXA.1R1','MQXB.A2R1','MQXB.B2R1','MQXA.3R1',	
  'MQXA.3L5','MQXB.B2L5','MQXB.A2L5','MQXA.1L5',
  'MQXA.1R5','MQXB.A2R5','MQXB.B2R5','MQXA.3R5',
  'MQXA.3L1','MQXB.B2L1','MQXB.A2L1','MQXA.1L1']
  
  data = {'NAME':magnet_names_list, 
          'K2L':[0 for i in magnet_names_list],
          'K2SL':[0 for i in magnet_names_list],
          'K3L':[0 for i in magnet_names_list],
          'K3SL':[0 for i in magnet_names_list]}
  
  rel_error_dataframe = pd.DataFrame(data)
  rel_error_dataframe = rel_error_dataframe.set_index("NAME")

  with MAD(mad_path = r"/afs/cern.ch/user/a/aborjess/work/public/mad-linux-0.9.7-pre", debug=True) as mad:
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
      # j=i=2, R=0.017 kref=k1 n=1
      # l is to make knl and not kn
      
      mad.send(f"""   
      local function calculate_error (element)
              
        !Calculate the absolute errors given a set of relative errors
        local k_ref = element.k1
        local ks_ref = element.k1
              
        local b3_err = 0.017*{1e4*k2_err}/k_ref/element.l/2
        local a3_err = 0.017*{1e4*k2s_err}/ks_ref/element.l/2
        local b4_err = (0.017^2)*{1e4*k3_err}/k_ref/element.l/6
        local a4_err = (0.017^2)*{1e4*k3s_err}/ks_ref/element.l/6
        
        py:send({{b3_err, a3_err, b4_err, a4_err}})

        end
      
      act = \e -> calculate_error(e)

      lhc:foreach{{action=act, pattern="{magnet_name}"}}
      """)
      dknl = mad.recv() # Signal pymadng to wait for madng
      rel_error_dataframe.loc[magnet_name,:] = dknl

  return rel_error_dataframe


def load_resp_matrix(matrix_path, XING):
  paths = glob.glob(matrix_path + "/*")

  orders = ['k2l', 'k2sl', 'k3l', 'k3sl']
  magnet_names = ['MQXA.1R1','MQXB.A2R1','MQXB.B2R1','MQXA.3R1',	
    'MQXA.3L5','MQXB.B2L5','MQXB.A2L5','MQXA.1L5',
    'MQXA.1R5','MQXB.A2R5','MQXB.B2R5','MQXA.3R5',
    'MQXA.3L1','MQXB.B2L1','MQXB.A2L1','MQXA.1L1']
    
  key_list = ['_'.join(params) for params in list(itertools.product(magnet_names, orders))]
  
  R = pd.DataFrame(columns=key_list)

  for path in tqdm(paths):
    try:
      sim_name = path.split('/')[-1][:-4]
      magnet_name = sim_name.split('_')[0]

      ks = sim_name.split('_')[1:]
      for idx, k in enumerate(ks):
         if k!='0':
          order = idx

      key = magnet_name +'_'+ orders[order]

      R[key] = load_sample(path, REL_TO_NOM=True, XING=XING)

      if 'k2l' in key or 'k2sl' in key:
        R[key]=R[key]/0.1

      if 'k3l' in key or 'k3sl' in key:
        R[key]=R[key]/2

    except pd.errors.EmptyDataError:
      print(f"Error: Empty dataframe{path}")  
  
  # Saving in a .npy format for quicker access afterwards
  #np.save(matrix_path + "/resp_matrix.npy", np_dataset)

  return R #R [Order][list of rdts]

def load_sample(path, REL_TO_NOM, XING):
    # Loads a single sample, taking the difference from nominal value, or not depending on REL_TO_NOM
    order_list = []
    X_df = pd.read_csv(path, sep="\t")

    if XING == True:
      X_nom_df = pd.read_csv('./RDT_BPMS_NOMINAL_B1_XING.csv', sep="\t")
    else:
      X_nom_df = pd.read_csv('./RDT_BPMS_NOMINAL_B1.csv', sep="\t")
    
    X_df["PART_OF_MEAS"] = X_df["NAME"].apply(check_distance_from_ip)
    X_nom_df["PART_OF_MEAS"] = X_nom_df["NAME"].apply(check_distance_from_ip)

    X_df = X_df[X_df["PART_OF_MEAS"]==True]
    X_nom_df = X_nom_df[X_nom_df["PART_OF_MEAS"]==True]

    if REL_TO_NOM == False:
      # If the sample is not measured wrt the nominal model
      X_nom_df.loc[:, :] = 0

    # List of all saved RDTS
    # Horizontal
    order_list.extend(list(X_df["RE_300000"]-X_nom_df["RE_300000"]))
    order_list.extend(list(X_df["IM_300000"]-X_nom_df["IM_300000"]))
    order_list.extend(list(X_df["RE_200200"]-X_nom_df["RE_200200"]))
    order_list.extend(list(X_df["IM_200200"]-X_nom_df["IM_200200"]))
    order_list.extend(list(X_df["RE_201000"]-X_nom_df["RE_201000"]))
    order_list.extend(list(X_df["IM_201000"]-X_nom_df["IM_201000"]))
    order_list.extend(list(X_df["RE_210100"]-X_nom_df["RE_210100"]))
    order_list.extend(list(X_df["IM_210100"]-X_nom_df["IM_210100"]))

    # Vertical
    order_list.extend(list(X_df["RE_102000"]-X_nom_df["RE_102000"]))
    order_list.extend(list(X_df["IM_102000"]-X_nom_df["IM_102000"]))
    order_list.extend(list(X_df["RE_012100"]-X_nom_df["RE_012100"]))
    order_list.extend(list(X_df["IM_012100"]-X_nom_df["IM_012100"]))
    order_list.extend(list(X_df["RE_003000"]-X_nom_df["RE_003000"]))
    order_list.extend(list(X_df["IM_003000"]-X_nom_df["IM_003000"]))
    order_list.extend(list(X_df["RE_022000"]-X_nom_df["RE_022000"]))
    order_list.extend(list(X_df["IM_022000"]-X_nom_df["IM_022000"]))

    return order_list

if __name__ == "__main__":
    main()
# %%
