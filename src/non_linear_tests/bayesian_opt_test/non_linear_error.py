#%%
import cpymad.madx as madx
import tfs
import numpy as np
import time

from pymadng import MAD
import pandas as pd
import scipy

import matplotlib.pyplot as plt

def main():

  loss_rdt_madng, rdt_madng, madng_mean_3000, madng_mean_4000 = calculate_rdt_pymadng([0,0])
  print("RDT 3000:", loss_rdt_madng, madng_mean_3000, madng_mean_4000)
  loss_rdt_ptc, rdt_ptc, ptc_mean_3000, ptc_mean_4000 = calculate_rdt_ptc([0])
  print("RDT 3000:", loss_rdt_ptc, ptc_mean_3000, ptc_mean_4000)

  rdt_ptc.columns = rdt_ptc.columns.str.upper()
  rdt_ptc['NAME'] = rdt_ptc['NAME'].apply(lambda x: x[:-2].upper() if 'bpm' in x else x)

  # Merge dataframes based on 'ID'
  merged_df = pd.merge(rdt_ptc, rdt_madng, on='NAME', how='inner')

  rdt_ptc = merged_df["GNFA_3_0_0_0_0_0"]
  rdt_madng = (merged_df["RE_300000"]**2 + merged_df["IM_300000"]**2)**0.5

  plt.plot(range(len(rdt_madng)), rdt_madng,"o", label="MADNG")
  plt.plot(range(len(rdt_ptc)), rdt_ptc,"o", label="PTC")

  plt.xlabel("BPM")
  plt.ylabel("$|f_{300000}|$")
  plt.legend()

  plt.show()
  
def calculate_rdt_ptc(corr):
  np.random.seed(seed=10000)

  corr = corr[0]

  mdx = madx.Madx()

  mdx.options(echo=False)
  mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx")
  mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx")
  mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2022.macros.madx")
  mdx.call(file="/afs/cern.ch/eng/lhc/optics/V6.5/errors/Esubroutines.madx")
  mdx.call(file = "/afs/cern.ch/eng/acc-models/lhc/2022/lhc.seq")

  mdx.options(echo=True)

  mdx.exec("define_nominal_beams(energy=6500)")
  mdx.call(file="/afs/cern.ch/eng/acc-models/lhc/2022/operation/optics/R2023a_A30cmC30cmA10mL200cm.madx")

  mdx.exec("cycle_sequences()")

  #Beam 1
  beam = 1

  mdx.use(sequence=f"LHCB{beam}")
  mdx.options(echo=False)
  mdx.exec(f"match_tunes(62.31, 60.32, {beam})")

  #Assigning corrections

  mdx.globals["ksd1.a23b1"] = mdx.globals["ksd1.a23b1"] + corr # Knob value + simulated error + iteration correction

  #Assigning error

  wise_data_frame = pd.read_csv("./seeds/std_optics_23.csv")
  for idx, magnet_row in wise_data_frame.iterrows():
    #print("MAGNET NAME", magnet_row['NAME'])
    if "MQX" in magnet_row['NAME']: #Testing only error in triplets
      std_a3 = float(magnet_row['a3'])
      std_b3 = float(magnet_row['b3'])
      std_a4 = float(magnet_row['a4'])
      std_b4 = float(magnet_row['b4'])

      magnet_name = magnet_row['NAME'].replace(".","\.")
      
      k2_err = 1E-4*std_b3*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
      k2s_err = 1E-4*std_a3*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
      k3_err = 1E-4*std_b4*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
      k3s_err = 1E-4*std_a4*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)

      mdx.input(f"""
                SELECT, FLAG=error, CLEAR;
                SELECT, FLAG=ERROR, PATTERN="{magnet_name}";
                !print, text= "k1_valuee";
                !k1 = MQXA.3L2->K1S;
                !PRINTF, text="%f",value=k1;
                EFCOMP, RADIUS=0.017, ORDER=1, DKNR={{0,0,{k2_err},{k3_err}}};
                EFCOMP, RADIUS=0.017, ORDER=1, DKSR={{0,0,{k2s_err},{k3s_err}}};
                """)
      
  mdx.select(flag="error", pattern="MQX.*")##
  mdx.esave(file="errortable.tfs")

  # To achieve the same as madng quit this!
  #mdx.exec(f"match_tunes(62.31, 60.32, {beam})") 

  mdx.input(f"""etable, table="final_error";""")

  #mdx.makethin(sequence=f"LHCB{beam}", style="TEAPOT")
  
  start = time.time()
  mdx.input("""
  !use, sequence=lhcb1;

  ptc_create_universe;
    ptc_create_layout, model=2, method=6, nst=1, exact=true, time=true; !method =4
    ptc_setswitch, madprint=true;
    ptc_twiss, normal=true, trackrdts=true, no=4, icase=56;
  ptc_end;
  """)

  #mdx.ptc_create_universe()
  #mdx.ptc_create_layout(MODEL=3, METHOD=2, NST=1)
  #start = time.time()
  #mdx.ptc_twiss(trackrdts=True, icase=4, no=4, file="b1_monitors.out", closed_orbit=True)

  #Uses last sequence used for chroma at least 2
  stop = time.time()
  #mdx.write(table="TWISSRDT", file="rdts.dat")
  #mdx.ptc_end()

  print('Execution time (s): ', stop-start)

  #tfs.writer.write_tfs(tfs_file_path=f"ptc_twiss.tfs", data_frame=mdx.table.ptc_twiss.dframe())
  #tfs.writer.write_tfs(tfs_file_path=f"ptc_rdt.tfs", data_frame=mdx.table.twissrdt.dframe())

  error_twiss = mdx.table.ptc_twiss.dframe()

  rdt_error_df = mdx.table.twissrdt.dframe()
  rdt_nom_df = tfs.reader.read_tfs("RDT_ptc_nominal_b1.tfs")

  # Filter out rows that do not contain "BPM"
  rdt_error_df = rdt_error_df[rdt_error_df['name'].str.contains('bpm')]
  rdt_nom_df = rdt_nom_df[rdt_nom_df['name'].str.contains('bpm')]

  rdt_f3000 = np.array(rdt_error_df["gnfa_3_0_0_0_0_0"])
  nom_f3000 = np.array(rdt_nom_df["gnfa_3_0_0_0_0_0"])  
  
  rdt_f4000 = np.array(rdt_error_df["gnfa_4_0_0_0_0_0"])
  nom_f4000 = np.array(rdt_nom_df["gnfa_4_0_0_0_0_0"])

  del_f3000 = np.mean(((rdt_f3000 - nom_f3000)/nom_f3000)**2)**0.5
  del_f4000 = np.mean(((rdt_f4000 - nom_f4000)/nom_f4000)**2)**0.5
  del_f = del_f3000 #+ del_f4000

  mdx.quit()

  return del_f, rdt_error_df, np.mean(rdt_f3000), np.mean(rdt_f4000)

def calculate_rdt_madng(ksd1_a23b1_corr):
  ksd1_a23b1_corr = ksd1_a23b1_corr[0]
  with MAD(mad_path = r"/home/alejandro/mad-linux-0.9.7-pre") as mad:
    mad.send(f"""assert(loadfile("rdt_bpm.mad"))({ksd1_a23b1_corr})""") # Error is 0.5
    mad.send("""py:send("Finish")""")
    finish = mad.recv() # Signal pymadng to wait for madng

    print(finish)

  rdt_df = pd.read_csv("RDT_BPMS_b1.csv", sep="\t")
  rdt_nom_df = pd.read_csv("RDT_BPMS_nominal_b1.csv", sep="\t")

  rdt_f3000 = np.mean((rdt_df["RE_300000"]**2 + rdt_df["IM_300000"]**2)**0.5)
  rdt_f0030 = np.mean((rdt_df["RE_003000"]**2 + rdt_df["IM_003000"]**2)**0.5)

  nom_f3000 = np.mean((rdt_nom_df["RE_300000"]**2 + rdt_nom_df["IM_300000"]**2)**0.5)
  nom_f0030 = np.mean((rdt_nom_df["RE_003000"]**2 + rdt_nom_df["IM_003000"]**2)**0.5)

  del_f3000 = rdt_f3000-nom_f3000
  del_f0030 = rdt_f0030-nom_f0030

  return del_f3000 + del_f0030

def calculate_rdt_pymadng(corr):
  np.random.seed(seed=10000)
  
  with MAD(mad_path = r"/home/alejandro/mad-linux-0.9.7-pre", debug=True) as mad:

    mad.load("MAD", "beam", "track", "twiss", "match", "damap", "option")
    mad.load("MAD.gphys", "normal")
    mad.load("MAD.gmath", "real", "imag", "abs")
    mad.load("MAD.utility", "tblcat", "printf")
    mad.load("MAD.element.flags", "observed")

    mad.send(r"""
    -- track and twiss columns
    tkcols = {'name','s','l','x','px','y','py'}
    twcols = {'name','s','beta11','beta22','mu1','mu2','dx','dy','x','y'}

    -- flag to run twiss checks
    twiss_check = 0 -- 1 do checks, -1 do checks and quit (no matching)

    -------------------------------------------------------------------------------o
    -- load LHCB1 and LHCB2                                                      --o
    -------------------------------------------------------------------------------o

    MADX:load("./lhc_data/lhcb1_saved.seq", "./lhc_data/lhcb1_saved.mad") -- convert on need
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
    lhc:  select(observed, {pattern="MO" })
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

    corr_k2 = corr[0]
    corr_k3 = corr[1]

    mad.send(f"""
      MADX:open_env()

      ! Change knob
      !ksd1_a23b1 = ksd1_a23b1+{corr};

      kcsx3_l2 = kcsx3_l2+{corr_k2};
      kcox3_r2 = kcox3_r2+{corr_k3};
      !MCSX.3L2->K2 = MCSX.3L2->K2 + {corr};

      MADX:close_env()
      
      lhc = MADX['lhc'..'b1']
      """)

    # Error generation
    wise_data_frame = pd.read_csv("./seeds/std_optics_23.csv")
    for idx, magnet_row in wise_data_frame.iterrows():
      #print("MAGNET NAME", magnet_row['NAME'])
      if "MQX" in magnet_row['NAME']: #Testing only error in triplets
        magnet_name = magnet_row['NAME'].replace(".","%.")

        std_a3 = float(magnet_row['a3'])
        std_b3 = float(magnet_row['b3'])
        std_a4 = float(magnet_row['a4'])
        std_b4 = float(magnet_row['b4'])

        k2_err = 1E-4*std_b3*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
        k2s_err = 1E-4*std_a3*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
        k3_err = 1E-4*std_b4*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
        k3s_err = 1E-4*std_a4*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
        
        # Formula for absolute errors from relative MADX manual 
        # j=i=2, R=0.017 kref=k1 n=1
        # l is to make knl and not kn

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

          !py:send(table.concat(element.knl, ", "))
          py:send(string.format("%s L:%f K2L:%E", element.name, element.l, k3sl_err))

        end

        local act = \e -> set_error(e)
        
        lhc:foreach{{action=act, pattern="{magnet_name}"}}

        """)

        dknl= mad.recv() # Signal pymadng to wait for madng
        print(dknl)

    start = time.time()
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

    --[[ ref values
    LHCB1
    q1  = 62.27504896
    q2  = 60.29512787
    dq1 = 15.1778898
    dq2 = 15.21652238

    LHCB2
    q1  = 62.27115164
    q2  = 60.29725754
    dq1 = 15.15613102
    dq2 = 15.23680003
    --]]

    -------------------------------------------------------------------------------o
    -- match                                                                     --o
    -------------------------------------------------------------------------------o


    -- knobs families and sectors names
    local knam = {'all_lhc'}
    local snam = {'error_triplets'}
    local gfs = {'400000','004000', '300000', '003000'}

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

    local function prt_rdt (nf, kind)
    local a = assert(nf[kind], "invalid kind '"..kind.."'")
    for i,k in ipairs(a) do
      local v = nf[kind][k]
      printf("%4d: %s[%s] A=% .6e, C=% .6e, S=% .6e\n",i,kind,k,abs(v),real(v),imag(v))
    end
    end

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

    -- print reference some values

    local f = assert(io.open(string.format("RDT_BPMS_%s.csv", bn), "w")) !Different csv for each beam

    !Writing column names
    f:write("NAME\t")    
    for j, rdt in ipairs(gfs) do
    f:write(string.format("RE_%s\tIM_%s\t", rdt, rdt))
    end
    f:write("\n")

    for i,obs_point in ipairs(obs_points) do
    f:write(string.format("%s\t", obs_point))
    !printf("%s: q1       = % -.6e\n", obs_point, nf[i]:q1{1}    )                      -- nf[i].q1                  )
    !printf("%s: q1j1     = % -.6e\n", obs_point, nf[i]:anhx{1,0})                      -- nf[i].anh["2100"..knbs[0]])
    !printf("%s: q2       = % -.6e\n", obs_point, nf[i]:q2{1}    )                      -- nf[i].q2                  )
    !printf("%s: q2j2     = % -.6e\n", obs_point, nf[i]:anhy{0,1})                      -- nf[i].anh["0021"..knbs[0]])
    for _,gf in ipairs(gfs) do
      local v = nf[i].gnf[gf..ks[0]]
      printf("%s: f%sr = % -.6e\n", obs_point, gf, real(v))                            -- real(nf[i].gnf[gf..knbs[0]]))
      printf("%s: f%si = % -.6e\n", obs_point, gf, imag(v))                            -- imag(nf[i].gnf[gf..knbs[0]]))
      f:write(string.format("%12.4f\t%12.4f\t", real(v), imag(v)))
    end
    !End of row
    f:write("\n")
    end
    f:close()

    -- run once and quit
    -- os.exit()

    end -- loop over lhcb1 and lhcb2
             
    py:send('Finish')
    """)
  
  finish = mad.recv() # Signal pymadng to wait for madng
  print(finish)

  stop = time.time()
  print('Execution time (s): ', stop-start)

  rdt_error_df = pd.read_csv("RDT_BPMS_b1.csv", sep="\t")
  rdt_nom_df = pd.read_csv("RDT_BPMS_nominal_b1.csv", sep="\t")

  rdt_f3000 = (rdt_error_df["RE_300000"]**2 + rdt_error_df["IM_300000"]**2)**0.5
  rdt_f4000 = (rdt_error_df["RE_400000"]**2 + rdt_error_df["IM_400000"]**2)**0.5

  nom_f3000 = (rdt_nom_df["RE_300000"]**2 + rdt_nom_df["IM_300000"]**2)**0.5
  nom_f4000 = (rdt_nom_df["RE_400000"]**2 + rdt_nom_df["IM_400000"]**2)**0.5

  del_f3000 = np.mean(((rdt_f3000-nom_f3000)/nom_f3000)**2)**0.5
  del_f4000 = np.mean(((rdt_f4000-nom_f4000)/nom_f4000)**2)**0.5

  del_f = del_f3000 #+ del_f4000

  return del_f, rdt_error_df, np.mean(rdt_f3000), np.mean(rdt_f4000)

if __name__ == "__main__":
   main()


# %%
