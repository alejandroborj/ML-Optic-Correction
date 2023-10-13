#%%
import cpymad.madx as madx
import tfs
import numpy as np
import time

from pymadng import MAD
import pandas as pd

import time
import scipy.stats


def main():
  start = time.time()
  calculate_rdt_ptc()
  stop = time.time()

  print(f"Time: {stop-start}(s)")

def calculate_rdt_ptc():

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

  #mdx.twiss(file="b1_monitors.dat")
  mdx.exec(f"match_tunes(62.31, 60.32, {beam})")

  #Error generation!
  wise_data_frame = pd.read_csv("./seeds/std_optics_23.csv")
  for idx, magnet_row in wise_data_frame.iterrows():
    #print("MAGNET NAME", magnet_row['NAME'])
    if "MQX" in magnet_row['NAME']: #Testing only error in triplets
      std_a3 = float(magnet_row['a3'])
      std_b3 = float(magnet_row['b3'])
      std_a4 = float(magnet_row['a4'])
      std_b4 = float(magnet_row['b4'])

      magnet_name = magnet_row['NAME'].replace(".","\.")
      
      k2_err = 1E-3#1E-4*std_b3*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
      k2s_err = 1E-3#1E-4*std_a3*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
      k3_err = 1E-3#1E-4*std_b4*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)
      k3s_err = 1E-3#1E-4*std_a4*scipy.stats.truncnorm.rvs(-3, 3, loc=0, scale=1)

      mdx.input(f"""
                SELECT, FLAG=error, CLEAR;
                SELECT, FLAG=ERROR, PATTERN="{magnet_name}";
                !print, text= "k1_valuee";
                !k1 = MQXA.3L2->K1S;
                !PRINTF, text="%f",value=k1;
                EFCOMP, RADIUS=0.017, ORDER=1, DKNR={{0,0,{k2_err},{k3_err}}}, DKSR={{0,0,{k2s_err},{k3s_err}}};
                """)
      
  mdx.select(flag="error", pattern="MQX.*")
  mdx.esave(file="errortable.tfs")
  mdx.etable(table="etable")
  print("Machine errors. ", mdx.table.etable.dframe())

  mdx.exec(f"match_tunes(62.31, 60.32, {beam})")

  mdx.etable(table="final_error")

  mdx.call(file = "./acc-models-lhc/toolkit/slice.madx")
  # Does this delete all errors too? There is a use command inside it seems like it does not
  #mdx.set(sequence="LHCB1") # Use deletes the errors so trying to use set
  mdx.use(sequence="lhcb1")
  #mdx.makethin(sequence=f"LHCB{beam}", style="TEAPOT")
  mdx.seterr(table="etable")
  mdx.exec(f"match_tunes(62.31, 60.32, {beam}")
  mdx.twiss()
  

  mdx.input(f"""
  betxac=table(twiss,MSIA.EXIT.B1,betx);
  betyac=table(twiss,MSIA.EXIT.B1,bety);

  ampx = 1.0E-6; ! Action
  ampy = 1.0E-6;
  !ampx = sqrt(180.0*betxac)*1e-5;
  !ampy = sqrt(177.0*betyac)*1e-5;
             
  TRACK, deltap=0.0, onetable, dump, file="trackone";
      START, FX=ampx, FY=ampy;
      CALL, file=observe_list.madx;
      RUN, turns = 1000;
  ENDTRACK;
  """) #Onepass is done to specify the coordinates wrt the reference orbit! is this only for initial coordinates?
  
  mdx.quit()


if __name__ == "__main__":
   main()


# %%
