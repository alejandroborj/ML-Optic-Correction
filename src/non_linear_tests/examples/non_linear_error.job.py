#%%
import cpymad.madx as madx
import tfs
import time


def main():
    calculate_rdts()

def calculate_rdts():
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
  #mdx.call(file="opticsfile.1")
  mdx.exec("cycle_sequences()")

  #Beam 1
  beam = 1

  mdx.use(sequence=f"LHCB{beam}")
  #mdx.twiss(sequence=f"LHCB{beam}", file="") #Chrom deltap=0.0

  mdx.options(echo=False)

  mdx.exec(f"match_tunes(62.31, 60.32, {beam})")
  #mdx.twiss(sequence=f"LHCB{beam}", file="") #Chrom deltap=0.0

  #Assigning errors
  mdx.select(flag="error", clear=True)
  mdx.select(flag="error", pattern = f"^MQ\..*B{beam}")

  mdx.globals.Rr = 0.017
  mdx.globals.B2r = 19
  mdx.globals.on_B2r = 0
  mdx.exec("SetEfcomp_Q")

  mdx.input("""
    !ktsx1_r2 = -ktsx1_l2;
    !kqsx3_r2 =  -1e-5;
    kqsx3_l2 =  1e-5;
  """)
  #mdx.twiss(sequence=f"LHCB{beam}", file="") #Chrom deltap=0.0

  mdx.exec(f"match_tunes(62.31, 60.32, {beam})")

  mdx.input(f"""etable, table="final_error";""")

  mdx.use(sequence=f"LHCB{beam}")
  mdx.makethin(sequence=f"LHCB{beam}", style="TEAPOT")
  mdx.ptc_create_universe()
  mdx.ptc_create_layout(MODEL=3, METHOD=2, NST=1) #check which order i should go to MAEL=6
  start = time.time()
  mdx.ptc_twiss(trackrdts=True, icase=4, no=4, file="b1_monitors.out", closed_orbit=True)
  #Uses last sequence used for chroma at least 2
  stop = time.time()
  mdx.write(table="TWISSRDT", file="rdts.dat")
  #tfs.writer.write_tfs(tfs_file_path=f"ptc_rdt.tfs", data_frame=mdx.table.twissrdt.dframe())
  #mdx.input("""SELECT, FLAG=twissrdt, CLEAR;
  #  WRITE, TABLE=twissrdt, file="ptc_rdt.tfs";""")
  mdx.ptc_end()

  print('Execution time (s): ', stop-start)

  #tfs.writer.write_tfs(tfs_file_path=f"ptc_twiss.tfs", data_frame=mdx.table.ptc_twiss.dframe())
  tfs.writer.write_tfs(tfs_file_path=f"ptc_rdt.tfs", data_frame=mdx.table.twissrdt.dframe())
  mdx.quit()

  #print(mdx.table.twissrdt.dframe()["GNFC_3_0_0_0_0_0"])

  return mdx.table.twissrdt.dframe()["GNFC_3_0_0_0_0_0"]

if __name__ == "__main__":
   main()


# %%
