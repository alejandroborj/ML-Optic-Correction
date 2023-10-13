# %%
import omc3, tfs, os
from pathlib import Path
from omc3 import hole_in_one
from omc3 import tbt_converter
from omc3 import model_creator

noise = "0"

tbt_converter.converter_entrypoint(
    files=["trackoneone"],
    tbt_datatype="trackone",
    outputdir="./",
    noise_levels=[noise])


#%%

#model_file = "b1_monitors.dat"

# Free and Forced tunes

qx, qy = 62.31, 60.32
turns = [0, 1000]
nat_tunes = [qx, qy, 0.]
dpp = 0.0

files =[f"./trackoneone_n{noise}.sdds"]
optics_folder="output"

# Call Harpy from OMC3 to do the frequency analysis
hole_in_one.hole_in_one_entrypoint(
    harpy=True,
    #clean = True,
    files=files,
    turns=turns,
    accel='lhc',
    model_dir='./lhc_model',
    to_write=['lin', 'full_spectra', 'bpm_summary'],
    unit='m',
    #tbt_datatype='trackone',
    tunes=nat_tunes,
    nat_tunes=nat_tunes,
    outputdir=optics_folder
    )

#%%
from omc3 import model_creator
from pathlib import Path
model_creator.create_instance_and_model(accel = "lhc",
                          year="2022",
                          beam=1,
                          energy=6.5,
                          nat_tunes=[62.31,60.32],
                          type="nominal",
                          modifiers=[Path("/afs/cern.ch/eng/acc-models/lhc/2022/operation/optics/R2023a_A30cmC30cmA10mL200cm.madx")],
                          outputdir=Path("./lhc_model"),
                          dpp=None
                          )
#print(parameter)
#model_creator.entrypoint(parameter=parameter)

# %%
# Do optics analysis

hole_in_one.hole_in_one_entrypoint(
    optics=True,
    files=[f"./output/trackoneone_n0.sdds"],
    accel='lhc',
    model_dir='./lhc_model',
    beam=1,
    year="2022",
    #nat_tunes=[qx, qy],
    outputdir=optics_folder,
    #three_bpm_method=True,
    nonlinear=["rdt"],
    compensation='none',
    #three_bpm_method=True
)

#%%
import matplotlib.pyplot as plt
import pandas as pd
import tfs

freq_data = tfs.read_tfs(f"./output/trackoneone_n{noise}.sdds.freqsx")
amp_data = tfs.read_tfs(f"./output/trackoneone_n{noise}.sdds.ampsx")

print(amp_data)

plt.yscale("log")
plt.plot(list(freq_data["LHCB1MSIA.EXIT.B1_P_"]), list(amp_data["LHCB1MSIA.EXIT.B1_P_"]))



# %%
import pandas as pd
import tfs

df = tfs.read("twiss.dat")

with open("observe_list.txt", "w") as f:
    i=0
    for name in df['NAME']:
        if i%5==0:
            f.write(f"OBSERVE, PLACE={name};\n")
        i+=1

# %%
'''
import os
model_file = "b1_monitors.dat"
# driven tunes
tunex = 62.31
tuney = 60.32
tunez = 0.0 # if longitudinal plane was also kicked, a value should be given here
# natural tunes 
nat_tunex = 62.31
nat_tuney = 60.32
nat_tunez = 0.0
# turns used for analysis
start = 0
end = 1000
sdds_files =[f"trackoneone_n{noise}.sdds"] 
# more information on the settings can be found in the documentation in the "hole_in_one.py" file 
for sdds_file in sdds_files:
    optics_folder = "output"
    # Harpy
    os.system("/afs/cern.ch/eng/sl/lintrack/omc_python3/bin/python3" # path to python3
    " /afs/cern.ch/eng/sl/lintrack/OMC_Repositories/omc3/omc3/hole_in_one.py" # path to omc3 
    " --harpy" # trigger harmonics analysis
    #" --clean" # trigger cleaning of the data
    " --files " + sdds_file + # file or files; files separtaed by comma, i.e. "file1,file2"
    " --outputdir " + str(optics_folder) + # output directory
    #" --model " + model_file + # model file
    " --tunes " + str(tunex) + " " + str(tuney) + " " + str(tunez)+ # driven tunes, if free oscillation equal to natural tunes
    " --nattunes " + str(nat_tunex) + " " + str(nat_tuney) + " " + str(nat_tunez)+ # natural tunes
    " --turns " + str(start) + " " + str(end) + # used turns for analysis
    #" --tolerance 0.15" # 
    " --unit m" # unit of the tracking data
    " --to_write full_spectra" # write full spectrum etc
    " --to_write lin" # write only lin files
    #" --max_peak 10.0" # max peak accepted by cleaning
    #" --sing_val 46" # used number of largest singular values
    #" --tune_clean_limit 1e-3" # outliers closer than this limit to the average tune will not be removed
    " --accel lhc" # define accelerator class
    )

    # Optics
    os.system("/afs/cern.ch/eng/sl/lintrack/omc_python3/bin/python3" # path to python3
    " /afs/cern.ch/eng/sl/lintrack/OMC_Repositories/omc3/omc3/hole_in_one.py" # path to omc3 
    " --optics" + # trigger optics analysis
    " --files " + "./" + str(optics_folder) + "/" + sdds_file + # path to lin files, if only optics analysis is required
    " --outputdir " + str(optics_folder) + # output directory
    " --nat_tunes " + str(nat_tunex) + " " + str(nat_tuney) + 
    # " --model_dir "+model_file+ # model file
    " --accel lhc" # define accelerator class
    #" --ring ler" # always LER for simulations, defines the beam direction
    " --compensation none" # no compensation since free oscillation was used
    " --beam 1"
    " --year 2022"
    " --nonlinear rdt " #crdt" # also measures rdts and crdts
    #" --rdt_magnet_order 8" # max magnet order
    )'''