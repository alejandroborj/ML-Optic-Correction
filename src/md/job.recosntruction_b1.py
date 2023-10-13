#%%
from cpymad import madx
import matplotlib.pyplot as plt
import numpy as np
import tfs
import pandas as pd

import joblib

from measurement_to_input import get_delta_and_meas_phase
from measurement_to_input import get_beta_beating_from_measurement

def main():

    measurement_name = "test_best_know"
    # Different measurement but correct error generation, to make an fake measurement as it will be on the DM
    measurement_path_b1 = "./measurements/B1_with_arc45_and_arc81_knobs/"
    #measurement_path_b1 = "./measurements/02-40-10_import_b1_30cm_beforeKmod"

    #measurement_path_b2 = "./measurements/b2_30cm_newvirginmeasurements/"
    measurement_path_b2 = "./measurements/19-06-04_import_b2_30cm_nocorrinarcs/"
    
    estimator = joblib.load("./md_models/arcb2_best_know_0.0001.pkl") 
    
    twiss_b2 = tfs.read_tfs("./data_analysis/b2_nominal_monitors.dat").set_index("NAME")
    twiss_b1 = tfs.read_tfs("./data_analysis/b1_nominal_monitors.dat").set_index("NAME")

    phasecut_x = 0.4
    phasecut_y = 0.4
    bbeat_cut = 150 #No bbeat_cut

    #Beta beating
    betabeatx_b1, betabeaty_b1 = get_beta_beating_from_measurement(measurement_path_b1)
    betabeatx_b2, betabeaty_b2 = get_beta_beating_from_measurement(measurement_path_b2)

    meas_betabeat_b1 = pd.DataFrame(columns=["NAME", "S", "BETX", "BETY"])
    meas_betabeat_b1.NAME = twiss_b1.NAME
    meas_betabeat_b1.S = twiss_b1.S
    meas_betabeat_b1.BETX = betabeatx_b1
    meas_betabeat_b1.BETY = betabeaty_b1

    meas_betabeat_b2 = pd.DataFrame(columns=["NAME", "S", "BETX", "BETY"])
    meas_betabeat_b2.NAME = twiss_b2.NAME
    meas_betabeat_b2.S = twiss_b2.S
    meas_betabeat_b2.BETX = betabeatx_b2
    meas_betabeat_b2.BETY = betabeaty_b2

    # Taking both inputs
    delta_meas_mux_b1, delta_meas_muy_b1, meas_mux_b1, meas_muy_b1 = get_delta_and_meas_phase(measurement_path_b1, twiss_b1, 
                                                              phasecut_x, phasecut_y, bbeat_cut)
    delta_meas_mux_b2, delta_meas_muy_b2, meas_mux_b2, meas_muy_b2 = get_delta_and_meas_phase(measurement_path_b2, twiss_b2, 
                                                              phasecut_x, phasecut_y, bbeat_cut)

    sample = delta_meas_mux_b1, delta_meas_muy_b1, delta_meas_mux_b2, delta_meas_muy_b2

    np.save(f'./measurements/meas_input_{measurement_name}.npy', np.array(sample, dtype=object))
    delta_meas_mux_b1, delta_meas_muy_b1, delta_meas_mux_b2, delta_meas_muy_b2 = np.load(f'./measurements/meas_input_{measurement_name}.npy', allow_pickle=True)
    meas_input = np.concatenate([delta_meas_mux_b2, delta_meas_muy_b2])
    
    #delta_meas_mux_b1, delta_meas_muy_b1, 
 
    
    pred_output = estimator.predict([meas_input])[0]

    error_tfs = save_np_errors_tfs(pred_output, "pred_best_know_err.tfs") #Savin errors in tfs file

    #mdx = madx.Madx()
    #tw_recons = recons_twiss("b1_pred_best_know_err.tfs", 1, mdx)
    #plot_betabeat_reconstruction(meas_betabeat_b1, tw_recons, 1)
    #mdx.quit()
    
    mdx = madx.Madx()
    tw_recons = recons_twiss("b2_pred_best_know_err.tfs", 2, mdx)
    plot_betabeat_reconstruction(meas_betabeat_b1, tw_recons, 2)
    mdx.quit()


def recons_twiss(error_file, beam, mdx):
    mdx.options(echo=False)
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2022.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/lhc/optics/V6.5/errors/Esubroutines.madx")
    mdx.call(file = "/afs/cern.ch/eng/acc-models/lhc/2022/lhc.seq")
    mdx.options(echo=True)

    mdx.input("exec, define_nominal_beams(energy=6500);")
    mdx.call(file="/afs/cern.ch/eng/acc-models/lhc/2022/operation/optics/R2023a_A30cmC30cmA10mL200cm.madx")
    mdx.input("exec, cycle_sequences();")

    mdx.use(sequence=f"LHCB{beam}")

    mdx.options(echo=False)

    mdx.input(f"exec, match_tunes(62.31, 60.32, {beam});")
    mdx.twiss(sequence=f"LHCB{beam}", file="") #Chrom deltap=0.0

    #Assigning errors
    mdx.input(f"""
                readtable, file = "/afs/cern.ch/eng/sl/lintrack/error_tables/Beam{beam}/error_tables_6.5TeV/MBx-0001.errors", table=errtab;
                seterr, table=errtab;
                READMYTABLE, file="/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/md/data_analysis/{error_file}", table=errtab;
                SETERR, TABLE=errtab;""")

    mdx.twiss(sequence=f"LHCB{beam}", file="") #Chrom deltap=0.0
    #mdx.input(f"match_tunes(62.31, 60.32, {beam});")
    mdx.input(f"""etable, table="final_error";""")
    
    mdx.twiss(sequence=f"LHCB{beam}", file="")    

    #print("Eroror", mdx.table.final_error.dframe())
    #tfs.writer.write_tfs(tfs_file_path=f"final_errors.tfs", data_frame=mdx.table.final_error.dframe())

    # Generate twiss with columns needed for training data
    mdx.input(f"""ndx := table(twiss,dx)/sqrt(table(twiss,betx));
                select, flag=twiss, clear;
                select, flag=twiss, pattern="^BPM.*B{beam}$", column=name, s, betx, bety, ndx,
                                                mux, muy;
                twiss, chrom, sequence=LHCB{beam}, deltap=0.0, file="";""")

    return mdx.table.twiss.dframe()

def plot_betabeat_reconstruction(tw_true, tw_recons, beam):
    if beam==1:
        tw_nominal = tfs.read_tfs("../generate_data/nominal_twiss/b1_nominal_monitors.dat").set_index("NAME")
    elif beam==2:
        tw_nominal = tfs.read_tfs("../generate_data/nominal_twiss/b2_nominal_monitors.dat").set_index("NAME")

    tw_recons = tw_recons.set_index("name") 
    tw_recons.index = [(idx.upper()).split(':')[0] for idx in tw_recons.index]
    tw_recons.columns = [col.upper() for col in tw_recons.columns]

    tw_recons = tw_recons[tw_recons.index.isin(tw_nominal.index)]
    
    bbeat_x_recons = 100*(np.array(tw_recons.BETX - tw_nominal.BETX))/tw_nominal.BETX
    bbeat_y_recons = 100*(np.array(tw_recons.BETY - tw_nominal.BETY))/tw_nominal.BETY

    fig, axs = plt.subplots(2)
    axs[0].plot(tw_true.S, tw_true.BETX, label="True", alpha=0.7)
    axs[0].plot(tw_recons.S, bbeat_x_recons, label="Rec", alpha=0.7)
    axs[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[0].set_ylabel(r"$\Delta \beta _x / \beta _x [\%]$")
    axs[0].set_xticklabels(labels=['IP2', 'IP3', 'IP4', 'IP5', 'IP6', 'IP7', 'IP8', 'IP1'])
    axs[0].set_xticks([i for i in np.linspace(0, int(tw_true.S[-1]), num=8)])
    axs[0].legend()

    axs[1].plot(tw_true.S, tw_true.BETY, label="True", alpha=0.7)
    axs[1].plot(tw_recons.S, bbeat_y_recons, label="Rec", alpha=0.7)
    axs[1].set_ylabel(r"$\Delta \beta _y / \beta _y [\%]$")
    axs[1].set_xlabel(r"Longitudinal location $[m]$")
    axs[1].legend()

    fig.suptitle(f"Beam {beam}")
    fig.savefig(f"../generate_data/figures/example_twiss_beam{beam}.pdf")
    fig.show()


def save_np_errors_tfs(np_errors, filename):
    #This is the tfs format that can be read, this model of file is then copied and filled
    error_tfs_model_b1 = tfs.read_tfs("./data_analysis/errors_b1.tfs")
    error_tfs_model_b2 = tfs.read_tfs("./data_analysis/errors_b2.tfs")

    #Function that takes np errors and outputs .tfs file with all error values
    with open("./data_analysis/mq_names_best_know.txt", "r") as f:
        lines = f.readlines()
        names = [name.replace("\n", "") for name in lines]

    # Recons_df is a dataframe with the correct names and errors but not format
    recons_df = pd.DataFrame(columns=["NAME","K1L","DS"])
    recons_df.K1L = np_errors[:-32]
    ds_errors = np_errors[-32:]
    recons_df.DS = [0 if i>=32 else ds_errors[i] for i, name in enumerate(names)]
    recons_df.NAME = names

    for beam, error_tfs_model in enumerate([error_tfs_model_b1, error_tfs_model_b2]):
        for i in range(len(error_tfs_model)):
            # check if the name is in recons_df
            if error_tfs_model.loc[i, 'NAME'] in list(recons_df['NAME']):
                error_tfs_model.loc[i, 'K1L'] = recons_df.loc[recons_df['NAME'] == error_tfs_model.loc[i, 'NAME']].values[0][1]
                error_tfs_model.loc[i, 'DS'] = recons_df.loc[recons_df['NAME'] == error_tfs_model.loc[i, 'NAME']].values[0][2]
            
    tfs.writer.write_tfs(tfs_file_path=f"./data_analysis/b1_{filename}", data_frame=error_tfs_model_b1)
    tfs.writer.write_tfs(tfs_file_path=f"./data_analysis/b2_{filename}", data_frame=error_tfs_model_b2)


if __name__ == "__main__":
    main()

# %%
