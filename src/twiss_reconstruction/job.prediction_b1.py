#%%
from cpymad import madx
import matplotlib.pyplot as plt
import numpy as np
import tfs
import pandas as pd

def main():
    mdx = madx.Madx()
    tw_recons = recons_twiss("b1_pred_example.tfs", 1, mdx)
    tw_true = recons_twiss("b1_true_example.tfs", 1, mdx)
    plot_betabeat_reconstruction(tw_true, tw_recons, 1)
    mdx.quit()

    mdx = madx.Madx()
    tw_recons = recons_twiss("b2_pred_example.tfs", 2, mdx)
    tw_true = recons_twiss("b2_true_example.tfs", 2, mdx)
    plot_betabeat_reconstruction(tw_true, tw_recons, 2)
    mdx.quit()


def recons_twiss(error_file, beam, mdx):
    mdx.options(echo=False)
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/beta_beat.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/sl/lintrack/Beta-Beat.src/madx/lib/lhc_runIII_2022.macros.madx")
    mdx.call(file="/afs/cern.ch/eng/lhc/optics/V6.5/errors/Esubroutines.madx")
    mdx.call(file = "/afs/cern.ch/eng/acc-models/lhc/2022/lhc.seq")
    mdx.options(echo=True)

    mdx.input("exec, define_nominal_beams();")
    mdx.call(file="/afs/cern.ch/eng/acc-models/lhc/2022/operation/optics/R2023a_A30cmC30cmA10mL200cm.madx")
    mdx.input("exec, cycle_sequences();")

    mdx.use(sequence=f"LHCB{beam}")

    mdx.options(echo=False)

    mdx.input(f"exec, match_tunes(62.31, 60.32, {beam});")

    #Assigning errors
    mdx.input(f"""readtable, file = "/afs/cern.ch/eng/sl/lintrack/error_tables/Beam{beam}/error_tables_6.5TeV/MBx-0001.errors", table=errtab;
                seterr, table=errtab;
                !select, flag=error, clear;
                READMYTABLE, file="/home/alejandro/Desktop/ML-Optic-Correction/src/twiss_reconstruction/{error_file}", table=errtab;
                SETERR, TABLE=errtab;""")

    mdx.twiss(sequence=f"LHCB{beam}", file="") #Chrom deltap=0.0

    #mdx.input(f"match_tunes(62.31, 60.32, {beam});")
            
    mdx.twiss(sequence=f"LHCB{beam}", file="")
    # Generate twiss with columns needed for training data
    mdx.input(f"""ndx := table(twiss,dx)/sqrt(table(twiss,betx));
                select, flag=twiss, clear;
                select, flag=twiss, pattern="^BPM.*B{beam}$", column=name, s, betx, bety, ndx,
                                                mux, muy;
                twiss, chrom, sequence=LHCB{beam}, deltap=0.0, file="";""")

    return mdx.table.twiss.dframe()

def plot_betabeat_reconstruction(tw_true, tw_recons, beam):
    if beam==1:
        tw_nominal = tfs.read_tfs("../generate_data/nominal_twiss/b1_nominal_monitors_30.dat").set_index("NAME")
    elif beam==2:
        tw_nominal = tfs.read_tfs("../generate_data/nominal_twiss/b2_nominal_monitors_30.dat").set_index("NAME")
    
    tw_true = tw_true.set_index("name") 
    tw_true.index = [(idx.upper()).split(':')[0] for idx in tw_true.index]
    tw_true.columns = [col.upper() for col in tw_true.columns]

    tw_recons = tw_recons.set_index("name") 
    tw_recons.index = [(idx.upper()).split(':')[0] for idx in tw_recons.index]
    tw_recons.columns = [col.upper() for col in tw_recons.columns]

    tw_true = tw_true[tw_true.index.isin(tw_nominal.index)]
    tw_recons = tw_recons[tw_recons.index.isin(tw_nominal.index)]

    bbeat_x = 100*(np.array(tw_true.BETX - tw_nominal.BETX))/tw_nominal.BETX
    bbeat_y = 100*(np.array(tw_true.BETY - tw_nominal.BETY))/tw_nominal.BETY
    
    bbeat_x_recons = 100*(np.array(tw_recons.BETX - tw_nominal.BETX))/tw_nominal.BETX
    bbeat_y_recons = 100*(np.array(tw_recons.BETY - tw_nominal.BETY))/tw_nominal.BETY

    fig, axs = plt.subplots(2)
    axs[0].plot(tw_true.S, bbeat_x, label="True")
    axs[0].plot(tw_recons.S, bbeat_x_recons, label="Rec")
    axs[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    axs[0].set_ylabel(r"$\Delta \beta _x / \beta _x [\%]$")
    axs[0].set_xticklabels(labels=['IP2', 'IP3', 'IP4', 'IP5', 'IP6', 'IP7', 'IP8', 'IP1'])
    axs[0].set_xticks([i for i in np.linspace(0, int(tw_true.S[-1]), num=8)])
    axs[0].legend()

    axs[1].plot(tw_true.S, bbeat_y, label="True")
    axs[1].plot(tw_recons.S, bbeat_y_recons, label="Rec")
    axs[1].set_ylabel(r"$\Delta \beta _y / \beta _y [\%]$")
    axs[1].set_xlabel(r"Longitudinal location $[m]$")
    axs[1].legend()

    fig.suptitle(f"Beam {beam}")
    fig.savefig(f"../generate_data/figures/example_twiss_beam{beam}.pdf")
    fig.show()

if __name__ == "__main__":
    main()

# %%
