from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
# Beta-Beat.src to import tfs_pandas
# Using python 3 --> import tfs
sys.path.append('./Beta-Beat.src/')
from tfs_files import tfs_pandas


# Is delta_phaseadv to model already written to getllm output?
def get_phase_diff_from_twiss(twiss_perturbed, model_path, measurement_path):
    getphasex = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getphasex_free2.out')).set_index("NAME")
    getphasey = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getphasey_free2.out')).set_index("NAME")
    mdl = tfs_pandas.read_tfs(model_path).set_index("NAME")
    mdl_x = mdl.loc[getphasex.index]
    mdl_y = mdl.loc[getphasey.index]
    twiss = tfs_pandas.read_tfs(twiss_perturbed).set_index("NAME")
    twiss_x = twiss.loc[getphasex.index]
    twiss_y = twiss.loc[getphasey.index]
    delta_mux = twiss_x.MUX - mdl_x.MUX
    delta_muy = twiss_y.MUY - mdl_y.MUY
    delta_phaseadv_x = np.diff(delta_mux)
    delta_phaseadv_y = np.diff(delta_muy)

    return delta_phaseadv_x, delta_phaseadv_y


def get_beta_beating_from_measurement(measurement_path):
    if os.path.isfile(os.path.join(measurement_path, 'getbetax_free2.out')):
        getbetax = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getbetax_free2.out')).set_index("NAME")
        getbetay = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getbetay_free2.out')).set_index("NAME")
    else:
        getbetax = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getbetax.out')).set_index("NAME")
        getbetay = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getbetay.out')).set_index("NAME")
    betabeatx = (getbetax.BETX - getbetax.BETXMDL) / getbetax.BETXMDL * 100
    betabeaty = (getbetay.BETY - getbetay.BETYMDL) / getbetay.BETYMDL * 100
    # plt.plot(range(len(betabeatx)), betabeatx, label="bbeatx")
    # plt.legend()
    # plt.show()
    # plt.plot(range(len(betabeaty)), betabeaty, label="bbeaty")
    # plt.legend()
    # plt.show()
    return betabeatx, betabeaty



def get_delta_and_meas_phase(measurement_path, twiss, phasecut_x, phasecut_y, bbeat_cut):
    if os.path.isfile(os.path.join(measurement_path, 'getphasex_free2.out')):
        getphasex_original = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getphasex_free2.out')).set_index("NAME")
        getphasey_original = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getphasey_free2.out')).set_index("NAME")
        getphasetotx_original = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getphasetotx_free2.out')).set_index("NAME")
        getphasetoty_original = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getphasetoty_free2.out')).set_index("NAME")
    else:
        getphasex_original = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getphasex.out')).set_index("NAME")
        getphasey_original = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getphasey.out')).set_index("NAME")
        getphasetotx_original = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getphasetotx.out')).set_index("NAME")
        getphasetoty_original = tfs_pandas.read_tfs(os.path.join(measurement_path, 'getphasetoty.out')).set_index("NAME")
    
    betabeatx, betabeaty = get_beta_beating_from_measurement(measurement_path)
    non_outliers_betabeatx = betabeatx[np.abs(betabeatx) < bbeat_cut].index
    non_outliers_betabeaty = betabeaty[np.abs(betabeaty) < bbeat_cut].index

    getphasex = getphasex_original.loc[non_outliers_betabeatx]
    getphasey = getphasey_original.loc[non_outliers_betabeaty]
    getphasextot = getphasetotx_original.loc[non_outliers_betabeatx]
    getphaseytot = getphasetoty_original.loc[non_outliers_betabeaty]

   
    delta_mux_raw = (getphasex.PHASEX - getphasex.PHXMDL)
    delta_muy_raw = (getphasey.PHASEY - getphasey.PHYMDL)
    # plt.plot(range(len(delta_mux_raw)), delta_mux_raw, label="phasex")
    # plt.legend()
    # plt.show()
    # plt.plot(range(len(delta_muy_raw)), delta_muy_raw, label="phasey")
    # plt.legend()
    # plt.show()
    meas_mux_raw = getphasextot.PHASEX
    meas_muy_raw = getphaseytot.PHASEY

    bpms_names_x = getphasex.index.values[np.abs(delta_mux_raw) < phasecut_x]
    bpms_names_y = getphasey.index.values[np.abs(delta_muy_raw) < phasecut_y]
   
    delta_mux_raw = delta_mux_raw[np.abs(delta_mux_raw) < phasecut_x]
    delta_muy_raw = delta_muy_raw[np.abs(delta_muy_raw) < phasecut_y]

    meas_mux_raw = meas_mux_raw.loc[bpms_names_x]
    meas_muy_raw = meas_muy_raw.loc[bpms_names_y]
    meas_mux_raw = meas_mux_raw[np.abs(meas_mux_raw) < 1.1]
    meas_muy_raw = meas_muy_raw[np.abs(meas_muy_raw) < 1.1]

    delta_mux = delta_mux_raw.reindex(twiss.index, fill_value=np.nan)
    delta_muy = delta_muy_raw.reindex(twiss.index, fill_value=np.nan)
    delta_mux_interpolated = delta_mux.interpolate()
    delta_muy_interpolated = delta_muy.interpolate()

    meas_mux = meas_mux_raw.reindex(twiss.index, fill_value=np.nan).interpolate()
    meas_muy = meas_muy_raw.reindex(twiss.index, fill_value=np.nan).interpolate()
    
    delta_mux_interpolated[np.abs(delta_mux_interpolated) > phasecut_x] = np.mean(np.abs(delta_mux_interpolated[np.abs(delta_mux_interpolated) < phasecut_x]))
    delta_muy_interpolated[np.abs(delta_muy_interpolated) > phasecut_y] = np.mean(np.abs(delta_muy_interpolated[np.abs(delta_muy_interpolated) < phasecut_y]))

    return delta_mux_interpolated, delta_muy_interpolated, meas_mux, meas_muy