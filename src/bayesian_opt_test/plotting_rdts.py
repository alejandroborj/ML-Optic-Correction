#%%
import matplotlib.pyplot as plt
import pandas as pd
import tfs

#plt.yscale("log")

ptc_rdt_data_err = tfs.reader.read_tfs("ptc_rdt_err1.tfs")
ptc_rdt_data_nom = tfs.reader.read_tfs("ptc_rdt_nom.tfs")

ptc_rdt_data = ptc_rdt_data_err[ptc_rdt_data_err['name'].str.contains("bpm")]
ptc_rdt_data = ptc_rdt_data_nom[ptc_rdt_data_nom['name'].str.contains("bpm")]

#plt.plot(range(len(rdt_data_met6)), (rdt_data_met6["RE_210000"]**2+rdt_data_met6["IM_210000"]**2)**0.5, alpha=0.7, label="Method 6")
#plt.plot(range(len(rdt_data_met4)), (rdt_data_met4["RE_210000"]**2+rdt_data_met4["IM_210000"]**2)**0.5, alpha=0.7, label="Method 4")
#plt.plot(range(len(rdt_data_met2)), (rdt_data_met2["RE_210000"]**2+rdt_data_met2["IM_210000"]**2)**0.5, alpha=0.7, label="Method 2")

#plt.plot(range(len(rdt_data)), (rdt_data["RE_300000"]**2+rdt_data["IM_300000"]**2)**0.5, alpha=0.7, label="MADNG")
plt.plot(range(len(ptc_rdt_data_nom)), ptc_rdt_data_nom["gnfa_3_0_0_0_0_0"],"o", label="NOM")
plt.plot(range(len(ptc_rdt_data_err)), ptc_rdt_data_err["gnfa_3_0_0_0_0_0"],"o", label="ERR")

plt.xlabel("BPM")
plt.ylabel("$|f_{300000}|$")
plt.legend()

plt.show()
# %%
