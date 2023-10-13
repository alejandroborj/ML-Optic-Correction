#%%
import matplotlib.pyplot as plt
import pandas as pd
import tfs

#plt.yscale("log")
rdt_data_met6 = pd.read_csv("RDT_BPMS_met6.csv", sep="\t")
rdt_data_met4 = pd.read_csv("RDT_BPMS_met4.csv", sep="\t")
rdt_data_met2 = pd.read_csv("RDT_BPMS_met2.csv", sep="\t")
rdt_data = pd.read_csv("RDT_BPMS.csv", sep="\t")
ptc_rdt_data = tfs.reader.read_tfs("ptc_rdt.tfs")
ptc_rdt_data = ptc_rdt_data[ptc_rdt_data['name'].str.contains("bpm")]
#gnfc_0_0_0_4_0_0

#plt.plot(range(len(rdt_data_met6)), (rdt_data_met6["RE_210000"]**2+rdt_data_met6["IM_210000"]**2)**0.5, alpha=0.7, label="Method 6")
#plt.plot(range(len(rdt_data_met4)), (rdt_data_met4["RE_210000"]**2+rdt_data_met4["IM_210000"]**2)**0.5, alpha=0.7, label="Method 4")
#plt.plot(range(len(rdt_data_met2)), (rdt_data_met2["RE_210000"]**2+rdt_data_met2["IM_210000"]**2)**0.5, alpha=0.7, label="Method 2")

#plt.plot(range(len(rdt_data)), (rdt_data["RE_300000"]**2+rdt_data["IM_300000"]**2)**0.5, alpha=0.7, label="MADNG")
plt.plot(range(len(ptc_rdt_data)), ptc_rdt_data["gnfa_3_0_0_0_0_0"],"o", label="PTC_A")
plt.plot(range(len(ptc_rdt_data)), ptc_rdt_data["gnfs_3_0_0_0_0_0"],"o",  label="PTC_R")
plt.plot(range(len(ptc_rdt_data)), ptc_rdt_data["gnfc_3_0_0_0_0_0"],"o",  label="PTC_I")

plt.xlabel("BPM")
plt.ylabel("$|f_{300000}|$")
plt.legend()

plt.show()
# %%
