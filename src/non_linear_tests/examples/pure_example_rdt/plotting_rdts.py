#%%
import matplotlib.pyplot as plt
import pandas as pd
import tfs

plt.yscale("log")
#ptc_rdt_data = tfs.reader.read_tfs("rdts")
ptc_tw_data = tfs.reader.read_tfs("twiss_b1")
ng_tw_data = tfs.reader.read_tfs("twiss_b1_ng.tfs")
ng_rdt_data = pd.read_csv("RDT_BPMS.csv", sep = "\t")

#ptc_rdt_data = ptc_rdt_data[ptc_rdt_data['NAME'].str.contains("BPM")]
ptc_tw_data = ptc_tw_data[ptc_tw_data['NAME'].str.contains("BPM")]

ptc_tw_data = ptc_tw_data.set_index("NAME")

ng_tw_data = ng_tw_data.set_index("name")

coinciding_indexes = ptc_tw_data.index.intersection(ng_tw_data.index)

ptc_tw_data = ptc_tw_data.loc[coinciding_indexes]
ng_tw_data = ng_tw_data.loc[coinciding_indexes]

print(ng_rdt_data)
plt.plot(range(len(ng_rdt_data)), (ng_rdt_data["RE_400000"]**2+ng_rdt_data["IM_400000"]**2)**0.5, alpha=0.7, label="Method 2")

#plt.plot(range(len(rdt_data)), (rdt_data["RE_300000"]**2+rdt_data["IM_300000"]**2)**0.5, alpha=0.7, label="MADNG")
#plt.plot(range(len(ptc_rdt_data)), ptc_rdt_data["GNFA_3_0_0_0_0_0"],"o", label="PTC_A")
#plt.plot(range(len(ptc_rdt_data)), ptc_rdt_data["GNFS_3_0_0_0_0_0"],"o",  label="PTC_S")
#plt.plot(range(len(ptc_rdt_data)), ptc_rdt_data["GNFC_3_0_0_0_0_0"],"o",  label="PTC_C")

plt.xlabel("BPM")
plt.ylabel("$|f_{300000}|$")
plt.legend()

plt.show()

plt.xlabel("BPM #")
plt.ylabel("Beta x")
plt.plot(range(len(ptc_tw_data)), ptc_tw_data["BETX"]-ng_tw_data["beta11"],"o", label="PTC BETX")
#plt.plot(range(len(ng_tw_data)), ng_tw_data["beta11"],"o", label="NG BETX")
plt.legend()
plt.show()


plt.ylabel("Beta y")
plt.plot(range(len(ptc_tw_data)), ptc_tw_data["BETY"],"o", label="PTC BETY")
#plt.plot(range(len(ng_tw_data)), ng_tw_data["beta22"],"o", label="NG BETY")
plt.legend()
plt.show()

# %%
