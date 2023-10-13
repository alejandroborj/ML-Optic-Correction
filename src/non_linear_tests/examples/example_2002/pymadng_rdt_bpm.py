#%%
from pymadng import MAD
import pandas as pd
import time
import numpy as np

with MAD(mad_path = r"/home/alejandro/mad-linux-0.9.7-pre") as mad:

    start = time.time()

    corr = 0.7

    mad.send(f"""assert(loadfile("rdt_bpm.mad"))({corr})""") # Error is 0.5
    mad.send("""py:send("Finish")""")

    finish = mad.recv()
    stop = time.time()

    rdt_df = pd.read_csv("RDT_BPMS_b1.csv", sep="\t")
    mean_f30000 = np.mean((rdt_df["RE_300000"]**2 +rdt_df["IM_300000"]**2)**0.5)


    print(f"Execution time: {stop-start} (s)")

# %%
