#%%
from pymadng import MAD
import time

with MAD() as mad:
    start = time.time()
    mad.loadfile("rdt-ip.mad")
    finish = mad.recv()
    stop = time.time()

    print(finish)
    print(f"Execution time: {stop-start} (s)")

# %%
