#%%
from cpymad.madx import Madx

madx = Madx()

# %%
madx.call(file='mad_gen_script.madx')

# %%
