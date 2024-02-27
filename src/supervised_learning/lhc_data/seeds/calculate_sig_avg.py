#%%
import pandas as pd
import glob
import tfs

seed_file_names = glob.glob('opticsfile.23-emfqcs-****.tfs')

sum_dataframe, sum_diff_squares = pd.DataFrame(), pd.DataFrame()

for seed_file in seed_file_names:
    seed_df = tfs.read_tfs(seed_file)
    seed_df = seed_df.set_index("NAME")

    seed_df = seed_df.loc[['MQX' in name for name in seed_df.index]]

    if sum_dataframe.empty:
        sum_dataframe = seed_df
    else:
        sum_dataframe += seed_df

mean_dataframe = sum_dataframe/len(seed_file_names)

for seed_file in seed_file_names:
    seed_df = tfs.read_tfs(seed_file)
    seed_df = seed_df.set_index("NAME")

    seed_df = seed_df.loc[['MQX' in name for name in seed_df.index]]

    if sum_diff_squares.empty:
        sum_diff_squares = (seed_df-mean_dataframe)**2
    else:
        sum_diff_squares += (seed_df-mean_dataframe)**2

std_dataframe = (sum_diff_squares/len(seed_file_names))**0.5

mean_dataframe.to_csv("mean_optics_23.csv")
std_dataframe.to_csv("std_optics_23.csv")
# %%
