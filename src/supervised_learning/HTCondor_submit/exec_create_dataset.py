import os
import sys
from create_dataset import create_dataset
import multiprocessing

instance = "%(INSTANCE)d"

# Change to the actual working directory to have access to all data
working_directory = '/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/supervised_learning'
job_directory = f"./datasets/tests/dataset_xing/job_{instance}"

os.chdir(working_directory)
sys.path.append(working_directory)
if not os.path.exists(job_directory):
    os.mkdir(job_directory)

n_processes = 2 # Number of parallel processes
dataset_name = "datasets/toy_dataset" # Folder name to save the data
n_samples = 30 # Number of samples for each process
XING = True # Whether to use a Xing angle setup or not

# Multiprocessing speeds up the process of generating data, for each HTCondor job multiple samples can be generated
# at once and asking for multiple CPUs. MADNG Is CPU hungry, so the n_processes should not be too big

with multiprocessing.Pool(processes=n_processes) as pool:
    args = [[dataset_name, XING, np.random.randint(0, 1E7), n_samples] for i in range(n_processes)]
    pool.map(create_dataset, args)