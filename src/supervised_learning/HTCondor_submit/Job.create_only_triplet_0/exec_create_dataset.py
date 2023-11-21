
import os
import sys

instance = "0"

# Change to the actual working directory to have access to all data
os.chdir('/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/non_linear_tests/supervised_learning')
sys.path.append('/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/non_linear_tests/supervised_learning')
os.mkdir(f"./htcondor_dataset/job_{instance}")
os.mkdir(f"./htcondor_dataset/job_{instance}/errors")
os.mkdir(f"./htcondor_dataset/job_{instance}/samples")

#print("Current working directory: {0}".format(os.getcwd()))

import create_dataset
create_dataset.main(dataset_name=f"htcondor_dataset/job_{instance}")