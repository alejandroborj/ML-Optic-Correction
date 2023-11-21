import os
import sys

instance = "%(INSTANCE)d"

# Change to the actual working directory to have access to all data
os.chdir('/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/supervised_learning')
sys.path.append('/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/supervised_learning')
os.mkdir(f"./xing_dataset/job_{instance}")
os.mkdir(f"./xing_dataset/job_{instance}/errors")
os.mkdir(f"./xing_dataset/job_{instance}/samples")

#print("Current working directory: {0}".format(os.getcwd()))

import create_dataset
create_dataset.main(dataset_name=f"xing_dataset/job_{instance}")