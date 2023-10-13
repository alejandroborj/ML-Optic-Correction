
import os
import sys

instace = %(INSTANCE)s
# Change to the actual working directory to have access to all data
os.chdir('/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/non_linear_tests/supervised_learning')
sys.path.append('/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/non_linear_tests/supervised_learning')

#print("Current working directory: {0}".format(os.getcwd()))

import create_dataset
create_dataset.main()