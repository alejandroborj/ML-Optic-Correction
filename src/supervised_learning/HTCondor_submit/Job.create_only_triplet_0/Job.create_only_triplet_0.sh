#!/bin/bash
mkdir /afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/non_linear_tests/supervised_learning/htcondor_dataset
/afs/cern.ch/user/a/aborjess/work/public/miniconda3/bin/python /afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/non_linear_tests/supervised_learning/HTCondor_submit/Job.create_only_triplet_0/exec_create_dataset.py 
