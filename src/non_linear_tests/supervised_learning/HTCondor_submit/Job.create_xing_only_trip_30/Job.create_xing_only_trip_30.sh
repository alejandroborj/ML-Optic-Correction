#!/bin/bash
mkdir /afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/non_linear_tests/supervised_learning/datasets/xing_dataset
/afs/cern.ch/user/a/aborjess/work/public/miniconda3/bin/python /afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/non_linear_tests/supervised_learning/HTCondor_submit/Job.create_xing_only_trip_30/exec_create_dataset.py 
