#%%
from pylhc_submitter.job_submitter import main as htcondor_submit

if __name__ == "__main__":
    htcondor_submit(
        executable="/afs/cern.ch/user/a/aborjess/work/public/miniconda3/bin/python",  # default pointing to the latest MAD-X on afs
        mask="exec_create_dataset.py",  # Code to execute
        replace_dict=dict(INSTANCE=[i for i in range(100)]),
        jobid_mask="create_xing_only_trip_%(INSTANCE)d",  # naming of the submitted jobfiles
        working_directory="/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/non_linear_tests/supervised_learning/HTCondor_submit",  # outputs
        #jobflavour="workday",  # htcondor flavour
        job_output_dir="/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/non_linear_tests/supervised_learning/xing_dataset",
        run_local=True
        )
#htc_arguments={"initialdir":"/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/non_linear_tests/supervised_learning"}

# %%
