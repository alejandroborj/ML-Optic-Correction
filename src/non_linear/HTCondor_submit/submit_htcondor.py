#%%
from pylhc_submitter.job_submitter import main as htcondor_submit

if __name__ == "__main__":
    htcondor_submit(
        executable="/afs/cern.ch/user/a/aborjess/work/public/miniconda3/bin/python",  # default pointing to the latest MAD-X on afs
        mask="exec_create_dataset.py",  # Code to execute
        replace_dict=dict(INSTANCE=[i for i in range(200)]),
        jobid_mask="create_new_dataset_%(INSTANCE)d",  # naming of the submitted jobfiles
        working_directory="/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/supervised_learning/HTCondor_submit",  # outputs
        #jobflavour="workday",  # htcondor flavour
        #job_output_dir="/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/supervised_learning/datasets/no_xing_dataset",
        run_local=False
        )
#htc_arguments={"initialdir":"/afs/cern.ch/user/a/aborjess/work/public/ML-Optic-Correction/src/supervised_learning"}

# %%
