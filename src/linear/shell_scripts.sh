
rsync -ar aborjess@lxplus:work/public/ML-Optic-Correction/src /home/alejandro/Desktop/ML-Optic-Correction/

rsync -ar aborjess@cs-ccr-dev1:/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2024-03-08/LHCB1/Results/B1_30cm_without_guessed_IP1_local_corrections /home/alejandro/Desktop/ML-Optic-Correction/afs/md_light/measurements

# Triplet measurements
/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2022-05-02/LHCB1/Results/02-38-46_import_4_files_beam1
/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2022-05-02/LHCB2/Results/02-38-46_import_b2_3files_12percent_30cm_beforecorrection
/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2022-05-26/LHCB2/Results/19-06-04_import_b2_30cm_nocorrinarcs

cd work/public/ML-Optic-Correction/src/generate_data

for i in {1..15}
do
    echo -e "\nInstance $i\n"
    python generate_data.py &
done
wait

