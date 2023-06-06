
rsync -ar /home/alejandro/Desktop/ML-Optic-Correction/src/generate_data aborjess@cs-ccr-dev3:work/public/ML-Optic-Correction/src

rsync -ar aborjess@cs-ccr-dev3:/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2022-05-02/LHCB1/Results/02-40-10_import_b1_30cm_beforeKmod /home/alejandro/Desktop/ML-Optic-Correction/src/twiss_reconstruction/measurement
cd work/public/ML-Optic-Correction/src/generate_data

for i in {1..40}
do
    echo -e "\nInstance $i\n"
    python generate_data.py &
done
wait
