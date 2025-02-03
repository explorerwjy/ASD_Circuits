#parallel -j 30 --eta python script_IQ_permutation.py -i {}  ::: $(seq 10000)
#parallel -j 30 --eta python script_Gender_permutation.py -i {}  ::: $(seq 10000)
parallel -j 30 --eta python script_MutType_permutation.py -i {}  ::: $(seq 10000)

