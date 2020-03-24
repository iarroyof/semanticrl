#! /bin/bash

FILE="srl_env_mi_tests.py"
NGRAMS="--ngrams ::: $(echo 1) ::: ::: $(echo 2) :::"
WSIZE="--wsize ::: $(echo {4..13}) :::"
SAMPLE="--sample ::: $(echo {10..200}) :::"
NJOBS="--njobs ::: $(echo -1) :::"
NSTEPS="--nsteps ::: $(echo 120) :::"
DENSITY="--density ::: $(echo expset gausset) :::"
BW="--bw ::: $(echo {1..20}) :::"
HITMISS="--hitmiss ::: $(echo {0..200}) :::"
BIAS="--bias ::: $(seq -s " " 1.0 20.0) :::"
IN_OIE="--in_oie ::: $(echo data/dis_train.txt.oie) :::"
IN_TXT="--in_txt ::: $(echo data/dis_train_.txt) :::"
OUTPUT_DIR="--dir ::: $(echo results_1st) :::"

eval "parallel python $FILE $NGRAMS $IN_OIE $IN_TXT $WSIZE $SAMPLE $NJOBS $NSTEPS $DENSITY $BW $HITMISS $BIAS $OUTPUT_DIR"

