#!/usr/bin/env bash

# This script assumes scores are stored in the following directory structure:
# logs
# ├── tuning-rad-markov_254293_1.o
# ├── tuning-rad-markov_254293_2.o
# ├── tuning-rad-markov_254293_3.o
# ├── tuning-rad-markov_254293_4.o
# ├── tuning-rad-markov_254293_5.o
# ├── ...
# └── tuning-rad-markov_254293_27.o

job=254293
max_task=27

cd logs
for i in `seq 1 ${max_task}`;
do
    echo tuning-rad-markov_${job}_${i}.o '->' tuning-rad-markov_${i}.g
    cat tuning-rad-markov_${job}_${i}.o | grep "eval" | sed -E 's/\| .*eval.* \| S: //; s/ \| ER: /,/;' > tuning-rad-markov_${i}.g
done
