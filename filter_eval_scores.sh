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

cd logs


#------------------

# rad-markov: 254293
# neev: 254416 272081 272853

# rad: 272418 272469 272470

jobname=tuning-rad-markov
min_task=18
max_task=10
for job in 254293;
do
    for i in `seq ${min_task} ${max_task}`;
    do
        if [ -f ${jobname}_${job}_${i}.o ]; then
            echo ${jobname}_${job}_${i}.o '->' ${jobname}_${i}.g
            cat ${jobname}_${job}_${i}.o | grep "eval" | sed -E 's/\| .*eval.* \| S: //; s/ \| ER: /,/;' > ${jobname}_${i}.g
            cat ${jobname}_${job}_${i}.o | grep "train" > ${jobname}_${i}.train
        fi
    done
done

#------------------

# sac: 2374889 2374891 2374892 2374893 2374894 2374895 2374896 2374897 2374898 2374899 2374900 2374901 2374902 2374905
# min_task=1
# max_task=18
# jobname=state-sac
# for job in 2374889 2374891 2374892 2374893 2374894 2374895 2374896 2374897 2374898 2374899 2374900 2374901 2374902 2374905
# do
#     for i in `seq ${min_task} ${max_task}`;
#     do
#         if [ -f ${jobname}_${job}_${i}.o ]; then
#             echo ${jobname}_${job}_${i}.o '->' ${jobname}_${i}.g
#             tr -s '\r' '\n' < ${jobname}_${job}_${i}.o | grep "eval" | sed -E 's/[^|]*\| .*eval.* \| S: //; s/ \| R: /,/;' > ${jobname}_${i}.g
#         fi
#     done
# done
