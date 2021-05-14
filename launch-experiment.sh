if [ -z ${exp_name+x} ]; then echo "exp_name is unset. try exp_name=exp0_default"; exit 1; fi
if [ -z ${tasks+x} ]; then echo "tasks is unset. try tasks=1-3"; exit 1; fi
venv="--venv ./venv"
if command -v sbatch &> /dev/null
then
    backend=slurm
elif command -v qsub &> /dev/null
    backend=gridengine
else
    echo "no backend detected"; exit 1
fi
settings="--backend ${backend} gridengine --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 ${venv}"
`echo onager launch ${settings} --jobname markov_${exp_name}_cartpole --tasklist ${tasks}`
`echo onager launch ${settings} --jobname markov_${exp_name}_ball_in_cup --tasklist ${tasks}`
`echo onager launch ${settings} --jobname markov_${exp_name}_finger --tasklist ${tasks}`
`echo onager launch ${settings} --jobname markov_${exp_name}_reacher --tasklist ${tasks}`
`echo onager launch ${settings} --jobname markov_${exp_name}_walker --tasklist ${tasks}`
`echo onager launch ${settings} --jobname markov_${exp_name}_cheetah --tasklist ${tasks}`
