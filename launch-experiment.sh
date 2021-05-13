exp_name=exp5_inv1.0_relu
tasks=1-3
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_cartpole --tasklist ${tasks}
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_ball_in_cup --tasklist ${tasks}
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_finger --tasklist ${tasks}
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_reacher --tasklist ${tasks}
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_walker --tasklist ${tasks}
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_cheetah --tasklist ${tasks}
