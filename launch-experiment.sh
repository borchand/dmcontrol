exp_name=fixenc
exp_number=4
tasks=1-3
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_cartpole_${exp_num} --tasklist ${tasks}
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_ball_in_cup_${exp_num} --tasklist ${tasks}
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_finger_${exp_num} --tasklist ${tasks}
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_reacher_${exp_num} --tasklist ${tasks}
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_walker_${exp_num} --tasklist ${tasks}
onager launch --backend slurm --mem 48 --gpus 1 --cpus 1 --duration 2-12:00:00 --jobname markov_${exp_name}_cheetah_${exp_num} --tasklist ${tasks}
