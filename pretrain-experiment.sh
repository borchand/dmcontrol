if [ -z ${exp_name+x} ]; then echo "exp_name is unset. try exp_name=exp0_default"; exit 1; fi
inv_coef=1.0
smoothness_coef=10.0
smoothness_dz=0.01
if command -v sbatch &> /dev/null;
then
    # backend=slurm
    prefix='PIPENV_IGNORE_VIRTUALENVS=1 xvfb-run -a pipenv run '
elif command -v qsub &> /dev/null;
then
    # backend=gridengine
    prefix="MUJOCO_GL=egl unbuffer xvfb-run -a "
else
    echo "no backend detected"; exit 1
fi
base_cmd="${prefix} python -m train --replicate --markov --init_steps 10000 --markov_catchup_steps 9000 --markov_pretrain_steps 100000 --markov_inv_coef ${inv_coef} --markov_smoothness_coef ${smoothness_coef} --markov_smoothness_max_dz ${smoothness_dz} --work_dir ./tmp/${exp_name}"
suffix="+tag --tag +tag-args '' +q"
onager prelaunch +command "${base_cmd} --domain_name cheetah --task_name run --markov_lr 2e-4" +arg --seed {1..10} +jobname markov_${exp_name}_cheetah ${suffix}
# onager prelaunch +command "${base_cmd} --domain_name cartpole --task_name swingup --markov_lr 1e-3" +arg --seed {1..10} +jobname markov_${exp_name}_cartpole ${suffix}
# onager prelaunch +command "${base_cmd} --domain_name ball_in_cup --task_name catch --markov_lr 1e-3" +arg --seed {1..10} +jobname markov_${exp_name}_ball_in_cup ${suffix}
onager prelaunch +command "${base_cmd} --domain_name finger --task_name spin --markov_lr 1e-3" +arg --seed {1..10} +jobname markov_${exp_name}_finger ${suffix}
onager prelaunch +command "${base_cmd} --domain_name reacher --task_name easy --markov_lr 1e-3" +arg --seed {1..10} +jobname markov_${exp_name}_reacher ${suffix}
onager prelaunch +command "${base_cmd} --domain_name walker --task_name walk --markov_lr 1e-3" +arg --seed {1..10} +jobname markov_${exp_name}_walker ${suffix}
