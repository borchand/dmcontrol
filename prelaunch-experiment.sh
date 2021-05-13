exp_name=exp5_inv1.0_relu
inv_coef=0.1
smoothness_coef=0.0
prefix='PIPENV_IGNORE_VIRTUALENVS=1 xvfb-run -a pipenv run '
# prefix='MUJOCO_GL=egl unbuffer xvfb-run -a '
onager prelaunch +command "${prefix} python -m train --domain_name cheetah --task_name run --replicate --markov_lr 2e-4 --markov_inv_coef ${inv_coef} --markov_smoothness_coef ${smoothness_coef} --work_dir ./tmp/${exp_name}" +arg --seed {1..10} +jobname markov_${exp_name}_cheetah +tag --tag +tag-args '' +q
onager prelaunch +command "${prefix} python -m train --domain_name cartpole --task_name swingup --replicate --markov_lr 1e-3 --markov_inv_coef ${inv_coef} --markov_smoothness_coef ${smoothness_coef} --work_dir ./tmp/${exp_name}" +arg --seed {1..10} +jobname markov_${exp_name}_cartpole +tag --tag +tag-args '' +q
onager prelaunch +command "${prefix} python -m train --domain_name ball_in_cup --task_name catch --replicate --markov_lr 1e-3 --markov_inv_coef ${inv_coef} --markov_smoothness_coef ${smoothness_coef} --work_dir ./tmp/${exp_name}" +arg --seed {1..10} +jobname markov_${exp_name}_ball_in_cup +tag --tag +tag-args '' +q
onager prelaunch +command "${prefix} python -m train --domain_name finger --task_name spin --replicate --markov_lr 1e-3 --markov_inv_coef ${inv_coef} --markov_smoothness_coef ${smoothness_coef} --work_dir ./tmp/${exp_name}" +arg --seed {1..10} +jobname markov_${exp_name}_finger +tag --tag +tag-args '' +q
onager prelaunch +command "${prefix} python -m train --domain_name reacher --task_name easy --replicate --markov_lr 1e-3 --markov_inv_coef ${inv_coef} --markov_smoothness_coef ${smoothness_coef} --work_dir ./tmp/${exp_name}" +arg --seed {1..10} +jobname markov_${exp_name}_reacher +tag --tag +tag-args '' +q
onager prelaunch +command "${prefix} python -m train --domain_name walker --task_name walk --replicate --markov_lr 1e-3 --markov_inv_coef ${inv_coef} --markov_smoothness_coef ${smoothness_coef} --work_dir ./tmp/${exp_name}" +arg --seed {1..10} +jobname markov_${exp_name}_walker +tag --tag +tag-args '' +q
