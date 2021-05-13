exp_name=exp5_inv1.0_relu
inv_coef=0.1
smoothness_coef=0.0
smoothness_dz=0.01
prefix='PIPENV_IGNORE_VIRTUALENVS=1 xvfb-run -a pipenv run '
# prefix='MUJOCO_GL=egl unbuffer xvfb-run -a '
base_cmd=${prefix} python -m train --replicate --markov_inv_coef ${inv_coef} --markov_smoothness_coef ${smoothness_coef} --markov_smoothness_max_dz ${smoothness_dz} --work_dir ./tmp/${exp_name}
onager prelaunch +command '"'${base_cmd} --domain_name cheetah --task_name run --markov_lr 2e-4'"' +arg --seed {1..10} +jobname markov_${exp_name}_cheetah +tag --tag +tag-args '' +q
onager prelaunch +command '"'${base_cmd} --domain_name cartpole --task_name swingup --markov_lr 1e-3'"' +arg --seed {1..10} +jobname markov_${exp_name}_cartpole +tag --tag +tag-args '' +q
onager prelaunch +command '"'${base_cmd} --domain_name ball_in_cup --task_name catch --markov_lr 1e-3'"' +arg --seed {1..10} +jobname markov_${exp_name}_ball_in_cup +tag --tag +tag-args '' +q
onager prelaunch +command '"'${base_cmd} --domain_name finger --task_name spin --markov_lr 1e-3'"' +arg --seed {1..10} +jobname markov_${exp_name}_finger +tag --tag +tag-args '' +q
onager prelaunch +command '"'${base_cmd} --domain_name reacher --task_name easy --markov_lr 1e-3'"' +arg --seed {1..10} +jobname markov_${exp_name}_reacher +tag --tag +tag-args '' +q
onager prelaunch +command '"'${base_cmd} --domain_name walker --task_name walk --markov_lr 1e-3'"' +arg --seed {1..10} +jobname markov_${exp_name}_walker +tag --tag +tag-args '' +q
