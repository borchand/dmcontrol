import glob
import json

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import seaborn as sns

WINDOW_SIZE = 5

dfs = []
unique_id = 0


exp_names = {
    'cheetah-run': [
        # 'exp7_markov_fix',
        'exp8_markov_relu',
        # 'exp9_markov_pretrain',
        # 'exp10_markov_pretrain_bs512',
        # 'exp11_markov_relu_inv0.1',
        # 'exp12_markov_pretrain_bs512_inv10',
        # 'exp13_markov_pretrain_bs512_inv1_lr5e-5',
        # 'exp14_markov_pretrain_bs512_inv1_dz0.1',
        # 'exp15_markov_pretrain_bs512_inv1_lr1e-3',
        'exp16_markov_pretrain_bs512_inv1_relu30',
        # 'exp17_markov_pretrain10k_bs512_inv1',
        # 'exp18_markov_pretrain10k_bs512_inv1_relu30',
        # 'exp19_markov_pretrain_bs512_beta0.5',
        'exp20_markov_pretrain_bs512_inv1_relu100',
        'exp21_markov_pretrain_bs512_inv1_relu300',
        'exp22_markov_pretrain_bs512_inv1_relu1k',
    ],
    'ball_in_cup-catch': [
        # 'exp7_markov_fix',
        # 'exp8_markov_relu',
        # 'exp9_markov_pretrain',
        'exp10_markov_pretrain_bs512',
        # 'exp11_markov_relu_inv0.1',
        'exp12_markov_pretrain_bs512_inv10',
        'exp23_markov_pretrain_bs512_inv30',
        'exp24_markov_pretrain_bs512_inv10_relu30',
        'exp26_markov_pretrain_bs512_inv30_relu30',
    ],
    'reacher-easy': [
        # 'exp7_markov_fix',
        # 'exp8_markov_relu',
        # 'exp9_markov_pretrain',
        'exp10_markov_pretrain_bs512',
        # 'exp11_markov_relu_inv0.1',
        'exp12_markov_pretrain_bs512_inv10',
        'exp23_markov_pretrain_bs512_inv30',
        'exp24_markov_pretrain_bs512_inv10_relu30',
        'exp26_markov_pretrain_bs512_inv30_relu30',
    ],
    'cartpole-swingup': [
        # 'exp7_markov_fix',
        # 'exp8_markov_relu',
        # 'exp9_markov_pretrain',
        'exp10_markov_pretrain_bs512',
        # 'exp11_markov_relu_inv0.1',
        'exp12_markov_pretrain_bs512_inv10',
        'exp25_markov_pretrain_bs512_inv1_relu30',
    ],
}

def rg(seed_first, seed_last):
    return range(seed_first, seed_last+1)

domain, seeds = [
    # ('cheetah-run', rg(1, 5)),
    # ('ball_in_cup-catch', rg(1,6)),
    # ('reacher-easy', rg(1,6)),
    ('cartpole-swingup', rg(1,6)),
][0]

#%%
for exp_name in exp_names[domain]:
    files = dict(enumerate([{
        'alg': exp_name,
        'domain': domain,
        'seed': seed,
        'log_filename': list(sorted(glob.glob('tmp/{}/{}-*_{}/eval.log'.format(exp_name, domain, seed))))[-1],
        'args_filename': list(sorted(glob.glob('tmp/{}/{}-*_{}/args.json'.format(exp_name, domain, seed))))[-1]
    } for seed in seeds]))

    for _, file_details in files.items():
        with open(file_details['args_filename']) as json_data:
            params = json.load(json_data)
        params['alg'] = file_details['alg']
        params['domain'] = file_details['domain']
        params['seed'] = file_details['seed']
        data = pd.read_json(file_details['log_filename'], lines=True)
        data = data.rename(columns={'step': 'steps', 'episode_reward': 'reward'}).set_index('steps')
        data['cumulative_reward'] = data.reward.cumsum()
        data['cumulative_reward_per_episode'] = data.cumulative_reward / data.episode * params['action_repeat']
        data['unique_id'] = unique_id
        unique_id += 1
        data.reward = data.reward.rolling(WINDOW_SIZE).mean()
        for k, v in params.items():
            if k == 'backup_file':
                continue
            if k == 'markov_params':
                for sub_k, sub_v in v.items():
                    data['markov_'+sub_k] = sub_v
                continue
            data[k] = v
        dfs.append(data)

#%%
data = pd.concat(dfs, axis=0)


#%%

subset = data
# subset = subset.query("alg == 'rad+markov'")
# subset = subset.query("domain == 'ball_in_cup-catch'")
# subset = subset.query("domain == 'cartpole-swingup'")
# subset = subset.query("domain == 'cheetah-run'")
# subset = subset.query("domain == 'finger-spin'")
# subset = subset.query("domain == 'reacher-easy'")
# subset = subset.query("domain == 'walker-walk'")
# subset = subset.query("domain in ['finger-spin', 'walker-walk', 'cheetah-run']")
subset = subset.query("domain != 'ball_in_cup-catch' or steps <= 100e3")
subset = subset.query("domain != 'cartpole-swingup' or steps <= 100e3")

subset.loc[subset.alg == 'rad+markov', 'alg'] = 'Markov+RAD'
subset.loc[subset.alg == 'rad', 'alg'] = 'RAD'
subset.loc[subset.alg == 'state-sac', 'alg'] = 'SAC (expert)'
subset.loc[subset.alg == 'curl', 'alg'] = 'CURL'
subset.loc[subset.alg == 'dbc', 'alg'] = "DBC"

subset.loc[subset.domain == 'ball_in_cup-catch', 'domain'] = 'Ball-in-cup, Catch'
subset.loc[subset.domain == 'cartpole-swingup', 'domain'] = 'Cartpole, Swingup'
subset.loc[subset.domain == 'cheetah-run', 'domain'] = 'Cheetah, Run'
subset.loc[subset.domain == 'finger-spin', 'domain'] = 'Finger, Spin'
subset.loc[subset.domain == 'reacher-easy', 'domain'] = 'Reacher, Easy'
subset.loc[subset.domain == 'walker-walk', 'domain'] = 'Walker, Walk'

subset = subset.rename(columns={'reward': 'Reward', 'alg': 'Agent', 'domain': 'Task'})
subset = subset.rename_axis(index={'steps':'Steps'})

# all_seeds_step_progress = {
#     task: subset.query("Task == @task and Agent == 'Markov+RAD'").groupby('seed').size().min()
#     for task in subset.Task.unique()
# }

# subset = subset.query("steps <= 100e3")

len(subset.unique_id.unique())
len(data.unique_id.unique())

p = sns.color_palette('Set1', n_colors=9, desat=0.5)
red, blue, green, purple, orange, yellow, brown, pink, gray = p

p = sns.color_palette('colorblind', n_colors=len(subset['Agent'].unique()))
# p[0] = blue
# p[0] = red
# p[1] = purple
# p[2] = orange
# p[3] = (.60,.57,.57) # reddish gray
# p[4] = green
g = sns.relplot(
    data=subset,
    x='Steps',
    y='Reward',
    hue='Agent',
    # hue_order=['Markov+RAD', 'RAD', 'CURL', 'SAC (expert)', 'DBC'],#, 'SAC (visual)' 'Markov+SAC (visual)',
    style='Agent',
    # style='markov_lr',
    col='Task',
    # col_wrap=3,
    # col_order=['Cartpole, Swingup', 'Ball-in-cup, Catch', 'Cheetah, Run', 'Finger, Spin', 'Reacher, Easy', 'Walker, Walk'],
    # style_order=['Markov+RAD', 'RAD', 'SAC (expert)', 'CURL', 'DBC'],#, 'SAC (visual)' 'Markov+SAC (visual)',
    kind='line',
    # units='seed',
    # estimator=None,
    # height=10,
    palette=p,
    facet_kws={'sharex': False, 'sharey': True},
)

# draw progress line
# for ax in g.axes:
#     domain = ax.get_title().replace('Task = ','')
#     xmin, xmax = ax.get_xlim()
#     xpos = min(all_seeds_step_progress[domain] * 1000, xmax)
#     ymin, ymax = ax.get_ylim()
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
#     ax.vlines(xpos, ymin=ymin, ymax=ymax, linestyles='dashed', colors='black')

leg = g._legend
leg.set_draggable(True)
# plt.title(d)
# plt.ylim([0,300])
plt.tight_layout()
plt.subplots_adjust(hspace=0.22)
plt.show()
