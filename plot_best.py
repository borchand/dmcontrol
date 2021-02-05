import json

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import seaborn as sns

WINDOW_SIZE = 5

sac_hyperparams = {
     1: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'cartpole-swingup', 'action_repeat': 8},
     2: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'cartpole-swingup', 'action_repeat': 8},
     3: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'cartpole-swingup', 'action_repeat': 8},
     4: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'ball_in_cup-catch', 'action_repeat': 4},
     5: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'ball_in_cup-catch', 'action_repeat': 4},
     6: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'ball_in_cup-catch', 'action_repeat': 4},
     7: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'cheetah-run', 'action_repeat': 4},
     8: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'cheetah-run', 'action_repeat': 4},
     9: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'cheetah-run', 'action_repeat': 4},
    10: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'finger-spin', 'action_repeat': 2},
    11: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'finger-spin', 'action_repeat': 2},
    12: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'finger-spin', 'action_repeat': 2},
    13: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'reacher-easy', 'action_repeat': 4},
    14: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'reacher-easy', 'action_repeat': 4},
    15: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'reacher-easy', 'action_repeat': 4},
    16: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'walker-walk', 'action_repeat': 2},
    17: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'walker-walk', 'action_repeat': 2},
    18: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'walker-walk', 'action_repeat': 2},
}

rad_hyperparams = {
   1: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad/ball_in_cup-catch-01-30-im108-b128-s1-pixel-tuning-rad_1/eval.log'},
   2: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad/ball_in_cup-catch-01-30-im108-b128-s2-pixel-tuning-rad_2/eval.log'},
   3: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad/ball_in_cup-catch-01-30-im108-b128-s3-pixel-tuning-rad_3/eval.log'},
   4: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad/cartpole-swingup-01-30-im108-b128-s1-pixel-tuning-rad_4/eval.log'},
   5: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad/cartpole-swingup-01-30-im108-b128-s2-pixel-tuning-rad_5/eval.log'},
   6: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad/cartpole-swingup-01-30-im108-b128-s3-pixel-tuning-rad_6/eval.log'},
   7: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad/cheetah-run-01-30-im108-b128-s1-pixel-tuning-rad_7/eval.log'},
   8: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad/cheetah-run-01-30-im108-b128-s2-pixel-tuning-rad_8/eval.log'},
   9: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad/cheetah-run-01-30-im108-b128-s3-pixel-tuning-rad_9/eval.log'},
  10: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad/finger-spin-01-31-im108-b128-s1-pixel-tuning-finger_1/eval.log'},
  11: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad/finger-spin-01-31-im108-b128-s2-pixel-tuning-finger_2/eval.log'},
  12: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad/finger-spin-01-31-im108-b128-s3-pixel-tuning-finger_3/eval.log'},
  13: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad/reacher-easy-01-31-im108-b128-s1-pixel-tuning-reacher_1/eval.log'},
  14: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad/reacher-easy-01-31-im108-b128-s2-pixel-tuning-reacher_2/eval.log'},
  15: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad/reacher-easy-01-31-im108-b128-s3-pixel-tuning-reacher_3/eval.log'},
  16: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad/walker-walk-01-31-im84-b128-s1-pixel-tuning-rad_1/eval.log'},
  17: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad/walker-walk-01-31-im84-b128-s2-pixel-tuning-rad_2/eval.log'},
  18: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad/walker-walk-01-31-im84-b128-s3-pixel-tuning-rad_3/eval.log'},
}

markov_hyperparams = {
   2: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'ball_in_cup-catch', 'action_repeat': 4},
   5: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'ball_in_cup-catch', 'action_repeat': 4},
   8: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'ball_in_cup-catch', 'action_repeat': 4},
  12: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 1, 'domain': 'cartpole-swingup', 'action_repeat': 8},
  15: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 2, 'domain': 'cartpole-swingup', 'action_repeat': 8},
  18: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 3, 'domain': 'cartpole-swingup', 'action_repeat': 8},
  81: {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 1, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s1-pixel-tuning-rad-markov-big-inv_1/eval.log'},
  82: {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 2, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s2-pixel-tuning-rad-markov-big-inv_2/eval.log'},
  83: {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 3, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s3-pixel-tuning-rad-markov-big-inv_3/eval.log'},
  29: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s1-pixel-tuning-rad-markov_29/eval.log'},
  32: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s2-pixel-tuning-rad-markov_32/eval.log'},
  35: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s3-pixel-tuning-rad-markov_35/eval.log'},
  38: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'reacher-easy', 'action_repeat': 4},
  41: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'reacher-easy', 'action_repeat': 4},
  44: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'reacher-easy', 'action_repeat': 4},
  91: {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-3, 'seed': 1, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/walker-walk-02-02-im84-b128-s1-pixel-tuning-rad-markov-big-inv2_7/eval.log'},
  92: {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-3, 'seed': 2, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/walker-walk-02-02-im84-b128-s2-pixel-tuning-rad-markov-big-inv2_8/eval.log'},
  93: {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-3, 'seed': 3, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/walker-walk-02-02-im84-b128-s3-pixel-tuning-rad-markov-big-inv2_9/eval.log'},
}

curl_hyperparams = {
    1: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-Ball_in_cupCatch-v0/curl-paper-results/curl/seed_001/scores.csv'},
    2: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-CartpoleSwingup-v0/curl-paper-results/curl/seed_001/scores.csv'},
    3: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-CheetahRun-v0/curl-paper-results/curl/seed_001/scores.csv'},
    4: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-FingerSpin-v0/curl-paper-results/curl/seed_001/scores.csv'},
    5: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-ReacherEasy-v0/curl-paper-results/curl/seed_001/scores.csv'},
    6: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-WalkerWalk-v0/curl-paper-results/curl/seed_001/scores.csv'},
}

additional_seeds = [
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 4, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/ball_in_cup-catch-02-01-im108-b128-s4-pixel-final-rad-markov-ball_in_cup_4/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 5, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/ball_in_cup-catch-02-01-im108-b128-s5-pixel-final-rad-markov-ball_in_cup_5/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 6, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/ball_in_cup-catch-02-01-im108-b128-s6-pixel-final-rad-markov-ball_in_cup_6/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 7, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/ball_in_cup-catch-02-01-im108-b128-s7-pixel-final-rad-markov-ball_in_cup_7/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 8, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/ball_in_cup-catch-02-01-im108-b128-s8-pixel-final-rad-markov-ball_in_cup_8/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 9, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/ball_in_cup-catch-02-01-im108-b128-s9-pixel-final-rad-markov-ball_in_cup_9/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 10, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/ball_in_cup-catch-02-01-im108-b128-s10-pixel-final-rad-markov-ball_in_cup_10/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 4, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad-markov/cartpole-swingup-02-01-im108-b128-s4-pixel-final-rad-markov-cartpole_4/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 5, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad-markov/cartpole-swingup-02-01-im108-b128-s5-pixel-final-rad-markov-cartpole_5/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 6, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad-markov/cartpole-swingup-02-01-im108-b128-s6-pixel-final-rad-markov-cartpole_6/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 7, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad-markov/cartpole-swingup-02-01-im108-b128-s7-pixel-final-rad-markov-cartpole_7/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 8, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad-markov/cartpole-swingup-02-01-im108-b128-s8-pixel-final-rad-markov-cartpole_8/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 9, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad-markov/cartpole-swingup-02-01-im108-b128-s9-pixel-final-rad-markov-cartpole_9/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 10, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad-markov/cartpole-swingup-02-01-im108-b128-s10-pixel-final-rad-markov-cartpole_10/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 4, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/cheetah-run-02-03-im108-b128-s4-pixel-markov-seeds-cheetah_1/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 5, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/cheetah-run-02-03-im108-b128-s5-pixel-markov-seeds-cheetah_2/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 6, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/cheetah-run-02-03-im108-b128-s6-pixel-markov-seeds-cheetah_3/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 7, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/cheetah-run-02-03-im108-b128-s7-pixel-markov-seeds-cheetah_4/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 8, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/cheetah-run-02-03-im108-b128-s8-pixel-markov-seeds-cheetah_5/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 9, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/cheetah-run-02-03-im108-b128-s9-pixel-markov-seeds-cheetah_6/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 10, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/cheetah-run-02-03-im108-b128-s10-pixel-markov-seeds-cheetah_7/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 4, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/finger-spin-02-01-im108-b128-s4-pixel-final-rad-markov-finger_4/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 5, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/finger-spin-02-01-im108-b128-s5-pixel-final-rad-markov-finger_5/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 6, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/finger-spin-02-01-im108-b128-s6-pixel-final-rad-markov-finger_6/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 7, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/finger-spin-02-01-im108-b128-s7-pixel-final-rad-markov-finger_7/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 8, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/finger-spin-02-01-im108-b128-s8-pixel-final-rad-markov-finger_8/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 9, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/finger-spin-02-01-im108-b128-s9-pixel-final-rad-markov-finger_9/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 10, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/finger-spin-02-01-im108-b128-s10-pixel-final-rad-markov-finger_10/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 4, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/reacher-easy-02-01-im108-b128-s4-pixel-final-rad-markov-reacher_4/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 5, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/reacher-easy-02-01-im108-b128-s5-pixel-final-rad-markov-reacher_5/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 6, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/reacher-easy-02-01-im108-b128-s6-pixel-final-rad-markov-reacher_6/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 7, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/reacher-easy-02-01-im108-b128-s7-pixel-final-rad-markov-reacher_7/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 8, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/reacher-easy-02-01-im108-b128-s8-pixel-final-rad-markov-reacher_8/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 9, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/reacher-easy-02-01-im108-b128-s9-pixel-final-rad-markov-reacher_9/eval.log'},
    {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 10, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad-markov/reacher-easy-02-01-im108-b128-s10-pixel-final-rad-markov-reacher_10/eval.log'},
    # {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-3, 'seed': 4, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/walker-walk-02-03-im84-b128-s4-pixel-markov-seeds-walker_1/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-3, 'seed': 5, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/walker-walk-02-03-im84-b128-s5-pixel-markov-seeds-walker_2/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-3, 'seed': 6, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/walker-walk-02-03-im84-b128-s6-pixel-markov-seeds-walker_3/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-3, 'seed': 7, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/walker-walk-02-03-im84-b128-s7-pixel-markov-seeds-walker_4/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-3, 'seed': 8, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/walker-walk-02-03-im84-b128-s8-pixel-markov-seeds-walker_5/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-3, 'seed': 9, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/walker-walk-02-03-im84-b128-s9-pixel-markov-seeds-walker_6/eval.log'},
    {'alg': 'rad+markov', 'inverse_coef': 10, 'markov_lr': 1e-3, 'seed': 10, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad-markov/walker-walk-02-03-im84-b128-s10-pixel-markov-seeds-walker_7/eval.log'},
] + [
    {'alg': 'rad', 'markov_lr': 0, 'seed': 4, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad/ball_in_cup-catch-02-02-im108-b128-s4-pixel-rad-seeds_1/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 5, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad/ball_in_cup-catch-02-02-im108-b128-s5-pixel-rad-seeds_2/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 6, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad/ball_in_cup-catch-02-02-im108-b128-s6-pixel-rad-seeds_3/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 7, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/rad/ball_in_cup-catch-02-02-im108-b128-s7-pixel-rad-seeds_4/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 4, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad/cartpole-swingup-02-02-im108-b128-s4-pixel-rad-seeds_5/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 5, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad/cartpole-swingup-02-02-im108-b128-s5-pixel-rad-seeds_6/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 6, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad/cartpole-swingup-02-02-im108-b128-s6-pixel-rad-seeds_7/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 7, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/rad/cartpole-swingup-02-02-im108-b128-s7-pixel-rad-seeds_8/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 4, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad/cheetah-run-02-02-im108-b128-s4-pixel-rad-seeds_9/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 5, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad/cheetah-run-02-02-im108-b128-s5-pixel-rad-seeds_10/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 6, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad/cheetah-run-02-02-im108-b128-s6-pixel-rad-seeds_11/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 7, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/rad/cheetah-run-02-02-im108-b128-s7-pixel-rad-seeds_12/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 4, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad/finger-spin-02-02-im108-b128-s4-pixel-rad-seeds_13/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 5, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad/finger-spin-02-02-im108-b128-s5-pixel-rad-seeds_14/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 6, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad/finger-spin-02-02-im108-b128-s6-pixel-rad-seeds_15/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 7, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/rad/finger-spin-02-02-im108-b128-s7-pixel-rad-seeds_16/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 4, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad/reacher-easy-02-02-im108-b128-s4-pixel-rad-seeds_17/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 5, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad/reacher-easy-02-02-im108-b128-s5-pixel-rad-seeds_18/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 6, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad/reacher-easy-02-02-im108-b128-s6-pixel-rad-seeds_19/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 7, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/rad/reacher-easy-02-02-im108-b128-s7-pixel-rad-seeds_20/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 4, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad/walker-walk-02-02-im84-b128-s4-pixel-rad-seeds_21/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 5, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad/walker-walk-02-02-im84-b128-s5-pixel-rad-seeds_22/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 6, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad/walker-walk-02-02-im84-b128-s6-pixel-rad-seeds_23/eval.log'},
    {'alg': 'rad', 'markov_lr': 0, 'seed': 7, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/rad/walker-walk-02-02-im84-b128-s7-pixel-rad-seeds_24/eval.log'},
]# + [
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/ball_in_cup-catch-02-02-im108-b128-s1-pixel-ablations-rad-markov-ball_in_cup_1/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/ball_in_cup-catch-02-02-im108-b128-s2-pixel-ablations-rad-markov-ball_in_cup_2/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/ball_in_cup-catch-02-02-im108-b128-s3-pixel-ablations-rad-markov-ball_in_cup_3/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 4, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/ball_in_cup-catch-02-02-im108-b128-s4-pixel-ablations-rad-markov-ball_in_cup_4/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-4, 'seed': 1, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/markov-no-rad/cartpole-swingup-02-02-im108-b128-s1-pixel-ablations-rad-markov-cartpole_1/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-4, 'seed': 2, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/markov-no-rad/cartpole-swingup-02-02-im108-b128-s2-pixel-ablations-rad-markov-cartpole_2/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-4, 'seed': 3, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/markov-no-rad/cartpole-swingup-02-02-im108-b128-s3-pixel-ablations-rad-markov-cartpole_3/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-4, 'seed': 4, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/markov-no-rad/cartpole-swingup-02-02-im108-b128-s4-pixel-ablations-rad-markov-cartpole_4/eval.log'},
#     # {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-4, 'seed': 1, 'inverse_coef': 10.0, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/cheetah-run-02-02-im108-b128-s4-pixel-rad-seeds_9/eval.log'},
#     # {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-4, 'seed': 2, 'inverse_coef': 10.0, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/cheetah-run-02-02-im108-b128-s5-pixel-rad-seeds_10/eval.log'},
#     # {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-4, 'seed': 3, 'inverse_coef': 10.0, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/cheetah-run-02-02-im108-b128-s6-pixel-rad-seeds_11/eval.log'},
#     # {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-4, 'seed': 4, 'inverse_coef': 10.0, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/cheetah-run-02-02-im108-b128-s7-pixel-rad-seeds_12/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/markov-no-rad/finger-spin-02-02-im108-b128-s1-pixel-ablations-rad-markov-finger_1/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/markov-no-rad/finger-spin-02-02-im108-b128-s2-pixel-ablations-rad-markov-finger_2/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/markov-no-rad/finger-spin-02-02-im108-b128-s3-pixel-ablations-rad-markov-finger_3/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 4, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/markov-no-rad/finger-spin-02-02-im108-b128-s4-pixel-ablations-rad-markov-finger_4/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/reacher-easy-02-02-im108-b128-s1-pixel-ablations-rad-markov-reacher_1/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/reacher-easy-02-02-im108-b128-s2-pixel-ablations-rad-markov-reacher_2/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/reacher-easy-02-02-im108-b128-s3-pixel-ablations-rad-markov-reacher_3/eval.log'},
#     {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 4, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/markov-no-rad/reacher-easy-02-02-im108-b128-s4-pixel-ablations-rad-markov-reacher_4/eval.log'},
#     # {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 1, 'inverse_coef': 10.0, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/markov-no-rad/walker-walk-02-02-im84-b128-s4-pixel-rad-seeds_21/eval.log'},
#     # {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 2, 'inverse_coef': 10.0, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/markov-no-rad/walker-walk-02-02-im84-b128-s5-pixel-rad-seeds_22/eval.log'},
#     # {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 3, 'inverse_coef': 10.0, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/markov-no-rad/walker-walk-02-02-im84-b128-s6-pixel-rad-seeds_23/eval.log'},
#     # {'alg': 'Markov+SAC (visual)', 'markov_lr': 1e-3, 'seed': 4, 'inverse_coef': 10.0, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/markov-no-rad/walker-walk-02-02-im84-b128-s7-pixel-rad-seeds_24/eval.log'},
# ] + [
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/visual/ball_in_cup-catch-02-03-im108-b128-s1-pixel-visual-seeds_1/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/visual/ball_in_cup-catch-02-03-im108-b128-s2-pixel-visual-seeds_2/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'ball_in_cup-catch', 'action_repeat': 4, 'backup_file': 'tmp/visual/ball_in_cup-catch-02-03-im108-b128-s3-pixel-visual-seeds_3/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-4, 'seed': 1, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/visual/cartpole-swingup-02-03-im108-b128-s1-pixel-visual-seeds_4/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-4, 'seed': 2, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/visual/cartpole-swingup-02-03-im108-b128-s2-pixel-visual-seeds_5/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-4, 'seed': 3, 'domain': 'cartpole-swingup', 'action_repeat': 8, 'backup_file': 'tmp/visual/cartpole-swingup-02-03-im108-b128-s3-pixel-visual-seeds_6/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-4, 'seed': 1, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/visual/cheetah-run-02-03-im108-b128-s1-pixel-visual-seeds_7/eval.log', 'inverse_coef': 10.0},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-4, 'seed': 2, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/visual/cheetah-run-02-03-im108-b128-s2-pixel-visual-seeds_8/eval.log', 'inverse_coef': 10.0},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-4, 'seed': 3, 'domain': 'cheetah-run', 'action_repeat': 4, 'backup_file': 'tmp/visual/cheetah-run-02-03-im108-b128-s3-pixel-visual-seeds_9/eval.log', 'inverse_coef': 10.0},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/visual/finger-spin-02-03-im108-b128-s1-pixel-visual-seeds_10/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/visual/finger-spin-02-03-im108-b128-s2-pixel-visual-seeds_11/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'finger-spin', 'action_repeat': 2, 'backup_file': 'tmp/visual/finger-spin-02-03-im108-b128-s3-pixel-visual-seeds_12/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/visual/reacher-easy-02-03-im108-b128-s1-pixel-visual-seeds_13/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/visual/reacher-easy-02-03-im108-b128-s2-pixel-visual-seeds_14/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'reacher-easy', 'action_repeat': 4, 'backup_file': 'tmp/visual/reacher-easy-02-03-im108-b128-s3-pixel-visual-seeds_15/eval.log'},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/visual/walker-walk-02-03-im84-b128-s1-pixel-visual-seeds_16/eval.log', 'inverse_coef': 10.0},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/visual/walker-walk-02-03-im84-b128-s2-pixel-visual-seeds_17/eval.log', 'inverse_coef': 10.0},
#     {'alg': 'SAC (visual)', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'walker-walk', 'action_repeat': 2, 'backup_file': 'tmp/visual/walker-walk-02-03-im84-b128-s3-pixel-visual-seeds_18/eval.log', 'inverse_coef': 10.0},
# ]

dfs = []
unique_id = 0
#%%
for i in sac_hyperparams.keys():
    filepath = 'logs/state-sac_%d.g' % i
    params = sac_hyperparams[i]
    data = pd.read_csv(filepath, names=['steps', 'reward'])
    data['episode'] = data.steps / 1000 * params['action_repeat']
    data = data.set_index('steps')
    data['cumulative_reward'] = (data.reward*10).cumsum()
    data['cumulative_reward_per_episode'] = data.cumulative_reward / data.episode * params['action_repeat']
    data['unique_id'] = unique_id
    unique_id += 1
    for k, v in params.items():
        data[k] = v
    dfs.append(data)

#%%
for i in rad_hyperparams.keys():
    filepath = 'logs/tuning-rad_%d.g' % i
    params = rad_hyperparams[i]
    if params.get('backup_file', False):
        data = pd.read_json(params['backup_file'], lines=True)
        data = data.rename(columns={'step': 'steps', 'mean_episode_reward': 'reward'}).drop(columns=['episode_reward', 'eval_time', 'best_episode_reward']).set_index('steps')
    else:
        data = pd.read_csv(filepath, names=['steps', 'reward'])
        data['episode'] = data.steps / 1000 * params['action_repeat']
        data = data.set_index('steps')
    data['cumulative_reward'] = data.reward.cumsum()
    data['cumulative_reward_per_episode'] = data.cumulative_reward / data.episode * params['action_repeat']
    data['unique_id'] = unique_id
    unique_id += 1
    data.reward = data.reward.rolling(WINDOW_SIZE).mean()
    for k, v in params.items():
        if k == 'backup_file':
            continue
        data[k] = v
    dfs.append(data)

#%%

for i in markov_hyperparams.keys():
    filepath = 'logs/tuning-rad-markov_%d.g' % i
    params = markov_hyperparams[i]
    if params.get('backup_file', False):
        data = pd.read_json(params['backup_file'], lines=True)
        data = data.rename(columns={'step': 'steps', 'mean_episode_reward': 'reward'}).drop(columns=['episode_reward', 'eval_time', 'best_episode_reward']).set_index('steps')
    else:
        data = pd.read_csv(filepath, names=['steps','reward'])
        data['episode'] = data.steps / 1000 * params['action_repeat']
        data = data.set_index('steps')
    data['cumulative_reward'] = data.reward.cumsum()
    data['cumulative_reward_per_episode'] = data.cumulative_reward / data.episode * params['action_repeat']
    data['unique_id'] = unique_id
    unique_id += 1
    data.reward = data.reward.rolling(WINDOW_SIZE).mean()
    for k, v in params.items():
        if k == 'backup_file':
            continue
        data[k] = v
    dfs.append(data)


for i, params in enumerate(additional_seeds):
    if params.get('backup_file', False):
        data = pd.read_json(params['backup_file'], lines=True)
        data = data.rename(columns={'step': 'steps', 'mean_episode_reward': 'reward'}).drop(columns=['episode_reward', 'eval_time', 'best_episode_reward']).set_index('steps')
    else:
        raise FileNotFoundError(str(i))
    data['cumulative_reward'] = data.reward.cumsum()
    data['cumulative_reward_per_episode'] = data.cumulative_reward / data.episode * params['action_repeat']
    data['unique_id'] = unique_id
    unique_id += 1
    data.reward = data.reward.rolling(WINDOW_SIZE).mean()
    for k, v in params.items():
        if k == 'backup_file':
            continue
        data[k] = v
    dfs.append(data)

#%%
# for i in curl_hyperparams.keys():
#     params = curl_hyperparams[i]
#     data = pd.read_csv(params['backup_file'])
#     data = data.rename(columns={'step': 'steps'})
#     data = data.set_index('steps')
#     data['unique_id'] = unique_id
#     unique_id += 1
#     for k, v in params.items():
#         if k == 'backup_file':
#             continue
#         data[k] = v
#     dfs.append(data)

#%%
data = pd.concat(dfs, axis=0)
data.loc[data.inverse_coef.isnull() & (data.markov_lr > 0), 'inverse_coef'] = 1
data.loc[data.inverse_coef.isnull() & (data.markov_lr == 0), 'inverse_coef'] = 0
# data.loc[((data['inverse_coef'] == 1) & (data['alg'] == 'rad+markov')), 'alg'] = 'rad+markov-inv_coef=1'
#%%
# for d in ['cartpole-swingup']:#list(data.domain.unique()):

subset = data
subset = subset.query("alg == 'rad'")
subset = subset.query("domain == 'reacher-easy'")
subset = subset.query("steps > 200000 and reward < 600")
subset.unique_id.unique()
subset.seed.unique()
#%%

subset = data
subset = subset.query("alg != 'curl'")
# subset = subset.query("domain == 'ball_in_cup-catch'")
# subset = subset.query("domain == 'cartpole-swingup'")
# subset = subset.query("domain == 'cheetah-run'")
# subset = subset.query("domain == 'finger-spin'")
# subset = subset.query("domain == 'reacher-easy'")
# subset = subset.query("domain == 'walker-walk'")
# subset = subset.query("domain in ['walker-walk', 'cheetah-run']")
subset = subset.query("domain != 'ball_in_cup-catch' or steps <= 100e3")
subset = subset.query("domain != 'cartpole-swingup' or steps <= 100e3")

subset.loc[subset.alg == 'rad+markov', 'alg'] = 'Markov+RAD'
subset.loc[subset.alg == 'rad', 'alg'] = 'RAD'
subset.loc[subset.alg == 'state-sac', 'alg'] = 'SAC (expert)'

subset.loc[subset.domain == 'ball_in_cup-catch', 'domain'] = 'Ball-in-cup, Catch'
subset.loc[subset.domain == 'cartpole-swingup', 'domain'] = 'Cartpole, Swingup'
subset.loc[subset.domain == 'cheetah-run', 'domain'] = 'Cheetah, Run'
subset.loc[subset.domain == 'finger-spin', 'domain'] = 'Finger, Spin'
subset.loc[subset.domain == 'reacher-easy', 'domain'] = 'Reacher, Easy'
subset.loc[subset.domain == 'walker-walk', 'domain'] = 'Walker, Walk'

subset = subset.rename(columns={'reward': 'Reward', 'alg': 'Agent', 'domain': 'Task'})
subset = subset.rename_axis(index={'steps':'Steps'})
list(subset.columns)

# subset = subset.query("steps <= 100e3")

len(subset.unique_id.unique())
len(data.unique_id.unique())

p = sns.color_palette('Set1', n_colors=len(subset['Agent'].unique()), desat=0.5)
p[0] = (0.696078431372549, 0.30000000000000004, 0.30392156862745096)
p[1] = (0.5343137254901962, 0.38921568627450975, 0.5558823529411765)
p[2] = (.60,.57,.57)
# p[3] = (0.34215686274509804, 0.4813725490196079, 0.5950980392156863)
# p[4] = (0.3950980392156863, 0.5872549019607843, 0.38921568627450986)
# p[-1] = (.5,.3,.3)
g = sns.relplot(
    data=subset,
    x='Steps',
    y='Reward',
    hue='Agent',
    hue_order=['Markov+RAD', 'RAD', 'SAC (expert)'],#, 'SAC (visual)' 'Markov+SAC (visual)',
    style='Agent',
    # style='markov_lr',
    col='Task',
    col_wrap=3,
    style_order=['Markov+RAD', 'RAD', 'SAC (expert)'],#, 'SAC (visual)' 'Markov+SAC (visual)',
    kind='line',
    # units='seed',
    # estimator=None,
    # height=10,
    palette=p,
    facet_kws={'sharex': False, 'sharey': True},
)
leg = g._legend
leg.set_draggable(True)
# plt.title(d)
# plt.ylim([0,300])
plt.tight_layout()
plt.show()
