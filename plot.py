import json

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import seaborn as sns

WINDOW_SIZE = 5

sac_hyperparams = {
     1: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'cartpole-swingup'},
     2: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'cartpole-swingup'},
     3: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'cartpole-swingup'},
     4: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'ball_in_cup-catch'},
     5: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'ball_in_cup-catch'},
     6: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'ball_in_cup-catch'},
     7: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'cheetah-run'},
     8: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'cheetah-run'},
     9: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'cheetah-run'},
    10: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'finger-spin'},
    11: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'finger-spin'},
    12: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'finger-spin'},
    13: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'reacher-easy'},
    14: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'reacher-easy'},
    15: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'reacher-easy'},
    16: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 1, 'domain': 'walker-walk'},
    17: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 2, 'domain': 'walker-walk'},
    18: {'alg': 'state-sac', 'markov_lr': 0, 'seed': 3, 'domain': 'walker-walk'},
}

rad_hyperparams = {
   1: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'ball_in_cup-catch',
        'backup_file': 'tmp/rad/ball_in_cup-catch-01-30-im108-b128-s1-pixel-tuning-rad_1/eval.log'},
   2: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'ball_in_cup-catch',
        'backup_file': 'tmp/rad/ball_in_cup-catch-01-30-im108-b128-s2-pixel-tuning-rad_2/eval.log'},
   3: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'ball_in_cup-catch',
        'backup_file': 'tmp/rad/ball_in_cup-catch-01-30-im108-b128-s3-pixel-tuning-rad_3/eval.log'},
   4: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'cartpole-swingup',
        'backup_file': 'tmp/rad/cartpole-swingup-01-30-im108-b128-s1-pixel-tuning-rad_4/eval.log'},
   5: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'cartpole-swingup',
        'backup_file': 'tmp/rad/cartpole-swingup-01-30-im108-b128-s2-pixel-tuning-rad_5/eval.log'},
   6: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'cartpole-swingup',
        'backup_file': 'tmp/rad/cartpole-swingup-01-30-im108-b128-s3-pixel-tuning-rad_6/eval.log'},
   7: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad/cheetah-run-01-30-im108-b128-s1-pixel-tuning-rad_7/eval.log'},
   8: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad/cheetah-run-01-30-im108-b128-s2-pixel-tuning-rad_8/eval.log'},
   9: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad/cheetah-run-01-30-im108-b128-s3-pixel-tuning-rad_9/eval.log'},
  10: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad/finger-spin-01-30-im108-b128-s1-pixel-tuning-rad_10/eval.log'},
  11: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad/finger-spin-01-30-im108-b128-s2-pixel-tuning-rad_11/eval.log'},
  12: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad/finger-spin-01-30-im108-b128-s3-pixel-tuning-rad_12/eval.log'},
  13: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'reacher-easy',
        'backup_file': 'tmp/rad/reacher-easy-01-30-im108-b128-s1-pixel-tuning-rad_13/eval.log'},
  14: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'reacher-easy',
        'backup_file': 'tmp/rad/reacher-easy-01-30-im108-b128-s2-pixel-tuning-rad_14/eval.log'},
  15: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'reacher-easy',
        'backup_file': 'tmp/rad/reacher-easy-01-30-im108-b128-s3-pixel-tuning-rad_15/eval.log'},
  16: {'alg': 'rad', 'markov_lr': 0, 'seed': 1, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad/walker-walk-01-31-im84-b128-s1-pixel-tuning-rad_1/eval.log'},
  17: {'alg': 'rad', 'markov_lr': 0, 'seed': 2, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad/walker-walk-01-31-im84-b128-s2-pixel-tuning-rad_2/eval.log'},
  18: {'alg': 'rad', 'markov_lr': 0, 'seed': 3, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad/walker-walk-01-31-im84-b128-s3-pixel-tuning-rad_3/eval.log'},
}

markov_hyperparams = {
   1: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 1, 'domain': 'ball_in_cup-catch'},
   2: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'ball_in_cup-catch'},
   3: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 1, 'domain': 'ball_in_cup-catch'},
   4: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 2, 'domain': 'ball_in_cup-catch'},
   5: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'ball_in_cup-catch'},
   6: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 2, 'domain': 'ball_in_cup-catch'},
   7: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 3, 'domain': 'ball_in_cup-catch'},
   8: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'ball_in_cup-catch'},
   9: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 3, 'domain': 'ball_in_cup-catch'},
  10: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 1, 'domain': 'cartpole-swingup'},
  11: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'cartpole-swingup'},
  12: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 1, 'domain': 'cartpole-swingup'},
  13: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 2, 'domain': 'cartpole-swingup'},
  14: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'cartpole-swingup'},
  15: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 2, 'domain': 'cartpole-swingup'},
  16: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 3, 'domain': 'cartpole-swingup'},
  17: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'cartpole-swingup'},
  18: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 3, 'domain': 'cartpole-swingup'},
  19: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-2, 'seed': 1, 'domain': 'cheetah-run'},
  20: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'cheetah-run'},
  21: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-4, 'seed': 1, 'domain': 'cheetah-run'},
  22: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-2, 'seed': 2, 'domain': 'cheetah-run'},
  23: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'cheetah-run'},
  24: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-4, 'seed': 2, 'domain': 'cheetah-run'},
  25: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-2, 'seed': 3, 'domain': 'cheetah-run'},
  26: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'cheetah-run'},
  27: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-4, 'seed': 3, 'domain': 'cheetah-run'},
  61: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-5, 'seed': 1, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s1-pixel-tuning-rad-markov-cheetah_1/eval.log'},
  62: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-5, 'seed': 1, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s2-pixel-tuning-rad-markov-cheetah_2/eval.log'},
  63: {'alg': 'rad+markov-inv_coef=1', 'markov_lr': 1e-5, 'seed': 1, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s3-pixel-tuning-rad-markov-cheetah_3/eval.log'},
  64: {'alg': 'rad+markov-inv_coef=6', 'markov_lr': 1e-5, 'inverse_coef': 6, 'seed': 2, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s1-pixel-tuning-rad-markov-cheetah_4/eval.log'},
  65: {'alg': 'rad+markov-inv_coef=6', 'markov_lr': 1e-5, 'inverse_coef': 6, 'seed': 2, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s2-pixel-tuning-rad-markov-cheetah_5/eval.log'},
  66: {'alg': 'rad+markov-inv_coef=6', 'markov_lr': 1e-5, 'inverse_coef': 6, 'seed': 2, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s3-pixel-tuning-rad-markov-cheetah_6/eval.log'},
  67: {'alg': 'rad+markov-inv_coef=6', 'markov_lr': 1e-4, 'inverse_coef': 6, 'seed': 3, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s1-pixel-tuning-rad-markov-cheetah_7/eval.log'},
  68: {'alg': 'rad+markov-inv_coef=6', 'markov_lr': 1e-4, 'inverse_coef': 6, 'seed': 3, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s2-pixel-tuning-rad-markov-cheetah_8/eval.log'},
  69: {'alg': 'rad+markov-inv_coef=6', 'markov_lr': 1e-4, 'inverse_coef': 6, 'seed': 3, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s3-pixel-tuning-rad-markov-cheetah_9/eval.log'},
  28: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 1, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s1-pixel-tuning-rad-markov_28/eval.log'},
  29: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s1-pixel-tuning-rad-markov_29/eval.log'},
  30: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 1, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s1-pixel-tuning-rad-markov_30/eval.log'},
  31: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 2, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s2-pixel-tuning-rad-markov_31/eval.log'},
  32: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s2-pixel-tuning-rad-markov_32/eval.log'},
  33: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 2, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s2-pixel-tuning-rad-markov_33/eval.log'},
  34: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 3, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s3-pixel-tuning-rad-markov_34/eval.log'},
  35: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s3-pixel-tuning-rad-markov_35/eval.log'},
  36: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 3, 'domain': 'finger-spin',
        'backup_file': 'tmp/rad-markov/finger-spin-01-30-im108-b128-s3-pixel-tuning-rad-markov_36/eval.log'},
  37: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 1, 'domain': 'reacher-easy'},
  38: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'reacher-easy'},
  39: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 1, 'domain': 'reacher-easy'},
  40: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 2, 'domain': 'reacher-easy'},
  41: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'reacher-easy'},
  42: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 2, 'domain': 'reacher-easy'},
  43: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 3, 'domain': 'reacher-easy'},
  44: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'reacher-easy'},
  45: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 3, 'domain': 'reacher-easy',
        'backup_file': 'tmp/rad-markov/reacher-easy-01-30-im108-b128-s3-pixel-tuning-rad-markov_45/eval.log'},
  46: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 1, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-01-31-im84-b128-s1-pixel-tuning-rad-markov_1/eval.log'},
  47: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 1, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-01-31-im84-b128-s1-pixel-tuning-rad-markov_2/eval.log'},
  48: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 1, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-01-31-im84-b128-s1-pixel-tuning-rad-markov_3/eval.log'},
  49: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 2, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-01-31-im84-b128-s2-pixel-tuning-rad-markov_4/eval.log'},
  50: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 2, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-01-31-im84-b128-s2-pixel-tuning-rad-markov_5/eval.log'},
  51: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 2, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-01-31-im84-b128-s2-pixel-tuning-rad-markov_6/eval.log'},
  52: {'alg': 'rad+markov', 'markov_lr': 1e-2, 'seed': 3, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-01-31-im84-b128-s3-pixel-tuning-rad-markov_7/eval.log'},
  53: {'alg': 'rad+markov', 'markov_lr': 1e-3, 'seed': 3, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-01-31-im84-b128-s3-pixel-tuning-rad-markov_8/eval.log'},
  54: {'alg': 'rad+markov', 'markov_lr': 1e-4, 'seed': 3, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-01-31-im84-b128-s3-pixel-tuning-rad-markov_9/eval.log'},
  71: {'alg': 'rad+markov-inv_coef=10', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 1, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-02-01-im84-b128-s1-pixel-tuning-rad-markov-big-inv_7/eval.log'},
  72: {'alg': 'rad+markov-inv_coef=10', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 2, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-02-01-im84-b128-s2-pixel-tuning-rad-markov-big-inv_8/eval.log'},
  73: {'alg': 'rad+markov-inv_coef=10', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 3, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-02-01-im84-b128-s3-pixel-tuning-rad-markov-big-inv_9/eval.log'},
  74: {'alg': 'rad+markov-inv_coef=30', 'inverse_coef': 30, 'markov_lr': 1e-4, 'seed': 1, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-02-01-im84-b128-s1-pixel-tuning-rad-markov-big-inv_10/eval.log'},
  75: {'alg': 'rad+markov-inv_coef=30', 'inverse_coef': 30, 'markov_lr': 1e-4, 'seed': 2, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-02-01-im84-b128-s2-pixel-tuning-rad-markov-big-inv_11/eval.log'},
  76: {'alg': 'rad+markov-inv_coef=30', 'inverse_coef': 30, 'markov_lr': 1e-4, 'seed': 3, 'domain': 'walker-walk',
        'backup_file': 'tmp/rad-markov/walker-walk-02-01-im84-b128-s3-pixel-tuning-rad-markov-big-inv_12/eval.log'},
  81: {'alg': 'rad+markov-inv_coef=10', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 1, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s1-pixel-tuning-rad-markov-big-inv_1/eval.log'},
  82: {'alg': 'rad+markov-inv_coef=10', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 2, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s2-pixel-tuning-rad-markov-big-inv_2/eval.log'},
  83: {'alg': 'rad+markov-inv_coef=10', 'inverse_coef': 10, 'markov_lr': 1e-4, 'seed': 3, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s3-pixel-tuning-rad-markov-big-inv_3/eval.log'},
  84: {'alg': 'rad+markov-inv_coef=30', 'inverse_coef': 30, 'markov_lr': 1e-4, 'seed': 1, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s1-pixel-tuning-rad-markov-big-inv_4/eval.log'},
  85: {'alg': 'rad+markov-inv_coef=30', 'inverse_coef': 30, 'markov_lr': 1e-4, 'seed': 2, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s2-pixel-tuning-rad-markov-big-inv_5/eval.log'},
  86: {'alg': 'rad+markov-inv_coef=30', 'inverse_coef': 30, 'markov_lr': 1e-4, 'seed': 3, 'domain': 'cheetah-run',
        'backup_file': 'tmp/rad-markov/cheetah-run-01-31-im108-b128-s3-pixel-tuning-rad-markov-big-inv_6/eval.log'},
}

curl_hyperparams = {
    1: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'ball_in_cup-catch',
        'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-Ball_in_cupCatch-v0/curl-paper-results/curl/seed_001/scores.csv'},
    2: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'cartpole-swingup',
        'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-CartpoleSwingup-v0/curl-paper-results/curl/seed_001/scores.csv'},
    3: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'cheetah-run',
        'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-CheetahRun-v0/curl-paper-results/curl/seed_001/scores.csv'},
    4: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'finger-spin',
        'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-FingerSpin-v0/curl-paper-results/curl/seed_001/scores.csv'},
    5: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'reacher-easy',
        'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-ReacherEasy-v0/curl-paper-results/curl/seed_001/scores.csv'},
    6: {'alg': 'curl', 'markov_lr': 0, 'seed': 1, 'domain': 'walker-walk',
        'backup_file': '../gridworlds/dmcontrol/experiments/dm2gym-WalkerWalk-v0/curl-paper-results/curl/seed_001/scores.csv'},
}

dfs = []

#%%
for i in range(1,18):
    filepath = 'logs/state-sac_%d.g' % i
    data = pd.read_csv(filepath, names=['reward'])
    data.index.name = 'steps'
    params = sac_hyperparams[i]
    for k, v in params.items():
        data[k] = v
    dfs.append(data)

#%%
for i in range(1,18):
    filepath = 'logs/tuning-rad_%d.g' % i
    data = pd.read_csv(filepath, names=['reward'])
    params = rad_hyperparams[i]
    data.index.name = 'steps'
    if data.empty and params.get('backup_file', False):
        data = pd.read_json(params['backup_file'], lines=True)
        data = data.rename(columns={'step': 'steps', 'mean_episode_reward': 'reward'}).drop(columns=['episode', 'episode_reward', 'eval_time', 'best_episode_reward']).set_index('steps')
    data.reward = data.reward.rolling(WINDOW_SIZE, center=True).mean()
    data = data.iloc[WINDOW_SIZE//2:-WINDOW_SIZE//2,:]
    for k, v in params.items():
        if k == 'backup_file':
            continue
        data[k] = v
    dfs.append(data)

#%%

for i in list(range(1,55))+list(range(61,70))+list(range(81,87))+list(range(71,77)):
    filepath = 'logs/tuning-rad-markov_%d.g' % i
    params = markov_hyperparams[i]
    if params.get('backup_file', False):
        data = pd.read_json(params['backup_file'], lines=True)
        data = data.rename(columns={'step': 'steps', 'mean_episode_reward': 'reward'}).drop(columns=['episode', 'episode_reward', 'eval_time', 'best_episode_reward']).set_index('steps')
    else:
        data = pd.read_csv(filepath, names=['reward'])
        data.index.name = 'steps'
    data.reward = data.reward.rolling(WINDOW_SIZE, center=True).mean()
    data = data.iloc[WINDOW_SIZE//2:-WINDOW_SIZE//2,:]
    for k, v in params.items():
        if k == 'backup_file':
            continue
        data[k] = v
    dfs.append(data)

#%%
for i in range(1,7):
    params = curl_hyperparams[i]
    data = pd.read_csv(params['backup_file'])
    data = data.rename(columns={'step': 'steps'})
    data = data.set_index('steps')
    for k, v in params.items():
        if k == 'backup_file':
            continue
        data[k] = v
    dfs.append(data)

#%%
data = pd.concat(dfs, axis=0)
data.loc[data.inverse_coef.isnull() & (data.markov_lr > 0), 'inverse_coef'] = 1
data.loc[data.inverse_coef.isnull() & (data.markov_lr == 0), 'inverse_coef'] = 0
data.loc[((data['inverse_coef'] == 1) & (data['alg'] == 'rad+markov')), 'alg'] = 'rad+markov-inv_coef=1'
#%%
# for d in ['cartpole-swingup']:#list(data.domain.unique()):
subset = data
# subset = subset.query("domain in 'cheetah-run'")
# subset = subset.query("domain == 'finger-spin'")
# subset = subset.query("domain == 'walker-walk'")
subset = subset.query("domain != 'ball_in_cup-catch' or markov_lr == 0.001 or alg in ['state-sac', 'rad', 'curl']")
subset = subset.query("domain != 'cartpole-swingup' or markov_lr == 0.0001 or alg in ['state-sac', 'rad', 'curl']")
subset = subset.query("domain != 'finger-spin' or markov_lr == 0.001 or alg in ['state-sac', 'rad', 'curl']")
subset = subset.query("domain != 'reacher-easy' or markov_lr == 0.001 or alg in ['state-sac', 'rad', 'curl']")
subset = subset.query("domain != 'cheetah-run' or (markov_lr == 0.0001 and inverse_coef == 10.0) or alg in ['state-sac', 'rad', 'curl']")
subset = subset.query("domain != 'walker-walk' or (markov_lr == 0.0001 and inverse_coef == 30.0) or alg in ['state-sac', 'rad', 'curl']")
# subset = subset.query("steps <= 100e3")
# subset = subset.query("alg in ['rad', 'state-sac'] or (domain == 'cartpole-swingup' and markov_lr == 0.0001) or (domain != 'cartpole-swingup' and markov_lr == 0.001) or (domain == 'cheetah-run')")
# subset = subset[subset.index % 5000 == 0]

p = sns.color_palette('viridis', n_colors=len(subset['markov_lr'].unique()), desat=0.5)
# p[0] = (0,0,0)
sns.relplot(
    data=subset,
    x='steps',
    y='reward',
    hue='markov_lr',
    style='alg',
    col='domain',
    col_wrap=3,
    # style_order=['markov', 'expert', 'visual'],
    kind='line',
    # units='seed',
    # estimator=None,
    # height=10,
    palette=p)
# plt.title(d)
# plt.ylim([0,300])
plt.tight_layout()
plt.show()
