# Author: Mikita Sazanovich

import math
from easydict import EasyDict as edict

CFG = edict()

# model params
CFG.early_drop_prob = 0.01
CFG.late_drop_prob = 0.05
CFG.max_norm_value = 4.0
CFG.regularizer = 1e-4
CFG.seed = 3930

# optimization params
CFG.batch_size = 64
CFG.steps_to_train_for = 1000000
CFG.learning_rate = 2.0 / math.sqrt(512)
CFG.learning_rate_warmup_steps = 16000
CFG.optimizer_adam_beta1 = 0.9
CFG.optimizer_adam_beta2 = 0.997
CFG.optimizer_adam_epsilon = 1e-9

# data params
CFG.train_data_ratio = 0.7
CFG.image_width = 64
CFG.image_height = 32
CFG.dataset_names = ['jetbrains', 'duckietown', 'loop_empty', 'udem1']
CFG.dataset_subsample = 1
