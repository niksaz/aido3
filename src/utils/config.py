# Author: Mikita Sazanovich

from easydict import EasyDict as edict

CFG = edict()

CFG.train_data_ratio = 0.7

CFG.batch_size = 64
CFG.epochs = 1000
CFG.learning_rate = 1e-4
CFG.regularizer = 1e-4
CFG.seed = 603930

CFG.model = 'CNN96Model'
CFG.image_width = 160
CFG.image_height = 90
