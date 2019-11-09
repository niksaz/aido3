# Author: Mikita Sazanovich

from easydict import EasyDict as edict

CFG = edict()

CFG.batch_size = 64
CFG.epochs = 100
CFG.lr = 1e-4
CFG.regularizer = 1e-2

CFG.image_width = 160  # 96
CFG.image_height = 80  # 48 

