import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from absl import logging
import copy
import datetime
import math

import einops
from einops import rearrange

import sys
import glob
import shutil
from torch.utils.tensorboard import SummaryWriter

UNSAVED_DIRS = ['outputs', 'checkpoint', 'checkpoints', 'workdir', 'build', '.git', '__pycache__', 'assets', 'samples']

def backup_code(work_dir, verbose=False):
    base_dir = './'

    dir_list = ["*.py", ]
    for file in os.listdir(base_dir):
        sub_dir = os.path.join(base_dir, file)
        if os.path.isdir(sub_dir):
            if file in UNSAVED_DIRS:
                continue

            for root, dirs, files in os.walk(sub_dir):
                for dir_name in dirs:
                    dir_list.append(os.path.join(root, dir_name)+"/*.py")

        elif file.split('.')[-1] == 'py':
            pass

    for pattern in dir_list:
        for file in glob.glob(pattern):
            src = os.path.join(base_dir, file)
            dst = os.path.join(work_dir, 'backup', os.path.dirname(file))

            if verbose:
                logging.info('Copying %s -> %s' % (os.path.relpath(src), os.path.relpath(dst)))
            
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src, dst)


def get_str_time():
    return str(datetime.datetime.now()).replace(':', '_').replace('.', '_').replace('-', "_").replace(' ', '_')


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)