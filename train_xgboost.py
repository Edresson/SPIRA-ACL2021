import os
import math
import torch
import torch.nn as nn
import traceback

import time
import numpy as np

import argparse

from utils.generic_utils import load_config, save_config_file
from utils.generic_utils import set_init_dict

from utils.generic_utils import NoamLR, binary_acc

from utils.generic_utils import save_best_checkpoint

from utils.tensorboard import TensorboardWriter

from utils.dataset import train_dataloader, eval_dataloader

from utils.audio_processor import AudioProcessor 

import xgboost as xgb



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file, for continue training")
    args = parser.parse_args()

    c = load_config(args.config_path)
    ap = AudioProcessor(**c.audio)

    log_path = os.path.join(c.train_config['logs_path'], c.model_name)
    os.makedirs(log_path, exist_ok=True)

    tensorboard = TensorboardWriter(os.path.join(log_path,'tensorboard'))

    train_dataloader = train_dataloader(c, ap)
    max_seq_len = train_dataloader.dataset.get_max_seq_lenght()
    c.dataset['max_seq_len'] = max_seq_len

    # save config in train dir, its necessary for test before train and reproducity
    save_config_file(c, os.path.join(log_path,'config.json'))

    eval_dataloader = eval_dataloader(c, ap, max_seq_len=max_seq_len)

    train(args, log_path, args.checkpoint_path, train_dataloader, eval_dataloader, tensorboard, c, c.model_name, ap, cuda=True)