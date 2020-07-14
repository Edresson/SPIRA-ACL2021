import os
import math
import torch
import torch.nn as nn
import traceback

import time
import numpy as np

import argparse

from utils.generic_utils import load_config
from utils.generic_utils import set_init_dict

from utils.tensorboard import TensorboardWriter

from utils.dataset import train_dataloader, eval_dataloader

from models.spiraconv import SpiraConv
from utils.audio_processor import AudioProcessor 


def validation(criterion, ap, model, testloader, tensorboard, step,  cuda):
    model.eval()
    losses = []
    for feature, target in testloader:
        #try:
        if cuda:
            feature = feature.cuda()
            target = target.cuda()

        output = model(feature)
        # Calculate loss
        loss = criterion(output, target).item()
        losses.append(loss)
    mean_loss = np.array(losses).mean()
    tensorboard.log_evaluation(mean_loss, step)
    model.train()

def train(args, log_dir, checkpoint_path, trainloader, testloader, tensorboard, c, model_name, ap, cuda=True):
    if(model_name == 'spiraconv'):
        model = SpiraConv(c)
    #elif(model_name == 'voicesplit'):
    else:
        raise Exception(" The model '"+model_name+"' is not suported")

    if c.train_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=c.train_config['learning_rate'])
    else:
        raise Exception("The %s  not is a optimizer supported" % c.train['optimizer'])

    step = 0
    if checkpoint_path is not None:
        print("Continue training from checkpoint: %s" % checkpoint_path)
        try:
            if c.train_config['reinit_layers']:
                raise RuntimeError
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            if cuda:
                model = model.cuda()
        except:
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint, c)
            model.load_state_dict(model_dict)
            del model_dict
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print(" > Optimizer state is not loaded from checkpoint path, you see this mybe you change the optimizer")
        
        step = checkpoint['step']
    else:
        print("Starting new training run")
    # convert model from cuda
    if cuda:
        model = model.cuda()

    # define loss function
    criterion = nn.BCELoss()

    model.train()
    for _ in range(c.train_config['epochs']):
        validation(criterion, ap, model, testloader, tensorboard, step,  cuda=cuda)
        for feature, target in trainloader:
                if cuda:
                    feature = feature.cuda()
                    target = target.cuda()

                output = model(feature)

                # Calculate loss
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    print("Loss exploded to %.02f at step %d!" % (loss, step))
                    break

                # write loss to tensorboard
                if step % c.train_config['summary_interval'] == 0:
                    tensorboard.log_training(loss, step)
                    print("Write summary at step %d" % step)

                # save checkpoint file  and evaluate and save sample to tensorboard
                if step % c.train_config['checkpoint_interval'] == 0:
                    save_path = os.path.join(log_dir, 'checkpoint_%d.pt' % step)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'config_str': str(c),
                    }, save_path)
                    print("Saved checkpoint to: %s" % save_path)
                    validation(criterion, ap, model, testloader, tensorboard, step,  cuda=cuda)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file, for continue training")
    args = parser.parse_args()

    c = load_config(args.config_path)
    ap = AudioProcessor(c.audio)

    log_path = os.path.join(c.train_config['logs_path'], c.model_name)
    os.makedirs(log_path, exist_ok=True)

    tensorboard = TensorboardWriter(os.path.join(log_path,'tensorboard'))

    train_dataloader = train_dataloader(c, ap)
    eval_dataloader = eval_dataloader(c, ap)

    train(args, log_path, args.checkpoint_path, train_dataloader, eval_dataloader, tensorboard, c, c.model_name, ap, cuda=True)