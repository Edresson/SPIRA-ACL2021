import os
import math
import torch
import torch.nn as nn
import traceback
import pandas as pd

import time
import numpy as np

import argparse

from utils.generic_utils import load_config, save_config_file
from utils.generic_utils import set_init_dict

from utils.generic_utils import NoamLR, binary_acc

from utils.generic_utils import save_best_checkpoint

from utils.tensorboard import TensorboardWriter

from utils.dataset import test_dataloader

from models.spiraconv import SpiraConvV1, SpiraConvV2
from utils.audio_processor import AudioProcessor 

def test(criterion, ap, model, c, testloader, step,  cuda, confusion_matrix=False):
    padding_with_max_lenght = c.dataset['padding_with_max_lenght']
    losses = []
    accs = []
    model.zero_grad()
    model.eval()
    loss = 0 
    acc = 0
    preds = []
    targets = []
    with torch.no_grad():
        for feature, target in testloader:       
            #try:
            if cuda:
                feature = feature.cuda()
                target = target.cuda()

            output = model(feature).float()

            # output = torch.round(output * 10**4) / (10**4)

            # Calculate loss
            if not padding_with_max_lenght:
                target = target[:, :output.shape[1],:target.shape[2]]
            loss += criterion(output, target).item()

            # calculate binnary accuracy
            y_pred_tag = torch.round(output)
            acc += (y_pred_tag == target).float().sum().item()
            preds += y_pred_tag.reshape(-1).int().cpu().numpy().tolist()
            targets += target.reshape(-1).int().cpu().numpy().tolist()
        if confusion_matrix:
            print("======== Confusion Matrix ==========")
            y_target = pd.Series(targets, name='Target')
            y_pred = pd.Series(preds, name='Predicted')
            df_confusion = pd.crosstab(y_target, y_pred, rownames=['Target'], colnames=['Predicted'], margins=True)
            print(df_confusion)
            
        mean_acc = acc / len(testloader.dataset)
        mean_loss = loss / len(testloader.dataset)
    print("Test\n Loss:", mean_loss, "Acurracy: ", mean_acc)
    return mean_acc


def run_test(args, checkpoint_path, testloader, c, model_name, ap, cuda=True):

    # define loss function
    criterion = nn.BCELoss(reduction='sum')

    padding_with_max_lenght = c.dataset['padding_with_max_lenght']
    if(model_name == 'spiraconv_v1'):
        model = SpiraConvV1(c)
    elif (model_name == 'spiraconv_v2'):
        model = SpiraConvV2(c)
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
        print("Loading checkpoint: %s" % checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("Model Sucessful Load !")
        except Exception as e:
            raise ValueError("You need pass a valid checkpoint, may be you need check your config.json because de the of this checkpoint cause the error: "+ e)       
        step = checkpoint['step']
    else:
        raise ValueError("You need pass a checkpoint_path")   

    # convert model from cuda
    if cuda:
        model = model.cuda()
    
    model.train(False)
    test_acc = test(criterion, ap, model, c, testloader, step, cuda=cuda, confusion_matrix=True)
        

if __name__ == '__main__':
    # python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1/spiraconv/checkpoint_1068.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1/spiraconv/config.json  --batch_size 5 --num_workers 2 --no_insert_noise True

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_csv', type=str, required=True,
                        help="test csv example: ../SPIRA_Dataset_V1/metadata_test.csv")
    parser.add_argument('-r', '--test_root_dir', type=str, required=True,
                        help="Test root dir example: ../SPIRA_Dataset_V1/")
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations get in checkpoint path")
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True,
                        help="path of checkpoint pt file, for continue training")
    parser.add_argument('--batch_size', type=int, default=20,
                        help="Batch size for test")
    parser.add_argument('--num_workers', type=int, default=10,
                        help="Number of Workers for test data load")
    parser.add_argument('--no_insert_noise', type=bool, default=False,
                        help=" No insert noise in test ?")
                        
    args = parser.parse_args()

    c = load_config(args.config_path)
    ap = AudioProcessor(**c.audio)
    
    if not args.no_insert_noise:
        c.data_aumentation['insert_noise'] = True
    else:
        c.data_aumentation['insert_noise'] = False
    print("Insert noise ?", c.data_aumentation['insert_noise'])

    c.dataset['test_csv'] = args.test_csv
    c.dataset['test_data_root_path'] = args.test_root_dir


    c.test_config['batch_size'] = args.batch_size
    c.test_config['num_workers'] = args.num_workers
    max_seq_len = c.dataset['max_seq_len'] 

    test_dataloader = test_dataloader(c, ap, max_seq_len=max_seq_len)

    run_test(args, args.checkpoint_path, test_dataloader, c, c.model_name, ap, cuda=True)