"""
Wrapper for report.ipynb
"""
# This code is bad, but it was an easy way to use the existing functions
# to test and report the results on a jupyter notebook.

import os
import math
import random
import time
import itertools
import traceback

import numpy as np
import pandas as pd
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import plotly.offline as py
import IPython
import IPython.display as ipd  # To play sound in the notebook
import seaborn as sns

from torch import stack
from sklearn.metrics import confusion_matrix

from models import *
from test import *
from utils.audio_processor import AudioProcessor  
from utils.dataset import Dataset, DataLoader, own_collate_fn   
from utils.generic_utils import load_config, save_config_file
from torch.nn.utils.rnn import pad_sequence
from utils.generic_utils import set_init_dict
from utils.generic_utils import NoamLR, binary_acc
from utils.generic_utils import save_best_checkpoint
from utils.tensorboard import TensorboardWriter
from utils.audio_processor import AudioProcessor



def TestDataloader(c, ap, max_seq_len=None):
    return DataLoader(dataset=Dataset(c,
                                      ap,
                                      max_seq_len=max_seq_len,
                                      train=False,
                                      test=True,
                                      return_paths=True),
                      collate_fn=lambda batch: own_collate_fn(batch, True),
                      batch_size=c.test_config['batch_size'],
                      shuffle=False,
                      num_workers=c.test_config['num_workers'])


def plot_confusion_matrix(real=None, 
                          preds=None,
                          unique_labels=None, 
                          cm=None,
                          show=True,
                          output=None, 
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    if real is not None and preds is not None:
        cm = confusion_matrix(real, preds)
    elif cm is None:
        raise ValueError('Must provide a confusion matrix or real and preds')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks, rotation=45)
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks() + 1).astype(str))
    plt.yticks(tick_marks)

    ax.set_xticklabels(unique_labels)
    ax.set_yticklabels(unique_labels)

    thresh = cm.max() / 1.4
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.1f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if output is not None:
        plt.savefig(output)
    if show:
        plt.show()
    plt.close()
    return output


def test(criterion, ap, model, c, testloader, step,  cuda,
         confusion_matrix=False, verbose=0):
    
    padding_with_max_lenght = c.dataset['padding_with_max_lenght']
    losses = []
    accs = []
    preds = []
    targets = []
    paths = []
    
    model.zero_grad()
    model.eval()
    
    loss = 0 
    acc = 0
    
    with torch.no_grad():
        for feature, target, path in testloader:       
            if cuda:
                feature = feature.cuda()
                target = target.cuda()

            output = model(feature).float()
            paths.append(path)
            
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
            y_target = pd.Series(targets, name='Target')
            y_pred = pd.Series(preds, name='Predicted')
            df_confusion = pd.crosstab(y_target, y_pred, 
                                       rownames=['Target'],
                                       colnames=['Predicted'],
                                       margins=True)
            
        mean_acc = acc / len(testloader.dataset)
        mean_loss = loss / len(testloader.dataset)
    print("Test\n Loss:", mean_loss, "Acurracy: ", mean_acc)
    return mean_acc, df_confusion, preds, targets, paths

  
def plot_waveform(b, show_title=False, name='Audio'):
    plt.figure()
    plt.plot(b)
    if show_title:
        plt.title('Wavform of ' + name, wrap=True)
    plt.show()
    plt.close()
    
  
def plot_feature(b, show_title=False, name='Audio'):
    plt.pcolormesh(b)
    if show_title:
        plt.title('Feature of ' + name, wrap=True)
    plt.show()
    plt.close()
    

def show_sample(path,
                target,
                pred,
                audio_processor,
                verbose=1,
                show_spec=True):
    """Show detailed information about the predicted sample, that is: its real
    class and predicted class, the audio and a visual representation 
    (aka spectrogram).

    Args:
        path (str): The path to the file.
        target: The real class.
        pred: The predicted class.
        audio_processor (AudioProcessor): Instance of AudioProcessor to load wav
            file and its feature.
        show_spec (bool, optional): If . Defaults to True.
    """
    sr, b = audio_processor.sample_rate, audio_processor.load_wav(path)
    if verbose > 0:
        print(f"\n\nArquivo {os.path.basename(path)}")
        print("REAL:",      "CONTROLE"  if target==0    else "PACIENTE")
        print("PREDITO:",   "CONTROLE"  if pred==0      else "PACIENTE")
    if verbose > 1:
        print('Playing', path)
    # IPython.display.display(ipd.Audio(path))
    IPython.display.display(ipd.Audio(data=b, rate=sr))
    plot_waveform(b[0], name=str(path), show_title=False)
    if show_spec:
        plot_feature(audio_processor.wav2feature(b)[0], show_title=False)


def get_faixa_etaria(idade, intervalo=10):
    for i in range(0, 100, 10):
        if i <= idade <= i+intervalo:
            return f'Entre {i} e {i+intervalo}'


def show_info_errors(csv, paths_errors, title=""):
    data = pd.read_csv(csv)
    data = data[~data['file_path'].isin(paths_errors)]

    # plt.rcParams["axes.titlesize"] = 8

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 7))
    fig.suptitle(title, fontsize=18)

    total = len(data['sexo'])
    sizes = [len(data[data['sexo']=='F'])/total,
             len(data[data['sexo']=='M'])/total]
    ax1.pie(sizes,
            labels=["Feminino", "Masculino"],
            autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    ax1.set_title("Erros de classificação - Sexo")
    
    
    controle = data[data['class']==0]
    total = len(controle['sexo'])
    sizes = [len(controle[controle['sexo']=='F'])/total,
             len(controle[controle['sexo']=='M'])/total]
    ax2.pie(sizes,
            labels=["Feminino", "Masculino"],
            autopct='%1.1f%%',
            shadow=True,
            startangle=90)
    ax2.axis('equal')
    ax2.set_title("Erros de classificação \n(apenas dados de controle) - Sexo")
    
    
    pacientes = data[data['class']==1]
    total = len(pacientes['sexo'])
    sizes = [len(pacientes[pacientes['sexo']=='F'])/total,
             len(pacientes[pacientes['sexo']=='M'])/total]
    ax3.pie(sizes,
            labels=["Feminino", "Masculino"],
            autopct='%1.1f%%',
            shadow=True,
            startangle=90)
    ax3.axis('equal')
    ax3.set_title("Erros de classificação \n(apenas dados de pacientes) - Sexo")
    
    evals = data['idade'].apply(get_faixa_etaria).value_counts()
    sns.barplot(evals.index, evals.values)
    plt.xticks(rotation=90)
    ax4.set_xlabel('Idade')
    ax4.set_ylabel('Frequência')
    ax4.set_title("Erros de classificação - Sexo")

    plt.show()
    plt.close()


def run_test(config_path,
             csv_data,
             data_root_dir,
             batch_size,
             num_workers,
             checkpoint_path,
             no_insert_noise=True,
             cuda=True,
             show_each_sample=True,
             show_info=True,
             verbose=0,
             *args,
             **kwargs):
    """Run the test

    General Args:
        config_path (str or os.path like): Path to the generated config file.
            See config.json.
            NOTE: this config should be the generated config file obtained after
            the trainning process. Otherwise some parameters might not be present. 
        csv_data (str or os.path like): CSV file containing instances for testing.
        data_root_dir (str or os.path like): Root dir to load the data.
        no_insert_noise (bool): Choose to whether to insert noise when testing or
            not. Defaults to False.

    Running Args:
        batch_size (int): batch size. 
        num_workers (int): number of workers. 
        cuda (bool, optional): Defaults to True.

    Reporting args:
        show_each_sample (bool, optional): If True will output information about
            each predictec sample. See show_sample. Defaults to True.
            *args and **kwargs are passed through this function.
        show_info (bool, optional): If True will show info about incorrect 
            predicted instances of the tested set. Defaults to True.
        verbose (int, optional): If > 0 will show
            some messages in the output stream. Defaults to 0. 
    """

    config = load_config(config_path)
    model = get_model_from_config(config)
    audio_processor = AudioProcessor(**config.audio)
    config.dataset['test_csv']                      = csv_data
    config.dataset['test_data_root_path']           = data_root_dir
    config.test_config['batch_size']                = batch_size
    config.test_config['num_workers']               = num_workers
    max_seq_len = config.dataset['max_seq_len'] 
    padding_with_max_lenght                         = config.dataset['padding_with_max_lenght']
    criterion = nn.BCELoss(reduction='sum')
    test_loader = TestDataloader(config,
                                 audio_processor,
                                 max_seq_len=max_seq_len)


    if verbose > 1:
        print("Insert noise ?", config.data_aumentation['insert_noise'])
    config.data_aumentation['insert_noise'] = not no_insert_noise

    if config.train_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.train_config['learning_rate'])
    else:
        raise Exception("The %s  not is a optimizer supported" % config.train['optimizer'])

    step = 0
    if checkpoint_path is not None:
        if verbose > 1:
            print("Loading checkpoint: %s" % checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            if verbose > 1:
                print("Model Sucessful Load !")
        except Exception as e:
            raise ValueError("You need pass a valid checkpoint "
                             "(you might need to check your config.json). "+ e)       
        step = checkpoint['step']
    else:
        raise ValueError("You need pass a checkpoint_path")   

    if cuda:
        model = model.cuda()
    
    model.train(False)

    mean_acc, df_confusion, preds, targets, paths = test(criterion, 
                                                         audio_processor,
                                                         model, config,
                                                         test_loader,
                                                         step,
                                                         cuda=cuda,
                                                         verbose=verbose,
                                                         confusion_matrix=True)
    if verbose > 0:
        print("Mean Acc:", mean_acc)
    if verbose > 1:
        print("Confusion matrix:", df_confusion)

    plot_confusion_matrix(real=targets,
                          preds=preds,
                          title=f"Confusion Matrix of {config.model_name} (ACC={mean_acc:.2g})",
                          show=True,
                          unique_labels=["CONTROLE", "PACIENTE"])

    paths_errors = []
    for path, target, pred in zip(paths, targets, preds):
        if target != pred:  # If incorrect!!!
            if show_each_sample:
                show_sample(path[0],
                            target,
                            pred,
                            audio_processor,
                            show_spec=True)
            paths_errors.append(path)

    if show_info:
        show_info_errors(csv_data,
                         paths_errors,
                         title=f"Dados preditos incorretamente com {config.model_name} (ACC={mean_acc:.2g})")
