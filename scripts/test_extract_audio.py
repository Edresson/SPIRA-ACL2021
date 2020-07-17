
import os
import sys
sys.path.append('.')
sys.path.append('../')
import random
import argparse
import json
import torch
import torch.utils.data
from utils.audio_processor import AudioProcessor 
from utils.generic_utils import load_config


if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--wavfile_path", required=True)
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='JSON file for configuration')
    args = parser.parse_args()

    config = load_config(args.config)

    filepath = args.wavfile_path

    # extract spectrogram
    config.audio['feature'] = 'spectrogram'
    
    ap = AudioProcessor(**config.audio)
    spectrogram = ap.get_feature_from_audio_path(filepath)
    print("Spectogram with shape:",spectrogram.shape)

    # extract spectrogram
    config.audio['feature'] = 'melspectrogram'
    ap = AudioProcessor(**config.audio)
    melspectrogram = ap.get_feature_from_audio_path(filepath)
    print("MelSpectogram with shape:",melspectrogram.shape)

    # extract spectrogram
    config.audio['feature'] = 'mfcc'
    ap = AudioProcessor(**config.audio)
    mfcc = ap.get_feature_from_audio_path(filepath)
    print("MFCC with shape:",mfcc.shape)  



