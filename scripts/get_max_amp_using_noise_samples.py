
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
import torchaudio

# audio_path, noise_start(in second), noise_end (in second)
samples = [[['../../dados-exemplo/202020/PTT-20200511-WA0005.wav',1.2, 1.75], ['../../dados-exemplo/202020/PTT-20200511-WA0005.wav', 3.4,4.05],
['../../dados-exemplo/202020/PTT-20200511-WA0005.wav', 5.3,5.75],  ['../../dados-exemplo/202020/PTT-20200511-WA0005.wav',6.5,7.15],
['../../dados-exemplo/202020/PTT-20200511-WA0005.wav', 9.95,10.85]]

,[['../../dados-exemplo/202020/PTT-20200511-WA0012.wav',2.3,3.1],
['../../dados-exemplo/202020/PTT-20200511-WA0012.wav',4.75,5.8], ['../../dados-exemplo/202020/PTT-20200511-WA0012.wav', 6.95,7.55]], 
[[ '../../dados-exemplo/202020/PTT-20200511-WA0014.wav', 0,1.4]],
[['../../dados-exemplo/202020/PTT-20200515-WA0002.wav', 0,1.2],['../../dados-exemplo/202020/PTT-20200515-WA0002.wav', 2.20,2.80]],
[['../../dados-exemplo/202020/PTT-20200624-WA0001.wav', 0, 1]]
]
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='JSON file for configuration')
    args = parser.parse_args()

    config = load_config(args.config)
    # extract spectrogram
    config.audio['feature'] = 'spectrogram'
    
    ap = AudioProcessor(**config.audio)
    max_amp = 0
    min_amp = 99999999999999999
    for samples_file in samples:
        for sample in samples_file:
            loc_max_amp = 0
            wav = ap.load_wav(sample[0])
            
            inicio = sample[1]
            fim = sample[2]
            slice_start = int(inicio*ap.sample_rate)
            slice_end = int(fim*ap.sample_rate)
            wav = wav[:,slice_start:slice_end]
            wav_max_amp = wav.max().numpy()
            if wav_max_amp > loc_max_amp:
                loc_max_amp = wav_max_amp
        if loc_max_amp < min_amp:
            min_amp = loc_max_amp
        if loc_max_amp > max_amp:
            max_amp = loc_max_amp    

        print('wav max amp for ',samples_file[0][0],' = ',loc_max_amp)
            #torchaudio.save('audio-1.wav', wav, ap.sample_rate)
    print('maxima das maxima amplitudes: ', max_amp, 'Minimima das maximas amplitudes:', min_amp)


