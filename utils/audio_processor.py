import torchaudio
from utils.hpcp import HPCP

class AudioProcessor(object):
    def __init__(self, feature, num_mels, num_mfcc, log_mels, mel_fmin, mel_fmax, normalize, sample_rate, n_fft, num_freq, hop_length, win_length, f_min, f_max, global_thr, local_thr, bins_per_octave, filter_width, harmonic_decay, harmonic_tolerance, final_thr):
        self.feature = feature
        self.num_mels = num_mels
        self.num_mfcc = num_mfcc
        self.log_mels = log_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.normalize = normalize
        self.num_freq = num_freq
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.global_thr = global_thr
        self.local_thr = local_thr
        self.bins_per_octave = bins_per_octave 
        self.filter_width = filter_width
        self.harmonic_decay = harmonic_decay
        self.harmonic_tolerance = harmonic_tolerance
        self.final_thr = final_thr
        
        print()

        valid_features = ['spectrogram', 'melspectrogram', 'mfcc', 'hpcp']
        if self.feature not in valid_features:
            raise ValueError("Invalid Feature: "+str(self.feature))

    def wav2feature(self, y):
        if self.feature == 'spectrogram':
            audio_class = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        
        elif self.feature == 'melspectrogram':
            audio_class = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, n_mels=self.num_mels, f_min=self.mel_fmin, f_max=self.mel_fmax)
        
        elif self.feature == 'mfcc':
            audio_class = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=self.num_mfcc, log_mels=self.log_mels, melkwargs={'n_fft':self.n_fft, 'win_length':self.win_length, 'hop_length':self.hop_length, 'n_mels':self.num_mels})
        
        elif self.feature == 'hpcp':
            audio_class = HPCP(sample_rate=self.sample_rate, win_size=self.win_length, hop_size=self.hop_length, f_min=self.f_min, f_max=self.f_max, global_thr=self.global_thr, local_thr=self.local_thr, bins_per_octave=self.bins_per_octave, filter_width=self.filter_width, harmonic_decay=self.harmonic_decay, harmonic_tolerance=self.harmonic_tolerance, final_thr=self.final_thr)

        feature = audio_class(y)
        return feature

    def get_feature_from_audio_path(self, audio_path):
        return self.wav2feature(self.load_wav(audio_path))

    def get_feature_from_audio(self, wav):
        feature = self.wav2feature(wav)
        return feature

    def load_wav(self, path):
        wav, sample_rate = torchaudio.load(path, normalization=self.normalize)
        # resample audio for specific samplerate
        if sample_rate != self.sample_rate:
            resample = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            wav = resample(wav)
        return wav