import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import pickle
import pandas as pd

activities_to_classify = [
  'Get/replace items from refrigerator/cabinets/drawers',
  'Peel a cucumber',
  'Clear cutting board',
  'Slice a cucumber',
  'Peel a potato',
  'Slice a potato',
  'Slice bread',
  'Spread almond butter on a bread slice',
  'Spread jelly on a bread slice',
  'Open/close a jar of almond butter',
  'Pour water from a pitcher into a glass',
  'Clean a plate with a sponge',
  'Clean a plate with a towel',
  'Clean a pan with a sponge',
  'Clean a pan with a towel',
  'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Stack on table: 3 each large/small plates, bowls',
  'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
  'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
]

# Sampling frequency is 160 Hz
# With 32 samples the frequency resolution after FFT is 160 / 32

n_fft = 32
win_length = None
hop_length = 4

spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    normalized=True
)


def cut_and_pad(signal, sampling_rate, seconds):
    required_length = sampling_rate * seconds
    padded_signal = torch.zeros((required_length,16))
   
    if signal.shape[0] < required_length:
        padded_signal[:signal.shape[0]] = signal
    else:
        padded_signal[:required_length] = signal[:required_length]

    return padded_signal

if __name__ == '__main__':

    next_key=0

    for split in ['train', 'test']:

        print(split)
        spectrograms={}
   
        emg_annotations = pd.read_pickle("./emg_data_preprocessed_"+split+".pkl")
        for sample in emg_annotations.values():
            signal = torch.from_numpy(sample['emg_data']).float()
            signal=cut_and_pad(signal,160,30)
            
            label=sample['label']
            if label== 'Get items from refrigerator/cabinets/drawers' or label== 'Replace items from refrigerator/cabinets/drawers' :
                spectrograms[next_key]={'spectrogram': signal, 'label': 0}
        
            elif label=='Open a jar of almond butter' or label=='Close a jar of almond butter':
                spectrograms[next_key]={'spectrogram': signal, 'label': 9}
        
            else:
                spectrograms[next_key]={'spectrogram': signal, 'label': activities_to_classify.index(sample['label'])}
            next_key+=1
        

        labels = [value['label'] for value in spectrograms.values()]

        print(labels)
        
        with open('./EMG_data/emg_spectrogram_'+split+'.pkl', 'wb') as f_pickle:
            pickle.dump(spectrograms, f_pickle)