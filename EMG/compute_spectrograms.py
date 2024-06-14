import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import pickle
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import os

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

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(len(specgram), 1, figsize=(16, 8))

    axs[0].set_title(title or "Spectrogram (db)")

    for i, spec in enumerate(specgram):
        im = axs[i].imshow(librosa.power_to_db(specgram[i]), origin="lower", aspect="auto")
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    axs[i].set_xlabel("Frame number")
    axs[i].get_xaxis().set_visible(True)
    plt.show(block=False)

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



if __name__ == '__main__':

    print(os.getcwd())
    for split in ['train', 'test']:
        next_key=0
        
        print(split)
        spectrograms={}
   
        emg_annotations = pd.read_pickle("./EMG_data/emg/emg_10s_"+split+".pkl")
        
        print(emg_annotations)
        for index, sample in emg_annotations.iterrows():
            signal = torch.from_numpy(sample['emg_data']).float()
            signals=[spectrogram(signal[:, i]) for i in range(16)]
         
            label =sample['description']
            if label== 'Get items from refrigerator/cabinets/drawers' or label== 'Replace items from refrigerator/cabinets/drawers' :
                
                if len(spectrograms)==0:
                    
                    spectrograms=pd.DataFrame({'spectrogram': [signals], 'label': [0], 'uid': [sample['uid']], 'file': [sample['file']]})
                
                else:
                    spectrograms.loc[len(spectrograms)] = {'spectrogram': signals, 'label': 0, 'uid': sample['uid'], 'file': sample['file']}
        
            elif label=='Open a jar of almond butter' or label=='Close a jar of almond butter':
                if len(spectrograms)==0:
                    spectrograms=pd.DataFrame({'spectrogram': [signals], 'label': [9], 'uid': [sample['uid']], 'file': [sample['file']]})       
                else:
                    spectrograms.loc[len(spectrograms)] = {'spectrogram': signals, 'label': 9, 'uid': sample['uid'], 'file': sample['file']}        
            else:
                if len(spectrograms)==0:
                    
                    spectrograms=pd.DataFrame({'spectrogram': [signals], 'label': [activities_to_classify.index(sample['description'])], 'uid': [sample['uid']], 'file': [sample['file']]})
                else:
                    spectrograms.loc[len(spectrograms)] = {'spectrogram': signals, 'label': activities_to_classify.index(sample['description']), 'uid': sample['uid'], 'file': sample['file']}
            next_key+=1
            
            
        with open('./EMG_data/emg_spectrogram_10s_'+split+'.pkl', 'wb') as f_pickle:
            pickle.dump(spectrograms, f_pickle)