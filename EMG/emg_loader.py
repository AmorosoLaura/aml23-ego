import torch
import torch.nn as nn
import torch.optim as optim
import scipy as sp
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os

# labels taken from the official code of the ActionSense dataset
baseline_label = 'None'
activities_to_classify = [
   baseline_label,
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
class EMG_dataset(Dataset):
    def __init__(self, directory, filename):
        self.df = pd.read_pickle(os.path.join(directory, filename))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        emg, label = self.df[idx]['emg_data'],self.df[idx]['label']
        if label== 'Get items from refrigerator/cabinets/drawers' or label== 'Replace items from refrigerator/cabinets/drawers' :
            return (emg, 1)
        elif label=='Open a jar of almond butter' or label=='Close a jar of almond butter':
            return (emg,10)
        return (emg, activities_to_classify.index(label))

