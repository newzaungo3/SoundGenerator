import torch
import torchaudio
import requests
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import Dataset,DataLoader
import librosa
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import librosa.display
import torch.nn as nn
import torch.nn.functional as F
import tarfile


data_path = "./Data/genres_original"
genres = os.listdir(data_path)


class audioData(Dataset):
    def __init__(self):
        super().__init__()
        self.data_path = "./Data/genres_original"
        self.genres = os.listdir(data_path)
        self.raw = []
        self.rate = []
        self.label = []
        for genre in genres:
            # filename path join datapath + genre = ./Data/genres_original/blues
            filenames = os.listdir(os.path.join(data_path, genre))

            for filename in filenames:
                # get filename in each genre ./Data/genres_original\blues\blues.00000.wav
                file_path = os.path.join(data_path, genre, filename)
                waveform, sample_rate = torchaudio.load(file_path)
                self.raw.append(waveform)
                self.rate.append(sample_rate)
                self.label.append(genre)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        return self.raw[idx], self.rate[idx],self.label[idx]
#trainset include rawaudio samplerate and label
train_set = audioData()

#dataloader for pytorch train
train_load = DataLoader(train_set,batch_size=15,shuffle=True)

#Number of class ex 2 class(jazz,country)
All_class = set(train_set.label)
numclass = len(All_class)

'''
class Model(nn.Module):
    def __init__(self,Numclass):
        super(Model, self).__init__()
        self.Numclass = Numclass
        self.C1 = nn.Conv2d()
'''
