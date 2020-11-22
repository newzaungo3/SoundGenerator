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
from tqdm import tqdm

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
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
            break

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        return self.raw[idx], self.rate[idx], self.label[idx]
#trainset include rawaudio samplerate and label
train_set = audioData()
print(train_set.raw[0])

melspec = torchaudio.transforms.MelSpectrogram()(train_set.raw[0])
print(melspec.shape)


#dataloader for pytorch train
train_load = DataLoader(train_set,batch_size=15,shuffle=True)



#Number of class ex 2 class(jazz,country)
All_class = set(train_set.label)
numclass = len(All_class)

#cnn
class Model(nn.Module):
    '''
        infomation
        if youwant to convo with melspectrogram
        input: mel(sft from waveform) [1,128,3309]

        conv1d(input_channel,output_channel,kernel_size,stride,padding)
        example conv1d(128,256,kennel_size=3)
        returns torch[1,256,3307]

        if you want to convo with raw waveform
        unsqueeze data first to be 3dimension [1,1,661794]
        input: waveform(raw audio) [1,1,661794]
        convo1d(1,128,kernel_size =3)
        return torch [1,128,661792]

        input:[minibatch_size,channel,width]
        in this case [15,1,667546]
    '''
    def __init__(self,audio_size=1,h_dim=128,z_dim=10):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(audio_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, audio_size)
        # 59049 x 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 19683 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 6561 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 2187 x 128
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 729 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 243 x 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3),
            nn.Dropout(0.5))
        # 81 x 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 27 x 256
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 9 x 256
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 3 x 256
        self.conv10 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        # 1 x 512
        self.conv11 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5))

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self,x):
        '''
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        '''
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)
        logit = out.view(x.shape[0],out.size(1) * out.size(2))
        fcout = self.fc1(logit)
        return out,logit,out.size(1),out.size(2),fcout
#model cuda input must be cuda || model cpu input must be cpu
model = Model().cpu()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
num_epoch = 2

for epoch in tqdm(range(num_epoch)):
    for batch,(raw,rate,label) in enumerate(train_load):
                # input [minibatchsize,channel,width]
                raw = raw.cpu()
                #label = label[0].to(device)
                #forward pass
                print(raw.shape)
                #raw = raw.view(-1,raw.shape[0])
                print(raw.view(-1,raw.shape[0]).shape)

                out,logit,out1,out2,fcout= model(raw)
                print("Out:",out)
                print("logit:",logit)
                print("Out shape:",out.shape)
                print("logit shape:",logit.shape)
                print("Out1:",out1)
                print("Out2:",out2)
                print("fcout:",fcout)
                print("fcout shape:",fcout.shape)
                #print(output.shape)
                #loss = criterion(output,label)

'''
#train
for batch,x in enumerate(train_load):
    # X = [minibatchsize,channel,width]
    conv2 = nn.Sequential(
        nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
        nn.BatchNorm1d(128),
        nn.ReLU())
    print(x[0].shape)
    #print(conv2(x[0]))
'''