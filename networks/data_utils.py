"""Data utility functions."""
import os

import numpy as np
import scipy.io
import torch
import torch.utils.data as data
import h5py


class ImdbData(data.Dataset):
    def __init__(self, X, y, w):
        self.X = X
        self.y = y
        self.w = w

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]
        weight = self.w[index]

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        weight = torch.from_numpy(weight)
        return img, label, weight

    def __len__(self):
        return len(self.y)


def get_imdb_data():
    # TODO: Need to change later
    NumClass = 9

    # Load DATA
    # Data = h5py.File('C:/Users/krive/PycharmProjects/relaynet_pytorch/datasets/Data.h5', 'r')
    # a_group_key = list(Data.keys())[0]
    # Data = list(Data[a_group_key])
    # Data = np.squeeze(np.asarray(Data))
    # Label = h5py.File('C:/Users/krive/PycharmProjects/relaynet_pytorch/datasets/label.h5', 'r')
    # a_group_key = list(Label.keys())[0]
    # Label = list(Label[a_group_key])
    # Label = np.squeeze(np.asarray(Label))
    # set = h5py.File('C:/Users/krive/PycharmProjects/relaynet_pytorch/datasets/set.h5', 'r')
    # a_group_key = list(set.keys())[0]
    # set = list(set[a_group_key])
    # set = np.squeeze(np.asarray(set))
    # sz = Data.shape
    # print(sz)
    # Data = Data.reshape([sz[0], 1, sz[1], sz[2]])
    # Data = Data[:, :, 61:573, :]
    # weights = Label[:, 1, 61:573, :]
    # Label = Label[:, 0, 61:573, :]
    # sz = Label.shape
    # Label = Label.reshape([sz[0], 1, sz[1], sz[2]])
    # weights = weights.reshape([sz[0], 1, sz[1], sz[2]])
    # train_id = set == 1
    # test_id = set == 3
    #
    # Tr_Dat = Data[train_id, :, :, :]
    # Tr_Label = np.squeeze(Label[train_id, :, :, :]) - 1  # Index from [0-(NumClass-1)]
    # Tr_weights = weights[train_id, :, :, :]
    # Tr_weights = np.tile(Tr_weights, [1, NumClass, 1, 1])
    #
    # Te_Dat = Data[test_id, :, :, :]
    # Te_Label = np.squeeze(Label[test_id, :, :, :]) - 1
    # Te_weights = weights[test_id, :, :, :]
    # Te_weights = np.tile(Te_weights, [1, NumClass, 1, 1])

    Data = scipy.io.loadmat('C:/Users/krive/PycharmProjects/relaynet_pytorch/datasets/imgTest.mat')
    Data = Data['img']
    Data = np.transpose(Data, (2, 0, 1))
    sz = Data.shape
    print(sz)
    Data = Data.reshape(sz[0], 1, sz[1], sz[2])
    Data = Data[:, :, :, :]
    lbl = scipy.io.loadmat('C:/Users/krive/PycharmProjects/relaynet_pytorch/datasets/lblgTest.mat')
    lbl = lbl['img']
    lbl = np.transpose(lbl, (2, 0, 1))
    sz = lbl.shape
    print(sz)
    lbl = lbl.reshape(sz[0], 1, sz[1], sz[2])
    lbl = lbl[:, :, :, :]
    weights = lbl

    return (ImdbData(Data, lbl, weights),
            ImdbData(Data, lbl, weights))
