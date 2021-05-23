import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import random

def split_data(data):
  # data = pd.read_csv(data_path, header=None)
  data.set_axis([*data.columns[:-1], "classes"], axis=1, inplace=True)

  source_samples = data[data['classes']==0].sample(frac=0.06, random_state=1)
  data = data.drop(source_samples.index)
  source_samples = source_samples.to_numpy()
  # print(samples)
  for i in range(1,31):
    src_samp_class_data = data[data['classes']==i].sample(frac=0.06, random_state=1)
    data = data.drop(src_samp_class_data.index)
    source_samples = np.vstack((source_samples,src_samp_class_data.to_numpy()))

  source_data = pd.DataFrame(source_samples)
  source_data.set_axis([*source_data.columns[:-1], "classes"], axis=1, inplace=True)

  return source_data, data              #data returned id target data


# returns a dataloader for a given dataframe obj argument
def data_loader(dataset, batch_size=3):
    features = dataset.iloc[:, 0:-1].values
    labels = dataset.iloc[:, -1:].values
    features = torch.tensor(features)
    labels = torch.IntTensor(labels)

    dataset = TensorDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    return data_loader

def get_dataloaders(data_path):
  dataset = pd.read_csv(data_path, header=None)
  source_data, target_data = split_data(dataset)
  source_loader = data_loader(source_data)
  target_loader_train = data_loader(target_data)
  target_loader_val = data_loader(target_data)
  target_loader_test = data_loader(target_data)

  return source_loader, target_loader_train, target_loader_val, target_loader_test
