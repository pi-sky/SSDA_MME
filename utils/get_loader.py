import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
import os

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

def get_dataloaders(data_path, domain = 'source'):
  dataset = pd.read_csv(data_path, header=None)
  if domain == 'source':
      source_data, unused_data1 = split_data(dataset)
      known_source_data, _ , unused_data2 = split_unknown(source_data)
      unlab_source_data, _, unused_data3 = split_unknown(unused_data1)
      source_loader = data_loader(known_source_data)
      source_unl_loader = data_loader(unlab_source_data)
      return source_loader, source_unl_loader
  elif domain=='target':
      unused_data3, unused_data4, target_data = split_unknown(dataset)
      target_loader_train = data_loader(target_data)
      target_loader_val = data_loader(target_data)
      target_loader_test = data_loader(target_data)
      return target_loader_train, target_loader_val, target_loader_test
  else:
    print("Specify domain for dataloader")
    return -1

def split_unknown(dataset):
    dataset.set_axis([*dataset.columns[:-1], "class"], axis=1, inplace=True)
    mask = dataset['class'] <= 19
    df1 = dataset[mask]
    df2 = dataset[~mask]
    df2.loc[:, ['class']] = 20.0
    known_data = df1.copy()
    unknown_data = df2.copy()
    dataset['class'].replace([x for x in range(21, 31)], [20.0 for x in range(21, 31)], inplace = True)

    return known_data, unknown_data, dataset.copy()

# path is of form  '...../domain_domain.extension'
def get_domain_from_path(path):
    folder, file = os.path.split(path)
    file_name = os.path.splitext(file)
    domain = file_name[0].split('_')
    return domain[0]