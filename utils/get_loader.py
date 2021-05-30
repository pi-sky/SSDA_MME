import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import random

def split_data(data, frac=0.06):
  # data = pd.read_csv(data_path, header=None)
  data.set_axis([*data.columns[:-1], "classes"], axis=1, inplace=True)

  source_samples = data[data['classes']==0].sample(frac=frac, random_state=1)
  data = data.drop(source_samples.index)
  source_samples = source_samples.to_numpy()
  # print(samples)
  for i in range(1,31):
    src_samp_class_data = data[data['classes']==i].sample(frac=frac, random_state=1)
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

def get_dataloaders(data_path, domain ='source', batch_size=3, frac=0.06):
  dataset = pd.read_csv(data_path, header=None)
  if domain=='source':
      source_data, source_un_data = split_data(dataset, frac)
      source_l_loader = data_loader(source_data, batch_size)
      source_un_loader = data_loader(source_un_data, batch_size)
      source_test_loader = data_loader(source_un_data, batch_size)

      return source_l_loader, source_un_loader, source_test_loader
  elif domain=='target':
      target_un_loader = data_loader(dataset, batch_size)
      target_loader_val = data_loader(dataset, batch_size)
      target_loader_test = data_loader(dataset, batch_size)

      return target_un_loader, target_loader_val, target_loader_test
  else:
      print("Specify domain for dataloader")
      return 0

# path is of form  '...../domain_domain.extension'
def get_domain_from_path(path):
    folder, file = os.path.split(path)
    file_name = os.path.splitext(file)
    domain = file_name[0].split('_')
    return domain[0]