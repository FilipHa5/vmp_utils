from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import torch
import os
import numpy as np
import random as r

class VMPData(Dataset):
  def __init__(self, path, transform=None):

    self.fileslist = os.listdir(path)
    # TODO: optimize this class to not read whole data at the same time
    self.files = [np.load(os.path.join(path, file), allow_pickle = True) for file in self.fileslist if file.endswith('.npy')]
    self.data = []
    self.filenames = []
    self.labels = []
    self.transform = transform
    for index, filename_r_x in enumerate(self.files):
        for element in filename_r_x:
            self.labels.append(torch.tensor(np.array(index), dtype = torch.long))
            self.filenames.append(element[0])
            self.data.append([element[1], element[2]])

  def __getitem__(self, index):
    if self.transform:
      return [self.transform(self.data[index]), self.labels[index], self.filenames[index]]
    return [self.data[index], self.labels[index], self.filenames[index]]

  def __len__(self):
    return len(self.data)

class VMPDataWideSlim(VMPData):
    def __getitem__(self, index):
      if self.transform:
        return [self.transform(self.data[index])[0], self.transform(self.data[index])[1], self.filenames[index]]
      return [self.data[index], self.data[index], self.filenames[index]]

class GetChannelsForWideSlimPred(object):
  def __init__(self, channels_wide, channels_slim, modality):
    self.channels_wide = channels_wide
    self.channels_slim = channels_slim

  def __call__(self, data):
    r_reads_wide = data[0][self.channels_wide]
    x_reads_wide = data[1][self.channels_wide]
    r_reads_slim = data[0][self.channels_slim]
    x_reads_slim = data[1][self.channels_slim]
    return [[r_reads_wide, x_reads_wide], [r_reads_slim, x_reads_slim]]

class StackSignalsForWideSlimPred(object):
  def __call__(self, data):
    r_stacked_wide = np.hstack(data[0][0])
    x_stacked_wide = np.hstack(data[0][1])
    r_stacked_slim = np.hstack(data[1][0])
    x_stacked_slim = np.hstack(data[1][1])
    joined_rx_wide = [r_stacked_wide, x_stacked_wide]
    joined_rx_slim = [r_stacked_slim, x_stacked_slim]
    return np.hstack(joined_rx_wide), np.hstack(joined_rx_slim)

class ToTensorWideSlim(object):
  def __call__(self, signal):
    return torch.tensor(signal[0], dtype = torch.float32), torch.tensor(signal[1], dtype = torch.float32)

class GetChannels(object):
  def __init__(self, channels):
    self.channels = channels

  def __call__(self, data):
    r_reads = data[0][self.channels]
    x_reads = data[1][self.channels]
    return [r_reads, x_reads]

class RollAllChannels(object):
  def __call__(self, r_x_reads):
    r_x_reads = r_x_reads
    r =  r_x_reads[0]
    x = r_x_reads[1]

    if np.random.rand() > .5:
      return r_x_reads

    self._calc_to_move(r[0])

    r = [np.array(np.roll(sig, self.to_move)) for sig in r]
    x = [np.array(np.roll(sig, self.to_move)) for sig in x]
    return [r, x]

  def _calc_to_move(self, signal):
    self.padding_size = self._calc_padding(signal)
    self.to_move = self._draw_number_to_move()

  def _draw_number_to_move(self):
    to_move = r.randint(0, self.padding_size)
    return to_move

  def _calc_padding(self, signal):
    start, stop = self._get_first_last_nonzero_index(signal)
    values_len = len(range(start, stop))
    pad_len = len(signal) - values_len
    return pad_len

  def _get_first_last_nonzero_index(self, array):
    nonzero_indxs = np.nonzero(array)
    first = nonzero_indxs[0][0]
    last = nonzero_indxs[0][-1]
    return first, last

class StackSignals(object):
  def __call__(self, data):
    r_stacked = np.hstack(data[0])
    x_stacked = np.hstack(data[1])
    joined_rx = [r_stacked, x_stacked]
    return np.hstack(joined_rx)

class ReduceSamplingRate():
  def __init__(self, reduce_factor):
    self.reduce_factor = reduce_factor

  def __call__(self, data):
    signal = data[::self.reduce_factor]
    return signal

class Absolute(object):
  def __call__(self, stacked_signal):
    return np.absolute(stacked_signal)

class Scale(object):
  def __init__(self, scale):
    self.scale = scale
  def __call__(self, stacked_signal):
    return stacked_signal * self.scale

class AddOffset(object):
  def __init__(self, offset):
    self.offset = offset
  def __call__(self, stacked_signal):
    return [sample + self.offset for sample in stacked_signal]

class ToTensor(object):
  def __call__(self, signal):
    return torch.tensor(signal, dtype = torch.float32)
