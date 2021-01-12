import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform

import pandas as pd
import numpy as np

class GameImagesDataset(Dataset):
    """Game Images dataset."""

    def __init__(self, rootf, csvf, num_class):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gameLabels = pd.read_csv(csvf, header=None)
        self.rootf = rootf
        self.transform = transform
        self.num_class = num_class

    def __len__(self):
        return self.gameLabels.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.rootf + self.gameLabels.iloc[idx,0] + '.png'
        image = io.imread(img_name)
        label = self.gameLabels.iloc[idx, 1:]
        label = np.array([label])
        label = label.astype('float').reshape(-1, self.num_class)

        image = transform.resize(image, (256,256))

        sample = {'image': image, 'label': label}

        return sample