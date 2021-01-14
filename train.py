import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage import color, io, transform

import resnet

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

        img_name = self.rootf + self.gameLabels.iloc[idx,0] + '.jpg'
        print(img_name)
        image = io.imread(img_name)
        label = self.gameLabels.iloc[idx, 2:]
        label = np.array([label])
        label = label.astype('float').reshape(-1, self.num_class).flatten()

        image = transform.resize(image, (256,256))

        if len(image.shape) < 3:
            image = color.gray2rgb(image)

        image = torch.from_numpy(image.transpose((2, 0, 1)))

        sample = (image,label)

        return sample

game_dataset = GameImagesDataset('game_images/', 'gameLabels.csv', 65)

"""
#####
Testing Dataset
#####
for i in range(len(game_dataset)):
    sample = game_dataset[i]

    print(i, sample['image'].shape, sample['label'].shape)

    if i == 3:
        break
"""

train_size = int(0.8 * len(game_dataset))
test_size = len(game_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(game_dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

def Average(lst): 
    return sum(lst) / len(lst) 

def train(train_loader, model, criterion, optimizer):
    model.train()
    loss_list = []

    for _, (images, labels) in enumerate(train_loader):
        # Run the forward pass

        outputs = model(images.float())

        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return Average(loss_list)

def test(test_loader, model, criterion):
    with torch.no_grad():
        model.eval()
        loss_list = []
        acc_list = []

        for _, (images, labels) in enumerate(test_loader):
            # Run the forward pass
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
    
    return Average(loss_list)

def main(train_loader, test_loader, model, optimizer):
    
    list_train_loss = []

    trloss = train(train_loader, model, criterion, optimizer)
    _ = test(test_loader, model, criterion)
    
    list_train_loss.append(trloss)
    
    print(f'Epoch 0: Train Loss {trloss}')

    for epoch in range(100):
        trloss = train(train_loader, model, criterion, optimizer)
        _ = test(test_loader, model, criterion)

        list_train_loss.append(trloss)
        
        print(f'Epoch {epoch + 1}: Train Loss {trloss}')
        
    return list_train_loss

model = resnet.ResNet(resnet.block, 65).float()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
criterion = nn.L1Loss()

train_losses = main(train_loader, test_loader, model, optimizer)

torch.save(model.state_dict(),'gameModel')

if False:
    model = resnet.ResNet(resnet.block, 65).float()
    model.load_state_dict(torch.load('gameModel'))
