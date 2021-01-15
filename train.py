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
        # print(img_name)
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
print(train_size, test_size)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def Average(lst): 
    return sum(lst) / len(lst) 

def accuracy(outputs, labels):
    outputs = outputs.flatten()
    labels = labels.flatten()
    cor = True
    for i in range(len(outputs)):
        if outputs[i] >= 0.5 and labels[i] == 0:
            cor = False
            return cor
        if outputs[i] < 0.5 and labels[i] == 1:
            cor = False
            return cor
    return cor
           

def train(train_loader, model, criterion, optimizer):
    print("training...")
    model.train()
    loss_list = []
    total = 0
    correct = 0
#    softmax = nn.LogSoftmax(dim=1)

    for k, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        print("loading image: " + str(k * 128))
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images.float())

#        outputs = softmax(outputs)
#        print("outputs:")
#        print(outputs)
#        print("labels:")
#        print(labels)
        loss = criterion(outputs, labels)
#        print(str(k) + ":" + str(loss.item()))
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        b = outputs.clone()
        for i in range(len(b)):
            for j in range(len(b[0])):
                if b[i][j] >= 0.5:
                    b[i][j] = 1
                else:
                    b[i][j] = 0
        total += int(labels.size(1)*labels.size(0))
#        print("predicted:")
#        print(predicted)
        correct += int(b.eq(labels).sum().item())
        del b
    print(correct, total)
    acc = 100. * correct / total
    return Average(loss_list),acc

def test(test_loader, model, criterion):
    print("testing...")
    with torch.no_grad():
        model.eval()
        loss_list = []
        total = 0
        correct = 0
#        softmax = nn.LogSoftmax(dim=1)

        for _, (images, labels) in enumerate(test_loader):
            # Run the forward pass
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images.float())
#            outputs = softmax(outputs)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            b = outputs.clone()
            for i in range(len(b)):
                for j in range(len(b[0])):
                    if b[i][j] >= 0.5:
                        b[i][j] = 1
                    else:
                        b[i][j] = 0
            total += int(labels.size(1)*labels.size(0))
            correct += int(b.eq(labels).sum().item())
            del b
    acc = 100. * correct / total

    return Average(loss_list),acc

def main(train_loader, test_loader, model, optimizer):
    
    list_train_loss = []
    list_train_acc = []
    list_test_acc = []
    print("start training...")
    trloss,tracc = train(train_loader, model, criterion, optimizer)
    _,testacc = test(test_loader, model, criterion)
    
    list_train_loss.append(trloss)
    list_train_acc.append(tracc)
    list_test_acc.append(testacc)
    
    print(f'Epoch 0: Train Loss {trloss}, Train Accuracy {tracc}, Test Accuracy {testacc}')

    for epoch in range(100):
        trloss,tracc = train(train_loader, model, criterion, optimizer)
        _,testacc = test(test_loader, model, criterion)

        list_train_loss.append(trloss)
        list_train_acc.append(tracc)
        list_test_acc.append(testacc)
        
        print(f'Epoch {epoch + 1}: Train Loss {trloss}, Train Accuracy {tracc}, Test Accuracy {testacc}')
        
    return list_train_loss,list_train_acc,list_test_acc

def save_accuracy(train_losses, train_accuracies, test_accuracies):
    outfile = open("accuracies.csv", "w")
    outfile.write("train_losses, train_accuracies, test_accuracies\n")
    for i in range(len(train_losses)):
        outfile.write(str(train_losses[i]))
        outfile.write(",")
        outfile.write(str(train_accuracies[i]))
        outfile.write(",")
        outfile.write(str(test_accuracies[i]))
        outfile.write("\n")
    outfile.close()


print("building model...")
model = resnet.ResNet(resnet.block, 65).float()
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
criterion = nn.L1Loss()

train_losses,train_accuracies,test_accuracies = main(train_loader, test_loader, model, optimizer)
save_accuracy(train_losses, train_accuracies, test_accuracies)

torch.save(model.state_dict(),'gameModel')

if False:
    model = resnet.ResNet(resnet.block, 65).float()
    model.load_state_dict(torch.load('gameModel'))
