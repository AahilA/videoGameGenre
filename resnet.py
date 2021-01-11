import torch
import torch.nn as nn

#Peter figure out padding math

class block(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(block,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chan)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chan)
        )

        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_chan != out_chan:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_chan)
            )

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_classes):
        super(ResNet, self).__init__()
        self.input_chan = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3,3) #wtf these numbers be

        #Resnet Layers
        self.layer1 = self._make_layer(block, 3, 64, 1)
        self.layer2 = self._make_layer(block, 4, 128, 2)
        self.layer3 = self._make_layer(block, 6, 256, 2)
        self.layer4 = self._make_layer(block, 3, 512, 2)

        self.avg_pool = nn.AvgPool2d(1,1)
        self.fc = nn.Linear(512, num_classes)


    def _make_layer(self, block, num_blocks, out_chan, stride):
        layers = []

        layers.append(block(self.input_chan, out_chan, stride))
        self.input_chan = out_chan

        for i in range(num_blocks - 1):
            layers.append(block(self.input_chan,out_chan,stride))

        return nn.Sequential(*layers)
    
    #TODO ADD POOLING
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)

        #Resnet
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x