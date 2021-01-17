import torch
import pandas as pd
from skimage import color, io, transform
import numpy as np

import resnet

gameLabels = pd.read_csv("gameLabels.csv", header=None)

img_no = 12

img_name = 'game_images/' + gameLabels.iloc[img_no,0] + '.jpg'
image = io.imread(img_name)
label = gameLabels.iloc[img_no, 2:]
name = gameLabels.iloc[img_no, 1]
label = np.array([label])
label = label.astype('float').reshape(-1, 65).flatten()

image = transform.resize(image, (256,256))

image = image[..., None]

image = torch.from_numpy(image.transpose((3, 2, 0, 1)))


model = resnet.ResNet(resnet.block, 65).float()
model.load_state_dict(torch.load('gameModel'))

output = model(image.float())
b = output.clone()

for i in range(0,65):
    if output[0][i] >= 0.5:
        output[0][i] = 1
    else:
        output[0][i] = 0

print(name)
print(label)
print(output)
print(b)



