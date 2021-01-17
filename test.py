import torch
import pandas as pd
from skimage import color, io, transform
import numpy as np

import resnet

gameLabels = pd.read_csv("gameLabels.csv", header=None)
thres = 0.5
tot = 150
gen = 65
tn = [0 for i in range(gen)]
tp = [0 for i in range(gen)]
fn = [0 for i in range(gen)]
fp = [0 for i in range(gen)]

totnum = 0

for img_no in range(tot):
    print("image: " + str(img_no))

    img_name = 'game_images/' + gameLabels.iloc[img_no,0] + '.jpg'
    image = io.imread(img_name)
    label = gameLabels.iloc[img_no, 2:]
    name = gameLabels.iloc[img_no, 1]
    label = np.array([label])
    label = label.astype('float').reshape(-1, gen).flatten()
    image = transform.resize(image, (256,256))
    
    image = image[..., None]
    print(np.array(image).shape)
    if len(np.array(image).shape) != 4:
        continue
    else:
        totnum += 1
    image = torch.from_numpy(image.transpose((3, 2, 0, 1)))
    
    
    model = resnet.ResNet(resnet.block, gen).float()
    model.load_state_dict(torch.load('gameModel'))
    
    output = model(image.float())
    b = output.clone()
    

    for i in range(0,gen):
        if output[0][i] >= thres:
            output[0][i] = 1
        else:
            output[0][i] = 0
        if output[0][i] == 1 and label[i] == 1:
            tp[i] += 1
        if output[0][i] == 1 and label[i] == 0:
            fp[i] += 1
        if output[0][i] == 0 and label[i] == 1:
            fn[i] += 1
        if output[0][i] == 0 and label[i] == 0:
            tn[i] += 1

#    print(name)
#    print(label)
#    print(output)
#    print(b)


tp = [tp[i] * 1.0 / totnum for i in range(gen)]
tn = [tn[i] * 1.0 / totnum for i in range(gen)]
fp = [fp[i] * 1.0 / totnum for i in range(gen)]
fn = [fn[i] * 1.0 / totnum for i in range(gen)]
tpr = [((tp[i] / (tp[i] + fn[i])) if ((tp[i] + fn[i]) != 0.) else 0.) for i in range(gen)]
fpr = [((fp[i] / (tn[i] + fp[i])) if ((tn[i] + fp[i]) != 0.) else 0.) for i in range(gen)]

print(tp)
print(tn)
print(fp)
print(fn)
print(tpr)
print(fpr)

