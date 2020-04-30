import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

from unet import UNet

NUM_EPOCHS = 50

class BobRossSegmentedImagesDataset(Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.dataroot = dataroot
        self.imgs = list((self.dataroot / 'train' / 'images').rglob('*.png'))
        self.segs = list((self.dataroot / 'train' / 'labels').rglob('*.png'))
        self.transform = transforms.Compose([
            transforms.Resize((164, 164)),
            transforms.Pad(46, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                            mean=(0.459387, 0.46603974, 0.4336706),
                            std=(0.06098535, 0.05802868, 0.08737113)
            )
        ])
        self.color_key = {
            3 : 0,
            5: 1,
            10: 2,
            14: 3,
            17: 4,
            18: 5,
            22: 6,
            27: 7,
            61: 8
        }
        assert len(self.imgs) == len(self.segs)
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        def translate(x):
            return self.color_key[x]
        translate = np.vectorize(translate)
        
        img = Image.open(self.imgs[i])
        img = self.transform(img)
        
        seg = Image.open(self.segs[i])
        seg = seg.resize((256, 256), Image.NEAREST)
        
        seg = translate(np.array(seg)).astype('int64')
        
        # Additionally, the original UNet implementation outputs a segmentation map
        # for a subset of the overall image, not the image as a whole! With this input
        # size the segmentation map targeted is a (164, 164) center crop.
        seg = seg[46:210, 46:210]
        
        return img, seg

dataroot = Path('/mnt/segmented-bob-ross-images/')
dataset = BobRossSegmentedImagesDataset(dataroot)
dataloader = DataLoader(dataset, shuffle=True, batch_size=8)

model = UNet()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=32)

for epoch in range(NUM_EPOCHS):
    losses = []

    for i, (batch, segmap) in enumerate(dataloader):
        optimizer.zero_grad()
        
        batch = batch.cuda()
        segmap = segmap.cuda()

        output = model(batch)
        loss = criterion(output, segmap)
        loss.backward()
        optimizer.step()
        scheduler.step()

        curr_loss = loss.item()
        losses.append(curr_loss)

    print(f'Finished epoch {epoch}.')

torch.save(model.state_dict(), '50_net.pth')
