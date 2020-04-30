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

# Instead of always starting the zeroeth epoch, check if the user passed a checkpoint.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--from-checkpoint', type=str, dest='checkpoint', default='')
args = parser.parse_args()
if args.checkpoint:
    first_remaining_epoch = int(args.checkpoint.split('_')[0]) + 1
    EPOCHS = range(first_remaining_epoch, NUM_EPOCHS)
else:
    EPOCHS = range(NUM_EPOCHS)

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

# Instead of always initializing an empty model, initialize from the checkpoints
# file if one is available.
model = UNet()
model.cuda()
if args.checkpoint:
    model.load_state_dict(torch.load(f'/mnt/checkpoints/{args.checkpoint}'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=32)

for epoch in EPOCHS:
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

    # Save the model checkpoints file every 5 epochs
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'{epoch}_net.pth')
        print(f'Saved model to {epoch}_net.pth.')
