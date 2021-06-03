# Model code adpated from:
# https://github.com/spellml/mobilenet-cifar10/blob/master/servers/eval_quantized_t4.py
import math
# import os
import time
import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from tvm_funcs import get_tvm_model, tune, time_it


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_model():
    mobilenet = MobileNetV2(width_mult=1, n_class=10, input_size=32)
    mobilenet.load_state_dict(
        torch.load("/mnt/checkpoints/model_10.pth", map_location=torch.device('cpu'))
    )
    # mobilenet.load_state_dict(torch.load("/spell/notebooks/mobilenet/checkpoints/model_10.pth"))
    return mobilenet


def get_dataloader():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomPerspective(),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.CIFAR10("/mnt/cifar10/", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader


def train(model):
    print(f"Training the model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    start_time = time.time()

    # NUM_EPOCHS = 0
    # NUM_EPOCHS = 1
    NUM_EPOCHS = 10
    for epoch in range(1, NUM_EPOCHS + 1):
        losses = []

        for i, (X_batch, y_cls) in enumerate(dataloader):
            optimizer.zero_grad()

            y = y_cls
            X_batch = X_batch
            # y = y_cls.cuda()
            # X_batch = X_batch.cuda()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            if i % 200 == 0:
                print(
                    f'Finished epoch {epoch}/{NUM_EPOCHS}, batch {i}. Loss: {curr_loss:.3f}.'
                )

            losses.append(curr_loss)

        print(
            f'Finished epoch {epoch}. '
            f'avg loss: {np.mean(losses)}; median loss: {np.min(losses)}'
        )
    print(f"Training done in {str(time.time() - start_time)} seconds.")


if __name__ == "__main__":
    mobilenet = get_model()
    dataloader = get_dataloader()
    X_ex, y_ex = next(iter(dataloader))

    train(mobilenet)

    print(f"Converting the model (post-training)...")
    start_time = time.time()
    quantized_mobilenet = torch.quantization.convert(mobilenet)
    print(f"Quantization done in {str(time.time() - start_time)} seconds.")
    torch.save(quantized_mobilenet.state_dict(), "quantized_model.pth")

    print("PyTorch (unquantized) timings:")
    print(time_it(lambda: mobilenet(X_ex)))

    print("PyTorch (quantized) timings:")
    print(time_it(lambda: quantized_mobilenet(X_ex)))

    # tvm part
    mod, params, module = get_tvm_model(quantized_mobilenet, X_ex)
    tvm_optimized_module = tune(mod, params, X_ex)

    print("TVM (Relay) timings:")
    print(time_it(lambda: module.run()))
    print("TVM (Tuned) timings:")
    print(time_it(lambda: tvm_optimized_module.run()))
