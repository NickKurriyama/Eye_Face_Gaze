import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torchvision import transforms
import pathlib

import numpy as np
import tqdm

from gaze_estimation import (GazeEstimationMethod, create_dataloader,
                             create_model)
from gaze_estimation.utils import compute_angle_error, load_config, save_config


class SpikeBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.downsample = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)

    def forward(self, x, mem1, mem2):
        out, mem1 = self.lif1(self.conv1(x), mem1)
        out, mem2 = self.lif2(self.conv2(out), mem2)
        out += self.downsample(x)
        return out, mem1, mem2


class ResNetSNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_in = nn.Conv2d(20, 64, 3, stride=1, padding=1)
       # self.conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

        self.layer1 = SpikeBlock(64, 64)
        self.layer2 = SpikeBlock(64, 128, stride=2)
        self.layer3 = SpikeBlock(128, 256, stride=2)

        self.fc = nn.Linear(256, num_classes)
        self.lif_fc = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps=25):
        num_steps = int(num_steps)  # fix lỗi không phải int

        mem1 = self.lif1.init_leaky()
        mems = [snn.Leaky(beta=0.9).init_leaky() for _ in range(6)]
        mem_fc = self.lif_fc.init_leaky()

        out_accum = 0
        for t in range(num_steps):
            x_t = torch.bernoulli(x)  # RATE ENCODING
            out = self.conv_in(x_t)
           # out = self.conv1(x_t)
            out, mem1 = self.lif1(out, mem1)

            out, mems[0], mems[1] = self.layer1(out, mems[0], mems[1])
            out, mems[2], mems[3] = self.layer2(out, mems[2], mems[3])
            out, mems[4], mems[5] = self.layer3(out, mems[4], mems[5])

            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            out, mem_fc = self.lif_fc(out, mem_fc)

            out_accum += out

        return out_accum / num_steps


def copy_weights_and_normalize(ann_model, snn_model):
    for ann_layer, snn_layer in zip(ann_model.modules(), snn_model.modules()):
        if isinstance(ann_layer, nn.Conv2d) and isinstance(snn_layer, nn.Conv2d):
            w = ann_layer.weight.data
            wmax = w.abs().max()
            snn_layer.weight.data = w / wmax
            if ann_layer.bias is not None:
                snn_layer.bias.data = ann_layer.bias.data / wmax

        elif isinstance(ann_layer, nn.Linear) and isinstance(snn_layer, nn.Linear):
            w = ann_layer.weight.data
            wmax = w.abs().max()
            snn_layer.weight.data = w / wmax
            if ann_layer.bias is not None:
                snn_layer.bias.data = ann_layer.bias.data / wmax

def load_h5_input(file_path):
    with h5py.File(file_path, 'r') as f:
        img = f['image'][:]
    img = torch.tensor(img, dtype=torch.float32)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    if img.max() > 1.0:
        img = img / 255.0
    return img

def test(model, test_loader, config):
    model.eval()
    device = torch.device(config.device)
    model.to(device)

    predictions = []
    gts = []

    with torch.no_grad():
        for images, poses, gazes in tqdm.tqdm(test_loader):
            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)
            outputs = model(images)

            predictions.append(outputs.cpu())
            gts.append(gazes.cpu())

    predictions = torch.cat(predictions)
    gts = torch.cat(gts)
    angle_error = float(compute_angle_error(predictions, gts).mean())
    return predictions, gts, angle_error


def main():
    config = load_config()
    test_loader = create_dataloader(config, is_train=False)

    checkpoint_name = pathlib.Path(config.test.checkpoint).stem
    ann_model = create_model(config)
    checkpoint = torch.load(config.test.checkpoint, map_location='cpu')
    ann_model.load_state_dict(checkpoint['model'])

    snn_model = ResNetSNN(num_classes=10)
    copy_weights_and_normalize(ann_model, snn_model)

    predictions, gts, angle_error = test(snn_model, test_loader, config)
    print(f'The mean angle error (deg): {angle_error:.2f}')


if __name__ == '__main__':
    main()
