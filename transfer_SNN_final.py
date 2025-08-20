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
from torch.cuda.amp import autocast
from gaze_estimation import (GazeEstimationMethod, create_dataloader,
                             create_model)
from gaze_estimation.utils import compute_angle_error, load_config, save_config

class ResNetSNN(nn.Module):
    def __init__(self, num_classes=2, skip_scale=0.7):
        super().__init__()

        self.numclasses = num_classes
        spike_grad = surrogate.fast_sigmoid(slope=10)

        self.conv1 = nn.Conv2d(1, 20, 3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(20, 50, 3, stride=2, padding=1)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.fc1 = nn.Linear(3600, 500)
        self.lif_fc1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.fc2 = nn.Linear(500, num_classes)
        self.lif_fc2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        # self.layer1 = SpikeBlock(50, 64, skip_scale=skip_scale)
        # self.layer2 = SpikeBlock(64, 128, stride=2, skip_scale=skip_scale)
        # self.layer3 = SpikeBlock(128, 256, stride=2, skip_scale=skip_scale)

        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(256, num_classes, bias=True)
        # self.lif_fc = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=10))

    def forward(self, x, num_steps=25, direct_input=False, return_dynamics=False):

        if num_steps is None:
            num_steps = 25

        num_steps = int(num_steps)
        mem1 = self.lif1.init_leaky().to(x.device)
        mem2 = self.lif2.init_leaky().to(x.device)
        mem_fc1 = self.lif_fc1.init_leaky().to(x.device)
        mem_fc2 = self.lif_fc2.init_leaky().to(x.device)

        out_accum = 0
        spk_record = []

        for t in range(num_steps):
            if direct_input:
                x_t = x 
            else:
                x_t = (torch.rand_like(x) < x).float()

            out = self.conv1(x_t)
            out1, mem1 = self.lif1(out, mem1)
            out1_pool = F.max_pool2d(out1, kernel_size=2, stride=2)

            out2 = self.conv2(out1_pool)
            out2, mem2 = self.lif2(out2, mem2)
            out2_pool = F.max_pool2d(out2, kernel_size=2, stride=2)

            flat = torch.flatten(out2_pool, start_dim=1)
            out3 = self.fc1(flat)
            out3, mem_fc1 = self.lif_fc1(out3, mem_fc1)

            out4 = self.fc2(out3)
            out4, mem_fc2 = self.lif_fc2(out4, mem_fc2)

            out_accum += out4
            spk_record.append(out4)

        if return_dynamics:
            return torch.stack(spk_record, dim=1)

        return out_accum / num_steps

def fold_bn_into_conv(conv, bn):
    w = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros(w.size(0), device=w.device)
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    w_fold = w * (gamma / torch.sqrt(var + eps)).reshape([-1, 1, 1, 1])
    b_fold = (b - mean) / torch.sqrt(var + eps) * gamma + beta
    return w_fold, b_fold



def copy_weights_and_normalize(ann_model, snn_model):
    for (ann_name, ann_layer), (snn_name, snn_layer) in zip(ann_model.named_modules(), snn_model.named_modules()):
        if isinstance(ann_layer, nn.Conv2d) and isinstance(snn_layer, nn.Conv2d):
            w = ann_layer.weight.data.clone()
            b = ann_layer.bias.data.clone() if ann_layer.bias is not None else torch.zeros(w.size(0))

            max_weight = w.abs().max()
            max_bias = b.abs().max()
            if max_weight > 0:
                w = w / max_weight
            if max_bias > 0:
                b = b / max_bias
            
            snn_layer.weight.data = w
            snn_layer.bias.data = b

        elif isinstance(ann_layer, nn.Linear) and isinstance(snn_layer, nn.Linear):
            w = ann_layer.weight.data.clone()
            b = ann_layer.bias.data.clone() if ann_layer.bias is not None else torch.zeros(w.size(0))

            max_weight = w.abs().max()
            max_bias = b.abs().max()
            if max_weight > 0:
                w = w / max_weight
            if max_bias > 0:
                b = b / max_bias
            
            snn_layer.weight.data = w
            snn_layer.bias.data = b

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
            with autocast():
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
