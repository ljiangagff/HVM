import torch
import torch.nn as nn
from .positionEncoding import PositionalEncoding
import torchvision.models as models
from torch import einsum, nn
import torch.nn.functional as F


class StemTransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, nhead=8, in_channels=1, out_channels=256):
        super().__init__()
        self.inc = in_channels
        self.hdim = out_channels

        # the initial image is a greyscale image
        self.pos = PositionalEncoding(self.hdim)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.gelu3 = nn.GELU()

        self.conv4 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.gelu4 = nn.GELU()
        self.init_weight()

        # transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hdim, nhead=nhead, dim_feedforward=4*self.hdim)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.hdim, self.hdim),
            nn.ReLU(inplace=True),
        )

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Extract the conv features
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.gelu4(x)

        b, c, h, w = x.shape
        # flatten the feature map to bsz, h*w, hdim
        x = x.flatten(2).transpose(1, 2)
        x = self.pos(x.permute(1, 0, 2))  # input_len, bsz, hdim
        encoded = self.transformer_encoder(x).permute(
            1, 0, 2)  # input_len, bsz, hdim

        y = encoded.contiguous().view(b, c, h, w)
        y = self.avg_pool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)

        return encoded, y


class VisionTransformer(nn.Module):
    def __init__(self, num_layers=6, nhead=8, in_channels=1, out_channels=256):
        super(VisionTransformer, self).__init__()
        self.inc = in_channels
        self.hdim = out_channels
        # the initial image is a greyscale image
        self.pos = PositionalEncoding(self.hdim)
        self.patchEmbed = nn.Conv2d(
            in_channels, out_channels, kernel_size=16, stride=16)        # transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hdim, nhead=nhead, dim_feedforward=4*self.hdim)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.hdim, self.hdim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Extract the conv features
        x = self.patchEmbed(x)
        b, c, h, w = x.shape
        # flatten the feature map to bsz, h*w, hdim
        x = x.flatten(2).transpose(1, 2)
        x = self.pos(x.permute(1, 0, 2))  # input_len, bsz, hdim
        encoded = self.transformer_encoder(x)  # input_len, bsz, hdim

        y = encoded.contiguous().view(b, c, h, w)
        y = self.avg_pool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)

        return encoded.permute(1, 0, 2), y


class DenseNet(nn.Module):
    def __init__(self, hdim=256) -> None:
        super().__init__()
        densenet = models.densenet121()
        self.hdim = hdim
        self.cf_layer = densenet.features
        self.ffn = nn.Linear(1024, hdim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.hdim, self.hdim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        x = self.cf_layer(x).permute(0, 2, 3, 1)
        x = self.ffn(x).permute(0, 3, 1, 2)
        b, c, h, w = x.shape

        y = self.avg_pool(x)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        # bsz, h*w, hdim = [b, n, c]
        result = x.contiguous().view(b, c, h*w).permute(0, 2, 1)
        return result, y



