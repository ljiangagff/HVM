import torch.nn as nn
import torch
from .positionEncoding import PositionalEncoding


class Pre(nn.Module):
    '''
    This class performs and padding and positionEncoding
    '''

    def __init__(self, hdim, window=4) -> None:
        super().__init__()
        self.pos = PositionalEncoding(hdim)
        self.window = window

    def forward(self, x):
        # pad the values that do not fullfill the whole window
        # e.g. [64, 256, 8, 39] to [64, 256, 8, 40]
        w0 = x.shape[3]
        exceed = w0 % self.window
        if exceed:
            padding = (0, self.window-exceed, 0, 0)
            x = nn.functional.pad(x, padding, mode='constant')

        b, c, h, w = x.shape
        # [b, c, h, w] to [n, b, c]
        x = x.view(b, c, -1).permute(2, 0, 1)
        x = self.pos(x)
        # [n, b, c] to [b, c, h, w]
        return x.permute(1, 2, 0).view(b, c, h, w)


class Stem(nn.Module):
    """
    Use Stem module for downsampling the image and introduce locality.

    First past through the image with a 2x2 convolution to reduce the image size.
    Then past throught two 1x1 convolution for better local information.

    Input:
        - x: (B, 1, H, W)
    Output:
        - result: (B, Stem_channel, H / 8, W / 8)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1, 32
        out_channels1 = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels1),
            nn.GELU()
        )
        # 32, 64
        out_channels2 = out_channels1*2
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels1, out_channels2,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels2),
            nn.GELU()
        )
        # 64, 128
        out_channels3 = out_channels2*2
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels2, out_channels3,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels3),
            nn.GELU()
        )
        # 128, 256
        out_channels4 = out_channels3*2
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels3, out_channels4,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels4),
            nn.GELU()
        )
        self.init_weight()

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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        result = self.conv4(x)
        return result


class HV_attention(nn.Module):
    """
    The horizontal-vertical attention

    Inputs:
        Q: [N, C, H, W]
        K: [N, C, H / stride, W / stride]
        V: [N, C, H / stride, W / stride]
    Outputs:
        X: [N, C, H, W]
    """

    def __init__(self, channels, pool_stride, heads, alpha=0.7, window=4, full_attn=False):
        super(HV_attention, self).__init__()
        self.channels = channels
        self.full_attn = full_attn
        self.alpha = 1.0 if full_attn else alpha
        self.channels1 = int(self.channels*self.alpha)
        self.channels2 = self.channels - self.channels1
        self.window = window

        if not full_attn:
            self.pool_k = nn.AvgPool2d(pool_stride, stride=pool_stride)
            self.pool_v = nn.AvgPool2d(pool_stride, stride=pool_stride)

        if self.channels2:
            self.hfc_o = nn.Linear(channels, self.channels1)
            self.vfc_o = nn.Linear(channels, self.channels2)

        self.h_attn = nn.MultiheadAttention(self.channels, heads)
        self.v_attn = nn.MultiheadAttention(self.channels, heads)

    def vertical_attention(self, x):
        b, c, h, w = x.shape
        window = self.window
        output = torch.empty(h, 0, b, c).to(x.device)
        for i in range(0, w, window):
            # each window width is sep
            xx = x[:, :, :, i: i+window].clone()
            # [b, c, h, w] to [n, b, c]
            xx = xx.view(b, c, -1).permute(2, 0, 1)
            # [n, b, c] to [h, w, b, c]
            sub_output = self.v_attn(xx, xx, xx)[0].view(h, window, b, c)
            output = torch.cat([output, sub_output], axis=1)
        return output.view(h*w, b, c)

    def horizontal_attention(self, x):
        b, c, h, w = x.shape
        q = x
        if self.full_attn:
            k = x
            v = x
        else:
            k = self.pool_k(x)
            v = self.pool_v(x)
        # [b, c, h, w] to [n, b, c]
        q = q.view(b, c, -1).permute(2, 0, 1)
        k = k.view(b, c, -1).permute(2, 0, 1)
        v = v.view(b, c, -1).permute(2, 0, 1)
        output = self.h_attn(q, k, v)[0]
        # [n, b, c]
        return output

    def forward(self, x):
        # input: [b, c, h, w]
        b, c, h, w = x.shape
        if self.channels2:
            ha_x = self.horizontal_attention(x)
            ha = self.hfc_o(ha_x)
            va_x = self.vertical_attention(x)
            va = self.vfc_o(va_x)
            # result [n, b, c]
            attn = torch.cat([ha, va], axis=2)
        else:
            attn = self.horizontal_attention(x)
        # [b, h, w, c] tp [b, c, h, w]
        attn = attn.view(h, w, b, c).permute(2, 3, 0, 1)
        result = attn + x
        result = torch.nn.functional.layer_norm(result, (c, h, w))
        return result, ha_x.view(h, w, b, c), va_x.view(h, w, b, c), attn


class IRFFN(nn.Module):
    """
    Inverted Residual Feed-forward Network
    """

    def __init__(self, in_channels, R):
        super(IRFFN, self).__init__()
        exp_channels = int(in_channels * R)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, kernel_size=1),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.dwconv = nn.Sequential(
            nn.Conv2d(exp_channels, exp_channels, kernel_size=3,
                      stride=1, padding=1, groups=exp_channels, bias=True),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(exp_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        result = x + self.conv2(self.dwconv(self.conv1(x)))
        return result


class FFN(nn.Module):
    '''
    Using nn.Conv1d replace nn.Linear to implements FFN.
    '''

    def __init__(self, hdim, R=4.0, dropout=0.1):
        super(FFN, self).__init__()
        hdim0 = int(hdim*R)
        self.ff1 = nn.Conv1d(hdim, hdim0, kernel_size=1)
        self.ff2 = nn.Conv1d(hdim0, hdim, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [b, c, h, w]
        b, c, h, w = x.shape
        residual = x
        x = x.view(b, c, -1)  # [batch, d_model, seq_len]
        x = self.ff1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ff2(x)
        x = x.view(b, c, h, w)  # [b, c, n]

        return nn.functional.layer_norm(residual + x, (c, h, w))


class CVT_layer(nn.Module):
    def __init__(self, pool_stride, num_heads, R=4.0, in_channels=32, alpha=0.75, full_attn=False, window=4):
        super(CVT_layer, self).__init__()

        # HV MHSA
        self.lmhsa = HV_attention(in_channels, pool_stride,
                                  num_heads, alpha=alpha, full_attn=full_attn, window=window)
        # Normal FFN
        self.ffn = FFN(in_channels, R)
        # Convolutional FFN
        # self.ffn = IRFFN(in_channels, R)

    def forward(self, x):
        x, ha_x, va_x, attn = self.lmhsa(x)
        x = self.ffn(x)
        self.cvt_attn = {
            "ha": ha_x,
            "va": va_x,
            "attn": attn
        }
        return x


class CVT(nn.Module):
    def __init__(self,
                 in_channels=3,
                 stem_channel=32,
                 transformer_channel=256,
                 block_layer=6,
                 R=4.0,
                 heads=8,
                 alpha=0.75,
                 full_attn=False,
                 window=4
                 ):
        super(CVT, self).__init__()

        # Stem output channel = stem_channel*8
        self.stem = Stem(in_channels, stem_channel)
        # Padding on the input, with window size(default 4)
        self.pre = Pre(transformer_channel, window=window)
        stage = []
        for _ in range(block_layer):
            cvt_layer = CVT_layer(
                pool_stride=(4, 1),
                num_heads=heads,
                R=R,
                in_channels=transformer_channel,
                alpha=alpha,
                full_attn=full_attn,
                window=window,
            )
            stage.append(cvt_layer)
        self.stage = nn.Sequential(*stage)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(transformer_channel, transformer_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pre(x)
        x = self.stage(x)

        y = self.avg_pool(x)
        y = torch.flatten(y, 1)
        y = self.fc(y)

        b, c, _, _ = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # b,n',c
        return x, y


def CVT_Small(params):
    alpha = params['alpha']
    full_attn = params['full_attn']
    model = CVT(
        in_channels=1,
        stem_channel=24,
        transformer_channel=192,
        block_layer=6,
        heads=4,
        R=3.6,
        window=4,
        alpha=alpha,
        full_attn=full_attn,
    )
    return model


def CVT_Base(params):
    alpha = params['alpha']
    full_attn = params['full_attn']
    model = CVT(
        in_channels=1,
        stem_channel=32,
        transformer_channel=256,
        block_layer=6,
        heads=8,
        R=4.0,
        window=4,
        alpha=alpha,
        full_attn=full_attn,
    )
    print(f'alpha {alpha}, full_attn {full_attn}')
    return model


def CVT_Large(params):
    alpha = params['alpha']
    full_attn = params['full_attn']
    model = CVT(
        in_channels=1,
        stem_channel=40,
        transformer_channel=320,
        block_layer=8,
        heads=8,
        R=4.0,
        window=8,
        alpha=alpha,
        full_attn=full_attn,
    )
    return model
