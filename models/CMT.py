import torch
import torch
import torch.nn as nn


class CMT(nn.Module):
    def __init__(self,
                 in_channels=1,
                 stem_channel=32,
                 pvt_channel=[32, 64, 128, 256],
                 block_layer=[2, 2, 4, 2],
                 R=4.0,
                 heads=8,
                 alpha=[0.7, 0.7, 0.7, 1.0]
                 ):
        super(CMT, self).__init__()

        # Stem layer, size = [img_size // 2]
        self.stem = CMT_Stem(in_channels, stem_channel)
        # Patch Aggregation Layer
        # Image size for each stage, size = [img_size // 4, img_size // 8, img_size // 16, img_size // 32]
        self.patch1 = PatchEmbed(stem_channel, pvt_channel[0])
        self.patch2 = PatchEmbed(pvt_channel[0], pvt_channel[1])
        self.patch3 = PatchEmbed(pvt_channel[1], pvt_channel[2])
        self.patch4 = PatchEmbed(pvt_channel[2], pvt_channel[3])

        # Block Layer
        stage1 = []
        for _ in range(block_layer[0]):
            pvt_layer = CMTBlock(
                pool_stride=(8, 2),
                vertical_heads=16,
                in_channels=pvt_channel[0], 
                d_k=pvt_channel[0],
                num_heads=1,
                R=R,
                alpha=alpha[0]
            )
            stage1.append(pvt_layer)
        self.stage1 = nn.Sequential(*stage1)

        stage2 = []
        for _ in range(block_layer[1]):
            pvt_layer = CMTBlock(
                pool_stride=(8, 2),
                vertical_heads=8,
                in_channels=pvt_channel[1],
                d_k=pvt_channel[1] // 2,
                num_heads=2,
                R=R,
                alpha=alpha[1]
            )
            stage2.append(pvt_layer)
        self.stage2 = nn.Sequential(*stage2)

        stage3 = []
        for _ in range(block_layer[2]):
            pvt_layer = CMTBlock(
                pool_stride=(4, 1),
                vertical_heads=4,
                in_channels=pvt_channel[2],
                d_k=pvt_channel[2] // 4,
                num_heads=4,
                R=R,
                alpha=alpha[2]
            )
            stage3.append(pvt_layer)
        self.stage3 = nn.Sequential(*stage3)

        stage4 = []
        for _ in range(block_layer[3]):
            pvt_layer = CMTBlock(
                pool_stride=(2, 1),
                vertical_heads=2,
                in_channels=pvt_channel[3],
                d_k=pvt_channel[3] // 8,
                num_heads=heads,
                R=R,
                alpha=alpha[3],
            )
            stage4.append(pvt_layer)
        self.stage4 = nn.Sequential(*stage4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(pvt_channel[3], pvt_channel[3]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.patch1(x)
        x = self.stage1(x)

        x = self.patch2(x)
        x = self.stage2(x)

        x = self.patch3(x)
        x = self.stage3(x)

        x = self.patch4(x)
        x = self.stage4(x)

        y = self.avg_pool(x)
        y = torch.flatten(y, 1)
        y = self.fc(y)

        b, c, _, _ = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # b,n',c
        return x, y


def init_CMT():
    model = CMT(
        in_channels=1,
        stem_channel=32,
        pvt_channel=[32, 64, 128, 256],
        block_layer=[1, 1, 6, 2],
        R=4.0,
    )
    return model


class Conv2x2(nn.Module):
    """
    2x2 Convolution
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv2x2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2,
                              stride=stride, padding=0, bias=True
                              )

    def forward(self, x):
        result = self.conv(x)
        return result


class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(DWCONV, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=True
                                   )

    def forward(self, x):
        result = self.depthwise(x)
        return result


class LPU(nn.Module):
    """
    Local Perception Unit to extract local infomation.
    LPU(X) = DWConv(X) + X
    """

    def __init__(self, in_channels, out_channels):
        super(LPU, self).__init__()
        self.DWConv = DWCONV(in_channels, out_channels)

    def forward(self, x):
        result = self.DWConv(x) + x
        return result


class LMHSA(nn.Module):
    """
    Lightweight Multi-head-self-attention module.

    Inputs:
        Q: [N, C, H, W]
        K: [N, C, H / stride, W / stride]
        V: [N, C, H / stride, W / stride]
    Outputs:
        X: [N, C, H, W]
    """

    def __init__(self, channels, d_k, pool_stride, vertical_heads, heads, dropout=0.1, alpha=0.75, window=4, full_attn=False):
        super(LMHSA, self).__init__()
        self.heads = heads
        self.channels = channels
        self.channels1 = int(self.channels*alpha)
        self.channels2 = self.channels - self.channels1
        self.vertical_heads = vertical_heads
        self.full_attn=full_attn
        self.window=window

        if not full_attn:
            self.pool_k = nn.AvgPool2d(pool_stride, stride=pool_stride)
            self.pool_v = nn.AvgPool2d(pool_stride, stride=pool_stride)

        self.d_k = d_k
        self.hfc_q = nn.Linear(self.channels, heads * self.d_k)
        self.hfc_k = nn.Linear(self.channels, heads * self.d_k)
        self.hfc_v = nn.Linear(self.channels, heads * self.d_k)
        self.hfc_o = nn.Linear(heads * self.d_k, self.channels1)

        if self.channels2:
            self.vfc_q = nn.Linear(self.channels, heads * self.d_k)
            self.vfc_k = nn.Linear(self.channels, heads * self.d_k)
            self.vfc_v = nn.Linear(self.channels, heads * self.d_k)
            self.vfc_o = nn.Linear(heads * self.d_k, self.channels2)

        self.dropout = nn.Dropout(p=dropout)

    def pre(self, x):
        # x: [b, c, h, w]
        exceed = x.shape[3] % self.window
        if exceed:
            padding = (0, self.window-exceed, 0, 0)
            x = nn.functional.pad(x, padding, mode='constant')
        return x

    def vertical_attention(self, x):
        b, c, h, w = x.shape
        # output: [b, h, w, c]
        output = torch.empty(b, h, 0, self.channels2).to(x.device)
        for i in range(0, w, self.window):
            # each window width is sep, [b, c, h, window]
            xx = x[:, :, :, i: i+self.window].clone()
            # [b, h, window, c]
            sub_output = self.self_attention(xx, xx, xx, v_attn=True)
            output = torch.cat([output, sub_output], axis=2)
        return output

    def horizontal_attention(self, x):
        q = x
        if self.full_attn:
            k = x
            v = x
        else:
            k = self.pool_k(x)
            v = self.pool_v(x)

        output = self.self_attention(q, k, v, v_attn=False)
        return output

    def self_attention(self, q, k, v, v_attn=False):
        b, c, h, w = q.shape
        d_k = self.d_k

        if not v_attn:
            fc_q = self.hfc_q
            fc_k = self.hfc_k
            fc_v = self.hfc_v
            fc_o = self.hfc_o
        else:
            fc_q = self.vfc_q
            fc_k = self.vfc_k
            fc_v = self.vfc_v
            fc_o = self.vfc_o

        scaled_factor = d_k ** -0.5

        q_reshape = q.view(b, c, h * w).permute(0, 2, 1)
        q_reshape = torch.nn.functional.layer_norm(q_reshape, (b, h * w, c))
        q = fc_q(q_reshape)
        q = q.view(b, h * w, self.heads, d_k).permute(0, 2, 1, 3)

        k_b, k_c, k_h, k_w = k.shape
        k = k.view(k_b, k_c, k_h * k_w).permute(0, 2, 1)
        k = fc_k(k)
        k = k.view(k_b, k_h * k_w, self.heads, d_k).permute(0, 2, 1, 3)

        v_b, v_c, v_h, v_w = v.shape
        v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1)
        v = fc_v(v)
        v = v.view(v_b, v_h * v_w, self.heads, d_k).permute(0, 2, 1, 3)

        # Attention
        attn = torch.einsum('... i d, ... j d -> ... i j',
                            q, k) * scaled_factor
        attn = torch.softmax(attn, dim=-1)  # [b, heads, h * w, k_h * k_w]
        attn = self.dropout(attn)
        # Attn mul v
        output = torch.matmul(attn, v).permute(0, 2, 1, 3)
        output = output.flatten(2).view(b, h, w, self.heads*d_k)
        output = fc_o(output)
        '''
        the output is Attn*V with [b, h, w, c]
        '''
        return output

    def forward(self, x):
        '''
        The input is [b, c, h, w], 
        the attention out is [b, h, w, c],
        and the output should be [b, c, h, w]
        '''
        x = self.pre(x)
        _, c, h, w = x.shape
        ha = self.horizontal_attention(x)
        if self.channels2:
            # [b, h, w, c]
            va = self.vertical_attention(x)
            result = torch.cat([ha, va], axis=3)
        else:
            result = ha
        # [b, h, w, c] tp [b, c, h, w]
        result = result.permute(0, 3, 1, 2) + x
        result = torch.nn.functional.layer_norm(result, (c, h, w))
        return result


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
            DWCONV(exp_channels, exp_channels),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(exp_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.dwconv(y)
        y = self.conv2(y)
        return x + y


class FFN(nn.Module):
    '''
    Using nn.Conv1d replace nn.Linear to implements FFN.
    '''

    def __init__(self, hdim, R=4.0, dropout=0.1):
        super(FFN, self).__init__()
        self.ff1 = nn.Conv1d(hdim, hdim*R, 1)
        self.ff2 = nn.Conv1d(hdim*R, hdim, 1)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(hdim)

    def forward(self, x):
        # x: [b, c, h, w]
        b, c, h, w = x.shape
        residual = x
        x = x.flatten(2)  # [batch, d_model, seq_len]
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        x = x.contiguous().view(b, c, h, w)  # [b, c, n]

        return self.layer_norm(residual + x)


class PatchEmbed(nn.Module):
    """
    Aggregate the patches into a single image.
    To produce the hierachical representation.

    Applied before each stage to reduce the size of intermediate features
    (2x downsampling of resolution), and project it to a larger dimension
    (2x enlargement of dimension).

    Input:
        - x: (B, In_C, H, W)
    Output:
        - x: (B, Out_C, H / 2, W / 2)
    """

    def __init__(self, in_channels, out_channels=None):
        super(PatchEmbed, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = Conv2x2(in_channels, out_channels, stride=2)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        _, c, h, w = x.shape
        result = torch.nn.functional.layer_norm(x, (c, h, w))
        return result


class CMT_Stem(nn.Module):
    """
    Use Stem module to process input image and overcome the limitation of the
    non-overlapping patches.

    First past through the image with a 2x2 convolution to reduce the image size.
    Then past throught two 1x1 convolution for better local information.

    Input:
        - x: (B, 3, H, W)
    Output:
        - result: (B, 32, H / 2, W / 2)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
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
        result = self.conv3(x)
        return result


class CMTBlock(nn.Module):
    def __init__(self, pool_stride, vertical_heads, d_k, num_heads, R=4.0, in_channels=32, alpha=0.75, full_attn=False):
        super(CMTBlock, self).__init__()

        # Local Perception Unit
        # self.lpu = LPU(in_channels, in_channels)
        # Lightweight MHSA
        self.lmhsa = LMHSA(in_channels, d_k, pool_stride,
                           vertical_heads, num_heads, dropout=0.1, alpha=alpha, full_attn=full_attn)
        self.irffn = IRFFN(in_channels, R)

    def forward(self, x):
        # x = self.lpu(x)
        x = self.lmhsa(x)
        x = self.irffn(x)
        return x
