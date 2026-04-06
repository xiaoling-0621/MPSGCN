import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels,reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x*self.channel_attention(x)
        x = x*self.spatial_attention(x)

        return x

class ADCB(nn.Module):
    def __init__(self, channels, scale=2.0):
        super(ADCB, self).__init__()
        self.cbam_H = CBAM(channels)
        self.cbam_F = CBAM(channels)
        self.scale = scale

    def forward(self, Ft, Ht_1):
        AttH = self.cbam_H(Ht_1)
        Ft_hat = self.scale * AttH * Ft

        AttF = self.cbam_F(Ft_hat)
        Ht_1_hat = self.scale * AttF * Ht_1

        return Ft_hat, Ht_1_hat

class ACFNCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, Lenth,Width,kernel_size=5):
        super(ACFNCell, self).__init__()
        padding = kernel_size // 2
        self.adcb = ADCB(in_channels)
        self.conv = nn.Conv2d(in_channels + hidden_channels, hidden_channels * 4, kernel_size, padding=padding)
        self.norm_cell = nn.LayerNorm([hidden_channels, Lenth, Width])
    def forward(self, Ft, Ht_1, Ct_1):
        Ft_hat, Ht_1_hat = self.adcb(Ft, Ht_1)
        combined = torch.cat([Ft_hat, Ht_1_hat], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        Ct = self.norm_cell(f * Ct_1 + i * g)
        Ht = o * torch.tanh(Ct)

        return Ht, Ct

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.encoder = nn.Conv2d(self.enc_in, self.d_model, kernel_size=1)
        self.decoder = nn.Conv2d(self.d_model, self.enc_in, kernel_size=1)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_layers = configs.e_layers
        self.Lenth = configs.Lenth
        self.Width = configs.Width
        # 堆叠多层 ACFNCell
        self.acfn_layers = nn.ModuleList([
            ACFNCell(self.d_model, self.d_model,self.Lenth,self.Width) for _ in range(self.num_layers)
        ])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if len(x_enc.shape)<5:
            x_enc = x_enc.unsqueeze(2)
        b, t, c, l, w = x_enc.shape
        # 初始化每一层的状态
        H = [torch.zeros(b, self.d_model, l, w).to(x_enc.device) for _ in range(self.num_layers)]
        C = [torch.zeros(b, self.d_model, l, w).to(x_enc.device) for _ in range(self.num_layers)]
        # ===== 编码阶段（处理输入序列） =====
        for i in range(self.seq_len):
            input_t = x_enc[:, i]
            input_t = self.encoder(input_t)
            for l in range(self.num_layers):
                input_t, C[l] = self.acfn_layers[l](input_t, H[l], C[l])
                H[l] = input_t  # 更新隐状态

        # ===== 解码阶段（递推 future pred_len -1 步）=====
        input_t = self.decoder(H[-1])  # 使用最后一层输出开始预测
        outputs = [input_t]
        for _ in range(self.pred_len-1):
            input_t = self.encoder(input_t)
            for l in range(self.num_layers):
                input_t, C[l] = self.acfn_layers[l](input_t, H[l], C[l])
                H[l] = input_t
            input_t = self.decoder(input_t)
            outputs.append(input_t)
        return torch.stack(outputs, dim=1).squeeze()  # (B, pred_len, C, H, W)