import torch
import torch.nn as nn
import torch.nn.functional as F

class SFPModule(nn.Module):
    """Spatiotemporal Feature Preprocessing Module"""
    def __init__(self, input_days, length, width, feature_dim):
        super(SFPModule, self).__init__()
        self.input_days = input_days
        self.length = length
        self.width = width
        self.feature_dim = feature_dim
        self.fc_time = nn.Linear(self.length * self.width, feature_dim)
        self.fc_space = nn.Linear(input_days, feature_dim)

    def forward(self, x):
        b, n, h, w = x.shape
        assert n == self.input_days
        x_merge = x.view(b, n, h * w)
        Itime = self.fc_time(x_merge)
        x_merge_t = x_merge.transpose(1, 2)
        Ispace = self.fc_space(x_merge_t)
        return Itime, Ispace

class AttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_out):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(dim_q, dim_out)
        self.key = nn.Linear(dim_k, dim_out)
        self.value = nn.Linear(dim_v, dim_out)

    def forward(self, q, k, v):
        Q = self.query(q)
        K = self.key(k)
        V = self.value(v)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        return out

class SIPGModule(nn.Module):
    """Spatiotemporal Information Pre-interaction Guidance"""
    def __init__(self, feature_dim):
        super(SIPGModule, self).__init__()
        self.temporal_self_attn = AttentionBlock(feature_dim, feature_dim, feature_dim, feature_dim)
        self.spatial_self_attn = AttentionBlock(feature_dim, feature_dim, feature_dim, feature_dim)
        self.temporal_guides_spatial = AttentionBlock(feature_dim, feature_dim, feature_dim, feature_dim)
        self.spatial_guides_temporal = AttentionBlock(feature_dim, feature_dim, feature_dim, feature_dim)

    def forward(self, Itime, Ispace, x_origin):
        FTT = self.temporal_self_attn(Itime, Itime, Itime)
        FSS = self.spatial_self_attn(Ispace, Ispace, Ispace)
        FTS = self.temporal_guides_spatial(Itime, Ispace, Ispace)
        FST = self.spatial_guides_temporal(Ispace, Itime, Itime)
        F = (FTT + FTS) @ (FSS + FST).transpose(1,2) + x_origin.view(x_origin.size(0), x_origin.size(1), -1)
        # print("Train Itime max/min:", Itime.max(), Itime.min())
        # print("Train Ispace max/min:", Ispace.max(), Ispace.min())
        # print("Train (FTT + FTS) max/min:", (FTT + FTS).max(), (FTT + FTS).min())
        # print("Train (FSS + FST) max/min:", (FSS + FST).max(), (FSS + FST).min())
        # print("Train F max/min:", F.max(), F.min())
        return F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 4, kernel_size, padding=padding)

    def forward(self, x, h, c,h_g = 0,c_g = 0):
        if len(x.shape)<=3:
            x = x.unsqueeze(1)
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g + c_g
        h_next = o * torch.tanh(c_next) + h_g
        return h_next, c_next

class GLConvLSTM(nn.Module):
    """Corrected Global-Local ConvLSTM Module"""
    def __init__(self, seq_len,feature_dim, hidden_dim, height, width, num_layers=2):
        super(GLConvLSTM, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.height = height
        self.width = width

        self.global_lstm = nn.ModuleList([
            ConvLSTMCell(self.seq_len, self.hidden_dim) for _ in range(num_layers)
        ])
        self.global_lstm = nn.ModuleList()
        self.global_lstm.append(ConvLSTMCell(self.seq_len, self.hidden_dim))
        self.local_lstm = nn.ModuleList()
        self.local_lstm.append(ConvLSTMCell(self.feature_dim, self.hidden_dim))
        for _ in range(num_layers-1):
            self.global_lstm.append(ConvLSTMCell(self.hidden_dim, self.hidden_dim))
            self.local_lstm.append(ConvLSTMCell(self.hidden_dim, self.hidden_dim))

    def forward(self, feats):
        b, t, f = feats.shape
        feats = feats.view(b, t, self.height, self.width)
        b,t,h,w = feats.shape
        h_global = [torch.zeros(b, self.hidden_dim, self.height, self.width, device=feats.device) for _ in range(self.num_layers)]
        c_global = [torch.zeros(b, self.hidden_dim, self.height, self.width, device=feats.device) for _ in range(self.num_layers)]
        h_local = [torch.zeros(b, self.hidden_dim, self.height, self.width, device=feats.device) for _ in range(self.num_layers)]
        c_local = [torch.zeros(b, self.hidden_dim, self.height, self.width, device=feats.device) for _ in range(self.num_layers)]

        feats_global = feats
        out = []
        for i in range(self.num_layers):
            h_global[i], c_global[i] = self.global_lstm[i](feats_global, h_global[i], c_global[i])
            for t_idx in range(t):
                h_local[i], c_local[i] = self.local_lstm[i](feats[:,t_idx], h_local[i], c_local[i],h_global[i],c_global[i])
                out.append(h_local[i])
            feats = torch.stack(out, dim=1)
            feats_global = h_global[i]
        return feats[:, -1]



class SpatialTransformer(nn.Module):
    def __init__(self, input_dim, nhead=2, num_layers=1):
        super(SpatialTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer(x)
        return x

class Model(nn.Module):
    """Full GL-ST Model"""
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.length = configs.Lenth
        self.width = configs.Width
        self.d_model = configs.d_model
        self.sfp = SFPModule(self.seq_len, self.length, self.width, self.d_model)
        self.sipg = SIPGModule(self.d_model)
        self.gl_lstm = GLConvLSTM(self.seq_len ,self.enc_in, self.seq_len, self.length, self.width)
        self.transformer = SpatialTransformer(self.length*self.width)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        X_min = x_enc.min(dim=1, keepdim=True)[0] # 沿S维度计算最小值
        X_max = x_enc.max(dim=1, keepdim=True)[0] # 沿S维度计算最大值
        x_enc = (x_enc - X_min)/(X_max-X_min+ 1e-10 )
        Itime, Ispace = self.sfp(x_enc)
        F = self.sipg(Itime, Ispace, x_enc)
        F_gl = self.gl_lstm(F)   # (B,d_model,w,h)
        #F_tr = self.transformer(x_enc.reshape(x_enc.shape[0],x_enc.shape[1],-1).permute(1,0,2)).permute(1,0,2)   #()
        F_tr = self.transformer(F.permute(1, 0, 2)).permute(1, 0, 2)
        out = F_gl + F_tr.reshape(F_tr.shape[0],F_tr.shape[1],self.length,self.width)
        last_output = out*((X_max-X_min))+X_min
        return last_output
