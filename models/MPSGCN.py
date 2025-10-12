import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layers.Embed import DataEmbedding


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncw,vw->ncv',(x,A))
        #x = torch.einsum('ncwl,wv->nclv',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv1d(c_in, c_out, kernel_size=1, bias=bias)

    def forward(self,x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)   # (bs, conv , node_num)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)           #(bs,skip,node_num)
        return ho


class GCN(torch.nn.Module):
    def __init__(self, node_num, d_model,in_dim,out_dim, node_dim , dropout, conv_channel, skip_channel,depth ,propalpha):
        super(GCN,self).__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.nodevec1 = nn.Parameter(torch.randn(node_num, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, node_num), requires_grad=True)
        self.start_conv = nn.Conv1d(in_dim , conv_channel, (d_model - node_num + 1))
        self.gconv1 = mixprop(conv_channel, skip_channel, depth, dropout, propalpha)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv1d(skip_channel, out_dim , kernel_size=1)
        self.linear = nn.Linear(node_num, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self,x):  # (bs ,in_dim, d_model)
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        out = x.reshape(x.shape[0],self.in_dim,self.d_model)
        out = self.start_conv(out)   # (bs,conv,node_num)
        out = self.gelu(self.gconv1(out , adp))   #(bs,skip,node_num)
        out = self.end_conv(out).squeeze()      #(bs, out_dim,node_num)
        out = self.linear(out)
        #out = self.norm(x + out) #(bs,out_dim,d_model)
        return  self.norm(out).reshape(out.shape[0],-1)

class Encoder(torch.nn.Module):
    def __init__(self, node_num, d_model, node_dim, conv_channel,skip_channel,depth,propalpha,num_layers, dropout):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.d_model = d_model
        self.conv = nn.ModuleList()
        self.conv.append(GCN(node_num, d_model, 3,6,node_dim , dropout, conv_channel, skip_channel,depth ,propalpha))
        for layer in range(num_layers-1):
            self.conv.append(GCN(node_num, d_model,4,6, node_dim , dropout, conv_channel, skip_channel,depth ,propalpha))
    def linear(self, input, batch_size, slice, Water2sea_slice_num,Original_slice_len,layer):
        a = self.conv[layer](input)
        # if slice < Original_slice_len:
        #     a[:batch_size * (slice + 1), :] = self.conv[layer](a[:batch_size * (slice + 1), :])
        # elif Original_slice_len<slice<Water2sea_slice_num:
        #     a[batch_size * (slice - Original_slice_len + 1):slice + 1, :] = self.conv[layer](a[batch_size * (slice - Original_slice_len + 1):slice + 1, :])
        # else:
        #     a[batch_size * -(Water2sea_slice_num + Original_slice_len - 1 - slice):, :] = (
        #         self.conv[layer](a[batch_size * -(Water2sea_slice_num + Original_slice_len - 1 - slice):, :]))
        return a

    def forward(self, input, flag):
        if flag == 1:  # cols > rows
            input = input.permute(2, 0, 1, 3)
        else:
            input = input.permute(1, 0, 2, 3)
        Water2sea_slice_num,batch_size, Original_slice_len,d_model = input.shape
        Water2sea_slice_len = Water2sea_slice_num + Original_slice_len - 1
        hidden_slice_row = torch.zeros(Water2sea_slice_num * batch_size, self.d_model).to(input.device)
        hidden_slice_col = torch.zeros(Water2sea_slice_num * batch_size, self.d_model).to(input.device)
        input_transfer = torch.zeros(Water2sea_slice_num, batch_size, Water2sea_slice_len, d_model).to(input.device)
        for r in range(Water2sea_slice_num):
            input_transfer[r, :, r:r + Original_slice_len, :] = input[r, :, :, :]
        hidden_row_all_list = []
        hidden_col_all_list = []
        for layer in range(self.num_layers):
            if layer == 0:
                a = input_transfer.reshape(Water2sea_slice_num * batch_size, Water2sea_slice_len, d_model)
            else:
                a = F.dropout(output_all_slice, self.dropout, self.training)
                if layer == 1:
                    layer0_output = a
                hidden_slice_row = hidden_slice_row * 0
                hidden_slice_col = hidden_slice_col * 0
            # start every for all slice
            output_all_slice_list = []
            for slice in range(Water2sea_slice_len):
                # gate generate
                gate = self.linear(torch.cat([hidden_slice_row, hidden_slice_col, a[:, slice, :]],
                                             dim=-1),batch_size, slice, Water2sea_slice_num,Original_slice_len,layer)
                # gate
                sigmod_gate, tanh_gate = torch.split(gate, 4 * self.d_model, dim=-1)
                sigmod_gate = torch.sigmoid(sigmod_gate)
                tanh_gate = torch.tanh(tanh_gate)
                update_gate_row, output_gate_row, update_gate_col, output_gate_col = sigmod_gate.chunk(4, dim=-1)
                input_gate_row, input_gate_col = tanh_gate.chunk(2, dim=-1)
                # gate effect
                hidden_slice_row = torch.tanh(
                    (1 - update_gate_row) * hidden_slice_row + update_gate_row * input_gate_row) * output_gate_row
                hidden_slice_col = torch.tanh(
                    (1 - update_gate_col) * hidden_slice_col + update_gate_col * input_gate_col) * output_gate_col
                # output generate
                output_slice = torch.cat([hidden_slice_row, hidden_slice_col], dim=-1)
                # save output
                output_all_slice_list.append(output_slice)
                # save row hidden
                if slice >= Original_slice_len - 1:
                    need_save_row_loc = slice - Original_slice_len + 1
                    hidden_row_all_list.append(
                        hidden_slice_row[need_save_row_loc * batch_size:(need_save_row_loc + 1) * batch_size, :])
                # save col hidden
                if slice >= Water2sea_slice_num - 1:
                    hidden_col_all_list.append(
                        hidden_slice_col[(Water2sea_slice_num - 1) * batch_size:, :])
                # hidden transfer
                hidden_slice_col = torch.roll(hidden_slice_col, shifts=batch_size, dims=0)
            if layer >= 1:  # layer-res
                output_all_slice = torch.stack(output_all_slice_list, dim=1) + layer0_output
            else:
                output_all_slice = torch.stack(output_all_slice_list, dim=1)
        hidden_row_all = torch.stack(hidden_row_all_list, dim=1)
        hidden_col_all = torch.stack(hidden_col_all_list, dim=1)
        hidden_row_all = hidden_row_all.reshape(batch_size, self.num_layers, Water2sea_slice_num,
                                                hidden_row_all.shape[-1])
        hidden_col_all = hidden_col_all.reshape(batch_size, self.num_layers, Original_slice_len,
                                                hidden_col_all.shape[-1])
        if flag == 1:
            return output_all_slice, hidden_col_all, hidden_row_all
        else:
            return output_all_slice, hidden_row_all, hidden_col_all

'''
class Decoder(torch.nn.Module):
    def __init__(self,configs):
        super(Decoder,self).__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.time_window_size = configs.time_window_size
        self.node_num = configs.node_num
        self.seq_len = configs.seq_len
        self.layer = configs.e_layers
        self.project = nn.Linear(self.d_model,self.node_num)
        self.predict = nn.Linear(self.layer*self.time_window_size+self.layer*(self.seq_len//self.time_window_size),self.pred_len)
    def forward(self,rows,cols):
        rows = rows.reshape(rows.shape[0], -1, self.d_model)
        cols = cols.reshape(cols.shape[0], -1, self.d_model)
        out = torch.cat([rows,cols],dim = 1)
        out = self.project(out)
        return self.predict(out.permute(0,2,1)).permute(0,2,1)
'''

class Decoder(torch.nn.Module):
    def __init__(self,configs):
        super(Decoder,self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.time_window_size = configs.time_window_size
        self.num_windows = self.seq_len//self.time_window_size
        self.node_num = configs.node_num
        self.layer = configs.e_layers
        self.project = nn.Linear(self.layer * 2 * self.d_model, self.num_windows * self.d_model)
        self.predict = nn.Linear(self.d_model, self.node_num)
    def forward(self,rows,cols):
        enc_hid_row = rows[:, :, -1:, :].expand(-1, -1, self.time_window_size, -1)
        output = torch.cat([enc_hid_row, cols], dim=-1).permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0],
                                output.shape[1], output.shape[2] * output.shape[3])
        last_output = self.project(output)
        last_output = last_output.reshape(last_output.shape[0], last_output.shape[1],
                                          self.num_windows, self.d_model).permute(0, 2, 1, 3)
        last_output = last_output.reshape(last_output.shape[0],
                                          last_output.shape[1] * last_output.shape[2], last_output.shape[3])

        last_output = self.predict(last_output)
        return last_output

    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.dec_in = configs.dec_in
        self.node_num =configs.node_num
        self.node_dim = configs.embed_dim
        self.conv_channel = configs.conv_channel
        self.skip_channel = configs.skip_channel
        self.depth  = configs.depth
        self.propalpha = configs.propalpha
        self.d_model = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        self.standard = configs.standard
        self.time_window_size = configs.time_window_size
        self.num_windows = int(configs.seq_len / self.time_window_size)
        # Encoder
        self.enc_embedding = DataEmbedding(configs.node_num, configs.d_model,
                                           configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder(self.node_num, self.d_model,self.node_dim,self.conv_channel, self.skip_channel,
                               self.depth, self.propalpha,self.num_layers,self.dropout)
        self.decoder = Decoder(configs)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.standard == 'type1':
            seq_last = x_enc[:, -1:, :].detach()
            x_enc = x_enc - seq_last
        elif self.standard == 'type2':
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        batch_size, _, node_num = enc_out.shape
        enc_out = enc_out.reshape(batch_size, self.num_windows, self.time_window_size, node_num)

        if self.num_windows <= self.time_window_size:
            flag = 0
        else:  # need permute
            flag = 1

        _, enc_hid_row, enc_hid_col = self.encoder(enc_out, flag)
        last_output = self.decoder(enc_hid_row,enc_hid_col)

        if self.standard == 'type1':
            last_output = last_output + seq_last
        elif self.standard == 'type2':
            last_output = last_output * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(
                          1, self.seq_len, 1))
            last_output = last_output + \
                      (means[:, 0, :].unsqueeze(1).repeat(
                          1, self.seq_len, 1))

        return last_output