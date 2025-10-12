import argparse
import os
import torch
from exp.exp_main import Exp_Main
from exp.exp_main_2d import Exp_Main_2d
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='PCN')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='GCN',
                        help='model name')


    # data loader
    parser.add_argument('--data_name', type=str, required=True, default='BOHAI', help='dataset type')
    parser.add_argument('--data', type=str, required=True, default='1D', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='BOHAI.csv', help='data file')
    parser.add_argument('--freq', type=str, default='h',help='freq for time features encoding, '
    'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
    'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--preprocess', type=str, default='normal', help='data prepocess')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=30, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=15, help='prediction sequence length')


    # model define
    parser.add_argument('--dec_type', type=str, default='Concat', help='type of dec')
    parser.add_argument('--PE_type', type=str, default='Concat', help='type of PE')

    parser.add_argument('--time_window_size', type=int, default=5, help='time_window_size')
    parser.add_argument('--standard', type=str, default='type1', help='type of standard')
    parser.add_argument('--depth', type=int, default=3, help='depth of GCN')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--enc_in', type=int, default=1, help='feature dimension')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--node_num', type=int, default=162, help='node_num')
    parser.add_argument('--embed_dim', type=int, default=5, help='embed_dim')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--conv_channel', type=int, default=32, help='dimension of conv_channel')
    parser.add_argument('--skip_channel', type=int, default=32, help='dimension of skip_channel')
    parser.add_argument('--propalpha', type=float, default=0.3, help='propalpha')
    parser.add_argument('--heads', type=int, default=2, help='heads of attentions')
    parser.add_argument('--dilation', type=int, default=1, help='conv of dilation')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--Lenth', type=int, default=16, help='Lenth')
    parser.add_argument('--Width', type=int, default=18, help='Width')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    # optimization
    parser.add_argument('--num_workers', type=int, default=-1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())
    if args.data == '1D':
        Exp = Exp_Main
    elif args.data == '2D':
        Exp = Exp_Main_2d
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = f'{args.model}_{args.data_name}_{args.seq_len}_{args.pred_len}_{args.model_id}_{ii}'

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = f'{args.model}_{args.data_name}_{args.seq_len}_{args.pred_len}_{args.model_id}_{ii}'
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
