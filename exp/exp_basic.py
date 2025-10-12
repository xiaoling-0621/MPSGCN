import os
import torch
from models import GCN,ASTGRN,EAGCN,Informer,WITRAN,MoE,MoECheb,MoEGRU,UNIMoE,GSTGAT,TransDtSt,ACFN,GLST,GRU,LSTM,ConvLSTM,MUSTAN,FEDformer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'GCN': GCN,
            'ASTGRN': ASTGRN,
            'EAGCN':EAGCN,
            'Informer':Informer,
            'WITRAN' : WITRAN,
            'MoE' :MoE,
            'MoECheb' :MoECheb,
            'MoEGRU' :MoEGRU,
            'UNIMoE':UNIMoE,
            'GSTGAT':GSTGAT,
            'TransDtSt':TransDtSt,
            'ACFN':ACFN,
            'GLST':GLST,
            'GRU':GRU,
            'LSTM':LSTM,
            'ConvLSTM':ConvLSTM,
            'MUSTAN' :MUSTAN,
            'FEDformer':FEDformer
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
