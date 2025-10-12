from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

def masked_loss(output, target, mask, loss_type='mse', eps=1e-6):
    """
    支持 MSE, MAE, RMSE, MAPE, MSPE 的掩码损失函数

    参数:
        output (Tensor): 模型预测值
        target (Tensor): 真实值
        mask (Tensor): 掩码，海洋区域为 1，陆地为 0
        loss_type (str): 'mse' | 'mae' | 'rmse' | 'mape' | 'mspe'
        eps (float): 防止除以 0 的最小值

    返回:
        标量误差
    """

    mask = mask.float()
    output = output * mask
    target = target * mask
    diff = output - target
    n,t,x,y = diff.shape
    node_num = t*n*mask.sum()
    if loss_type == 'mse':
        loss = (diff ** 2 * mask).sum() / node_num
    elif loss_type == 'mae':
        loss = (diff.abs() * mask).sum() / node_num
    elif loss_type == 'rmse':
        loss = torch.sqrt((diff ** 2 * mask).sum() / node_num)
    elif loss_type == 'mape':
        # 避免除以 0：将 target 小于 eps 的地方视为无效区域
        denom = torch.clamp(target.abs(), min=eps)
        percentage_error = (diff.abs() / denom) * mask
        loss = percentage_error.sum() / node_num
    elif loss_type == 'mspe':
        denom = torch.clamp(target.abs(), min=eps)
        percentage_squared_error = ((diff / denom) ** 2) * mask
        loss = percentage_squared_error.sum() / node_num
    else:
        raise ValueError(f"不支持的 loss_type：{loss_type}")
    return loss

class Exp_Main_2d(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_2d, self).__init__(args)
        self.area = args.data_name
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = masked_loss
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        mask = torch.load(f'datasets/{self.area}_flag.pth')['flag']
        mask = (~torch.isnan(torch.from_numpy(mask).float())).float()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :,:]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :,:], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :,:]
                batch_y = batch_y[:, -self.args.pred_len:, :,:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true,mask)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        mask = torch.load(f'datasets/{self.area}_flag.pth')['flag']
        mask = (~torch.isnan(torch.from_numpy(mask).float())).float()
        path = os.path.join(self.args.checkpoints + f'{self.args.model}_{self.args.model_id}', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        e_num = 0
        e_time = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :,:]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :,:], dec_inp], dim=1).float().to(self.device)
                
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:,:,:]
                batch_y = batch_y[:, -self.args.pred_len:,:,:]
                mask = mask.to(outputs.device)
                loss = criterion(outputs, batch_y,mask)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()
            e_num += 1
            e_time += time.time() - epoch_time

            print("Epoch: {} cost time: {} ".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        print(f's/Epoch :  {e_time / e_num} s/epoch')
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        mask = torch.load(f'datasets/{self.area}_flag.pth')['flag']
        mask = (~torch.isnan(torch.from_numpy(mask).float())).float()
        if test:
            print('loading model')
            path = os.path.join(self.args.checkpoints + f'{self.args.model}_{self.args.model_id}', setting)
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        preds = []
        trues = []

        if test:
            folder_path = f'./test_results/{self.args.model}/{self.args.model}_{self.args.model_id}/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :,:]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :,:], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :,:]
                batch_y = batch_y[:, -self.args.pred_len:, :,:]

                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()

                outputs = outputs[:, :, :,:]
                batch_y = batch_y[:, :, :,:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        print('test shape:', preds.shape, trues.shape)
        # for t in range(trues.shape[1]):
        #     mae, rmse, mape, _, _ = metric(preds[:, t, :], trues[:, t, :])
        #     print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
        #         t + 1, mae, rmse, mape*100))
        mae, mse, rmse, mape, mspe = (masked_loss(preds, trues,mask,'mae'),
                                      masked_loss(preds, trues,mask,'mse'),
                                      masked_loss(preds, trues, mask,'rmse'),
                                      masked_loss(preds, trues, mask,'mape'),
                                      masked_loss(preds, trues, mask,'mspe'))
        print('mse:{}, mae:{}, mape:{}, mspe:{}'.format(mse, mae, mape, mspe))
        f = open(f"{self.args.model}/{self.args.model}_{self.args.model_id}.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, mape:{}, mspe:{}'.format(mse, mae, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()
        # result save
        if test:
            folder_path = f'./results/{self.args.model}/{self.args.model}_{self.args.model_id}/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
        return
