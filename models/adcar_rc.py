import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from models import rcpna
from utils.constants import Cte
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
from utils import utils
import math


class PointDataset(Dataset):
    def __init__(self, x, u):
        self.x = x
        self.u = u

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.u[idx]


class ADCAR_RC(object):

    def __init__(self, cfg, input_dim, ad_model, model_vaca, data_module, intervention_features,
                 train_X, x_test, test_rc, rc_quantile=0.01, alpha=1, batch_size=64,
                 max_epoch=50, device='cuda:0', data='loan', cost_f=True, R_ratio=0.1, lr=1e-4):
        super().__init__()

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.device = device
        self.data = data
        self.ad_model = ad_model
        self.model_vaca = model_vaca.to(device)
        self.alpha = alpha
        self.data_module = data_module
        self.cfg = cfg
        self.intervention_features = intervention_features
        self.intervention_dim = len(intervention_features)
        self.train_X = train_X
        self.x_test = x_test
        self.test_rc = test_rc
        self.rc_quantile = rc_quantile
        if cost_f:
            self.cost_f = True
        else:
            self.cost_f = False

        self.model_vaca.eval()
        self.model_vaca.freeze()

        self.net = rcpna.RCPNA(self.cfg, self.data_module, self.intervention_features).to(self.device)

        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        self.loss_mse = nn.MSELoss()

        self.ad_model.net.eval()
        self.ad_net = self.ad_model.net.to(self.device)
        self.ad_net.load_state_dict(self.ad_model.net.state_dict())
        if ad_model.name == 'deepsvdd':
            self.ad_c = self.ad_model.c.to(self.device)
        self.param_name = f'{self.data}_{self.alpha}_{self.batch_size}_{self.max_epoch}_{self.intervention_features}_{self.cost_f}_{R_ratio}_{lr}'

        self.R_ratio = R_ratio

        self._get_scale()
        self._get_rc_res()


    def _get_rc_res(self):
        lst_rc_pred = []
        lst_rc_gt = []
        if self.data == 'loan':
            for i in self.test_rc:
                lst_rc_gt.extend(Cte.LOAN_RC_LIST[i])

            lst_prob = []
            for i in range(len(self.train_X)):
                prob = self.model_vaca.get_distribution(self.train_X[i].reshape(1, -1).to(self.device), self.data_module,
                    device=self.device)
                lst_prob.append(prob)
            lst_prob = np.array(lst_prob)
            self.prob_low = np.quantile(lst_prob, self.rc_quantile, axis=0)
            self.prob_high = np.quantile(lst_prob, (1 - self.rc_quantile), axis=0)
            for i in range(len(self.x_test)):
                prob = self.model_vaca.get_distribution(self.x_test[i].reshape(1, -1).to(self.device), self.data_module,
                    device=self.device)
                res = ((prob <= self.prob_high) == False).astype(int) + ((prob >= self.prob_low) == False).astype(int)
                res = np.sum(res[0], axis=1)
                res = np.where(res >= 1, 1, 0)
                idx_interven = np.zeros(len(res))
                for j in self.intervention_features:
                    idx_interven[j] += 1
                res = res * idx_interven
                lst_rc_pred.extend(res)
            print('Results for root cause analysis:')
            print(classification_report(y_true=lst_rc_gt, y_pred=lst_rc_pred, digits=5))
            print(confusion_matrix(y_true=lst_rc_gt, y_pred=lst_rc_pred))

        elif self.data == 'adult':
            for i in self.test_rc:
                lst_rc_gt.extend(Cte.ADULT_RC_LIST[i])

            lst_prob = []
            for i in range(len(self.train_X)):
                prob = self.model_vaca.get_distribution(F.pad(self.train_X[i].reshape(1,-1), (0,4,0,0)).to(self.device), self.data_module, device=self.device)
                lst_prob.append(prob)
            lst_prob = np.array(lst_prob)
            self.prob_low = np.quantile(lst_prob, self.rc_quantile, axis=0)
            self.prob_high = np.quantile(lst_prob, (1-self.rc_quantile), axis=0)
            for i in range(len(self.x_test)):
                prob = self.model_vaca.get_distribution(F.pad(self.x_test[i].reshape(1,-1), (0,4,0,0)).to(self.device), self.data_module, device=self.device)
                res = ((prob <= self.prob_high) == False).astype(int) + ((prob >= self.prob_low) == False).astype(int)
                res = np.sum(res[0], axis=1)[:-1]
                res = np.where(res >= 1, 1, 0)
                idx_interven = np.zeros(len(res))
                for j in self.intervention_features:
                    idx_interven[j] += 1
                res = res * idx_interven
                lst_rc_pred.extend(res)
            print('Results for root cause analysis:')
            print(classification_report(y_true=lst_rc_gt, y_pred=lst_rc_pred, digits=5))
            print(confusion_matrix(y_true=lst_rc_gt, y_pred=lst_rc_pred))

        elif self.data == 'donors':
            lst_prob = []
            for i in range(len(self.train_X)):
                prob = self.model_vaca.get_distribution(self.train_X[i].reshape(1, -1).to(self.device), self.data_module,
                    device=self.device)
                lst_prob.append(prob)
            lst_prob = np.array(lst_prob)
            self.prob_low = np.quantile(lst_prob, self.rc_quantile, axis=0)
            self.prob_high = np.quantile(lst_prob, (1 - self.rc_quantile), axis=0)
            for i in range(len(self.x_test)):
                prob = self.model_vaca.get_distribution(self.x_test[i].reshape(1, -1).to(self.device), self.data_module,
                    device=self.device)
                res = ((prob <= self.prob_high) == False).astype(int) + ((prob >= self.prob_low) == False).astype(int)
                res = np.sum(res[0], axis=1)[:-1]
                res = np.where(res >= 1, 1, 0)
                idx_interven = np.zeros(len(res))
                for j in self.intervention_features:
                    idx_interven[j] += 1
                res = res * idx_interven
                lst_rc_pred.extend(res)

        else:
            NotImplementedError



    def _get_scale(self):
        if self.data == 'loan':
            self.scale = (self.data_module.scaler.inverse_transform(
                [[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]]) - self.data_module.scaler.inverse_transform(
                [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))[0][3:].to(self.device)
        elif self.data == 'adult':
            lst_zeros = np.zeros(44)
            lst_1 = lst_zeros.copy()
            lst_2 = lst_zeros.copy()
            lst_1[[4, 16, 20]] = 1.0
            lst_2[[4, 16, 20]] = 2.0
            self.scale = \
                (self.data_module.scaler.inverse_transform([lst_2]) - self.data_module.scaler.inverse_transform(
                    [lst_1]))[
                    0][[3, 9, 10]].float().to(self.device)
        elif self.data == 'donors':
            self.scale = (self.data_module.scaler.inverse_transform(
                [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0]]) - self.data_module.scaler.inverse_transform(
                [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))[0][7:-1].to(self.device)
        else:
            print('Cannot get self.scale!!!')

    def _train(self, iterator):
        self.net.train()

        epoch_loss = 0
        epoch_dist_loss = 0
        epoch_l2_loss = 0

        for batch in iterator:
            org_x = batch[0].to(self.device)
            org_u = batch[1]

            self.optim.zero_grad()
            output = []

            for i in range(len(org_x)):
                if self.data == 'donors':
                    prob = self.model_vaca.get_distribution(
                        F.pad(org_x[i].reshape(1, -1), (0, 1, 0, 0)).to(self.device), self.data_module,
                        device=self.device)
                    res = ((prob <= self.prob_high) == False).astype(int) + ((prob >= self.prob_low) == False).astype(
                        int)
                    res = np.sum(res[0], axis=1)[:-1]
                    res = np.where(res >= 1, 1, 0)
                    idx_interven = np.zeros(len(res))
                    for j in self.intervention_features:
                        idx_interven[j] += 1
                    res = res * idx_interven
                    if sum(res) == 0:
                        res = idx_interven.copy()
                    inter_features = np.where(res == 1)[0].astype(int)
                    out = self.net.forward(F.pad(org_x[i].reshape(1, -1), (0, 1, 0, 0)).to(self.device), inter_features)
                    output.append(out)
                elif self.data == 'adult':
                    prob = self.model_vaca.get_distribution(
                        F.pad(org_x[i].reshape(1, -1), (0, 4, 0, 0)).to(self.device), self.data_module,
                        device=self.device)
                    res = ((prob <= self.prob_high) == False).astype(int) + ((prob >= self.prob_low) == False).astype(
                        int)
                    res = np.sum(res[0], axis=1)[:-1]
                    res = np.where(res >= 1, 1, 0)
                    idx_interven = np.zeros(len(res))
                    for j in self.intervention_features:
                        idx_interven[j] += 1
                    res = res * idx_interven
                    if sum(res) == 0:
                        res = idx_interven.copy()
                    inter_features = np.where(res == 1)[0].astype(int)
                    out = self.net.forward(F.pad(org_x[i].reshape(1, -1), (0, 4, 0, 0)).to(self.device), inter_features)
                    output.append(out)
                elif self.data == 'loan':
                    prob = self.model_vaca.get_distribution(org_x[i].reshape(1, -1).to(self.device),
                                                            self.data_module,
                                                            device=self.device)
                    res = ((prob <= self.prob_high) == False).astype(int) + ((prob >= self.prob_low) == False).astype(
                        int)
                    res = np.sum(res[0], axis=1)
                    res = np.where(res >= 1, 1, 0)
                    idx_interven = np.zeros(len(res))
                    for j in self.intervention_features:
                        idx_interven[j] += 1
                    res = res * idx_interven
                    if sum(res) == 0:
                        res = idx_interven.copy()
                    inter_features = np.where(res == 1)[0].astype(int)
                    out = self.net.forward(org_x[i].reshape(1, -1).to(self.device), inter_features)
                    output.append(out)
                else:
                    NotImplementedError
            output = torch.stack(output).reshape(-1, self.intervention_dim)
            x_cf_hat = self.model_vaca.get_changed(org_x.clone(), output, self.data_module,
                                                   self.data_module.likelihood_list, inverse=False, data=self.data,
                                                   device=self.device)

            if self.cost_f:
                l2_loss = self.loss_mse(output * self.scale, torch.zeros(output.size()).to(self.device))
            else:
                l2_loss = self.loss_mse(output, torch.zeros(output.size()).to(self.device))

            if self.ad_model.name == 'deepsvdd':
                dists_loss = torch.cdist(self.ad_model.net.forward(x_cf_hat), self.ad_c.view(1, -1), p=2).view(
                    -1).float()
            else:
                dists_loss = ((self.ad_model.net.forward(x_cf_hat) - x_cf_hat) ** 2).sum(axis=1)
            dists_loss = torch.where(dists_loss > self.R_ratio * self.ad_model.R, dists_loss,
                                     torch.tensor(0.0).float().to(self.device))
            total_dists_loss = torch.mean(dists_loss)
            # print(f'dist loss: {total_dists_loss}, l2 loss: {l2_loss}')
            loss = total_dists_loss + self.alpha * l2_loss
            loss.backward()
            self.optim.step()

            epoch_loss += loss.item()
            epoch_dist_loss += total_dists_loss.item()
            epoch_l2_loss += l2_loss.item()
        print(f'Epoch loss: {epoch_loss / len(iterator)}, epoch dist loss: {epoch_dist_loss / len(iterator)}, '
              f'epoch l2 loss: {epoch_l2_loss / len(iterator)}')

        return epoch_loss / len(iterator)

    def _evaluate(self, iterator):
        self.net.eval()
        epoch_loss = 0
        lst_pred_vaca = []  # lst for ad prediction vaca
        lst_pred_gt = []  # lst for ad prediction w causal
        lst_x_vaca = []  # lst for vaca x cf hat
        lst_x_gt = []  # lst for ground truth x cf which computed by delta x with causal
        lst_changes = []
        recon_mse_norm = 0.0  # vaca recoonstruction error
        recon_err_orig = 0.0  # vaca recoonstruction error
        recon_sse_norm = 0.0
        recon_sse_orig = 0.0
        with torch.no_grad():
            for batch in iterator:
                org_x = batch[0].to(self.device)
                org_u = batch[1]

                output = []

                for i in range(len(org_x)):
                    if self.data == 'donors':
                        prob = self.model_vaca.get_distribution(
                            F.pad(org_x[i].reshape(1, -1), (0, 1, 0, 0)).to(self.device), self.data_module,
                            device=self.device)
                        res = ((prob <= self.prob_high) == False).astype(int) + (
                                    (prob >= self.prob_low) == False).astype(
                            int)
                        res = np.sum(res[0], axis=1)[:-1]
                        res = np.where(res >= 1, 1, 0)
                        idx_interven = np.zeros(len(res))
                        for j in self.intervention_features:
                            idx_interven[j] += 1
                        res = res * idx_interven
                        if sum(res) == 0:
                            res = idx_interven.copy()
                        inter_features = np.where(res == 1)[0].astype(int)
                        out = self.net.forward(F.pad(org_x[i].reshape(1, -1), (0, 1, 0, 0)).to(self.device),
                                               inter_features)
                        output.append(out)
                    elif self.data == 'adult':
                        prob = self.model_vaca.get_distribution(
                            F.pad(org_x[i].reshape(1, -1), (0, 4, 0, 0)).to(self.device), self.data_module,
                            device=self.device)
                        res = ((prob <= self.prob_high) == False).astype(int) + (
                                    (prob >= self.prob_low) == False).astype(
                            int)
                        res = np.sum(res[0], axis=1)[:-1]
                        res = np.where(res >= 1, 1, 0)
                        idx_interven = np.zeros(len(res))
                        for j in self.intervention_features:
                            idx_interven[j] += 1
                        res = res * idx_interven
                        if sum(res) == 0:
                            res = idx_interven.copy()
                        inter_features = np.where(res == 1)[0].astype(int)
                        out = self.net.forward(F.pad(org_x[i].reshape(1, -1), (0, 4, 0, 0)).to(self.device),
                                               inter_features)
                        output.append(out)
                    elif self.data == 'loan':
                        prob = self.model_vaca.get_distribution(org_x[i].reshape(1, -1).to(self.device),
                                                                self.data_module,
                                                                device=self.device)
                        res = ((prob <= self.prob_high) == False).astype(int) + (
                                    (prob >= self.prob_low) == False).astype(
                            int)
                        res = np.sum(res[0], axis=1)
                        res = np.where(res >= 1, 1, 0)
                        idx_interven = np.zeros(len(res))
                        for j in self.intervention_features:
                            idx_interven[j] += 1
                        res = res * idx_interven
                        if sum(res) == 0:
                            res = idx_interven.copy()
                        inter_features = np.where(res == 1)[0].astype(int)
                        out = self.net.forward(org_x[i].reshape(1, -1).to(self.device), inter_features)
                        output.append(out)
                    else:
                        NotImplementedError
                output = torch.stack(output).reshape(-1, self.intervention_dim)
                # get x cf hat with vaca
                x_cf_hat = self.model_vaca.get_changed(org_x.clone(), output, self.data_module,
                                                       self.data_module.likelihood_list, inverse=False, data=self.data,
                                                       device=self.device)
                # get l2 loss
                if self.cost_f:
                    l2_loss = self.loss_mse(output * self.scale, torch.zeros(output.size()).to(self.device))
                else:
                    l2_loss = self.loss_mse(output, torch.zeros(output.size()).to(self.device))
                lst_changes.extend(
                    list((np.sum((output * self.scale).detach().cpu().numpy() ** 2, axis=1) ** 0.5).reshape(-1)))
                # get x cf hat ad prediction
                if self.ad_model.name == 'deepsvdd':
                    dists_loss_vaca = torch.cdist(self.ad_model.net.forward(x_cf_hat), self.ad_c.view(1, -1),
                                                  p=2).view(-1).float()
                else:
                    dists_loss_vaca = ((self.ad_model.net.forward(x_cf_hat) - x_cf_hat) ** 2).sum(axis=1)
                lst_dist_vaca = dists_loss_vaca.detach().cpu().numpy()
                lst_pred_vaca.extend(list(np.where(lst_dist_vaca < self.ad_model.R, 0, 1)))
                if self.data == 'loan':
                    lst_x_vaca.extend(x_cf_hat.detach().cpu().numpy().tolist())
                elif self.data == 'adult':
                    lst_x_vaca.extend(F.pad(x_cf_hat, (0, 4, 0, 0)).detach().cpu().numpy().tolist())
                elif self.data == 'donors':
                    lst_x_vaca.extend(F.pad(x_cf_hat, (0, 1, 0, 0)).detach().cpu().numpy().tolist())
                else:
                    NotImplementedError
                dists_loss_vaca = torch.where(dists_loss_vaca > self.R_ratio * self.ad_model.R, dists_loss_vaca,
                                              torch.tensor(0.0).float().to(self.device))
                total_dists_loss = torch.mean(dists_loss_vaca)
                loss = total_dists_loss + self.alpha * l2_loss
                epoch_loss += loss.item()

                # get gt results
                x_cf_gt = utils.get_summed_xGT(self.data_module, org_u.numpy(), output * self.scale, self.data)
                # get reconstruction error
                if self.data == 'loan':
                    recon_mse_norm += self.loss_mse(x_cf_hat.cpu(), x_cf_gt) / org_x.size()[1]
                    recon_err_orig += self.loss_mse(
                        self.data_module.scaler.inverse_transform(x_cf_hat.cpu()),
                        self.data_module.scaler.inverse_transform(x_cf_gt)) / org_x.size()[1]
                    recon_sse_norm += torch.std((x_cf_hat.cpu() - x_cf_gt), unbiased=False) / org_x.size()[1]
                    recon_sse_orig += torch.std(
                        self.data_module.scaler.inverse_transform(x_cf_hat.cpu()) -
                        self.data_module.scaler.inverse_transform(x_cf_gt), unbiased=False) / org_x.size()[1]
                    lst_x_gt.extend(x_cf_gt.numpy().tolist())
                elif self.data == 'adult':
                    recon_mse_norm += self.loss_mse(F.pad(x_cf_hat.cpu(), (0, 4, 0, 0)), F.pad(x_cf_gt, (0, 4, 0, 0))) / \
                                      org_x.size()[1]
                    recon_err_orig += self.loss_mse(
                        self.data_module.scaler.inverse_transform(F.pad(x_cf_hat.cpu(), (0, 4, 0, 0))),
                        self.data_module.scaler.inverse_transform(F.pad(x_cf_gt, (0, 4, 0, 0)))) / org_x.size()[1]
                    recon_sse_norm += torch.std((F.pad(x_cf_hat.cpu(), (0, 4, 0, 0)) - F.pad(x_cf_gt, (0, 4, 0, 0))),
                                                unbiased=False) / org_x.size()[1]
                    recon_sse_orig += torch.std(
                        self.data_module.scaler.inverse_transform(F.pad(x_cf_hat.cpu(), (0, 4, 0, 0))) -
                        self.data_module.scaler.inverse_transform(F.pad(x_cf_gt, (0, 4, 0, 0))), unbiased=False) / \
                                      org_x.size()[1]
                    lst_x_gt.extend(F.pad(x_cf_gt.cpu(), (0, 4, 0, 0)).numpy().tolist())
                elif self.data == 'donors':
                    recon_mse_norm = -1
                    recon_err_orig = -1
                    recon_sse_norm = -1
                    recon_sse_orig = -1
                else:
                    NotImplementedError

                # get x gt ad prediction
                if self.ad_model.name == 'deepsvdd':
                    dists_loss_gt = torch.cdist(self.ad_model.net.forward(x_cf_gt.to(self.device)),
                                                self.ad_c.view(1, -1),
                                                p=2).view(-1).float()
                else:
                    dists_loss_gt = ((self.ad_model.net.forward(x_cf_gt.to(self.device)) - x_cf_gt.to(
                        self.device)) ** 2).sum(axis=1)

                lst_dist_gt = dists_loss_gt.detach().cpu().numpy()
                lst_pred_gt.extend(list(np.where(lst_dist_gt < self.ad_model.R, 0, 1)))

            if self.data != 'donors':
                print(f'Reconstruction MSE norm: {recon_mse_norm / len(iterator)}')
                print(f'Reconstruction MSE orig: {recon_err_orig / len(iterator)}')
                print(f'Reconstruction SSE norm: {recon_sse_norm / len(iterator)}')
                print(f'Reconstruction SSE orig: {recon_sse_orig / len(iterator)}')
            print(epoch_loss)

        return epoch_loss / len(iterator), lst_pred_vaca, np.array(lst_x_vaca), lst_pred_gt, np.array(
            lst_x_gt), lst_changes

    def get_result(self, x, org_u):
        if self.data == 'loan':
            x_changed = self.net.forward(x.to(self.device))
            x_theta = (x_changed * self.scale).detach().cpu().numpy()
            x_cf_hat = self.model_vaca.get_changed(torch.unsqueeze(x, 0).to(self.device), torch.unsqueeze(x_changed, 0),
                                                   self.data_module, self.data_module.likelihood_list, inverse=False,
                                                   data=self.data, device=self.device)
            x_cf_hat = x_cf_hat.detach().cpu()[0].numpy()

            x_cf_gt = utils.get_summed_xGT(self.data_module, org_u, torch.unsqueeze(x_changed * self.scale, 0),
                                           self.data)
            x_cf_gt = x_cf_gt.detach().cpu()[0].numpy()

            x = x.numpy()
            x_inverse = self.data_module.scaler.inverse_transform([x])[0].numpy()
            x_cf_hat_inverse = self.data_module.scaler.inverse_transform([x_cf_hat])[0].numpy()
            x_cf_gt_inverse = self.data_module.scaler.inverse_transform([x_cf_gt])[0].numpy()
            lst = [x_inverse, x_cf_hat_inverse, x_cf_gt_inverse]
            df = pd.DataFrame(lst, columns=['G', 'A', 'E', 'L', 'D', 'I', 'S'])
            exp_val = -0.3 * (- df['L'] - df['D'] + df['I'] + df['S'] + df['I'] * df['S'])
            lst_y = []
            for i in range(len(df)):
                lst_y.append((1 + math.exp(exp_val[i])) ** (-1))
            df['y'] = lst_y

        elif self.data == 'adult':
            x_changed = self.net.forward(x.to(self.device))
            x_theta = (x_changed * self.scale).detach().cpu().numpy()
            x_cf_hat = self.model_vaca.get_changed(torch.unsqueeze(x, 0).to(self.device), torch.unsqueeze(x_changed, 0),
                                                   self.data_module, self.data_module.likelihood_list, inverse=False,
                                                   data=self.data, device=self.device)
            x = F.pad(x, (0, 4)).numpy()
            x_cf_hat = x_cf_hat.detach().cpu()[0]
            x_cf_hat = F.pad(x_cf_hat, (0, 4)).numpy()
            x_cf_gt = utils.get_summed_xGT(self.data_module, org_u, torch.unsqueeze(x_changed * self.scale, 0),
                                           self.data)
            x_cf_gt = x_cf_gt.detach().cpu()[0]
            x_cf_gt = F.pad(x_cf_gt, (0, 4)).numpy()
            x_inverse = self.data_module.scaler.inverse_transform([x])[0].numpy()
            x_cf_hat_inverse = self.data_module.scaler.inverse_transform([x_cf_hat])[0].numpy()
            x_cf_gt_inverse = self.data_module.scaler.inverse_transform([x_cf_gt])[0].numpy()
            lst = [x_inverse, x_cf_hat_inverse, x_cf_gt_inverse]
            df = pd.DataFrame(lst, \
                              columns=['R_0', 'R_1', 'R_2', 'A', 'N_0', 'N_1', 'N_2', 'N_3', 'S', 'E', 'H', \
                                       'W_0', 'W_1', 'W_2', 'W_3', 'M_0', 'M_1', 'M_2', 'O_0', 'O_1', 'O_2', \
                                       'L_0', 'L_1', 'L_2', 'I'])
            df[['R_0', 'R_1', 'R_2', 'N_0', 'N_1', 'N_2', 'N_3', 'S', 'W_0', 'W_1', 'W_2', 'W_3', 'M_0', 'M_1',
                'M_2', 'O_0', 'O_1', 'O_2', 'L_0', 'L_1', 'L_2']] = np.round(df[['R_0', 'R_1', 'R_2', 'N_0', 'N_1',
                                                                                 'N_2', 'N_3', 'S', 'W_0', 'W_1', 'W_2',
                                                                                 'W_3', 'M_0', 'M_1', 'M_2', 'O_0',
                                                                                 'O_1', 'O_2', 'L_0',
                                                                                 'L_1', 'L_2']].values)
            df['I'] = self.data_module.train_dataset.structural_eq['income'](u=np.array([org_u, org_u, org_u]),
                                                                             race=df[['R_0', 'R_1', 'R_2']].values,
                                                                             age=df[['A']].values,
                                                                             edu=df[['E']].values,
                                                                             occupation=df[
                                                                                 ['O_0', 'O_1', 'O_2']].values,
                                                                             work_class=df[
                                                                                 ['W_0', 'W_1', 'W_2', 'W_3']].values,
                                                                             maritial=df[['M_0', 'M_1', 'M_2']].values,
                                                                             hour=df[['H']].values,
                                                                             native_country=df[
                                                                                 ['N_0', 'N_1', 'N_2']].values,
                                                                             gender=df[['S']].values,
                                                                             relationship=df[
                                                                                 ['L_0', 'L_1', 'L_2']].values)
        elif self.data == 'donors':
            x_changed = self.net.forward(x.to(self.device))
            x_theta = (x_changed * self.scale).detach().cpu().numpy()
            x_cf_hat = self.model_vaca.get_changed(torch.unsqueeze(x, 0).to(self.device), torch.unsqueeze(x_changed, 0),
                                                   self.data_module, self.data_module.likelihood_list, inverse=False,
                                                   data=self.data, device=self.device)
            x = F.pad(x, (0, 1)).numpy()
            x_cf_hat = x_cf_hat.detach().cpu()[0]
            x_cf_hat = F.pad(x_cf_hat, (0, 1)).numpy()
            x_inverse = self.data_module.scaler.inverse_transform([x])[0].numpy()
            x_cf_hat_inverse = torch.round(self.data_module.scaler.inverse_transform([x_cf_hat])[0]).numpy()
            lst = [x_inverse, x_cf_hat_inverse]
            df = pd.DataFrame(lst, columns=['at_least_1_teacher_referred_donor', 'fully_funded',
                                            'at_least_1_green_donation', 'great_chat',
                                            'three_or_more_non_teacher_referred_donors',
                                            'one_non_teacher_referred_donor_giving_100_plus',
                                            'donation_from_thoughtful_donor', 'great_messages_proportion',
                                            'teacher_referred_count', 'non_teacher_referred_count', 'is_exciting'])
            df['is_exciting'] = -1
            df['is_exciting'] = df['fully_funded'].values * df['at_least_1_teacher_referred_donor'].values * \
                                df['great_chat'].values * df['at_least_1_green_donation'].values * \
                                (df['three_or_more_non_teacher_referred_donors'].values + \
                                 df['one_non_teacher_referred_donor_giving_100_plus'].values + \
                                 df['donation_from_thoughtful_donor'].values)
            df['is_exciting'].loc[df['is_exciting'] <= 0] = -1
            df['is_exciting'].loc[df['is_exciting'] >= 1] = 0
            df['is_exciting'].loc[df['is_exciting'] == -1] = 1
        else:
            NotImplementedError
        return x_theta, x_inverse, x_cf_hat_inverse, df

    def predict(self, test_x, test_u, thres_n=None):
        self.load_model()
        self.net.eval()
        pd_test = PointDataset(torch.tensor(test_x).float(), torch.tensor(test_u).float())
        test_iter = DataLoader(pd_test, self.batch_size, shuffle=False, worker_init_fn=np.random.seed(42))
        _, lst_pred, x_cf_hat, lst_pred_gt, x_gt, x_changes = self._evaluate(test_iter)
        print('VACA results:')
        print(f'{self.ad_model.name} pred: {(1 - sum(np.array(lst_pred)) / len(lst_pred))}')
        x_cf_hat_inverse = self.data_module.scaler.inverse_transform(x_cf_hat).numpy()
        if self.data == 'loan':
            test_x = self.data_module.scaler.inverse_transform(test_x).numpy()
            df = pd.DataFrame(x_cf_hat_inverse, columns=['G', 'A', 'E', 'L', 'D', 'I', 'S'])
            exp_val = -0.3 * (- df['L'] - df['D'] + df['I'] + df['S'] + df['I'] * df['S'])
            lst_y = []
            for i in range(len(df)):
                lst_y.append((1 + math.exp(exp_val[i])) ** (-1))
            df['y'] = lst_y
            print(f'GT CR: {len(df.loc[df["y"] > 0.9]) / len(df)}')
            assert len(x_changes) == len(df), 'Size error for x_changes'
            df['change_value_real'] = x_changes
            df['change_value_org'] = np.sum((df.iloc[:, :-2].values - test_x) ** 2, axis=1) ** 0.5
            print(f'Delta x Avg Real: {np.mean(df["change_value_real"].values)}')
            print(f'Ab GT Avg Real: {np.mean(df.loc[df["y"] > 0.9]["change_value_real"].values)}')
            print(f'Delta x Avg Org: {np.mean(df["change_value_org"].values)}')
            print(f'Ab GT Avg Org: {np.mean(df.loc[df["y"] > 0.9]["change_value_org"].values)}')
        elif self.data == 'adult':
            test_x = self.data_module.scaler.inverse_transform(F.pad(test_x, (0, 4, 0, 0))).numpy()
            df = pd.DataFrame(np.concatenate((x_cf_hat_inverse, test_u), axis=1), \
                              columns=['R_0', 'R_1', 'R_2', 'A', 'N_0', 'N_1', 'N_2', 'N_3', 'S', 'E', 'H', \
                                       'W_0', 'W_1', 'W_2', 'W_3', 'M_0', 'M_1', 'M_2', 'O_0', 'O_1', 'O_2', \
                                       'L_0', 'L_1', 'L_2', 'I', 'U_R', 'U_A', 'U_N', 'U_S', 'U_E', 'U_H', \
                                       'U_W', 'U_M', 'U_O', 'U_L', 'U_I'])
            df[['R_0', 'R_1', 'R_2', 'N_0', 'N_1', 'N_2', 'N_3', 'S', 'W_0', 'W_1', 'W_2', 'W_3', 'M_0', 'M_1',
                'M_2', 'O_0', 'O_1', 'O_2', 'L_0', 'L_1', 'L_2']] = np.round(df[['R_0', 'R_1', 'R_2', 'N_0', 'N_1',
                                                                                 'N_2', 'N_3', 'S', 'W_0', 'W_1', 'W_2',
                                                                                 'W_3',
                                                                                 'M_0', 'M_1', 'M_2', 'O_0', 'O_1',
                                                                                 'O_2', 'L_0',
                                                                                 'L_1', 'L_2']].values)

            df['I'] = self.data_module.train_dataset.structural_eq['income'](u=df[['U_I']].values,
                                                                             race=df[['R_0', 'R_1', 'R_2']].values,
                                                                             age=df[['A']].values,
                                                                             edu=df[['E']].values,
                                                                             occupation=df[
                                                                                 ['O_0', 'O_1', 'O_2']].values,
                                                                             work_class=df[
                                                                                 ['W_0', 'W_1', 'W_2', 'W_3']].values,
                                                                             maritial=df[['M_0', 'M_1', 'M_2']].values,
                                                                             hour=df[['H']].values,
                                                                             native_country=df[
                                                                                 ['N_0', 'N_1', 'N_2']].values,
                                                                             gender=df[['S']].values,
                                                                             relationship=df[
                                                                                 ['L_0', 'L_1', 'L_2']].values)
            df['y'] = [0 if val <= thres_n else 1 for val in df['I'].values]
            print(f'GT CR: {len(df.loc[df["y"] == 0]) / len(df)}')
            assert len(x_changes) == len(df), 'Size error for x_changes'
            df['change_value_real'] = x_changes
            df['change_value_org'] = np.sum((df.iloc[:, :24].values - test_x[:, :-1]) ** 2, axis=1) ** 0.5
            print(f'Delta x Avg Real: {np.mean(df["change_value_real"].values)}')
            print(f'Ab GT Avg Real: {np.mean(df.loc[df["y"] == 0]["change_value_real"].values)}')
            print(f'Delta x Avg Org: {np.mean(df["change_value_org"].values)}')
            print(f'Ab GT Avg Org: {np.mean(df.loc[df["y"] == 0]["change_value_org"].values)}')
        elif self.data == 'donors':
            test_x = self.data_module.scaler.inverse_transform(F.pad(test_x, (0, 1, 0, 0))).numpy()
            df = pd.DataFrame(x_cf_hat_inverse, columns=['at_least_1_teacher_referred_donor', 'fully_funded',
                                                         'at_least_1_green_donation', 'great_chat',
                                                         'three_or_more_non_teacher_referred_donors',
                                                         'one_non_teacher_referred_donor_giving_100_plus',
                                                         'donation_from_thoughtful_donor', 'great_messages_proportion',
                                                         'teacher_referred_count', 'non_teacher_referred_count',
                                                         'is_exciting'])
            df['is_exciting'] = -1
            df['is_exciting'] = df['fully_funded'].values * df['at_least_1_teacher_referred_donor'].values * \
                                df['great_chat'].values * df['at_least_1_green_donation'].values * \
                                (df['three_or_more_non_teacher_referred_donors'].values + \
                                 df['one_non_teacher_referred_donor_giving_100_plus'].values + \
                                 df['donation_from_thoughtful_donor'].values)
            df['is_exciting'].loc[df['is_exciting'] <= 0] = -1
            df['is_exciting'].loc[df['is_exciting'] >= 1] = 0
            df['is_exciting'].loc[df['is_exciting'] == -1] = 1
            print(f'GT CR: {len(df.loc[df["is_exciting"] == 0]) / len(df)}')
            assert len(x_changes) == len(df), 'Size error for x_changes'
            df['change_value_real'] = x_changes
            df['change_value_org'] = np.sum((df.iloc[:, :-2].values - test_x[:, :-1]) ** 2, axis=1) ** 0.5
            print(f'Delta x Avg Real: {np.mean(df["change_value_real"].values)}')
            print(f'Ab GT Avg Real: {np.mean(df.loc[df["is_exciting"] == 0]["change_value_real"].values)}')
            print(f'Delta x Avg Org: {np.mean(df["change_value_org"].values)}')
            print(f'Ab GT Avg Org: {np.mean(df.loc[df["is_exciting"] == 0]["change_value_org"].values)}')
        else:
            NotImplementedError

        print('-' * 20)
        print('GT results:')
        if self.data == 'loan':
            print(f'{self.ad_model.name} pred w causal: {(1 - sum(np.array(lst_pred_gt)) / len(lst_pred_gt))}')
            x_gt_inverse = self.data_module.scaler.inverse_transform(x_gt).numpy()
            df_gt = pd.DataFrame(x_gt_inverse, columns=['G', 'A', 'E', 'L', 'D', 'I', 'S'])
            exp_val = -0.3 * (- df_gt['L'] - df_gt['D'] + df_gt['I'] + df_gt['S'] + df_gt['I'] * df_gt['S'])
            lst_y = []
            for i in range(len(df_gt)):
                lst_y.append((1 + math.exp(exp_val[i])) ** (-1))
            df_gt['y'] = lst_y
            print(f'GT CR w causal: {len(df_gt.loc[df_gt["y"] > 0.9]) / len(df_gt)}')
            df_gt['change_value'] = np.sum((df_gt.iloc[:, :-1].values - test_x) ** 2, axis=1) ** 0.5
            print(f'GT x Avg w causal: {np.mean(df_gt["change_value"].values)}')
            print(f'Ab GT Avg w causal: {np.mean(df_gt.loc[df_gt["y"] > 0.9]["change_value"].values)}')
        elif self.data == 'adult':
            print(f'{self.ad_model.name} pred w causal: {(1 - sum(np.array(lst_pred_gt)) / len(lst_pred_gt))}')
            x_gt_inverse = self.data_module.scaler.inverse_transform(x_gt).numpy()
            df_gt = pd.DataFrame(np.concatenate((x_gt_inverse, test_u), axis=1), \
                                 columns=['R_0', 'R_1', 'R_2', 'A', 'N_0', 'N_1', 'N_2', 'N_3', 'S', 'E', 'H', \
                                          'W_0', 'W_1', 'W_2', 'W_3', 'M_0', 'M_1', 'M_2', 'O_0', 'O_1', 'O_2', \
                                          'L_0', 'L_1', 'L_2', 'I', 'U_R', 'U_A', 'U_N', 'U_S', 'U_E', 'U_H', \
                                          'U_W', 'U_M', 'U_O', 'U_L', 'U_I'])
            df_gt[['R_0', 'R_1', 'R_2', 'N_0', 'N_1', 'N_2', 'N_3', 'S', 'W_0', 'W_1', 'W_2', 'W_3', 'M_0', 'M_1',
                   'M_2', 'O_0', 'O_1', 'O_2', 'L_0', 'L_1', 'L_2']] = np.round(
                df_gt[['R_0', 'R_1', 'R_2', 'N_0', 'N_1',
                       'N_2', 'N_3', 'S', 'W_0', 'W_1', 'W_2', 'W_3',
                       'M_0', 'M_1', 'M_2', 'O_0', 'O_1', 'O_2', 'L_0',
                       'L_1', 'L_2']].values)
            df_gt['I'] = self.data_module.train_dataset.structural_eq['income'](u=df_gt[['U_I']].values,
                                                                                race=df_gt[
                                                                                    ['R_0', 'R_1', 'R_2']].values,
                                                                                age=df_gt[['A']].values,
                                                                                edu=df_gt[['E']].values,
                                                                                occupation=df_gt[
                                                                                    ['O_0', 'O_1', 'O_2']].values,
                                                                                work_class=df_gt[['W_0', 'W_1', 'W_2',
                                                                                                  'W_3']].values,
                                                                                maritial=df_gt[
                                                                                    ['M_0', 'M_1', 'M_2']].values,
                                                                                hour=df_gt[['H']].values,
                                                                                native_country=df_gt[
                                                                                    ['N_0', 'N_1', 'N_2']].values,
                                                                                gender=df_gt[['S']].values,
                                                                                relationship=df_gt[
                                                                                    ['L_0', 'L_1', 'L_2']].values)
            df_gt['y'] = [0 if val <= thres_n else 1 for val in df_gt['I'].values]
            print(f'GT CR w causal: {len(df_gt.loc[df_gt["y"] == 0]) / len(df_gt)}')
            df_gt['change_value'] = np.sum((df_gt.iloc[:, :24].values - test_x[:, :-1]) ** 2, axis=1) ** 0.5
            print(f'GT x Avg w causal: {np.mean(df_gt["change_value"].values)}')
            print(f'Ab GT Avg w causal: {np.mean(df_gt.loc[df_gt["y"] == 0]["change_value"].values)}')
        elif self.data == 'donors':
            print('No GT results!')
        else:
            NotImplementedError

    def train_ADCAR_RC(self, train_x, train_u, valid_x, valid_u):
        pd_train = PointDataset(torch.tensor(train_x).float(), torch.tensor(train_u).float())
        pd_eval = PointDataset(torch.tensor(valid_x).float(), torch.tensor(valid_u).float())
        train_iter = DataLoader(pd_train, self.batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
        eval_iter = DataLoader(pd_eval, len(pd_eval), shuffle=True, worker_init_fn=np.random.seed(42))

        best_eval_loss = float('inf')
        for _ in tqdm(range(self.max_epoch)):

            train_loss = self._train(train_iter)
            eval_loss, _, _, _, _, _ = self._evaluate(eval_iter)
            # print(f'Training loss: {train_loss}, Evaluation loss {eval_loss}')
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(self.net.state_dict(), f'./saved_models/ADCAR_RC_{self.param_name}.pt')

    def load_model(self):
        self.net.load_state_dict(torch.load(f'./saved_models/ADCAR_RC_{self.param_name}.pt'))
        self.net.to(self.device)
