import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score


class PointDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx]


class AutoEncoder(object):

    def __init__(self, input_dim, hid_dim, batch_size=1024, max_epoch=50, nu=0.05, device='cuda:0', data='loan'):
        super().__init__()

        self.nu = nu
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.device = device
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.data = data
        self.name = 'autoencoder'

        self._set_net()
        self.optim = optim.Adam(self.net.parameters(), weight_decay=0.5e-3)
        self.loss_mse = nn.MSELoss()

    def _set_net(self):
        if self.data == 'adult':
            self.net = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.hid_dim),
                nn.Linear(self.hid_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.input_dim)
            ).to(self.device)
        else:
            self.net = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.input_dim, 500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Linear(500, 2000),
                nn.ReLU(),
                nn.Linear(2000, self.hid_dim),
                nn.Linear(self.hid_dim, 2000),
                nn.ReLU(),
                nn.Linear(2000, 500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Linear(500, self.input_dim)
            ).to(self.device)

    def _train(self, iterator):
        self.net.train()

        epoch_loss = 0

        for (i, batch) in enumerate(iterator):
            src = batch.to(self.device)
            self.optim.zero_grad()
            output = self.net(src)
            loss = self.loss_mse(src, output)
            loss.backward()

            self.optim.step()
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def _evaluate(self, iterator):

        self.net.eval()

        epoch_loss = 0

        lst_dist = []

        with torch.no_grad():
            for (i, batch) in enumerate(iterator):
                src = batch.to(self.device)
                output = self.net(src)
                loss = self.loss_mse(src, output)
                epoch_loss += loss.item()
                dist = ((output - src) ** 2).sum(axis=1)
                lst_dist.extend(dist.detach().cpu().numpy())

        return epoch_loss / len(iterator), lst_dist

    def train_AutoEncoder(self, train_x, valid_x):
        pd_train = PointDataset(torch.tensor(train_x).float())
        pd_eval = PointDataset(torch.tensor(valid_x).float())
        train_iter = DataLoader(pd_train, self.batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
        eval_iter = DataLoader(pd_eval, self.batch_size, shuffle=True, worker_init_fn=np.random.seed(42))

        best_eval_loss = float('inf')

        for _ in tqdm(range(self.max_epoch)):
            train_loss = self._train(train_iter)
            # print(f'Training loss: {train_loss}')
            eval_loss, lst_dist = self._evaluate(eval_iter)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(self.net.state_dict(), f'./saved_models/AutoEncoder_{self.data}.pt')

    def load_model(self):
        self.net.load_state_dict(torch.load(f'./saved_models/AutoEncoder_{self.data}.pt', map_location=self.device))
        self.net.to(self.device)

    def get_R(self, train_x, nu=None):
        pd = PointDataset(torch.tensor(train_x).float())
        train_iter = DataLoader(pd, self.batch_size, shuffle=False)
        _, lst_dist = self._evaluate(train_iter)
        if nu == None:
            self.R = np.quantile(np.array(lst_dist), 1 - self.nu)
        else:
            self.R = np.quantile(np.array(lst_dist), 1 - nu)

    def predict(self, test_x, label, result=0):
        self.load_model()
        self.net.eval()
        pd = PointDataset(torch.tensor(test_x).float())
        test_iter = DataLoader(pd, self.batch_size, shuffle=False)
        _, lst_dist = self._evaluate(test_iter)
        pred = [0 if i <= self.R else 1 for i in lst_dist]
        if result == 1:
            print(classification_report(y_true=label, y_pred=pred, digits=5))
            print(confusion_matrix(y_true=label, y_pred=pred))
            print(f'AUC ROC: {roc_auc_score(y_true=label, y_score=lst_dist)}')
            print(F'AUC PR: {average_precision_score(y_true=label, y_score=lst_dist)}')
        return lst_dist, pred

    def get_score(self, x):
        self.load_model()
        self.net.eval()
        pd = PointDataset(torch.tensor([x]).float())
        test_iter = DataLoader(pd, self.batch_size, shuffle=False)
        _, lst_dist = self._evaluate(test_iter)
        return lst_dist
