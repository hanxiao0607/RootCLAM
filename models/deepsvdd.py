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


def get_center(emb):
    return torch.mean(emb, 0)


class PointDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx]


class DeepSVDD(object):

    def __init__(self, input_dim, out_dim, batch_size=1024, eps=0.1, max_epoch=50, nu=0.05, device='cuda:0',
                 data='loan'):
        super().__init__()

        self.nu = nu
        self.R = 0.0
        self.c = None
        self.eps = eps
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.device = device
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.data = data
        self.name = 'deepsvdd'

        self._set_net()
        self.optim = optim.Adam(self.net.parameters(), weight_decay=0.5e-3)
        self.loss_mse = nn.MSELoss()

    def _set_net(self):
        if self.data == 'donors':
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, self.out_dim, bias=False)
            ).to(self.device)
        else:
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, 16, bias=False),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(16, self.out_dim, bias=False)

            ).to(self.device)

    def _train(self, iterator, center):
        self.net.train()

        epoch_loss = 0

        for (i, batch) in enumerate(iterator):
            src = batch.to(self.device)
            self.optim.zero_grad()
            output = self.net(src)
            center = center.to(self.device)
            loss = self.loss_mse(output, center)
            loss.backward()

            self.optim.step()
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def _evaluate(self, iterator, center, epoch):

        self.net.eval()

        epoch_loss = 0

        lst_dist = []

        with torch.no_grad():
            for (i, batch) in enumerate(iterator):
                src = batch.to(self.device)
                output = self.net(src)
                if i == 0:
                    lst_emb = output
                else:
                    lst_emb = torch.cat((lst_emb, output), dim=0)
                center = center.to(self.device)
                loss = self.loss_mse(output, center)
                epoch_loss += loss.item()
                lst_dist.extend(torch.cdist(output, center.view(1, -1), p=2).detach().cpu().numpy().flatten().tolist())

        if epoch < 10:
            center = get_center(lst_emb)
            center[(abs(center) < self.eps) & (center < 0)] = -self.eps
            center[(abs(center) < self.eps) & (center > 0)] = self.eps

        return epoch_loss / len(iterator), center, lst_dist

    def train_DeepSVDD(self, train_x, valid_x):
        pd_train = PointDataset(torch.tensor(train_x).float())
        pd_eval = PointDataset(torch.tensor(valid_x).float())
        train_iter = DataLoader(pd_train, self.batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
        eval_iter = DataLoader(pd_eval, self.batch_size, shuffle=True, worker_init_fn=np.random.seed(42))

        best_eval_loss = float('inf')

        for epoch in tqdm(range(self.max_epoch)):
            if epoch == 0:
                center = torch.Tensor([0.0 for _ in range(self.out_dim)])
            if epoch > 9:
                center = fixed_center
            train_loss = self._train(train_iter, center)
            # print(f'Training loss: {train_loss}')
            eval_loss, center, lst_dist = self._evaluate(eval_iter, center, epoch)

            if epoch == 9:
                fixed_center = center

            if eval_loss < best_eval_loss and epoch >= 9:
                best_eval_loss = eval_loss
                torch.save(self.net.state_dict(), f'./saved_models/DeepSVDD_{self.data}.pt')
                self.c = fixed_center.cpu()
                pd.DataFrame(fixed_center.cpu().numpy()).to_csv(f'./saved_models/DeepSVDD_center_{self.data}.csv')

    def load_model(self):
        self.net.load_state_dict(torch.load(f'./saved_models/DeepSVDD_{self.data}.pt', map_location=self.device))
        self.net.to(self.device)
        self.c = torch.Tensor(
            pd.read_csv(f'./saved_models/DeepSVDD_center_{self.data}.csv', index_col=0).iloc[:, 0])

    def get_R(self, train_x, nu=None):
        pd = PointDataset(torch.tensor(train_x).float())
        train_iter = DataLoader(pd, self.batch_size, shuffle=False)
        _, _, lst_dist = self._evaluate(train_iter, self.c, 20)
        if nu == None:
            self.R = np.quantile(np.array(lst_dist), 1 - self.nu)
        else:
            self.R = np.quantile(np.array(lst_dist), 1 - nu)

    def predict(self, test_x, label, result=0):
        self.load_model()
        self.net.eval()
        pd = PointDataset(torch.tensor(test_x).float())
        test_iter = DataLoader(pd, self.batch_size, shuffle=False)
        _, _, lst_dist = self._evaluate(test_iter, self.c, 20)
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
        _, _, lst_dist = self._evaluate(test_iter, self.c, 20)
        return lst_dist[0]
