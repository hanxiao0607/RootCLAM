import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from modules.pna import PNAModule
from utils.constants import Cte
import numpy as np
from utils.activations import get_activation
from utils.likelihoods import get_likelihood
from utils.probabilistic_model import ProbabilisticModelSCM
from torch_geometric.utils import dense_to_sparse

class RCPNA(nn.Module):
    def __init__(self, cfg, data_module, intervention_features, device='cuda:0'):

        super().__init__()

        self.cfg = cfg
        self.data_module = data_module
        self.device = device
        self.intervention_features = intervention_features
        self.dim_input_enc = self.cfg['model']['params']['h_dim_list_enc'][0]
        self.act_name = Cte.RELU
        self.drop_rate = 0.0

        self._encoder_embeddings = nn.ModuleList()
        self.likelihoods_x = data_module.likelihood_list

        self.num_nodes = len(self.likelihoods_x)

        self.prob_model_x = ProbabilisticModelSCM(likelihoods=self.likelihoods_x,
                                             embedding_size=self.cfg['model']['params']['h_dim_list_dec'][-1],
                                             act_name=self.act_name,
                                             drop_rate=self.drop_rate,
                                             norm_categorical=self.cfg['model']['params']['norm_categorical'],
                                             norm_by_dim=False)
        self.node_dim_max = max(self.prob_model_x.node_dim_list)

        self.x0_size = self.num_nodes * self.node_dim_max
        for lik_i in self.likelihoods_x:
            x_dim_i = np.sum([lik_ij.domain_size for lik_ij in lik_i])
            if x_dim_i > 2 * self.dim_input_enc:
                embed_i = nn.Sequential(nn.Linear(x_dim_i, 2 * self.dim_input_enc, bias=True),
                                        get_activation(self.act_name),
                                        nn.Dropout(self.drop_rate),
                                        nn.Linear(2 * self.dim_input_enc, self.dim_input_enc, bias=True),
                                        get_activation(self.act_name),
                                        nn.Dropout(self.drop_rate))
            else:
                embed_i = nn.Sequential(nn.Linear(x_dim_i, self.dim_input_enc, bias=True),
                                        get_activation(self.act_name),
                                        nn.Dropout(self.drop_rate))
            self._encoder_embeddings.append(embed_i)
        if self.data_module.dataset_name == 'adult':
            self.input_dim = 11
        elif self.data_module.dataset_name == 'loan':
            self.input_dim = 7
        elif self.data_module.dataset_name == 'donors':
            self.input_dim = 11
        else:
            NotImplementedError

        self._crate_nets()


    def _crate_nets(self):
        self.net = nn.ModuleDict()
        self.edge_attr = []

        for ind, val in enumerate(self.intervention_features):
            c_list = []
            c_list.extend(self.cfg['model']['params']['h_dim_list_enc'])
            c_list.append(1)

            deg = self.data_module.get_deg(indegree=True)
            edge_dim = self.data_module.edge_dimension

            self.net.add_module('pna'+str(ind), PNAModule(c_list=c_list,
                                            deg=deg,
                                            edge_dim=edge_dim,
                                            drop_rate=self.drop_rate,
                                            act_name=self.act_name,
                                            aggregators=None,
                                            scalers=None,
                                            residual=self.cfg['model']['params']['residual']))
            self.net.add_module('fc'+str(ind), nn.Linear(self.input_dim, 1))

    def encoder_embeddings(self, X):

        X_0 = X.view(-1, self.x0_size)
        embeddings = []
        for i, embed_i in enumerate(self._encoder_embeddings):
            X_0_i = X_0[:, (i * self.node_dim_max):((i + 1) * self.node_dim_max)]
            H_i = embed_i(X_0_i[:, :self.prob_model_x.node_dim_list[i]])
            embeddings.append(H_i)
        return torch.cat(embeddings, dim=1).view(-1, self.dim_input_enc)


    def forward(self, x_input, inter_features):
        for ind, val in enumerate(inter_features):
            pna_name = 'pna' + str(self.intervention_features.index(val))
            fc_name = 'fc' + str(self.intervention_features.index(val))
            x_emb = self.encoder_embeddings(x_input)
            if self.data_module.dataset_name == 'adult':
                pna_out = self.net[pna_name](x_emb, self.data_module.train_dataset[0].edge_index.to(self.device),
                                             edge_attr=self.data_module.train_dataset[0].edge_attr.to(self.device),
                                             return_mean=True, get_prob=True,
                                             node_ids=self.data_module.train_dataset[0].node_ids.to(self.device))
            elif self.data_module.dataset_name == 'loan':
                pna_out = self.net[pna_name](x_emb, self.data_module.train_dataset[0].edge_index.to(self.device),
                                             edge_attr=self.data_module.train_dataset[0].edge_attr.to(self.device),
                                             return_mean=True, get_prob=True,
                                             node_ids=self.data_module.train_dataset[0].node_ids.to(self.device))
            elif self.data_module.dataset_name == 'donors':
                pna_out = self.net[pna_name](x_emb, self.data_module.train_dataset[0].edge_index.to(self.device),
                                             edge_attr=self.data_module.train_dataset[0].edge_attr.to(self.device),
                                             return_mean=True, get_prob=True,
                                             node_ids=self.data_module.train_dataset[0].node_ids.to(self.device))
            else:
                NotImplementedError

            f_ind = self.intervention_features.index(val)
            b_ind = len(self.intervention_features)
            if ind == 0:
                out = F.pad(self.net[fc_name](pna_out.reshape(1,-1)), (f_ind, b_ind-1-f_ind, 0, 0))
            else:
                out += F.pad(self.net[fc_name](pna_out.reshape(1,-1)), (f_ind, b_ind-1-f_ind, 0, 0))
        try:
            return out
        except:
            print(inter_features)
            print(pna_out)