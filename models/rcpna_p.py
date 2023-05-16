import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from modules.pna import PNAModule
from utils.constants import Cte
from torch_geometric.utils import dense_to_sparse

class RCPNA(nn.Module):
    def __init__(self, cfg, data_module, intervention_features):

        super().__init__()

        self.cfg = cfg
        self.data_module = data_module
        self.intervention_features = intervention_features
        self._crate_nets()

    def _crate_nets(self):
        self.net = nn.Module()
        self.edge_attr = []

        for ind, val in enumerate(self.intervention_features):
            if self.data_module.dataset_name == 'adult':
                dag = self.data_module.train_dataset.dag[:-1,:-1]
                input_dim_mpl = 4
                deg_list = self.data_module.get_deg(indegree=True)[:10]
            elif self.data_module.dataset_name == 'loan':
                dag = self.data_module.train_dataset.dag
                input_dim_mpl = 1
                deg_list = self.data_module.get_deg(indegree=True)[:7]
            elif self.data_module.dataset_name == 'donors':
                dag = self.data_module.train_dataset.dag
                input_dim_mpl = 1
                deg_list = self.data_module.get_deg(indegree=True)[:10]
            else:
                NotImplementedError
            c_list = [int(input_dim_mpl*sum(dag[val]))]
            c_list.extend(self.cfg['model']['params']['h_dim_list_dec'])
            c_list.append(int(input_dim_mpl*1))

            deg = deg_list[dag[val].index(1)]
            print(deg)
            edge_dim = sum(deg)
            print(edge_dim)

            self.edge_attr.append(self.data_module.train_dataset.adj_object.edge_attr[:val+1, :val+1])
            self.edge_index, _ = dense_to_sparse(torch.tensor(self.adj_matrix))
            self.edge_attr = torch.eye(edge_dim, edge_dim)
            self.net.add_module(str(val), PNAModule(c_list=c_list,
                                            deg=deg,
                                            edge_dim=edge_dim,
                                            drop_rate=0.0,
                                            act_name=Cte.RELU,
                                            aggregators=None,
                                            scalers=None,
                                            residual=self.cfg['model']['params']['residual']))
