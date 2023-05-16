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
                self.input_dim = 40
                deg = self.data_module.get_deg(indegree=True)[:10]
            elif self.data_module.dataset_name == 'loan':
                self.input_dim = 7
                deg = self.data_module.get_deg(indegree=True)[:7]
            elif self.data_module.dataset_name == 'donors':
                self.input_dim = 10
                deg = self.data_module.get_deg(indegree=True)[:10]
            else:
                NotImplementedError
            c_list = [self.input_dim]
            c_list.extend(self.cfg['model']['params']['h_dim_list_dec'])
            c_list.append(1)

            edge_dim = int(sum(deg))

            self.net.add_module(str(ind), PNAModule(c_list=c_list,
                                            deg=deg,
                                            edge_dim=edge_dim,
                                            drop_rate=0.0,
                                            act_name=Cte.RELU,
                                            aggregators=None,
                                            scalers=None,
                                            residual=self.cfg['model']['params']['residual']))

    def forward(self, x_input, inter_features):
        for ind, val in enumerate(inter_features):
            index = self.intervention_features.index(val)
            if ind == 0:
                if self.data_module.dataset_name == 'adult':
                    out = F.pad(self.net[index](x_input), (val*4, self.input_dim-1-(val*4), 0, 0))
                elif self.data_module.dataset_name == 'loan':
                    out = F.pad(self.net[index](x_input), (val, self.input_dim-1-val, 0, 0))
                elif self.data_module.dataset_name == 'donors':
                    out = F.pad(self.net[index](x_input), (val, self.input_dim - 1 - val, 0, 0))
                else:
                    NotImplementedError
            else:
                if self.data_module.dataset_name == 'adult':
                    out += F.pad(self.net[index](x_input), (val*4, self.input_dim-1-(val*4), 0, 0))
                elif self.data_module.dataset_name == 'loan':
                    out += F.pad(self.net[index](x_input), (val, self.input_dim-1-val, 0, 0))
                elif self.data_module.dataset_name == 'donors':
                    out += F.pad(self.net[index](x_input), (val, self.input_dim - 1 - val, 0, 0))
                else:
                    NotImplementedError

        return out