import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv

class PatientRepresentationGNN(nn.Module):
    def __init__(self, data, hidden_dim=64, out_dim=32):
        super().__init__()

        # --- Infer feature dims from graph ---
        patient_in = data['patient'].x.size(1)
        radiomic_in = data['radiomic'].x.size(1)
        gene_in = data['gene'].x.size(1)

        # --- Build HeteroConv with correct per-edge dims ---
        self.conv1 = HeteroConv({
            ('radiomic', 'to', 'patient'):
                GATConv((radiomic_in, patient_in), hidden_dim, add_self_loops=False),

            ('gene', 'to', 'patient'):
                GATConv((gene_in, patient_in), hidden_dim, add_self_loops=False),

            ('patient', 'similar', 'patient'):
                GATConv((patient_in, patient_in), hidden_dim, add_self_loops=False),
        }, aggr='sum')

        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        x_dict['patient'] = self.lin(x_dict['patient'])
        return x_dict

def graph_view_augment(data, drop_edge_rate=0.05):
    from copy import deepcopy
    view = deepcopy(data)

    # drop gene→patient
    e = view['gene', 'to', 'patient'].edge_index
    mask = torch.rand(e.size(1)) > drop_edge_rate
    view['gene', 'to', 'patient'].edge_index = e[:, mask]

    # drop radiomic→patient
    e = view['radiomic', 'to', 'patient'].edge_index
    mask = torch.rand(e.size(1)) > drop_edge_rate
    view['radiomic', 'to', 'patient'].edge_index = e[:, mask]

    # drop patient–patient similarity edges
    e = view['patient', 'similar', 'patient'].edge_index
    mask = torch.rand(e.size(1)) > drop_edge_rate
    view['patient', 'similar', 'patient'].edge_index = e[:, mask]

    return view

def contrastive_loss(z1, z2, temperature=0.5):
    """z1, z2: embeddings from two graph views"""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim_matrix = torch.mm(z1, z2.T) / temperature
    
    pos = torch.diag(sim_matrix)
    loss = -torch.log(torch.exp(pos) / torch.exp(sim_matrix).sum(dim=1))
    return loss.mean()