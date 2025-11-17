import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv

class PatientRepresentationGNN(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = HeteroConv({
            ('radiomic', 'to', 'patient'): GATConv(-1, hidden_dim),
            ('gene', 'to', 'patient'): GATConv(-1, hidden_dim),
            ('patient', 'to', 'patient'): GATConv(-1, hidden_dim)
        }, aggr='sum')

        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        x_dict['patient'] = self.lin(x_dict['patient'])
        return x_dict['patient'] 

def graph_view_augment(data, drop_edge_rate=0.2, mask_feature_rate=0.1):
    """Randomly drop edges and mask node features to create an augmented view."""
    from copy import deepcopy
    view = deepcopy(data)
    
    # Drop random patient-gene edges
    e = view['gene', 'to', 'patient'].edge_index
    mask = torch.rand(e.size(1)) > drop_edge_rate
    view['gene', 'to', 'patient'].edge_index = e[:, mask]
    
    # Drop random radiomic-patient edges
    e = view['radiomic', 'to', 'patient'].edge_index
    mask = torch.rand(e.size(1)) > drop_edge_rate
    view['radiomic', 'to', 'patient'].edge_index = e[:, mask]
    
    # Mask patient features (if any)
    mask = torch.rand_like(view['patient'].x) < mask_feature_rate
    view['patient'].x = view['patient'].x * (~mask)
    
    return view

def contrastive_loss(z1, z2, temperature=0.5):
    """z1, z2: embeddings from two graph views"""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim_matrix = torch.mm(z1, z2.T) / temperature
    
    pos = torch.diag(sim_matrix)
    loss = -torch.log(torch.exp(pos) / torch.exp(sim_matrix).sum(dim=1))
    return loss.mean()