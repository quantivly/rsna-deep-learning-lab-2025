import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


from torch_geometric.data import HeteroData
from rsna_deep_learning_lab_2025.model import PatientRepresentationGNN, graph_view_augment, contrastive_loss
from rsna_deep_learning_lab_2025.data_collection import DataCollection

##### Utility Functions #####

def build_patient_similarity_edges(patient_features, k=10, threshold=None):
    # compute cosine similarity
    sim = F.normalize(patient_features) @ F.normalize(patient_features).T
    sim.fill_diagonal_(0)  # remove self similarity

    if threshold is None:
        # Original k-NN approach: force exactly k connections per patient
        topk = torch.topk(sim, k, dim=1).indices
        row = torch.arange(len(patient_features)).repeat_interleave(k)
        col = topk.reshape(-1)
    else:
        # Threshold approach: only connect patients above similarity threshold
        edge_indices = (sim > threshold).nonzero(as_tuple=False)
        row = edge_indices[:, 0]
        col = edge_indices[:, 1]

    return torch.stack([row, col], dim=0)

def load_data(metadata_path='./data/clinical_metadata_TCGA.csv', radiomic_path='./data/radiomic_features_TCGA.csv', gene_assay_path='./data/multi_gene_assays.csv', threshold=None):
    data_collection = DataCollection(
        metadata_path=metadata_path,
        radiomic_path=radiomic_path,
        gene_assay_path=gene_assay_path,
    )
    
    # Initialize HeteroData object
    data = HeteroData()
    
    # Add patient nodes and features
    patient_features = data_collection.get_patient_metadata_features()
    patient_features = torch.tensor(patient_features, dtype=torch.float)
    pp_edges = build_patient_similarity_edges(patient_features, k=10, threshold=threshold)
    data['patient'].x = patient_features
    data['patient', 'similar', 'patient'].edge_index = pp_edges
    
    # Add radiomic nodes and features
    radiomic_features = data_collection.get_radiomic_features()
    data['radiomic'].x = torch.tensor(radiomic_features, dtype=torch.float)
    
    # Add gene assay nodes and features
    gene_assay_features = data_collection.get_gene_assay_features()
    data['gene'].x = torch.tensor(gene_assay_features, dtype=torch.float)
    
    # Add edges
    rad_src, rad_dst = data_collection.build_radiomic_to_patient_edges
    data['radiomic', 'to', 'patient'].edge_index = torch.tensor([rad_src, rad_dst], dtype=torch.long)
    
    gene_src, gene_dst = data_collection.build_gene_assay_to_patient_edges
    data['gene', 'to', 'patient'].edge_index = torch.tensor([gene_src, gene_dst], dtype=torch.long)
    
    # Add target labels for supervised tasks
    data.y_gene = torch.tensor(data_collection.get_gene_target(), dtype=torch.long)
    data.y = torch.tensor(data_collection.get_target(), dtype=torch.long)
    
    return {
        'data': data,
        'data_collection': data_collection
    }

def load_model_and_optimizer(data, hidden_dim=64, out_dim=32,lr=1e-4):
    model = PatientRepresentationGNN(data, hidden_dim=hidden_dim, out_dim=out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

##### Training Class #####
class Trainer:
    """Class to handle training of the GNN model using contrastive learning."""
    def __init__(self, model, optimizer, data):
        self.model = model
        self.optimizer = optimizer
        self.data = data
        

    def train(self, epochs=20, verbose: bool=True):
        self.model.train()
        losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Create two random augmentations
            view1 = graph_view_augment(self.data['data'])
            view2 = graph_view_augment(self.data['data'])
            
            # Encode both correctly
            z_dict1 = self.model(view1.x_dict, view1.edge_index_dict)
            z_dict2 = self.model(view2.x_dict, view2.edge_index_dict)
            
            z1 = z_dict1['patient']
            z2 = z_dict2['patient']
            
            # Compute contrastive loss
            loss = contrastive_loss(z1, z2)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            
            if verbose:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
                
        print("Training complete.")
        self.model.eval()
        return losses
    
    @property
    def get_patient_embeddings(self):
        self.model.eval()
        with torch.no_grad():
            z_dict = self.model(self.data['data'].x_dict, self.data['data'].edge_index_dict)
        return z_dict['patient']