import torch
import torch.nn.functional as F
import numpy as np
import plotly.express as px

from torch_geometric.data import HeteroData
from rsna_2025.data_collection import DataCollection

class DataViewer:
    """Utility class for analyzing and visualizing the heterogeneous graph data and raw features."""
    def __init__(self, dataset: HeteroData, original_data: DataCollection):
        self.dataset = dataset
        self.original_data = original_data

    @property
    def fetch_connected_gene_features(self):
        G_list = []
        patient_idx_with_genes = []

        num_patients = self.dataset['patient'].num_nodes

        edge_index = self.dataset['gene', 'to', 'patient'].edge_index  # shape [2, num_edges]

        for patient_id in range(num_patients):
            # get gene nodes connected to this patient
            gene_nodes = edge_index[0, edge_index[1] == patient_id]

            if len(gene_nodes) == 0:
                continue  # skip patients with no genes

            gene_feats = self.dataset['gene'].x[gene_nodes]
            G_list.append(gene_feats.mean(dim=0))
            patient_idx_with_genes.append(patient_id)

        # stack into matrix
        G = torch.stack(G_list)  # shape: num_valid_patients x gene_dim

        # patient_idx_with_genes maps G rows back to the original patient indices
        patient_idx_with_genes = torch.tensor(patient_idx_with_genes)
        return {
            'G': G,
            'patient_idx_with_genes': patient_idx_with_genes
        }

    @property
    def fetch_connected_radiomic_features(self):
        R_list = []
        patient_idx_with_radiomics = []

        num_patients = self.dataset['patient'].num_nodes

        edge_index = self.dataset['radiomic', 'to', 'patient'].edge_index  # shape [2, num_edges]

        for patient_id in range(num_patients):
            # get radiomic nodes connected to this patient
            radiomic_nodes = edge_index[0, edge_index[1] == patient_id]

            if len(radiomic_nodes) == 0:
                continue  # skip patients with no radiomics

            radiomic_feats = self.dataset['radiomic'].x[radiomic_nodes]
            R_list.append(radiomic_feats.mean(dim=0))
            patient_idx_with_radiomics.append(patient_id)

        # stack into matrix
        R = torch.stack(R_list)  # shape: num_valid_patients x radiomic_dim

        # patient_idx_with_radiomics maps R rows back to the original patient indices
        patient_idx_with_radiomics = torch.tensor(patient_idx_with_radiomics)
        return {
            'R': R,
            'patient_idx_with_radiomics': patient_idx_with_radiomics
        }
    
    @property
    def fetch_all_radiomic_features(self):
        return self.dataset['radiomic'].x 
    
    @property
    def fetch_all_gene_features(self):
        return self.dataset['gene'].x
    
    @property
    def fetch_all_patient_features(self):
        return self.dataset['patient'].x
    
    def fetch_cosine_similarity(self, graph):
        graph_norm = torch.nn.functional.normalize(graph)
        Sim_graph = (graph_norm @ graph_norm.T).cpu().numpy()
        return Sim_graph
    

    

    
