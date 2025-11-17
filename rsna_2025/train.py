from rsna_2025.model import PatientRepresentationGNN, graph_view_augment, contrastive_loss
from rsna_2025.data_collection import DicomDataCollection
import torch

def load_data():
    data_collection = DicomDataCollection(
        metadata_path='data/metadata.csv',
        radiomic_path='data/radiomic.csv',
        gene_assay_path='data/gene_assay.csv'
    )
    
    from torch_geometric.data import HeteroData
    data = HeteroData()

    patient_features = data_collection.get_patient_metadata_features()
    data['patient'].x = torch.tensor(patient_features, dtype=torch.float)

    
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
    
    return data

def load_model_and_optimizer():
    model = PatientRepresentationGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer

def train(epochs=20):
    data = load_data()
    model, optimizer = load_model_and_optimizer()

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Create two random augmentations
        view1 = graph_view_augment(data)
        view2 = graph_view_augment(data)
        
        # Encode both correctly
        z_dict1 = model(view1.x_dict, view1.edge_index_dict)
        z_dict2 = model(view2.x_dict, view2.edge_index_dict)
        
        z1 = z_dict1['patient']
        z2 = z_dict2['patient']
        
        # Compute contrastive loss
        loss = contrastive_loss(z1, z2)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
    print("Training complete.")
    
    return {
        'data':data, 
        'model':model,
        }
