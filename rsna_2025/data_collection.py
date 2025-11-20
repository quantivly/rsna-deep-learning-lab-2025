import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataCollection:
    """Class for loading and processing clinical metadata, radiomic features, and gene assay data."""
    def __init__(self, metadata_path: str, radiomic_path: str, gene_assay_path: str):
        self.metadata = pd.read_csv(metadata_path)
        self.radiomic = pd.read_csv(radiomic_path)
        self.gene_assay = pd.read_csv(gene_assay_path)
        self.set_radiomic_ids # set patient_id in radiomic dataframe so that it can be matched
        self.unique_ids = self.unique_patient_ids
    
    @property
    def set_radiomic_ids(self):
        ids = ['-'.join(n.split('.')[0].split('-')[:3]) for n in self.radiomic['Lesion Name'].to_list()]
        self.radiomic['patient_id'] = ids

    @property
    def unique_patient_ids(self):
        all_patient_ids = sorted(set(self.metadata['bcr_patient_barcode'].to_list())
                                 | set(self.radiomic['patient_id'].to_list())
                                 | set(self.gene_assay['CLID'].to_list()))
        return all_patient_ids
    
    @property
    def patient_node_mapping(self):
        patient_ids = self.unique_patient_ids
        mapping = {pid: idx for idx, pid in enumerate(patient_ids)}
        return mapping
    
    @property
    def radiomic_node_mapping(self):
        return {i: idx for i, idx in enumerate(self.radiomic.index)}
    
    @property
    def gene_assay_node_mapping(self):
        return {i: idx for i, idx in enumerate(self.gene_assay.index)}
    
    @property
    def build_radiomic_to_patient_edges(self):
        src_nodes = list(self.radiomic_node_mapping.values())
        dst_nodes = [self.patient_node_mapping[pid] for pid in self.radiomic['patient_id'].to_list()]
        return (src_nodes, dst_nodes)
    
    @property
    def build_gene_assay_to_patient_edges(self):
        src_nodes = list(self.gene_assay_node_mapping.values())
        dst_nodes = [self.patient_node_mapping[pid] for pid in self.gene_assay['CLID'].to_list()]
        return (src_nodes, dst_nodes)
    
    def get_radiomic_features(self, columns_to_drop: list=['Lesion Name', 'patient_id']):
        features = self.radiomic.drop(columns=columns_to_drop, errors='ignore')
        scaler = StandardScaler()
        return scaler.fit_transform(features.values)
    
    def get_gene_assay_features(self, columns_to_drop: list=['CLID', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18']):
        feature_cols = [c for c in self.metadata.columns if c not in columns_to_drop]
        numeric_cols = self.metadata[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = list(set(feature_cols) - set(numeric_cols))
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ]
        )
        return preprocessor.fit_transform(self.metadata[feature_cols])
    
    def get_patient_metadata_features(self, columns_to_drop: list=['bcr_patient_barcode', 'patient_id']):
        feature_cols = [c for c in self.metadata.columns if c not in columns_to_drop]
        numeric_cols = self.metadata[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = list(set(feature_cols) - set(numeric_cols))
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ]
        )
        return preprocessor.fit_transform(self.metadata[feature_cols])