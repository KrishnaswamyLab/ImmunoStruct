from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from collections import Counter
from .preprocess import preprocess_graphs, preprocess_graph, preprocess_sequence, preprocess_sequence_graph, preprocess_properties, preprocess_hla
from .preprocess import preprocess_properties_cancer_wt, preprocess_sequence_graph_cancer_wt, preprocess_sequence_graph_clinical
from .data_utils import RandomRotation, duplicate_check, RandomRotation
from .constants import AMINO_ACIDS, PADDING_CHAR

__all__ = ["ImmunoPredInferDataset", "ImmunoPredInferDatasetComparative", "ClinicalDataset"]

class ImmunoPredInferDataset(Dataset):
    def __init__(self, config, graph_directory, property_path, hla_path):

        self.config = config
        self.graph_directory = graph_directory
        self.property_path = property_path
        self.hla_path = hla_path

        graphs = preprocess_graphs(graph_directory)
        foreignness_dict, biochem_property_dict, immunogenicity_dict, mhc_pep_pair_list = preprocess_properties(property_path, True if "CEDAR" in graph_directory else False)
        name_mapper = preprocess_hla(mhc_pep_pair_list, hla_path)
        name_mapper, graph_mapper = preprocess_sequence_graph(name_mapper, graphs, immunogenicity_dict, foreignness_dict)
        graph_mapper = preprocess_graph(graph_mapper, config.feature_size, config.coord_size)
        encoded_full_sequence_map, encoded_peptide_map = preprocess_sequence(name_mapper, AMINO_ACIDS, PADDING_CHAR)

        self.organize(name_mapper, encoded_full_sequence_map, encoded_peptide_map, biochem_property_dict, immunogenicity_dict, foreignness_dict, graph_mapper)
        self.normalize()

    def organize(self, name_mapper, encoded_full_sequence_map, encoded_peptide_map, biochem_property_dict, immunogenicity_dict, foreignness_dict, graph_mapper):
        key_fullseq_list = [(key, mhc_seq + pep_seq) for key, (mhc_seq, pep_seq) in name_mapper.items()]

        raw_full_sequence = [full_seq for _, full_seq in key_fullseq_list]
        encoded_full_sequence = [encoded_full_sequence_map[key] for key, _ in key_fullseq_list]
        encoded_peptide_sequence = [encoded_peptide_map[key] for key, _ in key_fullseq_list]

        biochem_property_values = [biochem_property_dict[key] for key, _ in key_fullseq_list]
        immunogenicity_values = [immunogenicity_dict[key] for key, _ in key_fullseq_list]

        class_weights = Counter(immunogenicity_values)
        self.class_weights = class_weights

        foreignness_values = [foreignness_dict[key] for key, _ in key_fullseq_list]
        dgl_filtered_graphs = [graph_mapper[key] for key, _ in key_fullseq_list]

        duplicate_check(encoded_full_sequence, biochem_property_values, dgl_filtered_graphs)

        self.raw_full_sequence = np.array(raw_full_sequence)
        self.encoded_full_sequence = torch.tensor(np.array(encoded_full_sequence), dtype=torch.float32)
        self.encoded_peptide_sequence = torch.tensor(np.array(encoded_peptide_sequence), dtype=torch.float32)
        self.biochem_property = torch.tensor(np.array(biochem_property_values), dtype=torch.float32)
        self.immunogenicity = torch.tensor(np.array(immunogenicity_values), dtype=torch.float32)
        self.foreignness = torch.tensor(np.array(foreignness_values), dtype=torch.float32)

        self.graphs = dgl_filtered_graphs

        print("Preprocess Complete")

    def normalize(self):
        self.min = torch.min(self.foreignness)
        self.max = torch.max(self.foreignness)
        self.foreignness = 2 * (self.foreignness - (self.max+self.min)/2)/(self.max-self.min)

    def denormalize(self, output):
        return output / 2 * (self.max - self.min) + (self.max+self.min)/2

    def transform(self):
        return RandomRotation()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return (self.graphs[idx], self.encoded_full_sequence[idx], self.encoded_peptide_sequence[idx],
                self.biochem_property[idx], self.immunogenicity[idx], self.foreignness[idx])

class ImmunoPredInferDatasetComparative(Dataset):
    def __init__(self, config, graph_directory_cancer, graph_directory_wt, property_path_cancer, property_path_wt, hla_path):

        self.config = config
        self.graph_directory_cancer = graph_directory_cancer
        self.graph_directory_wt = graph_directory_wt
        self.property_path_cancer = property_path_cancer
        self.property_path_wt = property_path_wt
        self.hla_path = hla_path

        graphs_cancer = preprocess_graphs(graph_directory_cancer)
        graphs_wt = preprocess_graphs(graph_directory_wt)
        self.combined_df = preprocess_properties_cancer_wt(property_path_cancer, property_path_wt)
        name_mapper_cancer = preprocess_hla(self.combined_df['mhc_pep_pair_cancer'], hla_path)
        name_mapper_wt = preprocess_hla(self.combined_df['mhc_pep_pair_wt'], hla_path)

        self.combined_df, name_mapper_cancer, name_mapper_wt, graph_mapper_cancer, graph_mapper_wt = preprocess_sequence_graph_cancer_wt(
            self.combined_df, name_mapper_cancer, name_mapper_wt, graphs_cancer, graphs_wt)

        graph_mapper_cancer = preprocess_graph(graph_mapper_cancer, config.feature_size, config.coord_size)
        graph_mapper_wt = preprocess_graph(graph_mapper_wt, config.feature_size, config.coord_size)
        encoded_full_sequence_map_cancer, encoded_peptide_map_cancer = preprocess_sequence(name_mapper_cancer, AMINO_ACIDS, PADDING_CHAR)
        encoded_full_sequence_map_wt, encoded_peptide_map_wt = preprocess_sequence(name_mapper_wt, AMINO_ACIDS, PADDING_CHAR)

        self.organize(name_mapper_cancer, name_mapper_wt,
                      encoded_full_sequence_map_cancer, encoded_full_sequence_map_wt,
                      encoded_peptide_map_cancer, encoded_peptide_map_wt,
                      graph_mapper_cancer, graph_mapper_wt)
        self.normalize()

    def organize(self, name_mapper_cancer, name_mapper_wt,
                 encoded_full_sequence_map_cancer, encoded_full_sequence_map_wt,
                 encoded_peptide_map_cancer, encoded_peptide_map_wt,
                 graph_mapper_cancer, graph_mapper_wt):

        encoded_full_seq_cancer, encoded_peptide_seq_cancer = [], []
        peptide_property_cancer, immunogenicity_cancer, foreignness_cancer = [], [], []
        encoded_full_seq_wt, encoded_peptide_seq_wt = [], []
        peptide_property_wt, immunogenicity_wt, foreignness_wt = [], [], []
        dgl_graphs_cancer, dgl_graphs_wt = [], []

        key_fullseq_list = [(key, mhc_seq + pep_seq) for key, (mhc_seq, pep_seq) in name_mapper_cancer.items()]
        raw_full_sequence = [full_seq for _, full_seq in key_fullseq_list]

        for _, df_entry in self.combined_df.iterrows():
            mhc_pep_pair_cancer = df_entry['mhc_pep_pair_cancer']
            mhc_pep_pair_wt = df_entry['mhc_pep_pair_wt']

            encoded_full_seq_cancer.append(encoded_full_sequence_map_cancer[mhc_pep_pair_cancer])
            encoded_peptide_seq_cancer.append(encoded_peptide_map_cancer[mhc_pep_pair_cancer])
            peptide_property_cancer.append(tuple(df_entry[['Mprop1', 'Mprop2']].to_numpy().tolist()))
            immunogenicity_cancer.append(df_entry['immunogenicity'])
            foreignness_cancer.append(df_entry['smoothed_foreign'])
            dgl_graphs_cancer.append(graph_mapper_cancer[mhc_pep_pair_cancer])

            encoded_full_seq_wt.append(encoded_full_sequence_map_wt[mhc_pep_pair_wt])
            encoded_peptide_seq_wt.append(encoded_peptide_map_wt[mhc_pep_pair_wt])
            peptide_property_wt.append(tuple(df_entry[['Mprop1_wt', 'Mprop2_wt']].to_numpy().tolist()))
            immunogenicity_wt.append(0)
            foreignness_wt.append(self.combined_df['smoothed_foreign'].min())
            dgl_graphs_wt.append(graph_mapper_wt[mhc_pep_pair_wt])

        duplicate_check(encoded_full_seq_cancer, peptide_property_cancer, dgl_graphs_cancer)
        duplicate_check(encoded_full_seq_wt, peptide_property_wt, dgl_graphs_wt)

        class_weights = Counter(immunogenicity_cancer)
        self.class_weights = class_weights
        print(class_weights)

        self.raw_full_sequence = np.array(raw_full_sequence)
        self.encoded_full_seq_cancer = torch.tensor(np.array(encoded_full_seq_cancer), dtype=torch.float32)
        self.encoded_full_seq_wt = torch.tensor(np.array(encoded_full_seq_wt), dtype=torch.float32)
        self.encoded_peptide_seq_cancer = torch.tensor(np.array(encoded_peptide_seq_cancer), dtype=torch.float32)
        self.encoded_peptide_seq_wt = torch.tensor(np.array(encoded_peptide_seq_wt), dtype=torch.float32)
        self.peptide_property_cancer = torch.tensor(np.array(peptide_property_cancer), dtype=torch.float32)
        self.peptide_property_wt = torch.tensor(np.array(peptide_property_wt), dtype=torch.float32)
        self.immunogenicity_cancer = torch.tensor(np.array(immunogenicity_cancer), dtype=torch.float32)
        self.immunogenicity_wt = torch.tensor(np.array(immunogenicity_wt), dtype=torch.float32)
        self.foreignness_cancer = torch.tensor(np.array(foreignness_cancer), dtype=torch.float32)
        self.foreignness_wt = torch.tensor(np.array(foreignness_wt), dtype=torch.float32)
        self.graphs_cancer = dgl_graphs_cancer
        self.graphs_wt = dgl_graphs_wt

        print("Preprocess Complete.")

    def normalize(self):
        self.min = torch.min(self.foreignness_cancer)
        self.max = torch.max(self.foreignness_cancer)
        self.foreignness_cancer = 2 * (self.foreignness_cancer - (self.max+self.min)/2)/(self.max-self.min)

    def denormalize(self, output):
        return output / 2 * (self.max - self.min) + (self.max + self.min)/2

    def __len__(self):
        return len(self.graphs_cancer)

    def __getitem__(self, idx):
        return (self.graphs_cancer[idx], self.graphs_wt[idx]), \
               (self.encoded_full_seq_cancer[idx], self.encoded_full_seq_wt[idx]), \
               (self.encoded_peptide_seq_cancer[idx], self.encoded_peptide_seq_wt[idx]), \
               (self.peptide_property_cancer[idx], self.peptide_property_wt[idx]), \
               self.immunogenicity_cancer[idx], \
               self.foreignness_cancer[idx]

class ClinicalDataset(Dataset):
    def __init__(self, config, graph_directory, seq_path):

        self.config = config
        self.graph_directory = graph_directory
        self.seq_path = seq_path

        name_mapper, graph_mapper = preprocess_sequence_graph_clinical(graph_directory, seq_path)
        graph_mapper = preprocess_graph(graph_mapper, config.feature_size, config.coord_size)
        encoded_full_sequence_map, encoded_peptide_map = preprocess_sequence(name_mapper, AMINO_ACIDS, PADDING_CHAR)

        self.organize(name_mapper, encoded_full_sequence_map, encoded_peptide_map, graph_mapper)

    def organize(self, name_mapper, encoded_full_sequence_map, encoded_peptide_map, graph_mapper):
        encoded_full_seq, encoded_peptide_seq, peptide_property, dgl_graphs = [], [], [], []

        seq_df = pd.read_csv(self.seq_path)

        first_valid_pep_pair = None
        for _, df_entry in seq_df.iterrows():
            mhc_pep_pair = df_entry['allele'] + "_" + df_entry['mut_pep']
            if mhc_pep_pair in name_mapper.keys():
                first_valid_pep_pair = mhc_pep_pair
                break

        for _, df_entry in seq_df.iterrows():
            mhc_pep_pair = df_entry['allele'] + "_" + df_entry['mut_pep']

            if mhc_pep_pair in name_mapper.keys():
                encoded_full_seq.append(encoded_full_sequence_map[mhc_pep_pair])
                encoded_peptide_seq.append(encoded_peptide_map[mhc_pep_pair])
                # NOTE: currently a hack because these values are not avaliable yet.
                peptide_property.append([0.4, 0.4])
                dgl_graphs.append(graph_mapper[mhc_pep_pair])

            else:
                # Put a NaN placeholder for sequence, structure and property.
                encoded_full_seq.append(np.ones_like(encoded_full_sequence_map[first_valid_pep_pair]) * np.nan)
                encoded_peptide_seq.append(np.ones_like(encoded_peptide_map[first_valid_pep_pair]) * np.nan)
                peptide_property.append(np.ones_like([0.4, 0.4]) * np.nan)
                dgl_graphs.append(graph_mapper[first_valid_pep_pair])

        self.class_weights = 0.5

        self.encoded_full_seq = torch.tensor(np.array(encoded_full_seq), dtype=torch.float32)
        self.encoded_peptide_seq = torch.tensor(np.array(encoded_peptide_seq), dtype=torch.float32)
        self.peptide_property = torch.tensor(np.array(peptide_property), dtype=torch.float32)
        self.graphs = dgl_graphs

        self.y_placeholder = torch.tensor([-1], dtype=torch.float32)

        print("Preprocess Complete.")

    def transform(self):
        return RandomRotation()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.encoded_full_seq[idx], self.encoded_peptide_seq[idx], self.peptide_property[idx], self.y_placeholder, self.y_placeholder
