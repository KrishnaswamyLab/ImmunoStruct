from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from collections import Counter
from .preprocess import preprocess_graphs, preprocess_graph, preprocess_sequence, preprocess_sequence_graph, preprocess_properties, preprocess_hla
from .preprocess import preprocess_properties_cancer_wt, preprocess_sequence_graph_cancer_wt, preprocess_sequence_graph_clinical
from .utils import RandomRotation, duplicate_check, RandomRotation
from .immmunopred_dataloader import AMINO_ACIDS, PADDING_CHAR

__all__ = ["ImmunoPredInferDataset", "ImmunoPredInferDatasetComparative", "ClinicalDataset"]

class ImmunoPredInferDataset(Dataset):
    def __init__(self, config, graph_directory, property_path, hla_path):

        self.config = config
        self.graph_directory = graph_directory
        self.property_path = property_path
        self.hla_path = hla_path

        graphs = preprocess_graphs(graph_directory)
        f_dict, fp2_dict, new_imm_dict, expanded_pep_pair = preprocess_properties(property_path, True if "Cancer" in graph_directory else False)
        name_mapper = preprocess_hla(expanded_pep_pair, hla_path)
        name_mapper, graph_mapper = preprocess_sequence_graph(name_mapper, graphs, new_imm_dict, f_dict)
        graph_mapper = preprocess_graph(graph_mapper, config.feature_size, config.coord_size)
        encoded_full_sequence_map, encoded_peptide_map = preprocess_sequence(name_mapper, AMINO_ACIDS, PADDING_CHAR)

        self.organize(name_mapper, encoded_full_sequence_map, encoded_peptide_map, fp2_dict, new_imm_dict, f_dict, graph_mapper)
        self.normalize()

    def organize(self, name_mapper, encoded_full_sequence_map, encoded_peptide_map, fp2_dict, new_imm_dict, f_dict, graph_mapper):
        names = [(x, a, b, c) for x, (a, b, c) in name_mapper.items()]

        raw_full_sequence = [x[1] for x in names]
        encoded_full_sequence = [encoded_full_sequence_map[x[0]] for x in names]
        encoded_peptide_sequence = [encoded_peptide_map[x[0]] for x in names]

        protein_reg_values = [fp2_dict[x[0]] for x in names]
        protein_immuno_values = [new_imm_dict[x[0]] for x in names]

        class_weights = Counter(protein_immuno_values)
        self.class_weights = class_weights
        print(class_weights)

        protein_reg_values_f = [f_dict[x[0]] for x in names]

        dgl_filtered_graphs = [graph_mapper[x[2]] for x in names]

        duplicate_check(encoded_full_sequence, protein_reg_values, dgl_filtered_graphs)

        self.raw_full_sequence = np.array(raw_full_sequence)
        self.encoded_full_sequence = torch.tensor(np.array(encoded_full_sequence), dtype=torch.float32)
        self.encoded_peptide_sequence = torch.tensor(np.array(encoded_peptide_sequence), dtype=torch.float32)
        self.regression_values = torch.tensor(np.array(protein_reg_values), dtype=torch.float32)
        self.binary_values = torch.tensor(np.array(protein_immuno_values), dtype=torch.float32)
        self.regression_values_f = torch.tensor(np.array(protein_reg_values_f), dtype=torch.float32)

        self.graphs = dgl_filtered_graphs

        print("Preprocess Complete")

    def normalize(self):
        self.min = torch.min(self.regression_values_f)
        self.max = torch.max(self.regression_values_f)
        self.regression_values_f = 2 * (self.regression_values_f - (self.max+self.min)/2)/(self.max-self.min)

    def denormalize(self, output):
        return output / 2 * (self.max - self.min) + (self.max+self.min)/2

    def transform(self):
        return RandomRotation()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.encoded_full_sequence[idx], self.encoded_peptide_sequence[idx], self.regression_values[idx], self.binary_values[idx], self.regression_values_f[idx]

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
        name_mapper_cancer = preprocess_hla(self.combined_df['pep_pair_cancer'], hla_path)
        name_mapper_wt = preprocess_hla(self.combined_df['pep_pair_wt'], hla_path)

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

        names = [(x, a, b, c) for x, (a, b, c) in name_mapper_cancer.items()]

        raw_full_sequence = [x[1] for x in names]

        for _, df_entry in self.combined_df.iterrows():
            pep_pair_cancer = df_entry['pep_pair_cancer']
            pep_pair_wt = df_entry['pep_pair_wt']

            encoded_full_seq_cancer.append(encoded_full_sequence_map_cancer[pep_pair_cancer])
            encoded_peptide_seq_cancer.append(encoded_peptide_map_cancer[pep_pair_cancer])
            peptide_property_cancer.append(tuple(df_entry[['Mprop1', 'Mprop2']].to_numpy().tolist()))
            immunogenicity_cancer.append(df_entry['immunogenicity'])
            foreignness_cancer.append(df_entry['smoothed_foreign'])
            dgl_graphs_cancer.append(graph_mapper_cancer[name_mapper_cancer[pep_pair_cancer][1]])

            encoded_full_seq_wt.append(encoded_full_sequence_map_wt[pep_pair_wt])
            encoded_peptide_seq_wt.append(encoded_peptide_map_wt[pep_pair_wt])
            peptide_property_wt.append(tuple(df_entry[['Mprop1_wt', 'Mprop2_wt']].to_numpy().tolist()))
            immunogenicity_wt.append(0)
            foreignness_wt.append(self.combined_df['smoothed_foreign'].min())
            dgl_graphs_wt.append(graph_mapper_wt[name_mapper_wt[pep_pair_wt][1]])

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

        seq_df = pd.read_table(self.seq_path)

        first_valid_pep_pair = None
        for _, df_entry in seq_df.iterrows():
            pep_pair = df_entry['combo']
            if pep_pair in name_mapper.keys():
                first_valid_pep_pair = pep_pair
                break

        for _, df_entry in seq_df.iterrows():
            pep_pair = df_entry['combo']

            if pep_pair in name_mapper.keys():
                encoded_full_seq.append(encoded_full_sequence_map[pep_pair])
                encoded_peptide_seq.append(encoded_peptide_map[pep_pair])
                # NOTE: currently a hack because these values are not avaliable yet.
                peptide_property.append([0.4, 0.4])
                dgl_graphs.append(graph_mapper[name_mapper[pep_pair][1]])

            else:
                encoded_full_seq.append(np.ones_like(encoded_full_sequence_map[first_valid_pep_pair]) * np.nan)
                encoded_peptide_seq.append(np.ones_like(encoded_peptide_map[first_valid_pep_pair]) * np.nan)
                peptide_property.append(np.ones_like([0.4, 0.4]) * np.nan)
                dgl_graphs.append(graph_mapper[name_mapper[first_valid_pep_pair][1]])  # placeholder

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
