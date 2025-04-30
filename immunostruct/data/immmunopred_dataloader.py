import os
from torch.utils.data import Dataset
import random
import torch
import argparse
import numpy as np
from .preprocess import preprocess_graphs, preprocess_graph, preprocess_sequence, preprocess_sequence_graph, preprocess_properties, preprocess_hla
from .preprocess import preprocess_properties_cancer_wt, preprocess_sequence_graph_cancer_wt
from .utils import duplicate_check, RandomRotation, one_hot_encode_sequence
from collections import Counter

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
PADDING_CHAR = 'J'

__all__ = ["ImmunoPredDataset", "ImmunoPredDatasetComparative"]

class ImmunoPredDataset(Dataset):
    def __init__(self, config, graph_directory, property_path, hla_path):

        self.config = config
        self.graph_directory = graph_directory
        self.property_path = property_path
        self.hla_path = hla_path

        self.sequence_pad_count = config.sequence_pad_count
        self.structure_pad_count = config.structure_pad_count

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
    
    def mask_sequence(self, full, peptide, padding_char):
        length = len(full)- len(peptide)

        inds = [i for i in range(length)]
        to_mask = random.sample(inds, self.sequence_pad_count)

        pad_one_hot = torch.tensor(np.array(one_hot_encode_sequence(padding_char, AMINO_ACIDS, PADDING_CHAR)), dtype=torch.float32)

        for i in to_mask:
            full[i] = pad_one_hot

        return full

    # perform this after self supervision on single structure
    def mask_structure(self, graph):
        inds = [i for i in range(len(graph.ndata['x']))]
        to_mask = random.sample(inds, self.structure_pad_count)

        for i in to_mask:
            if torch.sum(graph.ndata['x'][i,:-3]) > 1: # self supervision structure
                continue
            else:
                graph.ndata['x'][i,:-3] = torch.full(graph.ndata['x'][i,:-3].shape, 0)

        return graph

    def mask_single_structure(self, graph):
        inds = [i for i in range(len(graph.ndata['x']))]

        for _ in inds: # loop until we find a valid, non padded amino acid
            to_mask = random.choice(inds)
            amino_acid = torch.nonzero(graph.ndata['x'][to_mask,:-3], as_tuple=True)[0]
            if amino_acid.numel():
                graph.ndata['x'][to_mask,:-3] = torch.full(graph.ndata['x'][to_mask,:-3].shape, 1)
                return graph, amino_acid

        print("unmaskable graph: " , graph.ndata['x'])
        return graph, torch.tensor([0])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.encoded_full_sequence[idx], self.encoded_peptide_sequence[idx], self.regression_values[idx], self.binary_values[idx], self.regression_values_f[idx]


class ImmunoPredDatasetComparative(Dataset):
    def __init__(self, config, graph_directory_cancer, graph_directory_wt, property_path_cancer, property_path_wt, hla_path):

        self.config = config
        self.graph_directory_cancer = graph_directory_cancer
        self.graph_directory_wt = graph_directory_wt
        self.property_path_cancer = property_path_cancer
        self.property_path_wt = property_path_wt
        self.hla_path = hla_path

        self.sequence_pad_count = config.sequence_pad_count
        self.structure_pad_count = config.structure_pad_count

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
    
    def mask_sequence(self, full, full_wt, peptide, peptide_wt, padding_char):
        assert len(full) == len(full_wt)
        assert len(peptide) == len(peptide_wt)

        length = len(full)- len(peptide)

        inds = [i for i in range(length)]
        to_mask = random.sample(inds, self.sequence_pad_count)

        pad_one_hot = torch.tensor(np.array(one_hot_encode_sequence(padding_char, AMINO_ACIDS, PADDING_CHAR)), dtype=torch.float32)

        for i in to_mask:
            full[i] = pad_one_hot
            full_wt[i] = pad_one_hot

        return full, full_wt

    # perform this after self supervision on single structure
    def mask_structure(self, graph, graph_wt):

        inds = [i for i in range(len(graph.ndata['x']))]
        inds_wt = [i for i in range(len(graph_wt.ndata['x']))]

        for i in random.sample(inds, self.structure_pad_count):
            if torch.sum(graph.ndata['x'][i,:-3]) > 1: # self supervision structure
                continue
            else:
                graph.ndata['x'][i,:-3] = torch.full(graph.ndata['x'][i,:-3].shape, 0)

        for i in random.sample(inds_wt, self.structure_pad_count):
            if torch.sum(graph_wt.ndata['x'][i,:-3]) > 1: # self supervision structure
                continue
            else:      
                graph_wt.ndata['x'][i,:-3] = torch.full(graph_wt.ndata['x'][i,:-3].shape, 0)

        return graph, graph_wt

    def mask_single_structure(self, graph, graph_wt):
        inds = [i for i in range(len(graph.ndata['x']))]
        random.shuffle(inds)
        inds_wt = [i for i in range(len(graph_wt.ndata['x']))]
        random.shuffle(inds_wt)

        for to_mask in inds: # loop until we find a valid, non padded amino acid
            amino_acid = torch.nonzero(graph.ndata['x'][to_mask,:-3], as_tuple=True)[0]
            if amino_acid.numel():
                # find the same amino_acid in graph_wt
                for to_mask_wt in inds_wt:
                    amino_wt = torch.nonzero(graph_wt.ndata['x'][to_mask_wt,:-3], as_tuple=True)[0]
                    if amino_wt.numel() and amino_wt == amino_acid:
                        graph.ndata['x'][to_mask,:-3] = torch.full(graph.ndata['x'][to_mask,:-3].shape, 1)
                        graph_wt.ndata['x'][to_mask_wt,:-3] = torch.full(graph_wt.ndata['x'][to_mask_wt,:-3].shape, 1)
                        return graph, graph_wt, amino_acid

        print("unmaskable graph: " , graph.ndata['x'], graph_wt.ndata['x'])
        return graph, graph_wt, torch.tensor([0])

    def transform(self):
        return RandomRotation()

    def __len__(self):
        return len(self.graphs_cancer)

    def __getitem__(self, idx):
        return (self.graphs_cancer[idx], self.graphs_wt[idx]), \
               (self.encoded_full_seq_cancer[idx], self.encoded_full_seq_wt[idx]), \
               (self.encoded_peptide_seq_cancer[idx], self.encoded_peptide_seq_wt[idx]), \
               (self.peptide_property_cancer[idx], self.peptide_property_wt[idx]), \
               self.immunogenicity_cancer[idx], \
               self.foreignness_cancer[idx]

if __name__ == '__main__':
    ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--learning-rate-a', default=1e-3, type=float)
    parser.add_argument('--learning-rate-b', default=1e-4, type=float)
    parser.add_argument('--num-epochs', default=40, type=int)
    parser.add_argument('--batch-size', default=150, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--model', default='HybridModelv2_Comparative', type=str)
    parser.add_argument('--full-sequence', action='store_true')
    parser.add_argument('--sequence-loss', action='store_true')
    config = parser.parse_args()

    dataset = ImmunoPredDataset(config,
                                ROOT_DIR + 'data/cancer_graph_pyg/',
                                ROOT_DIR + 'data/cedar_data_final_with_mprop1_mprop2_v2.txt',
                                ROOT_DIR + 'data/HLA_27_seqs_csv.csv', binary=False)

    print(dataset[0], dataset[-1])
    generator1 = torch.Generator().manual_seed(42)
    train, val, test = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15], generator1)

    print(len(train), len(val), len(test))
