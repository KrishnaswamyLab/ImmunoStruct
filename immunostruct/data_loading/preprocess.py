from tqdm import tqdm
import os
import torch
import numpy as np
import pandas as pd
from .data_utils import pad_graph, to_dgl, pad_peptide_sequence, one_hot_encode_sequence

__all__ = [
    "preprocess_graphs", "preprocess_properties", "preprocess_properties_cancer_wt",
    "preprocess_hla", "preprocess_sequence_graph", "preprocess_sequence_graph_cancer_wt",
    "preprocess_sequence_graph_clinical",
    "preprocess_graph", "preprocess_sequence",
]

def preprocess_graphs(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]

    # Initialize an empty list to store the graphs
    graphs = []

    # Loop through the files and load each graph, showing a progress bar
    for file in tqdm(files, desc="Loading graphs"):
        file_path = os.path.join(directory, file)
        graph = torch.load(file_path)
        graphs.append(graph)

    print(f"Loaded {len(graphs)} graphs.")

    graphs = [x for x in graphs if ('NXVPMVATV' not in x.name) and ('X' not in x.name)]

    all_graphs = []
    names = set()

    for graph in graphs:
        if graph.name not in names:
            names.add(graph.name)
            all_graphs.append(graph)

    #cut off h-bonding features for now
    for data in all_graphs:
        data.x = data.x[:, :-2]

    return all_graphs

def preprocess_properties(table, cancer=False):
    expanded_df = pd.read_csv(table)

    if cancer:
        # used for cancer dataset
        expanded_df = expanded_df.dropna(subset='foreign')
        expanded_df['mhc_pep_pair'] = expanded_df['allele'] + "_" + expanded_df['mut_pep']
    else:
        # used for IEDB dataset
        expanded_df = expanded_df.dropna(subset='Foreignness_Score')
        expanded_df['mhc_pep_pair'] = expanded_df['allele'] + "_" + expanded_df['peptide']

    foreignness_dict = dict(zip(expanded_df['mhc_pep_pair'], expanded_df['smoothed_foreign']))
    properties_dict = dict(zip(expanded_df['mhc_pep_pair'], zip(expanded_df['Mprop1'], expanded_df['Mprop2'])))
    immunogenicity_dict = dict(zip(expanded_df['mhc_pep_pair'], expanded_df['immunogenicity']))

    mhc_pep_pair_list = expanded_df['mhc_pep_pair'].tolist()
    return foreignness_dict, properties_dict, immunogenicity_dict, mhc_pep_pair_list


def preprocess_properties_cancer_wt(table_cancer, table_wt):
    expanded_df_cancer = pd.read_csv(table_cancer)
    expanded_df_wt = pd.read_csv(table_wt)

    expanded_df_cancer = expanded_df_cancer.dropna(subset='foreign')
    expanded_df_cancer['mhc_pep_pair_cancer'] = expanded_df_cancer['allele'] + "_" + expanded_df_cancer['mut_pep']

    expanded_df_wt = expanded_df_wt.dropna(subset='foreign')
    expanded_df_wt['mhc_pep_pair_wt'] = expanded_df_wt['allele'] + "_" + expanded_df_wt['wt_pep']

    short_df_cancer = expanded_df_cancer[['mut_pep', 'wt_pep', 'allele', 'immunogenicity', 'mhc_pep_pair_cancer', 'smoothed_foreign', 'Mprop1', 'Mprop2']]
    short_df_wt = expanded_df_wt[['mut_pep', 'wt_pep', 'allele', 'immunogenicity', 'foreign', 'mhc_pep_pair_wt', 'Mprop1_wt', 'Mprop2_wt']]
    short_df_cancer = __dedup_property_df(short_df_cancer)
    short_df_wt = __dedup_property_df(short_df_wt)

    combined_df = pd.merge(short_df_cancer, short_df_wt, on=['mut_pep', 'wt_pep', 'allele', 'immunogenicity'])
    combined_df = combined_df[['mut_pep', 'wt_pep', 'allele', 'immunogenicity', 'mhc_pep_pair_cancer', 'mhc_pep_pair_wt', 'smoothed_foreign', 'Mprop1', 'Mprop1_wt', 'Mprop2', 'Mprop2_wt']]
    assert len(short_df_cancer) == len(short_df_wt) == len(combined_df)

    return combined_df

def __dedup_property_df(df):
    """
    In a few cases, there may be duplicates in entries (likely from different patients)
    that share the same ('mut_pep', 'wt_pep', 'allele') but have slightly different ('smoothed_foreign', 'Mprop1', 'Mprop2').
    We will deduplicate, keeping the entry with the highest foreignness if immunogenic and the lowest foreignness otherwise.
    """

    assert len(np.unique([str(item) for item in df[['mut_pep', 'wt_pep', 'allele', 'immunogenicity']].values])) \
        == len(np.unique([str(item) for item in df[['mut_pep', 'wt_pep', 'allele']].values])), \
            "`__dedup_property_df`: same ('mut_pep', 'wt_pep', 'allele') but different immunogenicity!"

    tuple_list = [str(item) for item in df[['mut_pep', 'wt_pep', 'allele']].values]
    duplicate_items = []
    duplicate_rows_list = []
    for item in tuple_list:
        if tuple_list.count(item) > 1:
            if item not in duplicate_items:
                duplicate_items.append(item)
                duplicate_rows_list.append([i for i, x in enumerate(tuple_list) if x == item])

    rows_to_drop = []
    for duplicate_rows in duplicate_rows_list:
        immunogenicity_arr = df.loc[duplicate_rows]['immunogenicity'].values
        assert len(np.unique(immunogenicity_arr)) == 1, \
            "`__dedup_property_df`: same ('mut_pep', 'wt_pep', 'allele') but different immunogenicity!"
        immunogenicity = immunogenicity_arr[0]
        foreign_key = 'smoothed_foreign' if 'smoothed_foreign' in df else 'foreign'
        foreignness_arr = df.loc[duplicate_rows][foreign_key].values
        if immunogenicity == 1:
            idx_to_keep = duplicate_rows[foreignness_arr.argmax()]
        else:
            assert immunogenicity == 0
            idx_to_keep = duplicate_rows[foreignness_arr.argmin()]
        rows_to_drop.extend(list(set(duplicate_rows) - set([idx_to_keep])))

    if len(rows_to_drop) > 0:
        df = df.drop(index=rows_to_drop)

    return df

def preprocess_hla(mhc_pep_pair_list, hla_path):
    hla_df = pd.read_csv(hla_path)
    hla_dict_true = dict(zip(hla_df['allele'], hla_df['seqs']))

    name_mapper = {}
    # NOTE: `MHC` and `HLA` mean the same thing in this context.
    for mhc_pep_pair in mhc_pep_pair_list:
        hla_code, pep_seq = mhc_pep_pair.split("_")
        assert hla_code.startswith("HLA-")
        mhc_seq = hla_dict_true[hla_code]
        name_mapper[mhc_pep_pair] = (mhc_seq, pep_seq)

    return name_mapper

def preprocess_sequence_graph(name_mapper, all_graphs, immunogenicity_dict, foreignness_dict):
    # Step 1. Remove peptide-MHC pairs that do not have graphs.
    mhc_pep_pair_with_graphs = set([x.name for x in all_graphs])
    to_remove = []
    for k in name_mapper.keys():
        if k not in mhc_pep_pair_with_graphs:
            to_remove.append(k)

    for i in to_remove:
        del name_mapper[i]
    print("Filter peptide-MHC pairs without graphs. Remaining: {}, removed {}".format(len(name_mapper), len(to_remove)))

    # Step 2. Remove graphs that are not recorded in peptide-MHC pairs.
    to_remove = set()
    for i in mhc_pep_pair_with_graphs:
        if i not in name_mapper.keys():
            to_remove.add(i)
    print("Filter graphs without peptide-MHC information. Remaining: {}, removed {}".format(len(mhc_pep_pair_with_graphs), len(to_remove)))

    all_graphs = [x for x in all_graphs if x.name not in to_remove]
    mhc_pep_pair_to_graph = {x.name: x for x in all_graphs}

    for k in name_mapper.keys():
        immunogenicity = immunogenicity_dict[k]
        foreignness = foreignness_dict[k]
        graph = mhc_pep_pair_to_graph[k]

        graph.y = torch.tensor([immunogenicity, foreignness], dtype=torch.float)  # We use a one-element tensor for each graph-level label
        graph.x = torch.cat([graph.x, graph.coords], dim=-1)

        graph.x = graph.x.to(dtype=torch.float32)
        graph.y = graph.y.to(dtype=torch.float32)

    return name_mapper, mhc_pep_pair_to_graph

def preprocess_sequence_graph_cancer_wt(combined_df, name_mapper_cancer, name_mapper_wt, graphs_cancer, graphs_wt):
    """
    `name_mapper_cancer` and `name_mapper_wt`:
        key: mhc_pep_pair
        value: (mhc_seq, pep_seq)
    """

    to_remove_cancer_all = set()
    to_remove_wt_all = set()

    # Step 1. Remove peptide-MHC pairs that do not have graphs.
    mhc_pep_pair_with_graphs_cancer = set([x.name for x in graphs_cancer])
    to_remove_cancer = set()
    for k in name_mapper_cancer.keys():
        if k not in mhc_pep_pair_with_graphs_cancer:
            to_remove_cancer.add(k)
            to_remove_cancer_all.add(k)
    for k in to_remove_cancer:
        del name_mapper_cancer[k]
    print("(Cancer) Filter peptide-MHC pairs without graphs. Remaining: {}, removed {}".format(len(name_mapper_cancer), len(to_remove_cancer)))

    mhc_pep_pair_with_graphs_wt = set([x.name for x in graphs_wt])
    to_remove_wt = set()
    for k in name_mapper_wt.keys():
        if k not in mhc_pep_pair_with_graphs_wt:
            to_remove_wt.add(k)
            to_remove_wt_all.add(k)
    for k in to_remove_wt:
        del name_mapper_wt[k]
    print("(WT) Filter peptide-MHC pairs without graphs. Remaining: {}, removed {}".format(len(name_mapper_wt), len(to_remove_wt)))

    # Step 2. Remove graphs that are not recorded in peptide-MHC pairs.
    to_remove_cancer = set()
    for k in mhc_pep_pair_with_graphs_cancer:
        if k not in name_mapper_cancer.keys():
            to_remove_cancer.add(k)
            to_remove_cancer_all.add(k)
    for k in to_remove_cancer:
        del name_mapper_cancer[k]
    print("(Cancer) Filter graphs without peptide-MHC information. Remaining: {}, removed {}".format(len(name_mapper_cancer), len(to_remove_cancer)))

    to_remove_wt = set()
    for k in mhc_pep_pair_with_graphs_wt:
        if k not in name_mapper_wt.keys():
            to_remove_wt.add(k)
            to_remove_wt_all.add(k)
    for k in to_remove_wt:
        del name_mapper_wt[k]
    print("(WT) Filter graphs without peptide-MHC information. Remaining: {}, removed {}".format(len(name_mapper_wt), len(to_remove_wt)))

    # Cross check cancer vs. wt and remove unmatched sequences and graphs.
    cancer_wt_mapper = dict(zip(combined_df['mhc_pep_pair_cancer'], combined_df['mhc_pep_pair_wt']))
    wt_cancer_mapper = dict(zip(combined_df['mhc_pep_pair_wt'], combined_df['mhc_pep_pair_cancer']))

    # Step 3. Remove unmatched data in cancer and wildtype.
    to_remove_cancer = set()
    for k, v in name_mapper_cancer.items():
        k_wt = cancer_wt_mapper[k]
        if k_wt not in name_mapper_wt.keys():
            to_remove_cancer.add(k)
            to_remove_cancer_all.add(k)
    for k in to_remove_cancer:
        del name_mapper_cancer[k]

    to_remove_wt = set()
    for k, v in name_mapper_wt.items():
        k_cancer = wt_cancer_mapper[k]
        if k_cancer not in name_mapper_cancer.keys():
            to_remove_wt.add(k)
            to_remove_wt_all.add(k)
    for k in to_remove_wt:
        del name_mapper_wt[k]

    print("After cross-checking (cancer vs. wt), final list size: {}, removed {} from cancer and {} from wt".format(
        len(name_mapper_cancer), len(to_remove_cancer), len(to_remove_wt)))

    # Remove corresponding rows from `combinded_df`.
    for k in to_remove_cancer_all:
        combined_df = combined_df[combined_df['mhc_pep_pair_cancer'] != k]
    for k in to_remove_wt_all:
        combined_df = combined_df[combined_df['mhc_pep_pair_wt'] != k]

    # Organize the graph dicts.
    graphs_cancer = [item for item in graphs_cancer if item.name not in to_remove_cancer_all]
    graphs_wt = [item for item in graphs_wt if item.name not in to_remove_wt_all]
    graph_mapper_cancer = {item.name: item for item in graphs_cancer}
    graph_mapper_wt = {item.name: item for item in graphs_wt}

    for k in name_mapper_cancer.keys():
        k_wt = cancer_wt_mapper[k]

        df_entry = combined_df[np.logical_and(combined_df['mhc_pep_pair_cancer'] == k, combined_df['mhc_pep_pair_wt'] == k_wt)]
        assert len(df_entry) == 1
        immunogenicity = df_entry['immunogenicity'].item()
        foreignness = df_entry['smoothed_foreign'].item()

        graph_cancer = graph_mapper_cancer[k]
        graph_cancer.x = torch.cat([graph_cancer.x, graph_cancer.coords], dim=-1)
        graph_cancer.y = torch.tensor([immunogenicity, foreignness], dtype=torch.float)  # We use a one-element tensor for each graph-level label
        graph_cancer.x = graph_cancer.x.to(dtype=torch.float32)
        graph_cancer.y = graph_cancer.y.to(dtype=torch.float32)

        graph_wt = graph_mapper_wt[k_wt]
        if graph_wt.x.shape[1] < graph_cancer.x.shape[1]:
            graph_wt.x = torch.cat([graph_wt.x, graph_wt.coords], dim=-1)
            graph_wt.y = torch.tensor([0, combined_df['smoothed_foreign'].min()], dtype=torch.float)  # We use a one-element tensor for each graph-level label
            graph_wt.x = graph_wt.x.to(dtype=torch.float32)
            graph_wt.y = graph_wt.y.to(dtype=torch.float32)
            assert graph_wt.x.shape[1] == graph_cancer.x.shape[1]
        else:
            # In this case, the same graph has already been iterated. Move on.
            assert graph_wt.x.shape[1] == graph_cancer.x.shape[1]

    return combined_df, name_mapper_cancer, name_mapper_wt, graph_mapper_cancer, graph_mapper_wt

def preprocess_sequence_graph_clinical(graph_directory, seq_path):
    all_graphs = preprocess_graphs(graph_directory)

    seq_df = pd.read_csv(seq_path)
    name_mapper = {}
    for _, row in seq_df.iterrows():
        pep_seq = row['mut_pep']
        hla_code = row['allele']
        mhc_seq = row['hla_seq']
        mhc_pep_pair = hla_code + "_" + pep_seq
        name_mapper[mhc_pep_pair] = (mhc_seq, pep_seq)

    # Step 1. Remove peptide-MHC pairs that do not have graphs.
    mhc_pep_pair_with_graphs = set([x.name for x in all_graphs])
    to_remove = []
    for k in name_mapper.keys():
        if k not in mhc_pep_pair_with_graphs:
            to_remove.append(k)

    for i in to_remove:
        del name_mapper[i]
    print("Filter peptide-MHC pairs without graphs. Remaining: {}, removed {}".format(len(name_mapper), len(to_remove)))

    # Step 2. Remove graphs that are not recorded in peptide-MHC pairs.
    to_remove = set()
    for i in mhc_pep_pair_with_graphs:
        if i not in name_mapper.keys():
            to_remove.add(i)
    print("Filter graphs without peptide-MHC information. Remaining: {}, removed {}".format(len(mhc_pep_pair_with_graphs), len(to_remove)))

    all_graphs = [x for x in all_graphs if x.name not in to_remove]
    mhc_pep_pair_to_graph = {x.name: x for x in all_graphs}

    for k in name_mapper.keys():
        graph = mhc_pep_pair_to_graph[k]
        graph.x = torch.cat([graph.x, graph.coords], dim=-1)
        graph.x = graph.x.to(dtype=torch.float32)

    return name_mapper, mhc_pep_pair_to_graph


def preprocess_graph(graph_mapper, feature_size, coord_size):
    max_nodes = max(graph.num_nodes for graph in graph_mapper.values())
    padded_graphs = {key: pad_graph(graph, max_nodes, feature_size, coord_size) for key, graph in graph_mapper.items()}
    graph_mapper = {key: to_dgl(graph) for key, graph in padded_graphs.items()}
    return graph_mapper

def preprocess_sequence(name_mapper, amino_acids, padding_char):
    max_full_length = max(len(pep_mhc_pair[0]) + len(pep_mhc_pair[1]) for _, pep_mhc_pair in name_mapper.items())
    max_pep_length = max(len(pep_mhc_pair[0]) for _, pep_mhc_pair in name_mapper.items())

    pad_seq_mapper = {key: (pad_peptide_sequence(pep_seq, max_pep_length, padding_char),
                            pad_peptide_sequence(mhc_seq + pep_seq, max_full_length, padding_char)) for key, (mhc_seq, pep_seq) in name_mapper.items()}

    encoded_full_sequence_map = {key: one_hot_encode_sequence(padded_full_seq, amino_acids, padding_char) for key, (_, padded_full_seq) in pad_seq_mapper.items()}
    encoded_peptide_map = {key: one_hot_encode_sequence(padded_pep_seq, amino_acids, padding_char) for key, (padded_pep_seq, _) in pad_seq_mapper.items()}

    return encoded_full_sequence_map, encoded_peptide_map
