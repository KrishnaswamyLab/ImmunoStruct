import torch
import dgl
import numpy as np
import hashlib

__all__ = [
    "pad_graph", "find_matching_allele", "get_z_peps", "to_dgl",
    "pad_peptide_sequence", "one_hot_encode_sequence",
    "duplicate_check", "dedupe", "get_hash", "RandomRotation",
    "collate", "collate_amino_acid"
]

def pad_graph(graph, max_nodes, feature_size, coord_size):
    num_nodes_to_add = max_nodes - graph.num_nodes

    if graph.x.shape[1] != feature_size:
        print("mismatch", graph.x, graph.x.shape)
        raise ValueError("`pad_graph`: graph.x shape mismatch.")

    if num_nodes_to_add > 0:
        # Pad node features
        zero_features = torch.zeros(num_nodes_to_add, feature_size)
        padded_features = torch.cat([graph.x, zero_features], dim=0)

        # Pad coordinates
        zero_coords = torch.zeros(num_nodes_to_add, coord_size)
        padded_coords = torch.cat([graph.coords, zero_coords], dim=0)

        # Update the graph
        graph.x = padded_features
        graph.coords = padded_coords
        graph.num_nodes = max_nodes
    return graph

# Assuming df is your existing DataFrame
# Assuming expanded_pep_pair is your list of expanded peptide pairs
# Function to find the matching allele
def find_matching_allele(peptide, alleles, expanded_pep_pair):
    for allele in alleles:
        combo = peptide + allele
        if combo in expanded_pep_pair:
            return combo
    return 0  # Return None if no match is found

def get_z_peps(expanded_df, seq_df):
    z_peps = []

    for x in seq_df[seq_df['combo2'] == 0]['peptide'].tolist():
        ndf = expanded_df[expanded_df['peptide'] == x]
        if ndf['immunogenicity'].sum() == 0:
            z_peps.append(x)
    return z_peps

def to_dgl(pt_geometric_graph):
     # The number of edges is half the size of the second dimension of edge_index
    num_edges = pt_geometric_graph.edge_index.size(1)

    # Create a tensor of ones with the size equal to the number of edges
    # Assuming all edges have a single feature, which is set to 1
    pt_geometric_graph.edge_attr = torch.ones((num_edges, 1))

    # Convert to DGL graph
    src, dst = pt_geometric_graph.edge_index
    dgl_graph = dgl.graph((src, dst), num_nodes=pt_geometric_graph.num_nodes)
    dgl_graph.ndata['x'] = pt_geometric_graph.x  # Node features
    dgl_graph.edata['edge_attr'] = pt_geometric_graph.edge_attr  # Edge attributes
    return dgl_graph

# Function to pad peptide sequences
def pad_peptide_sequence(sequence, max_length=11, padding_char='J'):
    # Pad the sequence with the padding character to reach the max length
    padded_sequence = sequence.ljust(max_length, padding_char)
    return padded_sequence

def one_hot_encode_sequence(sequence, amino_acids = 'ACDEFGHIKLMNPQRSTVWY', padding_char='J'):
    # Create a dictionary mapping each amino acid and padding character to an integer
    char_to_int = dict((c, i) for i, c in enumerate(amino_acids + padding_char))

    # Initialize the one-hot encoded matrix for the sequence
    one_hot_encoded = np.zeros((len(sequence), len(char_to_int)))

    # Fill the one-hot encoded matrix with appropriate values
    for i, char in enumerate(sequence):
        if char in char_to_int:  # Only encode known characters
            one_hot_encoded[i, char_to_int[char]] = 1
        else:
            print("unknown character: {}", char)

    return one_hot_encoded

def duplicate_check(encoded_full_sequence, protein_reg_values, dgl_filtered_graphs):
    to_remove = set()
    cache = dict()
    dupe = 0
    double_dupe = 0

    for n, (a, b) in enumerate(zip(encoded_full_sequence, protein_reg_values)):
        overlap = (tuple(map(tuple, a)), b)
        if overlap in cache:
            dupe+=1
            if (dgl_filtered_graphs[cache[overlap]].num_nodes() == dgl_filtered_graphs[n].num_nodes() and
                dgl_filtered_graphs[cache[overlap]].num_edges() == dgl_filtered_graphs[n].num_edges() and
                dgl_filtered_graphs[cache[overlap]].ndata['x'].tolist() == dgl_filtered_graphs[n].ndata['x'].tolist() and
                dgl_filtered_graphs[cache[overlap]].edata['edge_attr'].tolist() == dgl_filtered_graphs[n].edata['edge_attr'].tolist() and
                dgl_filtered_graphs[cache[overlap]].edges()[0].tolist() == dgl_filtered_graphs[n].edges()[0].tolist()):
                double_dupe+=1
                to_remove.add(n)
        else:
            cache[overlap] = n
    print("dupes", dupe, double_dupe)

def dedupe(encoded_sequences, protein_reg_values, protein_immuno_values, protein_reg_values_f, dgl_filtered_graphs):
    to_remove = set()
    cache = dict()
    dupe = 0
    double_dupe = 0

    for n, (a, b) in enumerate(zip(encoded_sequences, protein_reg_values)):
        overlap = (tuple(map(tuple, a)), b)
        if overlap in cache:
            dupe+=1
            if (dgl_filtered_graphs[cache[overlap]].num_nodes() == dgl_filtered_graphs[n].num_nodes() and
                dgl_filtered_graphs[cache[overlap]].num_edges() == dgl_filtered_graphs[n].num_edges() and
                dgl_filtered_graphs[cache[overlap]].ndata['x'].tolist() == dgl_filtered_graphs[n].ndata['x'].tolist() and
                dgl_filtered_graphs[cache[overlap]].edata['edge_attr'].tolist() == dgl_filtered_graphs[n].edata['edge_attr'].tolist() and
                dgl_filtered_graphs[cache[overlap]].edges()[0].tolist() == dgl_filtered_graphs[n].edges()[0].tolist()):
                double_dupe+=1
                to_remove.add(n)
        else:
            cache[overlap] = n
    print("dupes", dupe, double_dupe)

    new_encoded_sequences = []
    new_protein_reg_values = []
    new_protein_immuno_values = []
    new_protein_reg_values_f = []
    new_dgl_filtered_graphs = []
    for n, (a, b, c, d, e) in enumerate(zip(encoded_sequences, protein_reg_values, protein_immuno_values, protein_reg_values_f, dgl_filtered_graphs)):
        if n not in to_remove:
            new_encoded_sequences.append(a)
            new_protein_reg_values.append(b)
            new_protein_immuno_values.append(c)
            new_protein_reg_values_f.append(d)
            new_dgl_filtered_graphs.append(e)

    return new_encoded_sequences, new_protein_reg_values, new_protein_immuno_values, new_protein_reg_values_f, new_dgl_filtered_graphs

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        return x @ Q

def get_hash(x):
  return hashlib.sha1(x.encode()).hexdigest()

def collate(samples):
    graphs, seq_data, labels, labelsf = map(list, zip(*samples))
    if isinstance(graphs[0], dgl.DGLGraph):
        batched_graph = dgl.batch(graphs)
        seq_data = torch.stack(seq_data, dim=0)
        labels = torch.stack(labels, dim=0)
        labelsf = torch.stack(labelsf, dim=0)
    else:
        # Using the Comparative learning method.
        batched_graph = (dgl.batch([item[0] for item in graphs]),
                         dgl.batch([item[1] for item in graphs]))
        seq_data = (torch.stack([item[0] for item in seq_data], dim=0),
                    torch.stack([item[1] for item in seq_data], dim=0))
        labels = torch.stack(labels, dim=0)
        labelsf = (torch.stack([item[0] for item in labelsf], dim=0),
                   torch.stack([item[1] for item in labelsf], dim=0))
    return batched_graph, seq_data, labels, labelsf

def collate_amino_acid(samples):
    graphs, seq_data, labels, labelsf, amino_acid = map(list, zip(*samples))
    amino_acid = torch.stack(amino_acid, dim=0).flatten()
    
    if isinstance(graphs[0], dgl.DGLGraph):
        batched_graph = dgl.batch(graphs)
        seq_data = torch.stack(seq_data, dim=0)
        labels = torch.stack(labels, dim=0)
        labelsf = torch.stack(labelsf, dim=0)
    else:
        # Using the Comparative learning method.
        batched_graph = (dgl.batch([item[0] for item in graphs]),
                         dgl.batch([item[1] for item in graphs]))
        seq_data = (torch.stack([item[0] for item in seq_data], dim=0),
                    torch.stack([item[1] for item in seq_data], dim=0))
        labels = torch.stack(labels, dim=0)
        labelsf = (torch.stack([item[0] for item in labelsf], dim=0),
                   torch.stack([item[1] for item in labelsf], dim=0))
    return batched_graph, seq_data, labels, labelsf, amino_acid
