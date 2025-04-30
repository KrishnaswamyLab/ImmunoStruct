import torch
import random
import gc
import os
from glob import glob

from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_peptide_bonds, add_hydrophobic_interactions, add_ionic_interactions
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.features.nodes.amino_acid import hydrogen_bond_acceptor
from graphein.protein.features.nodes.amino_acid import hydrogen_bond_donor
from graphein.protein.subgraphs import extract_subgraph_by_sequence_position
from graphein.protein.graphs import read_pdb_to_dataframe


ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])

def mask_sequence(seq_one_hot, percentage=10):
    # incase we want to mask, but for now we can set this to 0
    total_len = len(seq_one_hot)
    mask_len = int(total_len * percentage / 100)

    mask_indices = random.sample(range(total_len), mask_len)

    masked_seq = seq_one_hot.clone()  # Create a copy of the original sequence
    for idx in mask_indices:
        masked_seq[idx] = torch.tensor(enc_dict['MASK'])

    return masked_seq

if __name__ == '__main__':
    # TODO: Update the folders here.
    # alphafold_folder = ROOT_DIR + '/data/alphafold_pdb_Cancer/'
    # save_folder = ROOT_DIR + '/data/graph_pyg_Cancer/'

    # alphafold_folder = ROOT_DIR + '/data/alphafold_pdb_Cancer_WT/'
    # save_folder = ROOT_DIR + '/data/graph_pyg_Cancer_WT/'

    alphafold_folder = ROOT_DIR + '/data/alphafold_pdb_Clinical/'
    save_folder = ROOT_DIR + '/data/graph_pyg_Clinical/'
    os.makedirs(save_folder, exist_ok=True)

    # Generate different edge constructions
    new_edge_funcs = {
        "edge_construction_functions": [
            add_peptide_bonds,
            add_hydrogen_bond_interactions,
            add_hydrophobic_interactions,
            add_ionic_interactions
            ],
        "node_metadata_functions": [
            amino_acid_one_hot,
            hydrogen_bond_acceptor,
            hydrogen_bond_donor],
        "granularity": "CA",
        "exclude_waters": False}

    config = ProteinGraphConfig(**new_edge_funcs)

    config.dict()

    # Graphein does not appear to encode the peptide sequence uniquely, redefine an encoding sequence
    enc_dict = {
        'GLY': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'HIS': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'MET': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ARG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'PHE': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'VAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'GLU': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ILE': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ALA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ASP': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'GLN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'TRP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'LEU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ASN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'CYS': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'MASK': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    pygs = []
    g2s = []
    peptide_order_list = []
    convertor = GraphFormatConvertor(src_format = 'nx', dst_format = 'pyg')
    file_list = sorted(glob(alphafold_folder + '*.pdb'))
    for filename in file_list:
        try:
            filename_no_extension = os.path.basename(filename).replace('.pdb', '')
            save_filename = save_folder + filename_no_extension + '.pt'
            # if os.path.isfile(save_filename):
            #     print('Already processed {}'.format(filename_no_extension))
            #     continue

            g = construct_graph(config = config, path = filename);
            g2 = extract_subgraph_by_sequence_position(g, list(range(1, 180)) + list(range(273, 1000)))
            g_pyg = convertor(g2)

            #READ IN PDBS FOR TROUBLE SHOOTING
            tdf = read_pdb_to_dataframe(filename)

            tdfa = tdf[tdf['chain_id'] == 'A']
            tdfa = tdfa.drop_duplicates('residue_number')
            tdfa = tdfa[:179]
            sequence_a = tdfa['residue_name'].tolist()

            tdfb = tdf[tdf['chain_id'] == 'A']
            tdfb = tdfb.drop_duplicates('residue_number')
            tdfb = tdfb[272:]
            sequence_b = tdfb['residue_name'].tolist()

            #add node features to each graph here
            n_hdonors = torch.tensor([d['hbond_donors'] for n, d in g2.nodes(data=True)])
            n_hacceptors = torch.tensor([d['hbond_acceptors'] for n, d in g2.nodes(data=True)])
            #aa_one_hot = torch.tensor([d['amino_acid_one_hot'] for n, d in g2.nodes(data=True)])
            #gphein_seq_a = torch.tensor([d['amino_acid_one_hot'] for n, d in g2a.nodes(data=True)])
            #gphein_seq_b = torch.tensor([d['amino_acid_one_hot'] for n, d in g2b.nodes(data=True)])

            #amino acid one-hot encoding
            aa_one_hot_a = torch.tensor([enc_dict[x] for x in sequence_a])
            aa_one_hot_b = torch.tensor([enc_dict[x] for x in sequence_b])

            # Apply masking with a X% chance of masking each amino acid (applied only to peptide, increase percentage argument (eg, to 50) to mask %50 of the peptide)
            masked_aa_one_hot_b = mask_sequence(aa_one_hot_b, percentage=0)
            aa_one_hot = torch.cat([aa_one_hot_a, masked_aa_one_hot_b], dim=0)

            g2s.append(g2)

            # Use the masked amino acid one-hot encoding as node features, along with number of H-donors/acceptors
            node_feats = torch.cat([aa_one_hot, n_hdonors, n_hacceptors], dim =1)
            g_pyg.x = node_feats

            pygs.append(g_pyg)

            # Save the PyTorch graph to a file if desired
            torch.save(g_pyg, save_filename)

            print('done creating graph {}'.format(filename_no_extension))

            # Delete temporary variables to free memory
            del g, g2, g_pyg
            gc.collect()  # Explicitly call garbage collecto

        except Exception as e:
            log_message = 'Error creating graph {}. Encountered exception {}'.format(filename_no_extension, e)
            print(log_message)

            log_file = ROOT_DIR + "data/error_log.txt"
            with open(log_file, 'a') as file:
                file.write(log_message + '\n')
