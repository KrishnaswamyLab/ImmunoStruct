import torch
import random
import gc
import os
from glob import glob
from tqdm import tqdm

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

GRAPH_CONFIG = ProteinGraphConfig(
    edge_construction_functions=[
        add_peptide_bonds,
        add_hydrogen_bond_interactions,
        add_hydrophobic_interactions,
        add_ionic_interactions,
    ],
    node_metadata_functions=[
        amino_acid_one_hot,
        hydrogen_bond_acceptor,
        hydrogen_bond_donor,
    ],
    granularity="CA",
    exclude_waters=False,
)

# Graphein does not encode the peptide sequence uniquely; use our own encoding.
AA_ENCODING = {
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
    'MASK': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

_CONVERTOR = GraphFormatConvertor(src_format='nx', dst_format='pyg')


def mask_sequence(enc_dict, seq_one_hot, percentage: float = 10):
    total_len = len(seq_one_hot)
    mask_len = int(total_len * percentage / 100)

    mask_indices = random.sample(range(total_len), mask_len)

    masked_seq = seq_one_hot.clone()
    for idx in mask_indices:
        masked_seq[idx] = torch.tensor(enc_dict['MASK'])

    return masked_seq


def pdb_to_pyg(pdb_path, mask_percentage=0):
    """Convert a single AlphaFold peptide-MHC PDB to a PyG Data object.

    Extracts the MHC extracellular domain (first 179 residues of chain A)
    and the peptide (chain B). Node features are the amino-acid one-hot
    encoding concatenated with hydrogen-bond donor/acceptor counts.

    Args:
        pdb_path: Path to the PDB file.
        mask_percentage: Percentage of peptide residues to mask (0 = no masking).

    Returns:
        A torch_geometric ``Data`` object.
    """
    g = construct_graph(config=GRAPH_CONFIG, path=pdb_path)

    # First 179 residues are the extracellular domain of the MHC. 273 and beyond are the peptide.
    g2 = extract_subgraph_by_sequence_position(g, list(range(1, 180)) + list(range(273, 1000)))
    g_pyg = _CONVERTOR(g2)

    tdf = read_pdb_to_dataframe(pdb_path)

    # Chain A is the MHC.
    tdfa = tdf[tdf['chain_id'] == 'A']
    tdfa = tdfa.drop_duplicates('residue_number')
    tdfa = tdfa[:179]
    sequence_a = tdfa['residue_name'].tolist()

    # Chain B is the peptide.
    tdfb = tdf[tdf['chain_id'] == 'B']
    tdfb = tdfb.drop_duplicates('residue_number')
    sequence_b = tdfb['residue_name'].tolist()

    n_hdonors = torch.tensor([d['hbond_donors'] for n, d in g2.nodes(data=True)])
    n_hacceptors = torch.tensor([d['hbond_acceptors'] for n, d in g2.nodes(data=True)])

    aa_one_hot_a = torch.tensor([AA_ENCODING[x] for x in sequence_a])
    aa_one_hot_b = torch.tensor([AA_ENCODING[x] for x in sequence_b])

    # Apply masking to peptide (percentage=0 means no masking).
    masked_aa_one_hot_b = mask_sequence(AA_ENCODING, aa_one_hot_b, percentage=mask_percentage)
    aa_one_hot = torch.cat([aa_one_hot_a, masked_aa_one_hot_b], dim=0)

    node_feats = torch.cat([aa_one_hot, n_hdonors, n_hacceptors], dim=1)
    g_pyg.x = node_feats

    return g_pyg


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    file_list = sorted(glob(os.path.join(args.input_dir, '*.pdb')))
    print(f"Converting {len(file_list)} PDB files to PyG graphs...")
    for filename in tqdm(file_list):
        try:
            filename_no_extension = os.path.basename(filename).replace('.pdb', '')
            save_filename = os.path.join(args.output_dir, filename_no_extension + '.pt')
            if os.path.exists(save_filename):
                print(f"Graph already exists for {filename_no_extension}")
                continue

            g_pyg = pdb_to_pyg(filename, mask_percentage=0)
            torch.save(g_pyg, save_filename)

            print('done creating graph {}'.format(filename_no_extension), flush=True)

            del g_pyg
            gc.collect()

        except Exception as e:
            log_message = 'Error creating graph {}. Encountered exception {}'.format(filename_no_extension, e)
            print(log_message)

            log_file = os.path.join(args.output_dir, "error_log.txt")
            with open(log_file, 'a') as file:
                file.write(log_message + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert AlphaFold PDBs to PyG graphs.")
    parser.add_argument("--input-dir", default=f"{ROOT_DIR}/data/alphafold2_pdb_IEDB/", type=str)
    parser.add_argument("--output-dir", default=f"{ROOT_DIR}/data/graph_pyg_IEDB/", type=str)
    args = parser.parse_args()
    main(args)