import argparse
import os
import re
import pandas as pd
import jax
from tqdm import tqdm
from alphafold.data import parsers
import pickle

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])


def _detect_device():
    if jax.local_devices()[0].platform == "cpu":
        print("WARNING: no GPU detected, will be using CPU")
        return "cpu"
    print("Running on GPU")
    return "gpu"


def _parse_a3m_compat(a3m_lines):
    parsed = parsers.parse_a3m(a3m_lines)
    if isinstance(parsed, tuple) and len(parsed) == 2:
      return parsed
    return parsed.sequences, parsed.deletion_matrix


def _pad(ns, vals, mode, seq_lengths):
    _blank_seq = ["-" * L for L in seq_lengths]
    _blank_mtx = [[0] * L for L in seq_lengths]
    if mode == "seq": _blank = _blank_seq.copy()
    if mode == "mtx": _blank = _blank_mtx.copy()
    if isinstance(ns, list):
        for n, val in zip(ns, vals): _blank[n] = val
    else: _blank[ns] = vals
    if mode == "seq": return "".join(_blank)
    if mode == "mtx": return sum(_blank, [])


def standardize_hla(allele: str) -> str:
    """Standardize HLA allele strings to the canonical 'HLA-A*01:01' format."""
    allele = allele.strip().upper()
    match = re.match(r"^HLA-([A-Z])\*?(\d{2,3}):?(\d{2,3})$", allele)
    if match:
        locus, group, field = match.groups()
        return f"HLA-{locus}*{group}:{field}"
    match = re.match(r"^HLA-([A-Z])\*?(\d{2,3})$", allele)
    if match:
        locus, group = match.groups()
        return f"HLA-{locus}*{group}"
    return allele


def read_a3m(a3m_files, sequences):
    """
    Currently only works for MHC-peptide pairs, assuming 2 sequences.
    """
    assert len(sequences) == 2
    M = None
    Ms = [101, 102]

    a3m_lines = {}
    for a3m_file in a3m_files:
        for line_idx, line in enumerate(open(a3m_file, "r")):
            if line_idx < 3:
                continue
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                if line.startswith(">") and line[1:].rstrip().isnumeric():
                    metadata = int(line[1:].rstrip())
                    if metadata in Ms and metadata not in a3m_lines.keys():
                        M = metadata
                        a3m_lines[M] = []
                # unpad
                # NOTE: `line_idx % 2 == 0` is not a very good check.
                if "-" in line and M is not None and line_idx % 2 == 0:
                    if M == 101:
                        assert "-" * len(sequences[1]) in line
                        # NOTE: replace only once.
                        line = line.replace("-" * len(sequences[1]), "", 1)
                    else:
                        assert "-" * len(sequences[0]) in line
                        # NOTE: replace only once.
                        line = line.replace("-" * len(sequences[0]), "", 1)
                a3m_lines[M].append(line)
    a3m_lines_list = ["".join(a3m_lines[n]) for n in Ms]
    return a3m_lines_list


def a3m_to_msa(a3m_lines_list, sequences):
    msa_list, deletion_matrices = [], []
    for n, seq in enumerate(sequences):
        a3m_lines = a3m_lines_list[n]
        msa, mtx = _parse_a3m_compat(a3m_lines)
        msas_, mtxs_ = [msa], [mtx]
        # pad sequences
        for msa_, mtx_ in zip(msas_, mtxs_):
            seq_concat = "".join(sequences)
            seq_lengths = [len(item) for item in sequences]
            msa, mtx = [seq_concat], [[0]*len(seq_concat)]
            for s, m in zip(msa_, mtx_):
                msa.append(_pad(n, s, "seq", seq_lengths))
                mtx.append(_pad(n, m, "mtx", seq_lengths))
            msa_list.append(msa)
            deletion_matrices.append(mtx)
    return msa_list, deletion_matrices


def main(args):
    _detect_device()

    allele_sequence_csv = pd.read_csv(args.allele_sequence_csv)
    allele_to_sequence = allele_sequence_csv.set_index("allele")["seqs"].to_dict()

    folding_input_df = pd.read_csv(args.input_csv)
    folding_input_df = folding_input_df.sort_values(by=[args.allele_col_name, args.peptide_col_name])

    n_unique_pairs = folding_input_df[[args.allele_col_name, args.peptide_col_name]].drop_duplicates().shape[0]
    print(f"\nUnique [allele, peptide] pairs: {n_unique_pairs} (total rows: {len(folding_input_df)})", flush=True)

    folding_input_df = folding_input_df.reset_index(drop=True)

    for _, row in tqdm(folding_input_df.iterrows(), total=len(folding_input_df)):
        allele = row[args.allele_col_name]
        allele = standardize_hla(allele)
        peptide = row[args.peptide_col_name]
        MHC_sequence = allele_to_sequence[allele]

        if "hla_seq" in row.keys():
            double_check_seq = row["hla_seq"]
            assert double_check_seq == MHC_sequence, "HLA sequence does not match the expected sequence."

        full_sequence_folding = f"{MHC_sequence}:{peptide}"
        full_sequence_name = f"{allele}:{peptide}"
        full_sequence_name = re.sub(r'\W+', '_', full_sequence_name)

        a3m_files = [os.path.join(args.input_dir, f"{full_sequence_name}.a3m")]
        output_dir = os.path.join(args.output_dir, full_sequence_name) if args.output_dir else None
        msa_path = os.path.join(output_dir, "msa.pickle") if output_dir else None

        if msa_path and os.path.exists(msa_path):
            print(f"\nSkipping {full_sequence_name}: msa.pickle already exists.", flush=True)
            continue

        sequences = full_sequence_folding.split(':')
        a3m_lines_list = read_a3m(a3m_files, sequences)
        msa_list, deletion_matrices = a3m_to_msa(a3m_lines_list, sequences)
        os.makedirs(os.path.dirname(msa_path), exist_ok=True)
        pickle.dump({"msas": msa_list, "deletion_matrices": deletion_matrices}, open(os.path.join(msa_path), "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a3m files to MSA files.")
    parser.add_argument("--input-dir", required=True, help="Input directory for a3m files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for MSA files.")
    parser.add_argument("--input-csv", required=True, help="CSV with all sequences and peptides.")
    parser.add_argument("--allele-sequence-csv", default=f"{ROOT_DIR}/data/HLA_allele_sequences.csv")
    parser.add_argument("--allele-col-name", type=str, default="allele")
    parser.add_argument("--peptide-col-name", type=str, default="peptide")
    main(parser.parse_args())
