import argparse
import os
import re

import pandas as pd

ROOT_DIR = "/".join(os.path.realpath(__file__).split("/")[:-3])


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


def main(args):
    allele_sequence_csv = pd.read_csv(args.allele_sequence_csv)
    allele_to_sequence = allele_sequence_csv.set_index("allele")["seqs"].to_dict()

    folding_input_df = pd.read_csv(args.input_csv)
    folding_input_df = folding_input_df.sort_values(by=[args.allele_col_name, args.peptide_col_name])
    folding_input_df = folding_input_df.drop_duplicates(subset=[args.allele_col_name, args.peptide_col_name])
    folding_input_df = folding_input_df.reset_index(drop=True)

    n_unique = len(folding_input_df)
    print(f"\nUnique [allele, peptide] pairs: {n_unique} | total rows: {len(folding_input_df)}", flush=True)

    output_dir = os.path.dirname(args.output_fasta)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_fasta, "w", encoding="utf-8") as fout:
        for _, row in folding_input_df.iterrows():
            allele = row[args.allele_col_name]
            allele = standardize_hla(allele)
            peptide = row[args.peptide_col_name]
            MHC_sequence = allele_to_sequence[allele]

            if "hla_seq" in row.keys():
                double_check_seq = row["hla_seq"]
                assert (double_check_seq == MHC_sequence), "HLA sequence does not match the expected sequence."

            full_sequence_folding = f"{MHC_sequence}:{peptide}"
            full_sequence_name = f"{allele}:{peptide}"

            # NOTE: Keep naming behavior identical to step1_sequence_to_msa.py.
            full_sequence_name = re.sub(r"\W+", "_", full_sequence_name)

            fout.write(f">{full_sequence_name}\n")
            fout.write(f"{full_sequence_folding}\n")

    print(f"\nWrote FASTA records to: {args.output_fasta}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write all unique sequences to one FASTA file.")
    parser.add_argument("--input-csv", required=True, help="CSV with all sequences and peptides.")
    parser.add_argument("--output-fasta", required=True, help="Output FASTA file path for all records.")
    parser.add_argument("--allele-sequence-csv", default=f"{ROOT_DIR}/data/HLA_allele_sequences.csv")
    parser.add_argument("--allele-col-name", type=str, default="allele")
    parser.add_argument("--peptide-col-name", type=str, default="peptide")
    main(parser.parse_args())
