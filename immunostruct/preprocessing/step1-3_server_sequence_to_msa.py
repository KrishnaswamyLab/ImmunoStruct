import argparse
import os
import re
import pandas as pd
from tqdm import tqdm

import colabfold_alphafold as cf_af

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])


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

        # NOTE: ColabFold only allows alpha-numeric characters or underscores.
        full_sequence_name = re.sub(r'\W+', '_', full_sequence_name)

        output_dir = os.path.join(args.output_dir, full_sequence_name) if args.output_dir else None
        msa_path = os.path.join(output_dir, "msa.pickle") if output_dir else None

        if msa_path and os.path.exists(msa_path):
            print(f"\nSkipping {full_sequence_name}: msa.pickle already exists.", flush=True)
            continue

        I = cf_af.prep_inputs(full_sequence_folding, full_sequence_name, args.homooligomer, output_dir=output_dir, clean=False)
        print(f"\nPrepared inputs for {full_sequence_name}.", flush=True)
        cf_af.prep_msa(
            I,
            msa_method=args.msa_method,
            add_custom_msa=args.add_custom_msa,
            msa_format=args.msa_format,
            pair_mode=args.pair_mode,
            pair_cov=args.pair_cov,
            pair_qid=args.pair_qid,
            TMP_DIR=args.tmp_dir,
        )
        print(f"\nPrepared MSA for {full_sequence_name}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use online MSA server to perform MSA.")
    parser.add_argument("--input-csv", required=True, help="CSV with all sequences and peptides.")
    parser.add_argument("--output-dir", required=True, help="Output directory for MSA files.")
    parser.add_argument("--tmp-dir", default="/tmp/", help="Temporary directory for ColabFold.")
    parser.add_argument("--allele-sequence-csv", default=f"{ROOT_DIR}/data/HLA_allele_sequences.csv")
    parser.add_argument("--allele-col-name", type=str, default="allele")
    parser.add_argument("--peptide-col-name", type=str, default="peptide")
    parser.add_argument("--homooligomer", default="1:1")
    parser.add_argument("--msa-method", default="mmseqs2")
    parser.add_argument("--add-custom-msa", action="store_true")
    parser.add_argument("--msa-format", default="fas")
    parser.add_argument("--pair-mode", default="unpaired")
    parser.add_argument("--pair-cov", type=int, default=50)
    parser.add_argument("--pair-qid", type=int, default=20)
    main(parser.parse_args())
