import argparse
import os
import re
import pandas as pd
import jax
from tqdm import tqdm

import colabfold_alphafold as cf_af

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])


def _detect_device():
    if jax.local_devices()[0].platform == "cpu":
        print("WARNING: no GPU detected, will be using CPU")
        return "cpu"
    print("Running on GPU")
    return "gpu"


def _build_options(feature_dict, args):
    max_msa_clusters, max_extra_msa = [int(x) for x in args.max_msa.split(":")]
    return {
        "N": len(feature_dict["msa"]),
        "L": len(feature_dict["residue_index"]),
        "use_ptm": args.use_ptm,
        "use_turbo": args.use_turbo,
        "num_relax": args.num_relax,
        "max_recycles": args.max_recycles,
        "tol": 0.0,
        "num_ensemble": args.num_ensemble,
        "max_msa_clusters": max_msa_clusters,
        "max_extra_msa": max_extra_msa,
        "is_training": args.is_training,
    }


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
    _detect_device()
    if not args.use_ptm and args.rank_by == "pTMscore":
        print("\nWARNING: models will be ranked by pLDDT, 'use_ptm' is needed to compute pTMscore", flush=True)
        args.rank_by = "pLDDT"

    allele_sequence_csv = pd.read_csv(args.allele_sequence_csv)
    allele_to_sequence = allele_sequence_csv.set_index("allele")["seqs"].to_dict()

    runner = None
    folding_input_df = pd.read_csv(args.input_csv)
    folding_input_df = folding_input_df.sort_values(by=[args.allele_col_name, args.peptide_col_name])

    n_unique_pairs = folding_input_df[[args.allele_col_name, args.peptide_col_name]].drop_duplicates().shape[0]
    print(f"\nUnique [allele, peptide] pairs: {n_unique_pairs} (total rows: {len(folding_input_df)})", flush=True)

    folding_input_df = folding_input_df.reset_index(drop=True)

    print("\nRunning these samples: " +  str(args.start) + " to " + str(args.end) + " of " + str(len(folding_input_df)))

    folding_input_df = folding_input_df.iloc[args.start:args.end]
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
        output_dir = None
        if args.output_dir:
            output_dir = os.path.join(args.output_dir, full_sequence_name)

        if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, f"{full_sequence_name}.pdb")):
            print(f"\nSkipping {full_sequence_name} because it already exists.", flush=True)
            continue

        I = cf_af.prep_inputs(full_sequence_folding, full_sequence_name, args.homooligomer, output_dir=output_dir, clean=False)
        print(f"\nPrepped inputs for {full_sequence_name}.", flush=True)
        mod_I = cf_af.prep_msa(
            I,
            args.msa_method,
            args.add_custom_msa,
            args.msa_format,
            args.pair_mode,
            args.pair_cov,
            args.pair_qid,
            TMP_DIR=args.tmp_dir,
        )
        print(f"\nPrepped MSA for {full_sequence_name}.", flush=True)

        feature_dict = cf_af.prep_feats(mod_I, clean=False)
        opt = _build_options(feature_dict, args)

        if args.use_turbo:
            if runner is not None:
                runner = cf_af.prep_model_runner(opt, old_runner=runner, params_loc=args.params_loc)
            else:
                runner = cf_af.prep_model_runner(opt, params_loc=args.params_loc)

        _, _ = cf_af.run_alphafold(
            feature_dict,
            opt,
            runner,
            args.num_models,
            args.num_samples,
            args.subsample_msa,
            rank_by=args.rank_by,
            show_images=args.show_images,
            params_loc=args.params_loc,
        )
        print(f"\nRan AlphaFold2 for {full_sequence_name}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ColabFold/AlphaFold on sequence pairs.")
    parser.add_argument("--input-csv", required=True, help="CSV with sequences and peptides.")
    parser.add_argument("--start", type=int, required=True, help="Start index (0-based, inclusive).")
    parser.add_argument("--end", type=int, required=True, help="End index (0-based, inclusive).")
    parser.add_argument("--params-loc", required=True, help="Path to AlphaFold params directory.")
    parser.add_argument("--tmp-dir", default="/tmp/", help="Please use different tmp directory for each job.")
    parser.add_argument("--output-dir", help="Base output directory for predictions.")
    parser.add_argument("--allele-sequence-csv", default=f"{ROOT_DIR}/data/HLA_allele_sequences.csv")
    parser.add_argument("--allele-col-name", type=str, default="allele", help="Column name for allele.")
    parser.add_argument("--peptide-col-name", type=str, default="peptide", help="Column name for peptide.")
    parser.add_argument("--homooligomer", default="1:1", help="Homooligomer string passed to ColabFold.")
    parser.add_argument("--msa-method", default="mmseqs2", help="MSA method.")
    parser.add_argument("--add-custom-msa", action="store_true", help="Enable custom MSA.")
    parser.add_argument("--msa-format", default="fas", help="MSA format.")
    parser.add_argument("--pair-mode", default="unpaired", help="Pairing mode.")
    parser.add_argument("--pair-cov", type=int, default=50, help="Pairing coverage.")
    parser.add_argument("--pair-qid", type=int, default=20, help="Pairing QID.")
    parser.add_argument("--num-relax", type=int, default=None, help="Number of relax steps.")
    parser.add_argument("--rank-by", default="pLDDT", help="Ranking metric.")
    parser.add_argument("--no-use-turbo", action="store_false", dest="use_turbo", help="Disable turbo mode.")
    parser.set_defaults(use_turbo=True)
    parser.add_argument("--max-msa", default="256:512", help="Max MSA in 'clusters:extra' format.")
    parser.add_argument("--show-images", action="store_true", help="Show images during inference.")
    parser.add_argument("--num-models", type=int, default=1, help="Number of models.")
    parser.add_argument("--no-use-ptm", action="store_false", dest="use_ptm", help="Disable pTM model.")
    parser.set_defaults(use_ptm=True)
    parser.add_argument("--num-ensemble", type=int, default=1, help="Number of ensembles.")
    parser.add_argument("--max-recycles", type=int, default=3, help="Maximum recycles.")
    parser.add_argument("--is-training", action="store_true", help="Run in training mode.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples.")
    parser.add_argument("--no-subsample-msa", action="store_false", dest="subsample_msa", help="Disable MSA subsampling.")
    parser.set_defaults(subsample_msa=True)
    main(parser.parse_args())
