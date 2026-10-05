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


def _pdb_exists(folder_path: str) -> bool:
    """Check if any PDB file exists in the folder."""
    if not os.path.isdir(folder_path):
        return False
    pdb_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdb")]
    return len(pdb_files) > 0


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

    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else len(folding_input_df)
    print("\nRunning these samples: " + str(start) + " to " + str(end) + " of " + str(len(folding_input_df)), flush=True)
    folding_input_df = folding_input_df.iloc[start:end]

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

        output_dir = os.path.join(args.output_dir, full_sequence_name) if args.output_dir else None
        msa_path = os.path.join(output_dir, "msa.pickle") if output_dir else None

        if not msa_path or not os.path.exists(msa_path):
            print(f"\nSkipping {full_sequence_name}: msa.pickle not found (run step1 first).", flush=True)
            continue

        if _pdb_exists(output_dir):
            print(f"\nSkipping {full_sequence_name}: PDB already exists.", flush=True)
            continue

        I = cf_af.prep_inputs(full_sequence_folding, full_sequence_name, args.homooligomer, output_dir=output_dir, clean=False)
        print(f"\nPrepared inputs for {full_sequence_name}.", flush=True)
        mod_I = cf_af.prep_msa(
            I,
            msa_method="precomputed",
            precomputed=msa_path,
        )
        print(f"\nPrepared MSA for {full_sequence_name}.", flush=True)

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
    parser = argparse.ArgumentParser(description="Run AlphaFold2 on precomputed MSAs (step 2: MSA → PDB).")
    parser.add_argument("--input-csv", required=True, help="CSV with all sequences and peptides.")
    parser.add_argument("--output-dir", required=True, help="Output directory for PDB files.")
    parser.add_argument("--params-loc", required=True, help="Path to AlphaFold2 params directory.")
    parser.add_argument("--start", type=int, default=None, help="Start index (0-based). Process all if not set.")
    parser.add_argument("--end", type=int, default=None, help="End index (0-based, exclusive). Process all if not set.")
    parser.add_argument("--allele-sequence-csv", default=f"{ROOT_DIR}/data/HLA_allele_sequences.csv")
    parser.add_argument("--allele-col-name", type=str, default="allele")
    parser.add_argument("--peptide-col-name", type=str, default="peptide")
    parser.add_argument("--homooligomer", default="1:1")
    parser.add_argument("--rank-by", default="pLDDT")
    parser.add_argument("--no-use-turbo", action="store_false", dest="use_turbo")
    parser.set_defaults(use_turbo=True)
    parser.add_argument("--max-msa", default="256:512")
    parser.add_argument("--show-images", action="store_true")
    parser.add_argument("--num-models", type=int, default=1)
    parser.add_argument("--no-use-ptm", action="store_false", dest="use_ptm")
    parser.set_defaults(use_ptm=True)
    parser.add_argument("--num-relax", type=int, default=None)
    parser.add_argument("--num-ensemble", type=int, default=1)
    parser.add_argument("--max-recycles", type=int, default=3)
    parser.add_argument("--is-training", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--no-subsample-msa", action="store_false", dest="subsample_msa")
    parser.set_defaults(subsample_msa=True)
    main(parser.parse_args())
