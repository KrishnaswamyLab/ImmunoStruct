import argparse
import os

ROOT_DIR = "/".join(os.path.realpath(__file__).split("/")[:-3])


def main(args):
    os.system(f"colabfold_search {args.input_fasta} {args.msa_database_dir} {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA files to a3m files.")
    parser.add_argument("--input-fasta", required=True, help="Input FASTA file path for all records.")
    parser.add_argument("--msa-database-dir", required=True, help="Directory of MSA database containing `colabfold_envdb_202108_db` and `uniref30_2302_db`.")
    parser.add_argument("--output-dir", required=True, help="Output directory for a3m files.")
    main(parser.parse_args())
