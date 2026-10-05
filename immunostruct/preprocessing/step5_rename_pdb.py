import argparse
import os
import shutil

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])


def _find_single_pdb(folder_path: str) -> str:
    pdb_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdb")]
    if len(pdb_files) != 1:
        raise ValueError(f"Expected 1 PDB in {folder_path}, found {len(pdb_files)}")
    return os.path.join(folder_path, pdb_files[0])


def _standardize_folder_name(folder: str) -> str:
    """
    Standardize the folder name to the format:
    HLA-A*01:01_PEPTIDE
    """
    parts = folder.split("_")
    if len(parts) >= 5 and parts[0] == "HLA":
        allele = f"HLA-{parts[1]}*{parts[2]}:{parts[3]}"
        peptide = "_".join(parts[4:])
        return f"{allele}_{peptide}"
    return folder


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    subfolders = [d for d in os.listdir(args.input_dir)
                  if os.path.isdir(os.path.join(args.input_dir, d))]

    for folder in sorted(subfolders):
        out_name = _standardize_folder_name(folder)
        folder_path = os.path.join(args.input_dir, folder)
        pdb_path = _find_single_pdb(folder_path)
        out_path = os.path.join(args.output_dir, f"{out_name}.pdb")
        shutil.copy2(pdb_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy and rename PDB files by folder name.")
    parser.add_argument("--input-dir", default=f"{ROOT_DIR}/data/pdb_files/IEDB/", type=str)
    parser.add_argument("--output-dir", default=f"{ROOT_DIR}/data/alphafold2_pdb_IEDB/", type=str)
    main(parser.parse_args())
