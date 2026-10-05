"""
Compute the ImmunoStruct biochemical meta-properties (Mprop1 / Mprop2).

Mprop1 and Mprop2 are the 2-d property vector each model consumes.  This module
builds them from a peptide sequence and, where the spec calls for it, a pMHC
structure, in four stages:

    1. 84 biochemical properties per peptide from the `peptides` package
       (75 descriptor scales + 9 scalars).
    2. Solvent-accessible surface area per structure, via mdtraj Shrake-Rupley.
    3. The foreignness score, then a quantile transform of it and Gaussian
       smoothing of both foreignness and SASA.
    4. Mprop1, Mprop2 and master_property_score, each the mean of a min-max
       scaled subset of the columns above.

Which columns feed which meta-property is set per dataset in SPECS.

The foreignness score of stage 3 is computed by the foreignness module, which
implements the Luksza et al. multistate thermodynamic model; pass
--compute-foreignness, or supply the column yourself.  See that module's
docstring for the reference data it needs.

    pip install pandas numpy scipy scikit-learn peptides mdtraj

mdtraj is only needed with --pdb-dir; the cedar_wt spec uses no structures.
--compute-foreignness additionally needs blastp on PATH.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

__all__ = [
    "DESCRIPTORS_75",
    "SCALARS_9",
    "SPECS",
    "add_mprops",
    "add_smoothed",
    "compute_descriptors",
    "compute_sasa",
    "normalize_allele",
    "sasa_for_table",
    "structure_keys",
]

# data/HLA_27_seqs.csv, relative to this file (repo_root/immunostruct/preprocessing/).
DEFAULT_HLA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "HLA_27_seqs.csv")


# ---------------------------------------------------------------------------
# The 84 biochemical properties
# ---------------------------------------------------------------------------

DESCRIPTORS_75 = (
    [f"BLOSUM{i}" for i in range(1, 11)]
    + ["PP1", "PP2", "PP3"]
    + [f"F{i}" for i in range(1, 7)]
    + [f"KF{i}" for i in range(1, 11)]
    + ["MSWHIM1", "MSWHIM2", "MSWHIM3"]
    + [f"E{i}" for i in range(1, 6)]
    + [f"ProtFP{i}" for i in range(1, 9)]
    + [f"SV{i}" for i in range(1, 5)]
    + [f"ST{i}" for i in range(1, 9)]
    + [f"T{i}" for i in range(1, 6)]
    + [f"VHSE{i}" for i in range(1, 9)]
    + [f"Z{i}" for i in range(1, 6)]
)

SCALARS_9 = [
    "aliphathic_index",
    "boman",
    "charge",
    "hphobic",
    "hphobic2",
    "instability",
    "iso_epoint",
    "mw",
    "mz",
]

# ---------------------------------------------------------------------------
# Per-dataset specs: which column each stage reads, and which columns are
# averaged into each meta-property.
#
#   peptide_col        column holding the peptide sequence
#   foreign_col        raw foreignness score, or None to skip stage 3
#   sasa_col           raw SASA column, or None to skip stage 2
#   sasa_sigma         Gaussian sigma applied to the SASA column
#   smoothed_sasa_col  name written by stage 3
#   mprop1 / mprop2    columns averaged into each meta-property
#   master             columns averaged into master_property_score
#   suffix             appended to the Mprop output names
# ---------------------------------------------------------------------------

SPECS = {
    "iedb": dict(
        peptide_col="peptide",
        foreign_col="Foreignness_Score",
        sasa_col="sasa_af",
        sasa_sigma=5,
        smoothed_sasa_col="smoothed_af_sasa",
        mprop1=["smoothed_foreign", "aliphathic_index", "instability", "ProtFP1", "hphobic2"],
        mprop2=["mw", "mz", "iso_epoint", "hphobic", "charge", "boman", "smoothed_af_sasa"],
        master=["mw", "mz", "aliphathic_index", "instability", "iso_epoint",
                "hphobic2", "hphobic", "ProtFP1", "charge", "boman", "smoothed_foreign"],
        suffix="",
    ),
    "cedar": dict(
        peptide_col="mut_pep",
        foreign_col="foreign",
        sasa_col="sasa_score",
        sasa_sigma=3,
        smoothed_sasa_col="smoothed_sasa_cedar",
        mprop1=["ProtFP1", "T1", "ProtFP2"],
        mprop2=["smoothed_sasa_cedar", "smoothed_foreign", "Z1", "ProtFP3", "Z2", "T2", "Z3"],
        master=["smoothed_sasa_cedar", "ProtFP1", "ProtFP2", "T1", "smoothed_foreign",
                "Z1", "ProtFP3", "Z2", "T2", "Z3"],
        suffix="",
    ),
    # Sequence descriptors only: no SASA, no foreignness, no smoothing.
    "cedar_wt": dict(
        peptide_col="wt_pep",
        foreign_col=None,
        sasa_col=None,
        sasa_sigma=None,
        smoothed_sasa_col=None,
        mprop1=["F1", "F2", "F3", "ProtFP1", "SV1", "VHSE1"],
        mprop2=["BLOSUM1", "E1", "VHSE8", "Z1"],
        master=["F1", "F2", "F3", "ProtFP1", "SV1", "VHSE1",
                "BLOSUM1", "E1", "VHSE8", "Z1"],
        suffix="_wt",
    ),
}


# ---------------------------------------------------------------------------
# Stage 1: the 84 properties
# ---------------------------------------------------------------------------

def compute_descriptors(seqs):
    """84 biochemical properties per peptide, as a DataFrame aligned to `seqs`."""
    import peptides

    rows = []
    for seq in seqs:
        pep = peptides.Peptide(seq)
        desc = pep.descriptors()
        row = {key: desc[key] for key in DESCRIPTORS_75}
        row.update(
            aliphathic_index=pep.aliphatic_index(),
            boman=pep.boman(),
            charge=pep.charge(pKscale="EMBOSS"),
            hphobic=pep.hydrophobic_moment(),
            hphobic2=pep.hydrophobicity(scale="Aboderin"),
            instability=pep.instability_index(),
            iso_epoint=pep.isoelectric_point(),
            mw=pep.molecular_weight(),
            mz=pep.mz(),
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=DESCRIPTORS_75 + SCALARS_9)


# ---------------------------------------------------------------------------
# Stage 2: SASA
# ---------------------------------------------------------------------------

def compute_sasa(complex_file, mode="complex", n_peptide_residues=None):
    """Shrake-Rupley SASA (nm^2) of one pMHC structure.

    mode="complex"       sum over every residue in the structure.
    mode="peptide_tail"  sum over the last `n_peptide_residues` residues, where
                         the peptide sits in a fused MHC-then-peptide chain.
                         Works on single- and multi-chain files.
    mode="peptide_chain" sum over chain 1 only; requires a multi-chain file.
    """
    import mdtraj as md

    trajectory = md.load(complex_file)
    sasa = md.shrake_rupley(trajectory, mode="residue",
                            probe_radius=0.14, n_sphere_points=960)

    if mode == "complex":
        return float(sasa[0].sum())

    if mode == "peptide_tail":
        if not n_peptide_residues:
            raise ValueError("mode='peptide_tail' needs n_peptide_residues")
        return float(sasa[0][-n_peptide_residues:].sum())

    topology = trajectory.topology
    if topology.n_chains < 2:
        raise ValueError(
            f"{os.path.basename(complex_file)} has a single fused chain, so "
            "mode='peptide_chain' cannot isolate the peptide; use 'complex' or "
            "'peptide_tail'."
        )
    interest_residues = [residue.index for residue in topology.chain(1).residues]
    return float(sasa[0][interest_residues].sum())


def structure_keys(df, spec, hla_path):
    """The 5-hex structure key for each row.

    Structure files are named `..._<5 hex>.pdb`, where the hex is
    sha1(hla_seq + peptide)[:5].  Tables carrying that key as `file_id_code` or
    `id` are used directly; otherwise it is derived from `allele` plus the
    peptide column, using the HLA sequences in `hla_path`.
    """
    if "file_id_code" in df.columns:
        return df["file_id_code"].astype(str)
    if "id" in df.columns:
        return df["id"].astype(str)

    if not hla_path:
        raise SystemExit(
            "this table has no 'file_id_code'/'id' column, so structure keys "
            "must be derived from allele + peptide; pass --hla-path."
        )

    # Imported here rather than at module scope: immunostruct.data pulls in
    # torch and dgl, which the other stages do not need.
    try:
        from ..data.utils import get_hash
    except ImportError:
        from immunostruct.data.utils import get_hash

    hla_df = pd.read_csv(hla_path)
    hla_seqs = dict(zip(hla_df["allele"], hla_df["seqs"]))

    allele = normalize_allele(df["allele"])
    unknown = sorted(set(allele) - set(hla_seqs))
    if unknown:
        raise SystemExit(f"alleles absent from {hla_path}: {unknown}")

    return pd.Series(
        [get_hash(hla_seqs[al] + pep)[:5]
         for al, pep in zip(allele, df[spec["peptide_col"]])],
        index=df.index,
    )


def normalize_allele(allele):
    """`HLA-A0201` -> `HLA-A*02:01`; already-normalized values pass through."""
    allele = allele.astype(str)
    if allele.str.contains(r"\*").all():
        return allele
    prefix, code = allele.str.split("-", n=1, expand=True)[0], \
                   allele.str.split("-", n=1, expand=True)[1]
    return prefix + "-" + code.str[0] + "*" + code.str[1:3] + ":" + code.str[3:]


def sasa_for_table(df, pdb_dir, spec, mode="peptide_tail", peptides_seq=None,
                   hla_path=None):
    """Attach the raw SASA column by matching each row to a PDB in `pdb_dir`.

    Structures are indexed by the 5-hex suffix of their filename and matched to
    rows by `structure_keys`.  Rows with no matching structure get NaN.
    """
    by_id = {}
    for name in os.listdir(pdb_dir):
        if name.endswith(".pdb"):
            by_id[name[:-4].rsplit("_", 1)[-1]] = os.path.join(pdb_dir, name)

    keys = structure_keys(df, spec, hla_path)

    # peptide_tail needs each row's peptide length, so it is not cacheable by
    # structure id alone; the other modes are.
    lengths = ([len(s) for s in peptides_seq] if peptides_seq is not None
               else [None] * len(keys))

    cache, out = {}, []
    for key, n_res in zip(keys, lengths):
        cache_key = (key, n_res)
        if cache_key not in cache:
            path = by_id.get(key)
            cache[cache_key] = (
                compute_sasa(path, mode=mode, n_peptide_residues=n_res)
                if path else np.nan
            )
        out.append(cache[cache_key])

    missing = int(np.isnan(out).sum())
    if missing:
        print(f"  warning: no structure found for {missing}/{len(out)} rows (NaN)")
    df[spec["sasa_col"]] = out
    return df


# ---------------------------------------------------------------------------
# Stage 3: quantile transform + Gaussian smoothing
# ---------------------------------------------------------------------------

def add_smoothed(df, spec):
    """Add quant_foreign, smoothed_foreign and the smoothed SASA column.

    The foreignness score is mapped to a normal distribution, then both it and
    the SASA column are smoothed with a Gaussian filter along the row axis, so
    each value is blended with those of its row-order neighbours.  Row order is
    part of the result: pass the table in the order the features were defined
    on.
    """
    if spec["foreign_col"] is not None:
        qt = QuantileTransformer(n_quantiles=50, random_state=0,
                                 output_distribution="normal")
        df["quant_foreign"] = qt.fit_transform(df[[spec["foreign_col"]]])
        df["smoothed_foreign"] = gaussian_filter1d(df["quant_foreign"].values, sigma=3)

    if spec["sasa_col"] is not None:
        df[spec["smoothed_sasa_col"]] = gaussian_filter1d(
            df[spec["sasa_col"]].values, sigma=spec["sasa_sigma"]
        )

    return df


# ---------------------------------------------------------------------------
# Stage 4: the meta-properties
# ---------------------------------------------------------------------------

def add_mprops(df, spec, scaler_out=None):
    """Add Mprop1, Mprop2 and master_property_score.

    Each is the mean of its spec's columns after min-max scaling.  The scaler is
    fit across the whole table, so the values are relative to the rows present.
    `scaler_out` writes the fitted per-column ranges to JSON.
    """
    suffix = spec["suffix"]
    ranges = {}

    for name, cols in (("Mprop1", spec["mprop1"]),
                       ("Mprop2", spec["mprop2"]),
                       ("master_property_score", spec["master"])):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"{name}: missing required columns {missing}")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[cols])
        out_col = name if name == "master_property_score" else name + suffix
        df[out_col] = scaled.mean(axis=1)
        ranges[out_col] = {
            "columns": cols,
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
        }

    if scaler_out:
        with open(scaler_out, "w") as handle:
            json.dump(ranges, handle, indent=2)
        print(f"  wrote scaler ranges -> {scaler_out}")

    return df


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recompute Mprop1/Mprop2 for an ImmunoStruct property table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, choices=sorted(SPECS),
                        help="which column spec to use")
    parser.add_argument("--in-table", required=True,
                        help="source table (.csv or tab-separated .txt)")
    parser.add_argument("--out-table", required=True)
    parser.add_argument("--pdb-dir",
                        help="compute SASA from these PDBs (needs mdtraj)")
    parser.add_argument("--sasa-mode", default="peptide_tail",
                        choices=["complex", "peptide_tail", "peptide_chain"],
                        help="which residues to sum SASA over: the trailing "
                             "peptide residues (default), the whole structure, "
                             "or chain 1")
    parser.add_argument("--hla-path", default=DEFAULT_HLA_PATH,
                        help="allele -> HLA sequence map, used to derive "
                             "structure keys for tables with no 'file_id_code' "
                             "column")
    parser.add_argument("--sasa-from",
                        help="CSV with a precomputed SASA column to join instead")
    parser.add_argument("--compute-foreignness", action="store_true",
                        help="compute the spec's foreignness column instead of "
                             "reading it from the table")
    parser.add_argument("--foreignness-db",
                        help="blastp database prefix for --compute-foreignness "
                             "(default: data/iedb.bdb)")
    parser.add_argument("--blast-threads", type=int, default=1,
                        help="threads for the blastp call")
    parser.add_argument("--scaler-out",
                        help="write the fitted min/max ranges here as JSON")
    parser.add_argument("--reuse-existing-props", action="store_true",
                        help="keep the table's own property columns instead "
                             "of recomputing them from sequence")
    args = parser.parse_args()

    spec = SPECS[args.dataset]

    sep = "\t" if args.in_table.endswith(".txt") else ","
    df = pd.read_csv(args.in_table, sep=sep)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].reset_index(drop=True)
    print(f"loaded {len(df)} rows from {args.in_table}")

    wanted = DESCRIPTORS_75 + SCALARS_9
    if args.reuse_existing_props and all(col in df.columns for col in wanted):
        print(f"reusing the table's own {len(wanted)} property columns")
    else:
        print("computing the 84 biochemical properties...")
        props = compute_descriptors(df[spec["peptide_col"]].tolist())
        for col in props.columns:
            if args.reuse_existing_props and col in df.columns:
                continue
            df[col] = props[col].values

    if spec["sasa_col"] is not None:
        if args.pdb_dir:
            print(f"computing SASA from {args.pdb_dir} (mode={args.sasa_mode})...")
            df = sasa_for_table(df, args.pdb_dir, spec, mode=args.sasa_mode,
                                peptides_seq=df[spec["peptide_col"]].tolist(),
                                hla_path=args.hla_path)
        elif args.sasa_from:
            print(f"joining SASA from {args.sasa_from}...")
            sasa_df = pd.read_csv(args.sasa_from)
            key = "id" if "id" in sasa_df.columns else "file_id_code"
            mapping = dict(zip(sasa_df[key].astype(str), sasa_df[spec["sasa_col"]]))
            join_on = "file_id_code" if "file_id_code" in df.columns else "id"
            df[spec["sasa_col"]] = df[join_on].astype(str).map(mapping)
        elif spec["sasa_col"] not in df.columns:
            raise SystemExit(
                f"spec '{args.dataset}' needs '{spec['sasa_col']}'; pass --pdb-dir "
                "or --sasa-from, or use a table that already has it."
            )
        else:
            print(f"reusing existing '{spec['sasa_col']}' column")

    if spec["foreign_col"] is not None:
        if args.compute_foreignness:
            # imported here: the foreignness module pulls in no new hard
            # dependency, but the other stages have no use for it.
            try:
                from .foreignness import DEFAULT_DATA_DIR, foreignness_for_table
            except ImportError:
                from foreignness import DEFAULT_DATA_DIR, foreignness_for_table

            db = args.foreignness_db or os.path.join(DEFAULT_DATA_DIR, "iedb.bdb")
            print("computing foreignness...")
            df[spec["foreign_col"]] = foreignness_for_table(
                df[spec["peptide_col"]].tolist(), db,
                threads=args.blast_threads,
            )
        elif spec["foreign_col"] not in df.columns:
            raise SystemExit(
                f"spec '{args.dataset}' needs '{spec['foreign_col']}'; pass "
                "--compute-foreignness, or use a table that already has it. "
                "The score is not part of the biochemical descriptors: see the "
                "foreignness module for how it is computed."
            )
        else:
            print(f"reusing existing '{spec['foreign_col']}' column")

    print("smoothing...")
    df = add_smoothed(df, spec)

    print("building meta-properties...")
    df = add_mprops(df, spec, scaler_out=args.scaler_out)

    out_sep = "\t" if args.out_table.endswith(".txt") else ","
    out_dir = os.path.dirname(args.out_table)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out_table, sep=out_sep, index=False)
    print(f"wrote {len(df)} rows -> {args.out_table}")


if __name__ == "__main__":
    main()
