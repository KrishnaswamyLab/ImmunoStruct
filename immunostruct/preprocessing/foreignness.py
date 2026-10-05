#!/usr/bin/env python3
r"""Peptide foreignness score -- Luksza et al. 2017 cross-reactivity model.

    Z(s) = sum_e exp( -k * (a - |s,e|) )     a = 26, k = 4.86936
    R(s) = Z / (1 + Z)                       in [0, 1]

``|s,e|`` is the best local alignment score (BLOSUM62, affine gaps: open 11,
extend 1) between peptide ``s`` and an IEDB immunogenic epitope ``e``, and ``R``
is the TCR-recognition probability under the multistate thermodynamic model of
Luksza et al. 2017 (Nature 551:517).

Following antigen.garnish, the sum runs over the HSPs that ``blastp`` returns
rather than over the whole IEDB -- see the implementation notes below this
docstring for why that distinction matters, and for how to fetch the frozen
reference data. NumPy is the only hard dependency, plus ``blastp`` on PATH.

USAGE
-----
    # score peptides, reproducing the released Foreignness_Score column
    python foreignness.py --peptides peps.txt --db data/iedb.bdb -o out.csv

    # built-in regression checks
    python foreignness.py --db data/Mu_iedb.fasta

``--peptides`` takes one peptide per line, or a delimited table with
``--peptide-col NAME``. Output is ``peptide,Foreignness_Score``, one row per
input row and in input order.

To score the peptides of a property table as part of the Mprop pipeline, use
``biochem_properties.py --compute-foreignness`` instead, which calls
``foreignness_for_table`` below.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# --------------------------------------------------------------------------- #
# IMPLEMENTATION NOTES
#
# --------------------------------------------------------------------------- #
#
# WHY THE SUM RUNS OVER BLAST HSPs
#   antigen.garnish does not score a peptide against every reference epitope:
#   it runs blastp, then re-scores only the aligned segments of the returned
#   HSPs (qseq vs sseq) and sums those. A peptide for which blastp returns no
#   HSP scores exactly 0.
#
#
#
# REFERENCE DATA
#   The released column used the HUMAN IEDB set from the antigen.garnish 2.3.0
#   data bundle, frozen 2020-09-25:
#
#     curl -fsSL https://s3.amazonaws.com/get.rech.io/antigen.garnish-2.3.0.tar.gz -o ag.tar.gz
#     tar -xzf ag.tar.gz -C data --strip-components=1 --wildcards '*iedb*'  # GNU tar
#     tar -xzf ag.tar.gz -C data --strip-components=1 '*iedb*'              # bsdtar (macOS)
#
#   That yields iedb.fasta (human, 2,554 unique epitopes) and iedb.bdb.* (its
#   blastp database), plus Mu_iedb.fasta (mouse, 11,682). Use the frozen bundle
#   rather than a fresh IEDB export: the IEDB grows continuously and a larger
#   reference set raises Z monotonically, so scores computed against today's
#   IEDB are comparable neither to the released column nor to each other across
#   dates.
#

__all__ = [
    "A_DEFAULT",
    "BLOSUM62",
    "K_DEFAULT",
    "encode",
    "foreignness_for_table",
    "foreignness_score_blast",
]

# data/, relative to this file (repo_root/immunostruct/preprocessing/).
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data")

# --------------------------------------------------------------------------- #
# model parameters
# --------------------------------------------------------------------------- #

A_DEFAULT = 26.0
K_DEFAULT = 4.86936

GAP_OPEN = 11
GAP_EXTEND = 1

# --------------------------------------------------------------------------- #
# BLOSUM62
# --------------------------------------------------------------------------- #

ALPHABET = "ARNDCQEGHILKMFPSTWYV"

_BLOSUM62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""

BLOSUM62 = np.array(
    [[int(x) for x in line.split()] for line in _BLOSUM62_ROWS.strip().splitlines()],
    dtype=np.int16,
)

assert BLOSUM62.shape == (20, 20), "BLOSUM62 must be 20x20"
assert (BLOSUM62 == BLOSUM62.T).all(), "BLOSUM62 must be symmetric"

# CODE[ord(aa)] -> index into ALPHABET, or -1 for non-canonical residues
CODE = np.full(128, -1, dtype=np.int8)
for _i, _aa in enumerate(ALPHABET):
    CODE[ord(_aa)] = _i

_ALLOWED = set(ALPHABET)


def encode(seq):
    """Encode a peptide string to int8 codes. Raises on non-canonical residues."""
    arr = CODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    if (arr < 0).any():
        bad = {c for c in seq if c not in ALPHABET}
        raise ValueError(f"non-canonical residue(s) {sorted(bad)} in {seq!r}")
    return arr


def _sanitize(seq):
    """Normalise one sequence, or return '' if it is not a usable peptide.

    Strips the '-' and '*' padding out of alignment output, the way
    antigen.garnish sanitises BLAST results; anything left outside ALPHABET
    disqualifies the sequence.
    """
    s = str(seq).strip().upper().replace("-", "").replace("*", "")
    return s if s and set(s) <= _ALLOWED else ""


# --------------------------------------------------------------------------- #
# local alignment
# --------------------------------------------------------------------------- #

def affine_batch(Q, T):
    """Exact affine-gap local alignment score for batches of equal-shape pairs.

    Q : (B, m) int8 codes, T : (B, n) int8 codes -- pair b is (Q[b], T[b])
    returns (B,) int32

    Gotoh's recurrence with one gap state per direction. A gap of length L costs
    GAP_OPEN + L, so the first gapped residue costs GAP_OPEN + GAP_EXTEND = 12 --
    the Biostrings and NCBI BLAST convention, and what garnish's
    gapOpening = -11, gapExtension = -1 means.
    """
    B, m = Q.shape
    n = T.shape[1]
    sub = BLOSUM62[Q[:, :, None], T[:, None, :]].astype(np.int32)  # (B, m, n)
    open_cost = GAP_OPEN + GAP_EXTEND
    neg = np.int32(-(10 ** 6))

    H_prev = np.zeros((B, n + 1), dtype=np.int32)
    F_prev = np.full((B, n + 1), neg, dtype=np.int32)
    best = np.zeros(B, dtype=np.int32)

    for i in range(m):
        H_cur = np.zeros((B, n + 1), dtype=np.int32)
        F_cur = np.full((B, n + 1), neg, dtype=np.int32)
        E = np.full(B, neg, dtype=np.int32)
        row = sub[:, i, :]
        for j in range(1, n + 1):
            E = np.maximum(H_cur[:, j - 1] - open_cost, E - GAP_EXTEND)
            F = np.maximum(H_prev[:, j] - open_cost, F_prev[:, j] - GAP_EXTEND)
            h = H_prev[:, j - 1] + row[:, j - 1]
            np.maximum(h, E, out=h)
            np.maximum(h, F, out=h)
            np.maximum(h, 0, out=h)
            H_cur[:, j] = h
            F_cur[:, j] = F
            np.maximum(best, h, out=best)
        H_prev, F_prev = H_cur, F_cur
    return best


def affine_pairs(Qseqs, Tseqs, chunk=2_000_000):
    """Affine-gap local alignment scores for a list of (query, target) code pairs.

    Groups pairs by (len(query), len(target)) and dispatches to affine_batch.
    """
    out = np.zeros(len(Qseqs), dtype=np.int32)
    groups = {}
    for idx, (q, t) in enumerate(zip(Qseqs, Tseqs)):
        groups.setdefault((len(q), len(t)), []).append(idx)
    for (m, n), idxs in groups.items():
        ii = np.asarray(idxs)
        for lo in range(0, ii.size, chunk):
            sl = ii[lo:lo + chunk]
            Q = np.stack([np.asarray(Qseqs[i]) for i in sl])
            T = np.stack([np.asarray(Tseqs[i]) for i in sl])
            out[sl] = affine_batch(Q, T)
    return out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

BLAST_OUTFMT = ("10 qseqid sseqid qseq qstart qend sseq sstart send length "
                "mismatch pident evalue bitscore")


def run_blastp(peptides, db, threads=1):
    """Run antigen.garnish's blastp call. Returns [(query_index, qseq, sseq), ...].

    Flags match garnish exactly, including the absence of -task blastp-short.
    Aligned segments come back raw; the caller sanitises them.
    """
    if shutil.which("blastp") is None:
        raise RuntimeError(
            "blastp not found on PATH. Install NCBI BLAST+ (or e.g. "
            "'module load BLAST+')."
        )
    tmp = tempfile.mkdtemp(prefix="foreignness_blast_")
    qfa = os.path.join(tmp, "q.fa")
    out = os.path.join(tmp, "out.csv")
    try:
        with open(qfa, "w") as fh:
            for i, pep in enumerate(peptides):
                fh.write(f">{i}\n{pep}\n")
        # BLAST reads -db as a space-separated LIST of databases, so a path
        # holding a space is silently split and the run fails. Quoting inside
        # the argument value is how BLAST+ escapes that.
        db_arg = f'"{db}"' if any(c.isspace() for c in str(db)) else str(db)
        cmd = [
            "blastp", "-query", qfa, "-db", db_arg,
            "-evalue", "100000000",
            "-matrix", "BLOSUM62",
            "-gapopen", "11", "-gapextend", "1",
            "-out", out,
            "-num_threads", str(threads),
            "-outfmt", BLAST_OUTFMT,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"blastp failed: {res.stderr.strip()[:500]}")
        hits = []
        if os.path.exists(out):
            with open(out) as fh:
                for line in fh:
                    f = line.rstrip("\n").split(",")
                    if len(f) >= 6:
                        hits.append((int(f[0]), f[2], f[5]))  # index, qseq, sseq
        return hits
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def foreignness_score_blast(peptides, db, a=A_DEFAULT, k=K_DEFAULT, threads=1):
    """Foreignness score R = Z/(1+Z) in [0, 1] for a list of clean peptides.

    Sums only over the HSPs blastp returns, re-scoring each HSP's aligned
    segments; peptides with no HSP score exactly 0. This is what reproduces the
    released column -- see the implementation notes at the top of this file.

    np.exp cannot overflow here for peptide-length queries: it would need an
    alignment score above a + 709/k ~ 171, and an 11-mer caps out near 121.
    """
    qi, qs, ss = [], [], []
    for i, q, s in run_blastp(peptides, db, threads=threads):
        q, s = _sanitize(q), _sanitize(s)
        if q and s:
            qi.append(i)
            qs.append(encode(q))
            ss.append(encode(s))
    Z = np.zeros(len(peptides), dtype=np.float64)
    if qi:
        sw = affine_pairs(qs, ss).astype(np.float64)
        np.add.at(Z, np.asarray(qi), np.exp(-k * (a - sw)))
    return Z / (1.0 + Z)


# --------------------------------------------------------------------------- #
# row-aligned entry point, for the Mprop pipeline
# --------------------------------------------------------------------------- #

def foreignness_for_table(seqs, db, a=A_DEFAULT, k=K_DEFAULT, threads=1):
    """Foreignness score per row, aligned to `seqs`. NaN where unscorable.

    Unlike `foreignness_score_blast`, which takes a clean peptide list, this
    preserves the caller's row order and row count: the unique canonical
    peptides are scored once and mapped back by sequence, and any row whose
    peptide is empty or holds a non-canonical residue gets NaN. The number of
    such rows is reported.

    NaN is not inert downstream. `add_smoothed` runs `gaussian_filter1d` over
    this column, which propagates NaN, so one unscorable row corrupts
    `smoothed_foreign` for roughly +/-3 sigma rows around it -- the same hazard
    the SASA stage already carries. Drop or repair those rows before smoothing.
    """
    seqs = list(seqs)

    cleaned = {}
    for i, s in enumerate(seqs):
        pep = _sanitize(s)
        if pep:
            cleaned[i] = pep

    out = np.full(len(seqs), np.nan, dtype=np.float64)
    dropped = len(seqs) - len(cleaned)
    if not cleaned:
        print(f"  warning: no scorable peptides in {len(seqs)} rows (all NaN)")
        return out

    unique = sorted(set(cleaned.values()))
    print(f"  scoring {len(unique)} unique peptides from {len(seqs)} rows")
    scores = foreignness_score_blast(unique, db, a=a, k=k, threads=threads)

    lut = dict(zip(unique, scores))
    for i, pep in cleaned.items():
        out[i] = lut[pep]

    if dropped:
        print(f"  warning: {dropped}/{len(seqs)} rows have an empty or "
              "non-canonical peptide (NaN)")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _read_queries(path, peptide_col=None):
    if peptide_col:
        import csv
        with open(path, newline="") as fh:
            first = fh.readline()
            fh.seek(0)
            delim = "\t" if "\t" in first else ","
            rows = list(csv.DictReader(fh, delimiter=delim))
        if not rows or peptide_col not in rows[0]:
            raise SystemExit(f"column {peptide_col!r} not found in {path}")
        return [r[peptide_col] for r in rows]
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--peptides",
                   help="one peptide per line, or a table with --peptide-col")
    p.add_argument("--peptide-col", default=None,
                   help="column name if --peptides is a delimited table")
    p.add_argument("--db", default=os.path.join(DEFAULT_DATA_DIR, "iedb.bdb"),
                   help="blastp database prefix")
    p.add_argument("-o", "--out", default="-", help="output CSV ('-' for stdout)")
    p.add_argument("-a", type=float, default=A_DEFAULT, help=f"default {A_DEFAULT}")
    p.add_argument("-k", type=float, default=K_DEFAULT, help=f"default {K_DEFAULT}")
    p.add_argument("--threads", type=int, default=1, help="blastp threads")
    args = p.parse_args(argv)
    if not args.peptides:
        p.error("--peptides is required")

    peptides = _read_queries(args.peptides, args.peptide_col)
    if not peptides:
        p.error("no peptides in input")
    if not os.path.exists(args.db + ".pin"):
        p.error(f"blast database not found: {args.db}\n"
                "See REFERENCE DATA in the notes at the top of this file.")

    print(f"{len(peptides)} rows, db={args.db} (a={args.a}, k={args.k})",
          file=sys.stderr)
    scores = foreignness_for_table(peptides, args.db, a=args.a, k=args.k,
                                   threads=args.threads)

    rows = ["peptide,Foreignness_Score\n"]
    rows += [f"{pep},{sc:.17g}\n" for pep, sc in zip(peptides, scores)]
    if args.out == "-":
        sys.stdout.writelines(rows)
    else:
        with open(args.out, "w") as fh:
            fh.writelines(rows)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
