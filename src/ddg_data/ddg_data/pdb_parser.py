"""
pdb_parser
==========
PDB file parsing utilities.

This module is adapted from ProteinMPNN's protein_mpnn_utils.py and faithfully
preserves SPURS's parsing logic. Two parser variants are provided:

    parse_PDB_biounits / parse_single_PDB:
        Original ProteinMPNN parser. Returns a dict mapping
        seq_chain_<X> / coords_chain_<X> for each chain.

    alt_parse_PDB_biounits / alt_parse_PDB:
        Variant that additionally returns a `resn_list` — the ordered list of
        raw residue numbers (including insertion codes) as seen in the PDB.
        Required by datasets that need to map mutation positions (e.g. "T13A")
        back to indices in the parsed sequence.

A small helper `parse_pdb_directory_to_json` builds a JSON cache from a folder
of PDB files, exactly mirroring SPURS's `parse_pdb` behaviour.

The `fermi_transform` / `inverse_fermi_transform` helpers are kept here so the
module is self-contained.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Fermi transform helpers (used by some training scripts)
# ──────────────────────────────────────────────────────────────────────────────

def fermi_transform(x: float) -> float:
    """Sigmoid mapping used by some downstream papers to convert ΔΔG → fitness."""
    alpha = 3.0
    beta  = 0.4
    return 1.0 / (1.0 + np.exp(-beta * (x - alpha)))


def inverse_fermi_transform(x: float) -> float:
    """Inverse of `fermi_transform`."""
    alpha = 3.0
    beta  = 0.4
    eps   = 1e-12
    if x == 1.0:
        return 40.0
    if 0.0 < x < 1.0:
        return (alpha * beta - np.log(-1.0 + 1.0 / x + eps)) / beta
    if x == 0.0:
        return -40.0
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Amino-acid alphabet tables (3-letter ↔ 1-letter ↔ index)
# ──────────────────────────────────────────────────────────────────────────────

_ALPHA_1 = list("ARNDCQEGHILKMFPSTWYV-")
_ALPHA_3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "GAP",
]

_AA_1_TO_N = {a: n for n, a in enumerate(_ALPHA_1)}
_AA_3_TO_N = {a: n for n, a in enumerate(_ALPHA_3)}
_AA_N_TO_1 = {n: a for n, a in enumerate(_ALPHA_1)}
_STATES    = len(_ALPHA_1)


def _n_to_aa(x: np.ndarray) -> List[str]:
    """Convert an int array of AA indices into a list of single-letter strings."""
    x = np.array(x)
    if x.ndim == 1:
        x = x[None]
    return ["".join([_AA_N_TO_1.get(a, "-") for a in y]) for y in x]


# ──────────────────────────────────────────────────────────────────────────────
# Original ProteinMPNN-style parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_PDB_biounits(
    pdb_path: str,
    atoms: Tuple[str, ...] = ("N", "CA", "C"),
    chain: Optional[str] = None,
) -> Tuple[Union[np.ndarray, str], Union[List[str], str]]:
    """
    Parse a single PDB file and return coords + sequence for one chain.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file.
    atoms : tuple of str
        Atom names to extract per residue (default: N, CA, C).
    chain : str or None
        Specific chain letter to extract. If None, returns all atoms regardless
        of chain assignment.

    Returns
    -------
    coords : np.ndarray of shape (L, len(atoms), 3) or "no_chain"
    sequence : list of str (length 1, the chain sequence) or "no_chain"
    """
    xyz, seq = {}, {}
    min_resn, max_resn = 1e6, -1e6

    with open(pdb_path, "rb") as f:
        for line in f:
            line = line.decode("utf-8", "ignore").rstrip()

            # Treat selenomethionine as methionine
            if line[:6] == "HETATM" and line[17:20] == "MSE":
                line = line.replace("HETATM", "ATOM  ").replace("MSE", "MET")

            if line[:4] != "ATOM":
                continue

            ch = line[21:22]
            if chain is not None and ch != chain:
                continue

            atom = line[12:16].strip()
            resi = line[17:20]
            resn = line[22:27].strip()
            x, y, z = [float(line[i : i + 8]) for i in (30, 38, 46)]

            # Handle insertion codes
            if resn[-1].isalpha():
                resa, resn = resn[-1], int(resn[:-1]) - 1
            else:
                resa, resn = "", int(resn) - 1

            if resn < min_resn:
                min_resn = resn
            if resn > max_resn:
                max_resn = resn

            xyz.setdefault(resn, {}).setdefault(resa, {})
            seq.setdefault(resn, {}).setdefault(resa, resi)

            if atom not in xyz[resn][resa]:
                xyz[resn][resa][atom] = np.array([x, y, z])

    seq_out: List[int] = []
    xyz_out: List[np.ndarray] = []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_out.append(_AA_3_TO_N.get(seq[resn][k], 20))
            else:
                seq_out.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_out.append(xyz[resn][k][atom])
                        else:
                            xyz_out.append(np.full(3, np.nan))
            else:
                for atom in atoms:
                    xyz_out.append(np.full(3, np.nan))

        coords = np.array(xyz_out).reshape(-1, len(atoms), 3)
        return coords, _n_to_aa(np.array(seq_out))
    except TypeError:
        return "no_chain", "no_chain"


def parse_single_PDB(
    pdb_path: str,
    input_chain_list: Optional[List[str]] = None,
    ca_only: bool = False,
) -> Optional[Tuple[str, dict]]:
    """
    Parse a single PDB file across all chains and produce a SPURS-style dict
    with `seq_chain_<X>` and `coords_chain_<X>` entries per chain.

    Returns
    -------
    (name, my_dict) : tuple
        name : str — PDB stem (filename without directory or '.pdb')
        my_dict : dict — SPURS-format parsed PDB
    """
    init_alphabet = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z",
    ]
    extra_alphabet = [str(i) for i in range(300)]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    my_dict: dict = {}
    s = 0
    concat_seq = ""

    for letter in chain_alphabet:
        sidechain_atoms = ["CA"] if ca_only else ["N", "CA", "C", "O"]
        xyz, seq = parse_PDB_biounits(pdb_path, atoms=sidechain_atoms, chain=letter)

        if isinstance(xyz, str):  # "no_chain"
            continue

        concat_seq += seq[0]
        my_dict[f"seq_chain_{letter}"] = seq[0]
        coords_dict_chain: dict = {}
        if ca_only:
            coords_dict_chain[f"CA_chain_{letter}"] = xyz.tolist()
        else:
            coords_dict_chain[f"N_chain_{letter}"]  = xyz[:, 0, :].tolist()
            coords_dict_chain[f"CA_chain_{letter}"] = xyz[:, 1, :].tolist()
            coords_dict_chain[f"C_chain_{letter}"]  = xyz[:, 2, :].tolist()
            coords_dict_chain[f"O_chain_{letter}"]  = xyz[:, 3, :].tolist()
        my_dict[f"coords_chain_{letter}"] = coords_dict_chain
        s += 1

    fi = pdb_path.rfind("/")
    my_dict["name"]          = pdb_path[(fi + 1) : -4]
    my_dict["num_of_chains"] = s
    my_dict["seq"]           = concat_seq

    if s <= len(chain_alphabet):
        return my_dict["name"], my_dict
    return None


def parse_pdb_dir(pdb_files: List[str]) -> Iterator[Tuple[str, dict]]:
    """Yield `(name, my_dict)` for each PDB in `pdb_files`."""
    for f in pdb_files:
        result = parse_single_PDB(f)
        if result is not None:
            yield result


def parse_pdb_directory_to_json(
    structure_dir: str,
    output_json_path: str,
    overwrite: bool = False,
) -> None:
    """
    Walk a directory of PDB files and write a JSON cache mapping
    pdb_stem -> parsed_dict.

    Mirrors SPURS's `parse_pdb()` exactly. Skipped if the JSON already exists
    (unless `overwrite=True`).
    """
    if os.path.exists(output_json_path) and not overwrite:
        return

    pdb_files = glob.glob(os.path.join(structure_dir, "*.pdb"))
    json_dict: dict = {}
    for name, parsed in tqdm(parse_pdb_dir(pdb_files), total=len(pdb_files),
                              desc=f"Parsing {structure_dir}"):
        json_dict[name] = parsed

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(json_dict, f)


# ──────────────────────────────────────────────────────────────────────────────
# Alt parser — adds resn_list (raw residue numbers including insertion codes)
# ──────────────────────────────────────────────────────────────────────────────

def alt_parse_PDB_biounits(
    pdb_path: str,
    atoms: Tuple[str, ...] = ("N", "CA", "C"),
    chain: Optional[str] = None,
) -> Tuple[Union[np.ndarray, str], Union[List[str], str], Union[List[str], str]]:
    """
    Variant of `parse_PDB_biounits` that additionally returns `resn_list`,
    the ordered list of raw residue numbers as seen in the PDB (including
    insertion codes).

    This list is critical for mapping mutation strings like "T13A" back to
    indices in the parsed sequence, since the PDB may use non-contiguous
    or insertion-coded numbering.
    """
    xyz, seq = {}, {}
    min_resn, max_resn = 1e6, -1e6
    resn_list: List[str] = []

    with open(pdb_path, "rb") as f:
        for line in f:
            line = line.decode("utf-8", "ignore").rstrip()

            if line[:6] == "HETATM" and line[17:20] == "MSE":
                line = line.replace("HETATM", "ATOM  ").replace("MSE", "MET")

            if line[:4] != "ATOM":
                continue

            ch = line[21:22]
            if chain is not None and ch != chain:
                continue

            atom = line[12:16].strip()
            resi = line[17:20]
            resn_raw = line[22:27].strip()

            if resn_raw not in resn_list:
                resn_list.append(resn_raw)   # raw, including insertion codes

            x, y, z = [float(line[i : i + 8]) for i in (30, 38, 46)]

            if resn_raw[-1].isalpha():
                resa, resn = resn_raw[-1], int(resn_raw[:-1]) - 1
            else:
                resa, resn = "", int(resn_raw) - 1

            if resn < min_resn:
                min_resn = resn
            if resn > max_resn:
                max_resn = resn

            xyz.setdefault(resn, {}).setdefault(resa, {})
            seq.setdefault(resn, {}).setdefault(resa, resi)

            if atom not in xyz[resn][resa]:
                xyz[resn][resa][atom] = np.array([x, y, z])

    seq_out, xyz_out = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_out.append(_AA_3_TO_N.get(seq[resn][k], 20))
            else:
                seq_out.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_out.append(xyz[resn][k][atom])
                        else:
                            xyz_out.append(np.full(3, np.nan))
            else:
                for atom in atoms:
                    xyz_out.append(np.full(3, np.nan))

        coords = np.array(xyz_out).reshape(-1, len(atoms), 3)
        return coords, _n_to_aa(np.array(seq_out)), list(dict.fromkeys(resn_list))
    except TypeError:
        return "no_chain", "no_chain", "no_chain"


def alt_parse_PDB(
    pdb_path: str,
    input_chain_list: Optional[List[str]] = None,
    ca_only: bool = False,
    side_chains: bool = False,
) -> List[dict]:
    """
    Alt variant of `parse_single_PDB`: returns a list of dicts (one per biounit,
    typically just one). Each dict contains a `resn_list` field.

    `side_chains=True` enables extraction of a larger atom set including
    side-chain atoms (CB/CG/CD/...).
    """
    init_alphabet = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z",
    ]
    extra_alphabet = [str(i) for i in range(300)]
    chain_alphabet = init_alphabet + extra_alphabet
    if input_chain_list:
        chain_alphabet = input_chain_list

    out: List[dict] = []
    my_dict: dict = {"resn_list": []}
    s = 0
    concat_seq = ""
    resn_list_final: List[str] = []

    for letter in chain_alphabet:
        if ca_only:
            sidechain_atoms = ["CA"]
        elif side_chains:
            sidechain_atoms = [
                "N", "CA", "C", "O", "CB",
                "CG", "CG1", "OG1", "OG2", "CG2", "OG", "SG",
                "CD", "SD", "CD1", "ND1", "CD2", "OD1", "OD2", "ND2",
                "CE", "CE1", "NE1", "OE1", "NE2", "OE2", "NE", "CE2", "CE3",
                "NZ", "CZ", "CZ2", "CZ3", "CH2", "OH", "NH1", "NH2",
            ]
        else:
            sidechain_atoms = ["N", "CA", "C", "O"]

        xyz, seq, resn_list = alt_parse_PDB_biounits(
            pdb_path, atoms=sidechain_atoms, chain=letter
        )
        if isinstance(xyz, str):  # "no_chain"
            continue

        concat_seq += seq[0]
        my_dict[f"seq_chain_{letter}"] = seq[0]
        coords_dict_chain: dict = {}
        if ca_only:
            coords_dict_chain[f"CA_chain_{letter}"] = xyz.tolist()
        else:
            coords_dict_chain[f"N_chain_{letter}"]  = xyz[:, 0, :].tolist()
            coords_dict_chain[f"CA_chain_{letter}"] = xyz[:, 1, :].tolist()
            coords_dict_chain[f"C_chain_{letter}"]  = xyz[:, 2, :].tolist()
            coords_dict_chain[f"O_chain_{letter}"]  = xyz[:, 3, :].tolist()
        my_dict[f"coords_chain_{letter}"] = coords_dict_chain
        s += 1
        resn_list_final = resn_list   # keep the last seen — matches MultimodalDDG

    fi = pdb_path.rfind("/")
    my_dict["name"]          = pdb_path[(fi + 1) : -4]
    my_dict["num_of_chains"] = s
    my_dict["seq"]           = concat_seq
    my_dict["resn_list"]     = list(resn_list_final) if resn_list_final else []

    if s <= len(chain_alphabet):
        out.append(my_dict)
    return out
