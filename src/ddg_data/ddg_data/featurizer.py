"""
featurizer
==========
ProteinMPNN-style featurisation and ESM-aware batch conversion.

Faithful reproduction of SPURS's `tied_featurize`, `get_pdb`, `Alphabet`,
`CoordBatchConverter`, and `Featurizer` — but with NO dependency on the
`spurs` package.

The only external dependency is `esm` (Meta's fair-esm) for the alphabet and
batch converter base class. If you cannot install esm, see the bottom of this
file for `_NoEsmAlphabet` — a minimal fallback that supports basic tokenisation.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import esm
    _HAS_ESM = True
except ImportError:
    esm = None
    _HAS_ESM = False


# ──────────────────────────────────────────────────────────────────────────────
# Alphabet wrapper (mirrors spurs.datamodules.datasets.data_utils.Alphabet)
# ──────────────────────────────────────────────────────────────────────────────

class Alphabet:
    """
    Thin wrapper around `esm.Alphabet`.

    Two modes are supported:
        - 'esm'          : ESM-1b architecture alphabet (default)
        - 'protein_mpnn' : ProteinMPNN's 20-AA alphabet with <pad>/<unk>

    Attribute access falls through to the underlying `esm.Alphabet` so that
    standard methods (`get_idx`, `padding_idx`, `mask_idx`, `prepend_bos`, ...)
    are available transparently.
    """

    def __init__(self, name: str = "esm", alphabet_cfg: Optional[Dict[str, Any]] = None):
        if not _HAS_ESM:
            raise ImportError(
                "fair-esm is required for the Alphabet class. "
                "Install with: pip install fair-esm"
            )
        alphabet_cfg = alphabet_cfg or {}
        self.name = name

        if name == "esm":
            self._alphabet           = esm.Alphabet.from_architecture("ESM-1b")
            self.add_special_tokens  = True
        elif name == "protein_mpnn":
            self._alphabet = esm.Alphabet(
                standard_toks=[
                    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
                    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
                ],
                prepend_toks=["<pad>", "<unk>"],
                append_toks=[],
                prepend_bos=False,
                append_eos=False,
            )
            self.add_special_tokens = False
        else:
            self._alphabet = esm.Alphabet(**alphabet_cfg)
            self.add_special_tokens = (
                self._alphabet.prepend_bos and self._alphabet.append_eos
            )

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._alphabet, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute `{name}`.")

    def __len__(self) -> int:
        return len(self._alphabet)

    def decode(
        self,
        batch_ids: torch.Tensor,
        return_as: str = "str",
        remove_special: bool = False,
    ) -> List[Any]:
        """Decode a tensor of token ids back to amino acid strings / lists."""
        ret: List[Any] = []
        for ids in batch_ids.cpu():
            if return_as == "str":
                line = "".join([self.get_tok(i) for i in ids])
                if remove_special:
                    line = (
                        line.replace(self.get_tok(self.mask_idx), "_")
                        .replace(self.get_tok(self.eos_idx), "")
                        .replace(self.get_tok(self.cls_idx), "")
                        .replace(self.get_tok(self.padding_idx), "")
                        .replace(self.get_tok(self.unk_idx), "-")
                    )
            else:
                line = [self.get_tok(i) for i in ids]
            ret.append(line)
        return ret


# ──────────────────────────────────────────────────────────────────────────────
# CoordBatchConverter — wraps ESM's BatchConverter with coord handling
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_ESM:
    _BATCH_CONVERTER_BASE = esm.data.BatchConverter
else:
    class _BatchConverterFallback:
        def __init__(self, alphabet): self.alphabet = alphabet
    _BATCH_CONVERTER_BASE = _BatchConverterFallback


class CoordBatchConverter(_BATCH_CONVERTER_BASE):
    """
    Mirrors `esm.inverse_folding.util.CoordBatchConverter` and SPURS's variant.

    Accepts ((coords, confidence), seq) tuples, tokenises the sequence with the
    ESM alphabet, and stacks coordinates with optional BOS/EOS padding.
    """

    def __init__(
        self,
        alphabet,
        coord_pad_inf: bool = False,
        coord_nan_to_zero: bool = True,
        to_pifold_format: bool = False,
    ):
        if not _HAS_ESM:
            raise ImportError("fair-esm is required for CoordBatchConverter.")
        super().__init__(alphabet)
        self.coord_pad_inf     = coord_pad_inf
        self.coord_nan_to_zero = coord_nan_to_zero
        self.to_pifold_format  = to_pifold_format

    def __call__(self, raw_batch, device: Optional[torch.device] = None):
        # Re-format batch entries
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.0
            if isinstance(confidence, (float, int)):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = "X" * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        if self.coord_pad_inf:
            # pad coordinates at BOS/EOS positions with NaN
            coords = [
                F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.nan)
                for cd, _ in coords_and_confidence
            ]
            confidence = [
                F.pad(torch.tensor(cf), (1, 1), value=-1.0)
                for _, cf in coords_and_confidence
            ]
        else:
            coords     = [torch.tensor(cd) for cd, _ in coords_and_confidence]
            confidence = [torch.tensor(cf) for _, cf in coords_and_confidence]

        coords     = self.collate_dense_tensors(coords,     pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.0)

        if self.to_pifold_format:
            coords, tokens, confidence = _to_pifold_format(coords, tokens, confidence)

        lengths = tokens.ne(self.alphabet.padding_idx).sum(1).long()
        if device is not None:
            coords, confidence = coords.to(device), confidence.to(device)
            tokens, lengths    = tokens.to(device), lengths.to(device)

        coord_padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask         = torch.isfinite(coords.sum([-2, -1]))
        confidence         = confidence * coord_mask + (-1.0) * coord_padding_mask

        if self.coord_nan_to_zero:
            coords[torch.isnan(coords)] = 0.0

        return coords, confidence, strs, tokens, lengths, coord_mask

    def from_lists(
        self,
        coords_list,
        confidence_list=None,
        seq_list=None,
        device: Optional[torch.device] = None,
    ):
        bsz = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * bsz
        if seq_list is None:
            seq_list = [None] * bsz
        return self(zip(coords_list, confidence_list, seq_list), device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        if not samples:
            return torch.Tensor()
        dims = set(x.dim() for x in samples)
        if len(dims) != 1:
            raise RuntimeError(f"Samples have varying dimensions: {[x.dim() for x in samples]}")
        (device,)  = tuple({x.device for x in samples})
        max_shape  = [max(s) for s in zip(*[x.shape for x in samples])]
        result     = torch.empty(len(samples), *max_shape,
                                  dtype=samples[0].dtype, device=device)
        result.fill_(pad_v)
        for i, t in enumerate(samples):
            result[i][tuple(slice(0, k) for k in t.shape)] = t
        return result


def _to_pifold_format(X, S, cfd):
    """Drop padded NaN rows so that all valid residues are contiguous."""
    mask    = torch.isfinite(torch.sum(X, [-2, -1]))
    numbers = torch.sum(mask, dim=1).long()

    S_new   = torch.zeros_like(S)
    X_new   = torch.zeros_like(X) + np.nan
    cfd_new = torch.zeros_like(cfd)
    for i, n in enumerate(numbers):
        X_new[i, :n]   = X[i][mask[i] == 1]
        S_new[i, :n]   = S[i][mask[i] == 1]
        cfd_new[i, :n] = cfd[i][mask[i] == 1]
    return X_new, S_new, cfd_new


# ──────────────────────────────────────────────────────────────────────────────
# tied_featurize — the ProteinMPNN packing routine
# ──────────────────────────────────────────────────────────────────────────────

def tied_featurize(
    batch: List[Dict],
    device: str,
    chain_dict: Optional[Dict] = None,
    fixed_position_dict: Optional[Dict] = None,
    omit_AA_dict: Optional[Dict] = None,
    tied_positions_dict: Optional[Dict] = None,
    pssm_dict: Optional[Dict] = None,
    bias_by_res_dict: Optional[Dict] = None,
    ca_only: bool = False,
):
    """
    Pack a list of parsed protein dicts into ProteinMPNN tensors.

    Each dict in `batch` must contain at minimum:
        - 'name'                : str
        - 'seq'                 : str (full concatenated sequence across chains)
        - 'seq_chain_<X>'       : str (sequence of chain X)
        - 'coords_chain_<X>'    : dict with 'N_chain_<X>', 'CA_chain_<X>',
                                  'C_chain_<X>', 'O_chain_<X>' (or just CA if ca_only)

    Returns a 20-tuple matching SPURS:
        X, S, mask, lengths, chain_M, chain_encoding_all,
        letter_list_list, visible_list_list, masked_list_list,
        masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
        residue_idx, dihedral_mask, tied_pos_list_of_lists_list,
        pssm_coef_all, pssm_bias_all, pssm_log_odds_all,
        bias_by_res_all, tied_beta
    """
    alphabet  = "ACDEFGHIKLMNPQRSTVWYX"
    B         = len(batch)
    lengths   = np.array([len(b["seq"]) for b in batch], dtype=np.int32)
    L_max     = int(max(lengths))

    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    else:
        X = np.zeros([B, L_max, 4, 3])

    residue_idx        = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M            = np.zeros([B, L_max], dtype=np.int32)
    pssm_coef_all      = np.zeros([B, L_max], dtype=np.float32)
    pssm_bias_all      = np.zeros([B, L_max, 21], dtype=np.float32)
    pssm_log_odds_all  = 10000.0 * np.ones([B, L_max, 21], dtype=np.float32)
    chain_M_pos        = np.zeros([B, L_max], dtype=np.int32)
    bias_by_res_all    = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    S                  = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask       = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)

    letter_list_list              = []
    visible_list_list             = []
    masked_list_list              = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list   = []

    # First pass: determine chain ordering per protein
    chain_order_per_protein = []
    for b in batch:
        if chain_dict is not None:
            masked_chains, visible_chains = chain_dict[b["name"]]
        else:
            masked_chains  = [k[-1:] for k in b if k.startswith("seq_chain_")]
            visible_chains = []
        masked_chains.sort()
        visible_chains.sort()
        chain_order_per_protein.append((masked_chains, visible_chains))

    tied_beta = np.ones(L_max)

    for i, b in enumerate(batch):
        masked_chains, visible_chains = chain_order_per_protein[i]
        all_chains = masked_chains + visible_chains

        x_chain_list             = []
        chain_mask_list          = []
        chain_seq_list           = []
        chain_encoding_list      = []
        c                        = 1
        letter_list              = []
        global_idx_start_list    = [0]
        visible_list             = []
        masked_list              = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list        = []
        pssm_coef_list           = []
        pssm_bias_list           = []
        pssm_log_odds_list       = []
        bias_by_res_list         = []
        l0, l1 = 0, 0

        for letter in all_chains:
            is_visible = letter in visible_chains
            letter_list.append(letter)
            if is_visible:
                visible_list.append(letter)
            else:
                masked_list.append(letter)

            chain_seq    = b[f"seq_chain_{letter}"]
            chain_seq    = "".join([a if a != "-" else "X" for a in chain_seq])
            chain_length = len(chain_seq)
            global_idx_start_list.append(global_idx_start_list[-1] + chain_length)

            chain_coords = b[f"coords_chain_{letter}"]
            if ca_only:
                x_chain = np.array(chain_coords[f"CA_chain_{letter}"])
                if x_chain.ndim == 2:
                    x_chain = x_chain[:, None, :]
            else:
                x_chain = np.stack(
                    [chain_coords[k] for k in (
                        f"N_chain_{letter}",  f"CA_chain_{letter}",
                        f"C_chain_{letter}",  f"O_chain_{letter}",
                    )],
                    axis=1,
                )
            chain_mask = np.zeros(chain_length) if is_visible else np.ones(chain_length)
            if not is_visible:
                masked_chain_length_list.append(chain_length)

            x_chain_list.append(x_chain)
            chain_mask_list.append(chain_mask)
            chain_seq_list.append(chain_seq)
            chain_encoding_list.append(c * np.ones(chain_length))
            l1 += chain_length
            residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
            l0 += chain_length
            c += 1

            # Fixed-position masking
            fixed_position_mask = np.ones(chain_length)
            if (not is_visible) and fixed_position_dict is not None:
                fixed_pos_list = fixed_position_dict[b["name"]][letter]
                if fixed_pos_list:
                    fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0
            fixed_position_mask_list.append(fixed_position_mask)

            # Omit-AA mask
            omit_AA_temp = np.zeros([chain_length, len(alphabet)], dtype=np.int32)
            if (not is_visible) and omit_AA_dict is not None:
                for item in omit_AA_dict[b["name"]][letter]:
                    idx_AA = np.array(item[0]) - 1
                    AA_idx = np.array(
                        [np.argwhere(np.array(list(alphabet)) == AA)[0][0]
                         for AA in item[1]]
                    ).repeat(idx_AA.shape[0])
                    pairs = np.array([[a, b_] for a in idx_AA for b_ in AA_idx])
                    omit_AA_temp[pairs[:, 0], pairs[:, 1]] = 1
            omit_AA_mask_list.append(omit_AA_temp)

            # PSSM
            pc  = np.zeros(chain_length)
            pb  = np.zeros([chain_length, 21])
            plo = 10000.0 * np.ones([chain_length, 21])
            if (not is_visible) and pssm_dict and pssm_dict[b["name"]][letter]:
                pc  = pssm_dict[b["name"]][letter]["pssm_coef"]
                pb  = pssm_dict[b["name"]][letter]["pssm_bias"]
                plo = pssm_dict[b["name"]][letter]["pssm_log_odds"]
            pssm_coef_list.append(pc)
            pssm_bias_list.append(pb)
            pssm_log_odds_list.append(plo)

            # Bias-by-residue
            if (not is_visible) and bias_by_res_dict:
                bias_by_res_list.append(bias_by_res_dict[b["name"]][letter])
            else:
                bias_by_res_list.append(np.zeros([chain_length, 21]))

        # Tied positions
        letter_list_np         = np.array(letter_list)
        tied_pos_list_of_lists = []
        if tied_positions_dict is not None:
            tied_pos_list = tied_positions_dict[b["name"]]
            if tied_pos_list:
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[
                            np.argwhere(letter_list_np == k)[0][0]
                        ]
                        if isinstance(v[0], list):
                            for j in range(len(v[0])):
                                one_list.append(start_idx + v[0][j] - 1)
                                tied_beta[start_idx + v[0][j] - 1] = v[1][j]
                        else:
                            for v_ in v:
                                one_list.append(start_idx + v_ - 1)
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

        # Concatenate per-chain pieces
        x_arr          = np.concatenate(x_chain_list,        axis=0)
        all_sequence   = "".join(chain_seq_list)
        m_arr          = np.concatenate(chain_mask_list,     axis=0)
        chain_encoding = np.concatenate(chain_encoding_list, axis=0)
        m_pos_arr      = np.concatenate(fixed_position_mask_list, axis=0)
        pc_arr         = np.concatenate(pssm_coef_list,      axis=0)
        pb_arr         = np.concatenate(pssm_bias_list,      axis=0)
        plo_arr        = np.concatenate(pssm_log_odds_list,  axis=0)
        bbr_arr        = np.concatenate(bias_by_res_list,    axis=0)

        l = len(all_sequence)

        X[i, :, :, :] = np.pad(
            x_arr, [[0, L_max - l], [0, 0], [0, 0]],
            mode="constant", constant_values=np.nan,
        )
        chain_M[i, :]            = np.pad(m_arr,     [[0, L_max - l]], constant_values=0.0)
        chain_M_pos[i, :]        = np.pad(m_pos_arr, [[0, L_max - l]], constant_values=0.0)
        omit_AA_mask[i]          = np.pad(
            np.concatenate(omit_AA_mask_list, axis=0),
            [[0, L_max - l], [0, 0]], constant_values=0.0,
        )
        chain_encoding_all[i, :] = np.pad(chain_encoding, [[0, L_max - l]], constant_values=0.0)
        pssm_coef_all[i, :]      = np.pad(pc_arr,  [[0, L_max - l]], constant_values=0.0)
        pssm_bias_all[i, :]      = np.pad(pb_arr,  [[0, L_max - l], [0, 0]], constant_values=0.0)
        pssm_log_odds_all[i, :]  = np.pad(plo_arr, [[0, L_max - l], [0, 0]], constant_values=0.0)
        bias_by_res_all[i, :]    = np.pad(bbr_arr, [[0, L_max - l], [0, 0]], constant_values=0.0)

        S[i, :l] = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)

        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)

    # Build mask + clean NaNs
    isnan = np.isnan(X)
    mask  = np.isfinite(np.sum(X, axis=(2, 3))).astype(np.float32)
    X[isnan] = 0.0

    # Dihedral mask from residue_idx gaps
    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    phi_mask   = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask   = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    dihedral_mask = np.concatenate(
        [phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]], axis=-1,
    )

    # → torch
    to = lambda arr, dtype: torch.from_numpy(arr).to(dtype=dtype, device=device)
    X_t                  = to(X.astype(np.float32),                  torch.float32)
    S_t                  = to(S.astype(np.int64),                    torch.long)
    mask_t               = to(mask.astype(np.float32),               torch.float32)
    chain_M_t            = to(chain_M.astype(np.float32),            torch.float32)
    chain_M_pos_t        = to(chain_M_pos.astype(np.float32),        torch.float32)
    omit_AA_mask_t       = to(omit_AA_mask.astype(np.float32),       torch.float32)
    residue_idx_t        = to(residue_idx.astype(np.int64),          torch.long)
    chain_encoding_all_t = to(chain_encoding_all.astype(np.int64),   torch.long)
    dihedral_mask_t      = to(dihedral_mask.astype(np.float32),      torch.float32)
    pssm_coef_all_t      = to(pssm_coef_all,                          torch.float32)
    pssm_bias_all_t      = to(pssm_bias_all,                          torch.float32)
    pssm_log_odds_all_t  = to(pssm_log_odds_all,                      torch.float32)
    bias_by_res_all_t    = to(bias_by_res_all,                        torch.float32)
    tied_beta_t          = to(tied_beta.astype(np.float32),           torch.float32)

    X_out = X_t[:, :, 0] if ca_only else X_t

    return (
        X_out, S_t, mask_t, lengths, chain_M_t, chain_encoding_all_t,
        letter_list_list, visible_list_list, masked_list_list,
        masked_chain_length_list_list, chain_M_pos_t, omit_AA_mask_t,
        residue_idx_t, dihedral_mask_t, tied_pos_list_of_lists_list,
        pssm_coef_all_t, pssm_bias_all_t, pssm_log_odds_all_t,
        bias_by_res_all_t, tied_beta_t,
    )


# ──────────────────────────────────────────────────────────────────────────────
# get_pdb — build a single-protein training dict
# ──────────────────────────────────────────────────────────────────────────────

def get_pdb(pdb: dict, wt_seq: str, wt_name: str, check_assert: bool = True) -> dict:
    """
    Build a single-protein training dict from a parsed PDB dict.

    Mirrors SPURS's `get_pdb`. The returned dict contains:
        seq, coords, name, chain_ids, mut_ids, ddG, append_tensors, mut_seq,
        X, S, mask, chain_M, chain_M_chain_M_pos, residue_idx,
        chain_encoding_all, randn_1

    When `check_assert=True`, asserts that `pdb["seq"]` matches `wt_seq`.
    """
    if check_assert:
        assert len(pdb["seq"]) == len(wt_seq), (
            f"PDB seq length {len(pdb['seq'])} != wt_seq length {len(wt_seq)}"
        )
        assert pdb["seq"] == wt_seq
        pdb["seq"] = wt_seq
    else:
        wt_seq = pdb["seq"]

    # Find the first chain
    chainid = None
    for k in pdb.keys():
        if k.startswith("seq_chain_"):
            chainid = k.split("_")[-1]
            break
    if chainid is None:
        raise ValueError(f"No seq_chain_<X> entries in PDB dict for {wt_name}")

    coords = pdb[f"coords_chain_{chainid}"]
    coords = {k.split("_")[0]: v for k, v in coords.items()}
    coords = {k: torch.FloatTensor(v) for k, v in coords.items()}

    protein: dict = {
        "seq":            wt_seq,
        "coords":         coords,
        "name":           wt_name,
        "chain_ids":      chainid,
        "mut_ids":        [],
        "ddG":            [],
        "append_tensors": [],
        "mut_seq":        [],
    }

    protein_info = {
        f"seq_chain_{chainid}":    wt_seq,
        f"coords_chain_{chainid}": pdb[f"coords_chain_{chainid}"],
        "seq":                     wt_seq,
        "name":                    wt_name,
    }

    (X, S, mask, lengths, chain_M, chain_encoding_all,
     _, _, _, _, chain_M_pos, _,
     residue_idx, _, _, _, _, _, _, _) = tied_featurize(
        batch=[protein_info],
        device="cpu",
        chain_dict=None,
        fixed_position_dict=None,
        omit_AA_dict=None,
        tied_positions_dict=None,
        pssm_dict=None,
        bias_by_res_dict=None,
        ca_only=False,
    )

    protein["X"]                   = X
    protein["S"]                   = S
    protein["mask"]                = mask
    protein["chain_M"]              = chain_M
    protein["chain_M_chain_M_pos"]  = chain_M * chain_M_pos
    protein["residue_idx"]          = residue_idx
    protein["chain_encoding_all"]   = chain_encoding_all
    protein["randn_1"]              = torch.randn(chain_M.shape, device=X.device)

    return protein


# ──────────────────────────────────────────────────────────────────────────────
# Featurizer — top-level batch featurizer used by MultimodalDDG datasets
# ──────────────────────────────────────────────────────────────────────────────

class Featurizer:
    """
    Tokenises sequences and packs coordinates for a batch.

    Used as the `collate_fn` for the SPURS-style datasets in this library.
    For mutation datasets the batch typically contains a single protein, with
    many mutations bundled into `mut_ids` + `append_tensors`.
    """

    def __init__(
        self,
        alphabet: Alphabet,
        to_pifold_format: bool = False,
        coord_nan_to_zero: bool = True,
        atoms: Tuple[str, ...] = ("N", "CA", "C", "O"),
        single_mut: bool = False,
        mut_seq:    bool = False,
    ):
        self.alphabet   = alphabet
        self.atoms      = atoms
        self.single_mut = single_mut
        self.mut_seq    = mut_seq
        self.batcher    = CoordBatchConverter(
            alphabet=alphabet,
            coord_pad_inf=alphabet.add_special_tokens,
            to_pifold_format=to_pifold_format,
            coord_nan_to_zero=coord_nan_to_zero,
        )
        self.cache: Dict[str, Any] = defaultdict(lambda: -1)

    def __call__(self, raw_batch):
        # MultimodalDDG convention: each "batch" is actually a single-protein dict
        # already containing the bundled mutations. Unwrap if needed.
        if not self.single_mut:
            if isinstance(raw_batch, list):
                raw_batch = raw_batch[0]

            if not self.mut_seq:
                seqs   = [raw_batch["seq"]]
                coords = [
                    np.stack([raw_batch["coords"][atom] for atom in self.atoms], 1)
                ]
            else:
                seqs   = [raw_batch["seq"]] + raw_batch["mut_seq"]
                coords = [
                    np.stack([raw_batch["coords"][atom] for atom in self.atoms], 1)
                ] * len(seqs)

            _, _, _, tokens, _, _ = self.batcher.from_lists(
                coords_list=coords, confidence_list=None, seq_list=seqs,
            )
            raw_batch["tokens"]     = tokens
            raw_batch["mut_tokens"] = None

            if "ddG" in raw_batch and isinstance(raw_batch["ddG"], torch.Tensor):
                raw_batch["ddG"] = raw_batch["ddG"].reshape(-1)
            return raw_batch

        # single_mut=True branch — separate features for each protein in batch
        for protein in raw_batch:
            name = protein["name"]
            if isinstance(self.cache[name], int):
                seqs   = [protein["seq"]]
                coords = [
                    np.stack([protein["coords"][atom] for atom in self.atoms], 1)
                ]
                _, _, _, tokens, _, _ = self.batcher.from_lists(
                    coords_list=coords, confidence_list=None, seq_list=seqs,
                )
                self.cache[name] = {"tokens": tokens, "mut_tokens": None}
            else:
                tokens     = self.cache[name]["tokens"]
            protein["tokens"]     = tokens
            protein["mut_tokens"] = None

        ddg = torch.stack([p["ddG"] for p in raw_batch])
        return {
            "raw_batch":      raw_batch,
            "mut_ids":        [p["mut_ids"]        for p in raw_batch],
            "append_tensors": torch.stack([p["append_tensors"] for p in raw_batch]),
            "ddG":            ddg,
            "name":           [p["name"] + p["chain_ids"] for p in raw_batch],
            "dataset":        [p["dataset"]        for p in raw_batch],
        }
