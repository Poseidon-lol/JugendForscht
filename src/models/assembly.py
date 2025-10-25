# src/models/assembly.py
"""
Fragment assembly utilities: Subgraph matching + Beam Search + simple MCTS
------------------------------------------------------------------------
Purpose:
- Given a set of fragment SMILES (e.g., from JT-VAE decoder), assemble
  chemically valid full molecules by finding attachment points and joining
  fragments while respecting valence rules.
- Provide two strategies:
  1) beam_search_assemble: deterministic beam search with scoring
  2) mcts_assemble: Monte-Carlo Tree Search with random rollouts and UCB

Notes / limitations:
- This is a practical, research-grade starting point. It handles many common
  cases (fragments with available H or explicit attachment sites) but will not
  cover all edge-cases of complex chemistries. Further tailoring to your
  OSC fragment set is recommended.
- Requires RDKit installed.

API (key functions):
- beam_search_assemble(fragment_smiles_list, beam_width=8, max_atoms=200, score_fn=None)
- mcts_assemble(fragment_smiles_list, iterations=500, rollout_depth=8, score_fn=None)

Example:
    assembled = beam_search_assemble(['c1ccccc1', 'C(=O)O', 'c1ccncc1'], beam_width=6)

"""

from typing import List, Tuple, Callable, Optional, Dict, Any
import math
import random
import copy
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import RWMol

# -------------------------
# Helper: valence table
# -------------------------
# Typical maximum valences for common atoms (approximate)
_TYPICAL_VALENCE = {
    'H': 1, 'B': 3, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 6, 'Cl': 1, 'Br': 1, 'I': 1
}


def _typical_valence(atom_symbol: str) -> int:
    return _TYPICAL_VALENCE.get(atom_symbol, 4)


# -------------------------
# Helper: find attachment points
# -------------------------

def find_attachment_points(mol: Chem.Mol) -> List[int]:
    """Return atom indices that are suitable attachment points.

    Heuristic used:
    - Atoms with at least one implicit H (atom.GetTotalNumHs() > 0)
    - Or atoms with explicit dummy/attachment atoms (e.g. '*' or 'R')
    - Or atoms whose current explicit valence < typical valence
    """
    attach = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        try:
            hydrogens = atom.GetTotalNumHs()
        except Exception:
            hydrogens = atom.GetNumImplicitHs() if hasattr(atom, 'GetNumImplicitHs') else 0
        # dummy atoms like '*' represented as atomic number 0
        if atom.GetAtomicNum() == 0:
            attach.append(atom.GetIdx())
            continue
        if hydrogens > 0:
            attach.append(atom.GetIdx())
            continue
        # fallback: check valence
        cur_val = atom.GetDegree()
        max_val = _typical_valence(sym)
        if cur_val < max_val:
            attach.append(atom.GetIdx())
    return attach


# -------------------------
# Helper: join two molecules at specified atom indices
# -------------------------

def attach_molecules(mol1: Chem.Mol, mol2: Chem.Mol, idx1: int, idx2: int, bond_type=Chem.rdchem.BondType.SINGLE) -> Optional[Chem.Mol]:
    """Attach mol2 to mol1 by creating a bond between atom idx1 (in mol1) and idx2 (in mol2).

    Returns a new sanitized RDKit Mol or None if invalid.
    The function merges the two molecules and adds a bond. It attempts to
    remove an H at the attachment site if present (implicit H), by using
    RDKit EditableMol operations. This is heuristic but works for many cases.
    """
    try:
        # create editable molecules
        m1 = RWMol(mol1)
        m2 = RWMol(mol2)
        # map indices of m2 into combined molecule
        offset = m1.GetNumAtoms()
        # copy atoms from m2
        for a in range(m2.GetNumAtoms()):
            atom = m2.GetAtomWithIdx(a)
            new_atom = atom.Clone()
            m1_idx = m1.AddAtom(new_atom)
        # copy bonds from m2
        for b in range(m2.GetNumBonds()):
            bo = m2.GetBondWithIdx(b)
            a1 = bo.GetBeginAtomIdx() + offset
            a2 = bo.GetEndAtomIdx() + offset
            m1.AddBond(a1, a2, bo.GetBondType())
        # attempt to remove an H at idx1 or idx2 (if implicit H, this is heuristic)
        # RDKit doesn't allow directly removing implicit Hs; instead we try to replace
        # by creating explicit bond and sanitizing.
        new_idx2 = idx2 + offset
        m1.AddBond(idx1, new_idx2, bond_type)
        newmol = m1.GetMol()
        Chem.SanitizeMol(newmol)
        # try to kekulize / cleanup
        try:
            Chem.Kekulize(newmol, clearAromaticFlags=True)
        except Exception:
            pass
        return newmol
    except Exception as e:
        # assembly failed
        return None


# -------------------------
# Scoring function utilities
# -------------------------

def default_score_fn(mol: Chem.Mol, cond: Optional[np.ndarray] = None, surrogate_score: Optional[float] = None) -> float:
    """Default score: penalize invalid molecules, prefer larger conjugated systems (heuristic),
    and (optionally) combine with external surrogate score (higher is better).
    """
    if mol is None:
        return -1e6
    try:
        # check sanitization
        Chem.SanitizeMol(mol)
    except Exception:
        return -1e6
    # heuristic: number of aromatic atoms (proxy for conjugation)
    aro = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    nat = mol.GetNumAtoms()
    score = aro - 0.1 * nat
    if surrogate_score is not None:
        score = 0.6 * surrogate_score + 0.4 * score
    # return higher is better
    return float(score)


# -------------------------
# Beam search assembly
# -------------------------

def beam_search_assemble(fragment_smiles_list: List[str], beam_width: int = 8, max_steps: Optional[int] = None,
                         cond: Optional[np.ndarray] = None, score_fn: Optional[Callable] = None,
                         surrogate_predictor: Optional[Callable[[str], float]] = None) -> List[str]:
    """Beam search to assemble fragments into full molecules.

    Args:
        fragment_smiles_list: list of fragment SMILES (decoded by JT-VAE). Order may be arbitrary.
        beam_width: how many partial assemblies to keep at each step.
        max_steps: maximum assembly steps (defaults to len(fragments)-1)
        cond: conditioning vector (passed to score_fn or surrogate_predictor)
        score_fn: function(mol, cond, surrogate_score) -> float
        surrogate_predictor: optional function(smiles)->score used to bias search

    Returns:
        list of assembled SMILES strings (top beam_width)
    """
    if score_fn is None:
        score_fn = default_score_fn

    fragments = fragment_smiles_list.copy()
    n_frag = len(fragments)
    if n_frag == 0:
        return []
    if max_steps is None:
        max_steps = max(0, n_frag - 1)

    # pre-parse fragments to RDKit mols
    frag_mols = []
    for smi in fragments:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            frag_mols.append(None)
        else:
            Chem.SanitizeMol(m, catchErrors=True)
            frag_mols.append(m)

    # initial beam: each state is (mol, used_indices_set, score)
    beam = []
    for i, m in enumerate(frag_mols):
        if m is None:
            continue
        score = score_fn(m, cond=cond, surrogate_score=None)
        beam.append({'mol': m, 'used': {i}, 'score': score, 'smiles': Chem.MolToSmiles(m)})
    # reduce to beam_width
    beam = sorted(beam, key=lambda x: -x['score'])[:beam_width]

    steps = 0
    while steps < max_steps:
        new_beam = []
        for state in beam:
            base_mol = state['mol']
            used = state['used']
            # try attaching any unused fragment
            for j, frag in enumerate(frag_mols):
                if j in used or frag is None:
                    continue
                # find attachment points
                attach_base = find_attachment_points(base_mol)
                attach_frag = find_attachment_points(frag)
                if len(attach_base) == 0 or len(attach_frag) == 0:
                    # fallback: try all atom pairs
                    attach_base = list(range(base_mol.GetNumAtoms()))
                    attach_frag = list(range(frag.GetNumAtoms()))
                # try top K pairs to limit branching
                K = 6
                pairs = []
                for a in attach_base:
                    for b in attach_frag:
                        pairs.append((a,b))
                # randomize order to increase diversity
                random.shuffle(pairs)
                for (a,b) in pairs[:K]:
                    newmol = attach_molecules(base_mol, frag, a, b)
                    if newmol is None:
                        continue
                    smi = Chem.MolToSmiles(newmol)
                    surrogate_score = None
                    if surrogate_predictor is not None:
                        try:
                            surrogate_score = surrogate_predictor(smi)
                        except Exception:
                            surrogate_score = None
                    sc = score_fn(newmol, cond=cond, surrogate_score=surrogate_score)
                    new_state = {'mol': newmol, 'used': used.union({j}), 'score': sc, 'smiles': smi}
                    new_beam.append(new_state)
        # if no expansions, break
        if len(new_beam) == 0:
            break
        # select top beam_width unique states (by canonical SMILES)
        # deduplicate
        uniq = {}
        for st in new_beam:
            try:
                cs = Chem.MolToSmiles(st['mol'], canonical=True)
            except Exception:
                cs = st['smiles']
            if cs not in uniq or st['score'] > uniq[cs]['score']:
                uniq[cs] = st
        new_beam = list(uniq.values())
        new_beam = sorted(new_beam, key=lambda x: -x['score'])[:beam_width]
        beam = new_beam
        steps += 1

    # return final beam SMILES sorted
    final = sorted(beam, key=lambda x: -x['score'])
    return [st['smiles'] for st in final]


# -------------------------
# Simple MCTS implementation
# -------------------------
class MCTSNode:
    def __init__(self, mol: Chem.Mol, used: set, parent=None, action=None):
        self.mol = mol
        self.used = used
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.action = action  # (frag_idx, attach_pair)

    def ucb_score(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)


def mcts_assemble(fragment_smiles_list: List[str], iterations: int = 300, rollout_depth: int = 6,
                  cond: Optional[np.ndarray] = None, score_fn: Optional[Callable] = None,
                  surrogate_predictor: Optional[Callable[[str], float]] = None) -> str:
    """Monte Carlo Tree Search assembly. Returns best found SMILES.

    Procedure:
    - Start from each fragment as root (try a few roots)
    - Iteratively expand nodes by attaching unused fragments
    - Rollout: perform random attachments up to rollout_depth, score final mol
    - Backpropagate reward
    """
    if score_fn is None:
        score_fn = default_score_fn

    fragments = fragment_smiles_list.copy()
    frag_mols = [Chem.MolFromSmiles(smi) for smi in fragments]
    roots = [i for i, m in enumerate(frag_mols) if m is not None]
    if len(roots) == 0:
        return ''
    # initialize root node with first fragment
    root_idx = roots[0]
    root_node = MCTSNode(frag_mols[root_idx], used={root_idx}, parent=None, action=None)
    root_node.visits = 1

    for it in range(iterations):
        # selection
        node = root_node
        while node.children:
            # pick child with highest UCB
            scores = [child.ucb_score() for child in node.children]
            best = max(range(len(scores)), key=lambda i: scores[i])
            node = node.children[best]
        # expansion: if not terminal, expand by adding a random unused fragment
        unused = [i for i in range(len(frag_mols)) if i not in node.used and frag_mols[i] is not None]
        if unused:
            j = random.choice(unused)
            # choose random attachment points
            attach_base = find_attachment_points(node.mol)
            attach_frag = find_attachment_points(frag_mols[j])
            if len(attach_base) == 0:
                attach_base = list(range(node.mol.GetNumAtoms()))
            if len(attach_frag) == 0:
                attach_frag = list(range(frag_mols[j].GetNumAtoms()))
            a = random.choice(attach_base)
            b = random.choice(attach_frag)
            newmol = attach_molecules(node.mol, frag_mols[j], a, b)
            if newmol is not None:
                child = MCTSNode(newmol, used=node.used.union({j}), parent=node, action=(j,(a,b)))
                node.children.append(child)
                node = child
        # rollout
        current = node
        depth = 0
        while depth < rollout_depth:
            unused = [i for i in range(len(frag_mols)) if i not in current.used and frag_mols[i] is not None]
            if not unused:
                break
            j = random.choice(unused)
            attach_base = find_attachment_points(current.mol)
            attach_frag = find_attachment_points(frag_mols[j])
            if len(attach_base) == 0:
                attach_base = list(range(current.mol.GetNumAtoms()))
            if len(attach_frag) == 0:
                attach_frag = list(range(frag_mols[j].GetNumAtoms()))
            a = random.choice(attach_base)
            b = random.choice(attach_frag)
            newmol = attach_molecules(current.mol, frag_mols[j], a, b)
            if newmol is None:
                break
            current = MCTSNode(newmol, used=current.used.union({j}), parent=current)
            depth += 1
        # scoring
        smi = None
        try:
            smi = Chem.MolToSmiles(current.mol, canonical=True)
        except Exception:
            smi = None
        surrogate_score = None
        if surrogate_predictor is not None and smi is not None:
            try:
                surrogate_score = surrogate_predictor(smi)
            except Exception:
                surrogate_score = None
        reward = score_fn(current.mol, cond=cond, surrogate_score=surrogate_score)
        # backpropagate
        node_to_update = node
        while node_to_update is not None:
            node_to_update.visits += 1
            node_to_update.value += reward
            node_to_update = node_to_update.parent
    # after iterations, pick best child (highest average value)
    best_node = root_node
    best_score = -1e9
    stack = [root_node]
    while stack:
        n = stack.pop()
        if n.visits > 0:
            avg = n.value / n.visits
            if avg > best_score:
                best_score = avg
                best_node = n
        for c in n.children:
            stack.append(c)
    try:
        return Chem.MolToSmiles(best_node.mol, canonical=True)
    except Exception:
        return ''


# -------------------------
# Example usage
# -------------------------
if __name__ == '__main__':
    frags = ['c1ccccc1', 'C(=O)O', 'c1ccncc1']
    print('Beam search assembled:')
    out = beam_search_assemble(frags, beam_width=6)
    print(out)
    print('MCTS assembled:')
    out2 = mcts_assemble(frags, iterations=200)
    print(out2)
