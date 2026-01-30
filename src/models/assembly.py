"""Fragment assembly utilities (beam search und simplen MCTS)"""

from typing import List, Tuple, Callable, Optional, Dict, Any
import math
import random
import copy
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import RWMol

# Helper: valence table
# typische Werte
_TYPICAL_VALENCE = {
    'H': 1, 'B': 3, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 6, 'Cl': 1, 'Br': 1, 'I': 1
}


def _typical_valence(atom_symbol: str) -> int:
    return _TYPICAL_VALENCE.get(atom_symbol, 4)


# Helper: find attachment points

def find_attachment_points(mol: Chem.Mol) -> List[int]:
    """Gibt die Atom-Indizes zurück, die als Ansatzpunkt taugen: 
    Atom hat noch ein implizites H, oder ist ein Dummy/Attachment (*/R), oder seine aktuelle Valenz liegt unter der typischen Valenz
    """
    attach = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        try:
            hydrogens = atom.GetTotalNumHs()
        except Exception:
            hydrogens = atom.GetNumImplicitHs() if hasattr(atom, 'GetNumImplicitHs') else 0
        # dummy atoms wie '*' representiert als Ordnugszahl 0
        if atom.GetAtomicNum() == 0:
            attach.append(atom.GetIdx())
            continue
        if hydrogens > 0:
            attach.append(atom.GetIdx())
            continue
        # fallback: checkt valence
        cur_val = atom.GetDegree()
        max_val = _typical_valence(sym)
        if cur_val < max_val:
            attach.append(atom.GetIdx())
    return attach


# Helper: join two molecules at specified atom indices

def attach_molecules(mol1: Chem.Mol, mol2: Chem.Mol, idx1: int, idx2: int, bond_type=Chem.rdchem.BondType.SINGLE) -> Optional[Chem.Mol]:
    """Verbindet mol2 mit mol1, indem ein Bindung zwischen Atom idx1 (in mol1) und idx2 (in mol2) erstellt wird

    """
    try:
        m1 = RWMol(mol1)
        m2 = RWMol(mol2)
        # map indices von m2 zu Kombinierten Moleküle
        offset = m1.GetNumAtoms()
        # kopiert Atome von m2
        for a in range(m2.GetNumAtoms()):
            atom = m2.GetAtomWithIdx(a)
            new_atom = atom.Clone()
            m1_idx = m1.AddAtom(new_atom)
        #  kopiert Bindungen von m2
        for b in range(m2.GetNumBonds()):
            bo = m2.GetBondWithIdx(b)
            a1 = bo.GetBeginAtomIdx() + offset
            a2 = bo.GetEndAtomIdx() + offset
            m1.AddBond(a1, a2, bo.GetBondType())
        # fügt die neue Bindung hinzu
        new_idx2 = idx2 + offset
        m1.AddBond(idx1, new_idx2, bond_type)
        newmol = m1.GetMol()
        Chem.SanitizeMol(newmol)
        # kekulize / cleanup
        try:
            Chem.Kekulize(newmol, clearAromaticFlags=True)
        except Exception:
            pass
        return newmol
    except Exception as e:
        # assembly verkackt
        return None


# Scoring function utilities

def default_score_fn(mol: Chem.Mol, cond: Optional[np.ndarray] = None, surrogate_score: Optional[float] = None) -> float:
    """tandard-Score: ungültige Moleküle werden bestraft, größere konjugierte Systeme werden bevorzugt,
      optional kombiniert mit einem externen Surrogat-Score (höher ist besser)
    """
    if mol is None:
        return -1e6
    try:
        # checkt sanitization
        Chem.SanitizeMol(mol)
    except Exception:
        return -1e6
    # Simpel gemacht: Anzahl aromatischer Atome (als Proxy für Konjugation)
    aro = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    nat = mol.GetNumAtoms()
    score = aro - 0.1 * nat
    if surrogate_score is not None:
        score = 0.6 * surrogate_score + 0.4 * score
    return float(score)


# Beam search assembly

def beam_search_assemble(fragment_smiles_list: List[str], beam_width: int = 8, max_steps: Optional[int] = None,
                         cond: Optional[np.ndarray] = None, score_fn: Optional[Callable] = None,
                         surrogate_predictor: Optional[Callable[[str], float]] = None) -> List[str]:
    """Sucht per Beam Search Fragmente zu ganzen Molekülen zusammen

    fragment_smiles_list: Liste der Fragment-SMILES (vom JT-VAE dekodiert), Reihenfolge egal
    beam_width: wie viele Teil-Assemblies pro Schritt behalten werden
    max_steps: maximale Assemblieschritte (Default len(fragments)–1)
    cond: Konditionsvektor (geht an score_fn oder surrogate_predictor)
    score_fn: Funktion (mol, cond, surrogate_score) -> float
    surrogate_predictor: optional Funktion(smiles) -> Score, um die Suche zu steuern
    
    Gibt zurück: Liste der zusammengebauten SMILES (top beam_width)
    """
    if score_fn is None:
        score_fn = default_score_fn

    fragments = fragment_smiles_list.copy()
    n_frag = len(fragments)
    if n_frag == 0:
        return []
    if max_steps is None:
        max_steps = max(0, n_frag - 1)

    # pre-parse fragments zu RDKit mols
    frag_mols = []
    for smi in fragments:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            frag_mols.append(None)
        else:
            Chem.SanitizeMol(m, catchErrors=True)
            frag_mols.append(m)

    # initial beam
    beam = []
    for i, m in enumerate(frag_mols):
        if m is None:
            continue
        score = score_fn(m, cond=cond, surrogate_score=None)
        beam.append({'mol': m, 'used': {i}, 'score': score, 'smiles': Chem.MolToSmiles(m)})
    # reduziert zu beam_width
    beam = sorted(beam, key=lambda x: -x['score'])[:beam_width]

    steps = 0
    while steps < max_steps:
        new_beam = []
        for state in beam:
            base_mol = state['mol']
            used = state['used']
            for j, frag in enumerate(frag_mols):
                if j in used or frag is None:
                    continue
                # findet attachment points
                attach_base = find_attachment_points(base_mol)
                attach_frag = find_attachment_points(frag)
                if len(attach_base) == 0 or len(attach_frag) == 0:
                    # fallback: alle atom paare erlauben
                    attach_base = list(range(base_mol.GetNumAtoms()))
                    attach_frag = list(range(frag.GetNumAtoms()))
                # versucht K Verbindungen
                K = 6
                pairs = []
                for a in attach_base:
                    for b in attach_frag:
                        pairs.append((a,b))
                # randomize order um vielfalt zu fördern
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
        # wenn keine expansions, break
        if len(new_beam) == 0:
            break
        # wählt top beam_width aus
        # dedupliziert nach SMILES, behält den besten Score
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

    # returned die finalen SMILES
    final = sorted(beam, key=lambda x: -x['score'])
    return [st['smiles'] for st in final]


# Simple MCTS implementation
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
    """Monte-Carlo-Tree-Search-Assembly, gibt das beste gefundene SMILES zurück

Ablauf:

    Startet mit jedem Fragment als Wurzel (ein paar Wurzeln probieren)
    Expandiert schrittweise, indem unbenutzte Fragmente angehängt werden
    Rollout: zufällige Anhänge bis rollout_depth, finale Mol bewerten
    Reward zurückpropagieren
    """
    if score_fn is None:
        score_fn = default_score_fn

    fragments = fragment_smiles_list.copy()
    frag_mols = [Chem.MolFromSmiles(smi) for smi in fragments]
    roots = [i for i, m in enumerate(frag_mols) if m is not None]
    if len(roots) == 0:
        return ''
    root_idx = roots[0]
    root_node = MCTSNode(frag_mols[root_idx], used={root_idx}, parent=None, action=None)
    root_node.visits = 1

    for it in range(iterations):
        # selection
        node = root_node
        while node.children:
            # nimmt den child mit dem höchsten UCB score
            scores = [child.ucb_score() for child in node.children]
            best = max(range(len(scores)), key=lambda i: scores[i])
            node = node.children[best]
        # Exppansion (wenn möglich)
        unused = [i for i in range(len(frag_mols)) if i not in node.used and frag_mols[i] is not None]
        if unused:
            j = random.choice(unused)
            # nimmt random attachment points
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
    # nach iterations: bestes Molekül finden
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


# Beispielnutzung
if __name__ == '__main__':
    frags = ['c1ccccc1', 'C(=O)O', 'c1ccncc1']
    print('beam search assembled')
    out = beam_search_assemble(frags, beam_width=6)
    print(out)
    print('mcts assembled')
    out2 = mcts_assemble(frags, iterations=200)
    print(out2)
