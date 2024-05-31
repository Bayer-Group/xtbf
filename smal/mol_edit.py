import copy

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom

from smal.io import from_smi, to_smi, random_fle


def remove_atom_at_index(mol: Chem.Mol, atm_idx: int) -> Chem.Mol:
    """
    Removes the atom at the given index atm_idx.

    >>> m = from_smi('FCC[Cl]')
    >>> for i in range(m.GetNumAtoms()):
    ...     print(to_smi(remove_atom_at_index(m,i)))
    CCCl
    CCl.F
    CF.Cl
    CCF
    """
    mol = copy.deepcopy(mol)
    if isinstance(atm_idx,list) or isinstance(atm_idx,tuple):
        for ai in atm_idx:
            mol.GetAtomWithIdx(ai).SetAtomicNum(0)
    else:
        mol.GetAtomWithIdx(atm_idx).SetAtomicNum(0)
    mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts("[#0]"))
    return mol


def single_non_ring_bonds(mol: Chem.Mol) -> list:
    """
    >>> m = from_smi('C1CCCCC1CCC=CC')
    >>> len(single_non_ring_bonds(m))
    4
    """
    return [
        bond
        for bond in mol.GetBonds()
        if round(bond.GetBondTypeAsDouble()) == 1 and not bond.IsInRing()
    ]


def remove_bond_at_index(
    mol: Chem.Mol,
    bnd_idx: int,
) -> Chem.Mol:
    """
    Removes the bond at the given index bnd_idx.

    >>> m = from_smi('FCC[Cl]')
    >>> for i in range(m.GetNumBonds()):
    ...     print(to_smi(remove_bond_at_index(m,i)))
    CCCl.F
    CCl.CF
    CCF.Cl
    """
    emol = Chem.EditableMol(mol)
    bnd = mol.GetBondWithIdx(bnd_idx)
    emol.RemoveBond(bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx())
    return emol.GetMol()


def embed_molecule(
    mol: "Chem.Mol", enforceChirality=None, maxAttempts=None
) -> "Chem.Mol":
    if enforceChirality is None:
        enforceChirality = 0
    if maxAttempts is None:
        maxAttempts = 100
    mol = Chem.AddHs(mol)
    ps = rdDistGeom.ETKDGv3()
    ps.enforceChirality = False
    AllChem.EmbedMolecule(
        mol,
        enforceChirality=enforceChirality,
        maxAttempts=maxAttempts,
    )
    return mol


def mol_to_xyzstring(
    mol: "Chem.Mol",
    do_embed=None,
    enforceChirality=None,
    maxAttempts=None,
) -> "tuple[Chem.Mol, str]":
    """
    Embeds the given molecule in 3D and returns the
    resulting molecule with 3D coordinates and the
    corresponding xyz string.

    >>> mol = from_smi('CCCO')
    >>> mol, xyz = mol_to_xyzstring(mol)
    >>> print(xyz) #doctest:+SKIP
    12
    <BLANKLINE>
    C     -1.285327   -0.056776    0.434662
    C     -0.175447    0.695786   -0.299881
    C      0.918409   -0.342619   -0.555572
    O      1.309356   -0.801512    0.717050
    H     -1.923389    0.626223    0.994971
    H     -0.850209   -0.851665    1.084577
    H     -1.831209   -0.618285   -0.370858
    H     -0.509218    1.121896   -1.245390
    H      0.216108    1.472826    0.409811
    H      0.439516   -1.203877   -1.086785
    H      1.744456    0.056257   -1.140213
    H      1.946955   -0.098256    1.057627
    <BLANKLINE>
    """
    if do_embed is None:
        do_embed = True

    if do_embed:
        mol = embed_molecule(
            mol,
            enforceChirality=enforceChirality,
            maxAttempts=maxAttempts,
        )

    rand_fle = random_fle("xyz")
    Chem.MolToXYZFile(mol, str(rand_fle))
    if rand_fle.exists():
        xyz_string = rand_fle.read_text()
        rand_fle.unlink()
        return mol, xyz_string
    else:
        return None, None


def find_radical_positions(
    mol: Chem.Mol,
) -> "list[int]":
    # see https://asteeves.github.io/blog/2015/01/14/editing-in-rdkit/
    for atm_idx in range(mol.GetNumAtoms()):
        atm = mol.GetAtomWithIdx(atm_idx)
        if atm.GetNumRadicalElectrons() > 0:
            yield atm_idx


def mark_radical_position(
    mol: Chem.Mol,
    marker=None,
) -> Chem.Mol:
    """
    Marks the radical position in the given molecule
    so a fingerprint can be generated to mark the
    radical local environment.

    This is necessary because radicals will not show up in the ecfp,
    but we want them to show up.

    >>> to_smi(mark_radical_position(from_smi("CC([NH])CN")))
    'CC(CN)[NH][As]'
    """
    # see https://asteeves.github.io/blog/2015/01/14/editing-in-rdkit/
    if marker is None:
        marker = "[As]"
    for atm_idx in range(mol.GetNumAtoms()):
        atm = mol.GetAtomWithIdx(atm_idx)
        if atm.GetNumRadicalElectrons() > 0:
            marker = from_smi(marker)

            end_parent = mol.GetNumAtoms()
            mol = Chem.CombineMols(mol, marker)
            edmol = Chem.EditableMol(mol)
            edmol.AddBond(atm_idx, end_parent, order=Chem.rdchem.BondType.SINGLE)
            mol = edmol.GetMol()
            break
    return mol


def all_h_abstraction_fragments(mol: Chem.Mol) -> "list[Chem.Mol]":
    """
    >>> m = from_smi('CCCCOC')
    >>> [to_smi(Chem.RemoveHs(f[0])) for f in all_h_abstraction_fragments(m)]
    ['[CH2]CCCOC', '[CH2]CCCOC', '[CH2]CCCOC', 'C[CH]CCOC', 'C[CH]CCOC', 'CC[CH]COC', 'CC[CH]COC', 'CCC[CH]OC', 'CCC[CH]OC', '[CH2]OCCCC', '[CH2]OCCCC', '[CH2]OCCCC']
    """
    mol = Chem.AddHs(mol)

    def h(mol):
        for atm_idx in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(atm_idx).GetSymbol() == "H":
                yield remove_atom_at_index(mol, atm_idx), atm_idx

    return list(h(mol))


def all_hydrolysis_candidates(mol: Chem.Mol, only_atm_idx=None) -> "list[Chem.Mol]":
    """

    >>> mol = from_smi("CCN")
    >>> [to_smi((cand)) for cand in all_hydrolysis_candidates(mol)]
    ['CCO.N']
    >>> mol = from_smi("CC[Cl]")
    >>> [to_smi((cand)) for cand in all_hydrolysis_candidates(mol)]
    ['CCO.Cl']
    """
    N = mol.GetNumAtoms()
    for atm_idx in range(N):
        if only_atm_idx is not None:
            if only_atm_idx != atm_idx:
                continue
        atm = mol.GetAtomWithIdx(atm_idx)
        symb = atm.GetSymbol()
        if symb.lower() in ["h", "o", "cl", "br", "i"]:
            continue  # can't hydroxylate those elements
        else:
            # This atom could (potentially) by hydroxylated
            for bnd in atm.GetBonds():
                if round(bnd.GetBondTypeAsDouble()) != 1:
                    continue  # don't break double/triple bonds

                [other_idx] = list(
                    {bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()} - {atm_idx}
                )
                other_symb = mol.GetAtomWithIdx(other_idx).GetSymbol()
                if other_symb.lower() in ["h", "c"]:
                    continue  # not valid leaving groups

                end_parent = mol.GetNumAtoms()
                hydroxyl = from_smi("O")
                cand = copy.deepcopy(mol)  # avoid mutations
                cand = remove_bond_at_index(cand, bnd.GetIdx())
                cand = Chem.CombineMols(cand, hydroxyl)
                edmol = Chem.EditableMol(cand)
                edmol.AddBond(atm_idx, end_parent, order=Chem.rdchem.BondType.SINGLE)
                cand = edmol.GetMol()
                # now the molecule is hydroxylated at atom with index <atm_idx>
                # and the bond <bnd> has been homolytically split.
                # Finally, we add the missing hydrogens, and we are done:
                # cand = Chem.AddHs(cand)
                yield cand

def all_hydrolysis_candidates_2(mol: Chem.Mol, only_atm_idx=None) -> "list[Chem.Mol]":
    """

    >>> mol = from_smi("CC(=O)OC")
    >>> [to_smi((cand)) for cand in all_hydrolysis_candidates_2(mol)]
    ['COC(C)(O)O']
    """
    N = mol.GetNumAtoms()
    for atm_idx in range(N):
        if only_atm_idx is not None:
            if only_atm_idx != atm_idx:
                continue
        atm = mol.GetAtomWithIdx(atm_idx)
        symb = atm.GetSymbol()
        if symb.lower() in ["h", "o", "cl", "br", "i"]:
            continue  # can't hydroxylate those elements
        else:
            # This atom could (potentially) by hydroxylated
            for bnd in atm.GetBonds():
                if round(bnd.GetBondTypeAsDouble()) not in [2,3]:
                    continue  # only break double/triple bonds...

                [other_idx] = list(
                    {bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()} - {atm_idx}
                )
                other_symb = mol.GetAtomWithIdx(other_idx).GetSymbol()
                if other_symb.lower() in ["c"]:
                    continue  # not valid "leaving groups"

                end_parent = mol.GetNumAtoms()
                hydroxyl = from_smi("O")
                cand = copy.deepcopy(mol)  # avoid mutations
                cand = remove_bond_at_index(cand, bnd.GetIdx())
                cand = Chem.CombineMols(cand, hydroxyl)
                edmol = Chem.EditableMol(cand)
                # attach hydroxyl
                edmol.AddBond(atm_idx, end_parent, order=Chem.rdchem.BondType.SINGLE)
                # converting the double-bond to single-bond
                edmol.AddBond(atm_idx, other_idx, order=Chem.rdchem.BondType.SINGLE)
                cand = edmol.GetMol()
                # now the molecule is hydroxylated at atom with index <atm_idx>
                # and the bond <bnd> has been homolytically split.
                # Finally, we add the missing hydrogens, and we are done:
                # cand = Chem.AddHs(cand)
                yield cand
