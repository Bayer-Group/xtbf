import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def _fingerprint_to_numpy(fp):
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def ecfp_fingerprint(
    mol: Chem.Mol,
    radius=2,
    n_bits=2048,
) -> np.ndarray:
    """
    Computes the ecfp fingerprint and returns
    it as a numpy array

    >>> ecfp_fingerprint(Chem.MolFromSmiles("CCCCO"))
    array([0, 0, 0, ..., 0, 0, 0], dtype=int8)

    >>> ecfp_fingerprint(Chem.MolFromSmiles("CCCCO")).shape
    (2048,)

    >>> ecfp_fingerprint(Chem.MolFromSmiles("OCCCCO")).max()
    1

    """
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
    )
    return _fingerprint_to_numpy(fp)


def ecfp_count_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """

    >>> ecfp_fingerprint(Chem.MolFromSmiles("OCCCCO")).max()
    1
    >>> ecfp_count_fingerprint(Chem.MolFromSmiles("OCCCCO")).max()
    3
    >>> ecfp_count_fingerprint(Chem.MolFromSmiles("OCCCCO")).shape
    (2048,)

    """
    fpgen = AllChem.GetRDKitFPGenerator()
    fp = fpgen.GetCountFingerprint(mol)
    return _fingerprint_to_numpy(fp)


def ecfp_bin_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """ """
    fpgen = AllChem.GetRDKitFPGenerator()
    return fpgen.GetFingerprintAsNumPy(mol)

