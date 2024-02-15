"""
**S**mall **m**olecule **a**bstraction **l**ayer

A collection of utilities for working with small molecules.

The idea is to provide nice shortcuts to things that we often need in our day to day.
So, instead of having to type something like
>>> import numpy as np 
>>> from rdkit import Chem
>>> from rdkit.Chem import AllChem, DataStructs
>>> mol = Chem.MolFromSmiles('CCCOCCN')
>>> radius = 2
>>> n_bits = 2048
>>> fp = AllChem.GetMorganFingerprintAsBitVect(
...     mol,
...     radius=radius,
...     nBits=n_bits,
...     )
>>> fp_array = np.zeros((0,), dtype=np.int8)
>>> DataStructs.ConvertToNumpyArray(fp, fp_array)
>>> fp_array
array([0, 0, 0, ..., 0, 0, 0], dtype=int8)

It is just
>>> from smal.all import *
>>> ecfp_fingerprint(from_smi('CCCOCCN'),radius=radius,n_bits=n_bits)
array([0, 0, 0, ..., 0, 0, 0], dtype=int8)


"""
