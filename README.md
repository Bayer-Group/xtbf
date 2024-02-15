# xtbf
A minimal, functional interface to the semiempirical extended tight-binding (xtb) program package (https://github.com/grimme-lab/xtb).

For a read on the theoretical background and applicability of the method, see https://wires.onlinelibrary.wiley.com/doi/full/10.1002/wcms.1493. 

The goal is to make it easy to produce physical descriptors to feed those into machine learning models as informative features.

Two examples of it's usefulness:

## \#1 Example: Octanol / Water Partitioning Coefficients

With the high-level functionality exposed by xtbf, the water/octanol
partitioning coefficient can be computed within below 20 lines of code:
```python
from xtbf import run_xtb
from xtbf.parsers import total_energy

def xtb_logp_ow(mol) -> float:
    """
    Computes the octanol/water partitioning coefficient logP_ow
    or None in case any computation fails.
    Takes an rdkit molecule as input.
    """
    en_success, en_water = run_xtb("--alpb water",mol,1)
    if not en_success:
        return None
    en_water = total_energy(en_water)
    en_success, en_octanol = run_xtb("--alpb octanol",mol,1)
    if not en_success:
        return None
    en_octanol = total_energy(en_octanol)

    en_delta = en_water - en_octanol
    k = 3.166811563E-6
    T = 298
    return - np.log10( np.e ** ( - en_delta / (k * T)) )
```
and already gives good qualitative results:

<img src="/doc/log_p_parity_plot.png" width="300">

using the following code to evaluate:
```python
from smal.all import *
import seaborn as sns

data_logp = pd.DataFrame([
    {"iupac": "acetamide", "logp": -1.16,},
    {"iupac": "methanol", "logp": -0.81,},
    {"iupac": "formic acid", "logp": -0.41,},
    {"iupac": "diethylether", "logp": 0.83,},
    {"iupac": "p-dichlorobenzene", "logp": 3.37,},
    {"iupac": "hexamethylbenzene", "logp": 4.61,},
    {"iupac": "2,2',4,4',5-Pentachlorobiphenyl", "logp": 6.41,},
]) # source https://en.wikipedia.org/wiki/Partition_coefficient

data_logp["smiles"] = data_logp["iupac"].apply(iupac_to_smiles)
data_logp["mol"] = data_logp["smiles"].apply(from_smi)
data_logp["pred_logp"] = data_logp["mol"].apply(xtb_logp)
sns.scatterplot(
data=data_logp, x="pred_logp", y="logp",
)
```

## \#2 Example: Partial Charges
The following example shows how to compute partial charges:
```python
    >>> from xtbf import *
    >>> from xtbf.shortcuts import *
    >>> mol = Chem.MolFromSmiles("NCCCO")
    >>> mol = embed_molecule(mol)
    >>> add_xtb_charges(mol)
    True
    >>> for atm in mol.GetAtoms():
    ...     print(atm.GetSymbol(),"<>",atm.GetDoubleProp('xtb_partial_charge'))
    N <> -0.343
    C <> 0.021
    C <> -0.074
    C <> 0.075
    O <> -0.442
    H <> 0.138
    H <> 0.137
    H <> 0.007
    H <> 0.035
    H <> 0.034
    H <> 0.043
    H <> 0.049
    H <> 0.036
    H <> 0.284

```


# Getting Started
Installation is possible via pip
```bash
pip install xtbf
```

To run XTB-related functionality, XTB needs to be installed from:
```
https://github.com/grimme-lab/xtb/releases
```
or alternatively via conda:
```
conda install -c conda-forge xtb
```
(see here: https://anaconda.org/conda-forge/xtb)

To run the iupac-to-smiles conversion, opsin needs to be installed:
```bash
conda install bioconda::opsin
```


## Dependencies
You need to have the following minimal dependencies:
```
joblib, tqdm, numpy, pandas
```
aka:
```
	pip install joblib
	pip install tqdm
	pip install numpy 
	pip install pandas 
```
alternatively, run from make:
```
make install-deps
```


For documentation gen, pdoc needs to be installed:
```
pip install pdoc
```

# Running Tests
```
make doctests
```
runs all doctests.

# Documentation
Found in ```doc/``` folder. Interactively generated using
```
make doc-show
```



