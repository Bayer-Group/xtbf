import os

import pandas as pd

from smal.config import _DIR_EXAMPLES
from smal.io import download_file

"""
Module that provides example molecule datasets.
"""
KNOWN_SETS = [
    "HIV.csv",
    "Lipophilicity.csv",
    "SAMPL.csv",
    "delaney-processed.csv",
]


def example_smiles(example_sets: "list[str]" = None) -> "list[str]":
    """
    Returns example smiles for the given example set.

    Useful for quickly getting access to a random list of
    (realistic, drug-like) molecules for testing.

    >>> smis = example_smiles()
    >>> print('\\n'.join(smis[:10]))
    CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2
    C(=Cc1ccccc1)C1=[O+][Cu-3]2([O+]=C(C=Cc3ccccc3)CC(c3ccccc3)=[O+]2)[O+]=C(c2ccccc2)C1
    CC(=O)N1c2ccccc2Sc2c1ccc1ccccc21
    Nc1ccc(C=Cc2ccc(N)cc2S(=O)(=O)O)c(S(=O)(=O)O)c1
    O=S(=O)(O)CCS(=O)(=O)O
    CCOP(=O)(Nc1cccc(Cl)c1)OCC
    O=C(O)c1ccccc1O
    CC1=C2C(=COC(C)C2C)C(O)=C(C(=O)O)C1=O
    O=[N+]([O-])c1ccc(SSc2ccc([N+](=O)[O-])cc2[N+](=O)[O-])c([N+](=O)[O-])c1
    O=[N+]([O-])c1ccccc1SSc1ccccc1[N+](=O)[O-]
    """
    if example_sets is None or example_sets == "*":
        example_sets = KNOWN_SETS
    assert set(example_sets) <= set(
        KNOWN_SETS
    ), f"Unknown example sets: {set(example_sets) - set(KNOWN_SETS)}"

    _DIR_EXAMPLES.mkdir(exist_ok=True)
    url_templ = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}"
    for example in example_sets:
        if not (_DIR_EXAMPLES / example).exists():
            download_file(url_templ.format(example), str(_DIR_EXAMPLES / example))

    df = []
    for example_set in example_sets:
        if not example_set.endswith(".csv"):
            example_set = example_set + ".csv"
        df.append(pd.read_csv(_DIR_EXAMPLES / example_set))

    return pd.concat(df)["smiles"].tolist()
