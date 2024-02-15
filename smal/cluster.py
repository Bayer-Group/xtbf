import random
import pandas as pd


import random
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator

from typing import Iterable


import hashlib

from smal.io import iupac_to_smiles


def split_in_chunks(
    df: pd.DataFrame,
    chunk_size: int,
) -> Iterable[pd.DataFrame]:
    """
    Yields an iterable of chunk_sized chunks of the given
    dataframe.

    Useful to reduce the memory load in e.g. predict calls.

    >>> df = pd.DataFrame({"x": list(range(20)), "y":[itm**2 for itm in range(20)]})
    >>> rslt = list(split_in_chunks(df,8))
    >>> rslt
    [   x   y
    0  0   0
    1  1   1
    2  2   4
    3  3   9
    4  4  16
    5  5  25
    6  6  36
    7  7  49,      x    y
    8    8   64
    9    9   81
    10  10  100
    11  11  121
    12  12  144
    13  13  169
    14  14  196
    15  15  225,      x    y
    16  16  256
    17  17  289
    18  18  324
    19  19  361]
    >>> (pd.concat(rslt) == df).all().all()
    True

    """
    N = len(df)
    start = 0
    while start < N:
        end = start + chunk_size
        yield df.iloc[start:end]
        start = end


def sha1_hash(s: str) -> str:
    hash_object = hashlib.sha1(s.encode("utf8"))
    return hash_object.hexdigest()


def clusters_to_split(
    clusters: "list[list[int]]",
    split_ratio: "list[int]",
    eta=0.000001,
    random_seed=None,
):
    """
    Given the clusters in the format cluster-index->instance-indices, this method
    produces a new split with given split ratios.

    For example, given three clusters 1,2,3,4,5,...,10 where 1 contains five instances,
    2 contains 3 instances and 3-10 contain a single instance:
    >>> clusters = [[129,23,111,590,42],[40,27,8],[44],[99],[139],[1],[7],[333],[222],[98]]

    We can use this method to create a (roughly) 50:50 train/test split:
    >>> splits = clusters_to_split(clusters,split_ratio=[0.5,0.5],random_seed=123)
    >>> splits
    [[0, 2, 8, 9], [1, 3, 4, 5, 6, 7]]

    And these represent indeed roughly a 50:50 split:
    >>> [sum(len(clusters[idx]) for idx in split) for split in splits]
    [8, 8]

    >>> for loop in range(10):
    ...     clusters = [[i] for i in range(100)]
    ...     splits = clusters_to_split(clusters,split_ratio=[0.2,0.8],random_seed=123+loop)
    ...     print([len(split) for split in splits])
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]
    [20, 80]

    >>> clusters = [[11],[12],[13],[3,2,1],[4,5,6],[8],[7],[9]]
    >>> splits = clusters_to_split(clusters,split_ratio=[0.5,0.1,0.4],random_seed=123)
    >>> splits
    [[0, 1, 3], [5, 6], [2, 4, 7]]
    >>> n_instances = sum([len(clu) for clu in clusters])
    >>> [sum([round(100 * len(clusters[clus]) / n_instances) for clus in split]) for split in splits]
    [41, 16, 41]


    """
    if random_seed:
        random.seed(random_seed)
    n_instances = sum(map(len, clusters))
    n_splits = len(split_ratio)
    splits = [[] for _ in range(n_splits)]
    for cluster_idx, cluster in enumerate(clusters):
        while True:
            split = random.randint(0, n_splits - 1)
            if random.random() < split_ratio[split]:
                splits[split].append(cluster_idx)
                split_ratio[split] -= len(cluster) / n_instances
                split_ratio[split] = max(split_ratio[split], eta)
                break
    return splits


def tanimoto_distance_matrix(fp_list):
    """Calculate distance matrix for fingerprint list"""
    dissimilarity_matrix = []
    for i in range(1, len(fp_list)):
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix


def simple_butina_clustering(
    fps,
    cutoff=0.4,
):
    dist_mat = tanimoto_distance_matrix(fps)
    clusters = Butina.ClusterData(dist_mat, len(fps), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


def add_split_by_butina_clustering(
    df,
    col="smiles",
    amount_train=0.6,
    amount_val=0.2,
    amount_test=0.2,
    random_seed=None,
    butina_radius=0.4,
    relabel=True,
):
    """
    Creates a train/val/test split on the given dataframe
    by utilizing the butina clustering.

    In normal operation relabel should always be True. If set
    to False, then the original butina clustering labels are
    kept! This is more for debugging/demonstration purposes,
    or really in any case where the raw butina clustering
    labels should be accessed.

    >>> names = ["methylanthracene","ethylanthracene","propylanthracene", \
         "furane", "2-methyl-furane", "3-methyl-furane", "2-ethyl-furane", \
         "glucose", "fructose", "galactose", \
         "bromobenzene", "chlorobenzene", "fluorobenzene", "iodobenzene", \
        ] 
    >>> smiles = [iupac_to_smiles(name) for name in names]
    >>> df = pd.DataFrame({"smiles": smiles})
    >>> print('\\n'.join([str((row.smiles[0:-1],row.split)) for _,row in add_split_by_butina_clustering(df,random_seed=123).iterrows()])) # doctest:+NORMALIZE_WHITESPACE
    ('CC1=CC=CC2=CC3=CC=CC=C3C=C12', 'train')
    ('C(C)C1=CC=CC2=CC3=CC=CC=C3C=C12', 'train')
    ('C(CC)C1=CC=CC2=CC3=CC=CC=C3C=C12', 'train')
    ('O1C=CC=C1', 'test')
    ('CC=1OC=CC1', 'train')
    ('CC1=COC=C1', 'test')
    ('C(C)C=1OC=CC1', 'train')
    ('O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO', 'train')
    ('OCC(=O)[C@@H](O)[C@H](O)[C@H](O)CO', 'train')
    ('O=C[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO', 'train')
    ('BrC1=CC=CC=C1', 'val')
    ('ClC1=CC=CC=C1', 'test')
    ('FC1=CC=CC=C1', 'val')
    ('IC1=CC=CC=C1', 'val')
    """
    smiles = list(sorted(set(df[col].tolist())))
    random.seed(random_seed)
    random.shuffle(smiles)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    # ecfps =[ecfp_fingerprint(mol) for mol in mols]
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    fingerprints = [rdkit_gen.GetFingerprint(mol) for mol in mols]
    butina_clusters = simple_butina_clustering(
        fingerprints,
        cutoff=butina_radius,
    )

    splits = clusters_to_split(
        butina_clusters,
        split_ratio=[amount_train, amount_val, amount_test],
        random_seed=random_seed,
    )

    smi_to_clu = {}
    for clu_idx, clu in enumerate(butina_clusters):
        for mol_idx in clu:
            smi_to_clu[smiles[mol_idx]] = clu_idx

    clu_to_split = {}
    for split_idx, clusters in enumerate(splits):
        for cluster in clusters:
            clu_to_split[cluster] = split_idx

    if relabel:
        smi_to_split = {smi: clu_to_split[smi_to_clu[smi]] for smi in smiles}
    else:
        smi_to_split = {smi: smi_to_clu[smi] for smi in smiles}

    df["split"] = df[col].map(smi_to_split)

    if relabel:
        df.split = df.split.map(
            {
                0: "train",
                1: "val",
                2: "test",
            }
        )

    return df


def add_cv_by_butina_clustering(
    df,
    col="smiles",
    n_cvs=5,
    random_seed=None,
    butina_radius=0.4,
):
    """

    >>> df = pd.DataFrame({"smiles": ["CCCPCC","CCCCPCC","COCC","COCCC","c1ccc1C=O","c1c(C)cc1C=O"]})
    >>> add_cv_by_butina_clustering(df,random_seed=123)#doctest:+NORMALIZE_WHITESPACE
            smiles  cross_fold
    0        CCCPCC           0
    1       CCCCPCC           0
    2          COCC           1
    3         COCCC           1
    4     c1ccc1C=O           3
    5  c1c(C)cc1C=O           3
    >>> add_cv_by_butina_clustering(df,random_seed=456,) #doctest:+NORMALIZE_WHITESPACE
            smiles  cross_fold
    0        CCCPCC           1
    1       CCCCPCC           1
    2          COCC           3
    3         COCCC           3
    4     c1ccc1C=O           0
    5  c1c(C)cc1C=O           0
    """
    smiles = list(sorted(set(df[col].tolist())))
    random.seed(random_seed)
    random.shuffle(smiles)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    # ecfps =[ecfp_fingerprint(mol) for mol in mols]
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    fingerprints = [rdkit_gen.GetFingerprint(mol) for mol in mols]
    butina_clusters = simple_butina_clustering(
        fingerprints,
        cutoff=butina_radius,
    )

    splits = clusters_to_split(
        butina_clusters,
        split_ratio=[1 / n_cvs for _ in range(n_cvs)],
        random_seed=random_seed,
    )

    smi_to_clu = {}
    for clu_idx, clu in enumerate(butina_clusters):
        for mol_idx in clu:
            smi_to_clu[smiles[mol_idx]] = clu_idx

    clu_to_split = {}
    for split_idx, clusters in enumerate(splits):
        for cluster in clusters:
            clu_to_split[cluster] = split_idx

    smi_to_split = {smi: clu_to_split[smi_to_clu[smi]] for smi in smiles}

    df["cross_fold"] = df[col].map(smi_to_split)

    return df


def add_split_by_col(
    df: pd.DataFrame,
    col: str,
    amount_train=0.7,
    amount_test=0.3,
    amount_val=0.0,
    random_seed=None,
) -> None:
    """
    A simple utility that splits by the given column col.
    Useful because one usually never wants to split on the
    iloc of the dataframe (leakage guaranteed iff replicates present),
    but almost always on a given column.

    >>>

    """
    if random_seed:
        random.seed(random_seed)
    split_col = random.choices(
        ["train", "test", "val"],
        weights=[amount_train, amount_test, amount_val],
        k=df[col].nunique(),
    )
    split_dct = {k: v for v, k in zip(split_col, df[col].unique())}
    df["split"] = df[col].map(split_dct)
    return None


def add_cv_by_col(
    df,
    col="smiles",
    n_cvs=5,
    random_seed=None,
) -> None:
    if random_seed:
        random.seed(random_seed)
    split_col = random.choices(
        list(range(n_cvs)),
        weights=[1 / n_cvs for _ in range(n_cvs)],
        k=df[col].nunique(),
    )
    split_dct = {k: v for v, k in zip(split_col, df[col].unique())}
    df["cv"] = df[col].map(split_dct)
    return None
