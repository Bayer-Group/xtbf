from contextlib import redirect_stderr, redirect_stdout
import io
import multiprocessing
import uuid
from joblib import Parallel, delayed
import tqdm
import random
import re
import numpy as np
import pandas as pd
import shutil
import os
from pathlib import Path


from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

import tempfile

# Path to the xtb binary

BIN_XTB = os.getenv("XTBF__BIN_XTB") or "xtb"

import subprocess
import time
import psutil


class Silencer:
    """
    A useful tool for silencing stdout and stderr.
    Usage:
    >>> with Silencer() as s:
    ...         print("kasldjf")

    >>> print("I catched:",s.out.getvalue())
    I catched: kasldjf
    <BLANKLINE>

    Note that nothing was printed and that we can later
    access the stdout via the out field. Similarly,
    stderr will be redirected to the err field.
    """

    def __init__(self):
        self.out = io.StringIO()
        self.err = io.StringIO()

    def __enter__(self):
        self.rs = redirect_stdout(self.out)
        self.re = redirect_stderr(self.err)
        self.rs.__enter__()
        self.re.__enter__()
        return self

    def __exit__(self, exctype, excinst, exctb):
        self.rs.__exit__(exctype, excinst, exctb)
        self.re.__exit__(exctype, excinst, exctb)



try:
    with Silencer() as s:
        xtb_version = subprocess.check_output(
            [f"{BIN_XTB}", "--version"],
            stderr=subprocess.DEVNULL,
        )
    if not isinstance(xtb_version, str):
        xtb_version = xtb_version.decode("utf-8")
    xtb_version = re.findall(r"(version \d+.\d+.\d+)", xtb_version)
except:
    print("could not determine xtb version.")
    print(
        "most likely no xtb binary is installed. See: https://xtb-docs.readthedocs.io/en/latest/setup.html"
    )
    raise

if xtb_version:
    xtb_version = xtb_version[0]
    if xtb_version >= "6.5.1":
        print("xtb version:", xtb_version)
    else:
        assert (
            f"detected outdated xtb version: '{xtb_version}'. Please install version >= 6.5.1."
            "see https://xtb-docs.readthedocs.io/en/latest/setup.html"
        )
else:
    print("could not determine xtb version.")
    print(
        "most likely no xtb binary is installed. See: https://xtb-docs.readthedocs.io/en/latest/setup.html"
    )
    exit(1)


# Try to prioritize memory mapped file system
# to improve speed and reduce strain,
# fallback to potentially memory mapped, or
# non-mem mapped file system otherwise...
TMP_ROOT = Path("/dev/shm/") # nosec
if not TMP_ROOT.exists():
    print("Warning: could not find /dev/shm/ mem-mapped io not possible")
    TMP_ROOT = Path("/tmp") # nosec
if not TMP_ROOT.exists():
    TMP_ROOT = tempfile.gettempdir()

XTB_TMP_DIR = TMP_ROOT / Path("xtbf")

XTB_OUTER_JOBS = min(32,multiprocessing.cpu_count()-1)
XTB_INNER_JOBS = 1
XTB_TIMEOUT_SECONDS = int(os.environ.get("XTBF_TIMEOUT",1200)) # 20 minutes


def temp_dir():
    """
    Creates a path to a temporary directory
    """
    if Path("/tmp").exists(): # nosec
        tmp_dir = Path("/tmp") / "xtbf" # nosec
    else:
        tmp_dir = Path(tempfile.gettempdir()) / "xtbf"
    tmp_dir.mkdir(
        exist_ok=True,
    )
    return tmp_dir


def random_fle(
    suf: str,
):
    """
    Creates a random file with the given suffix suf
    """
    fle_name = f"{uuid.uuid4().hex}.{suf}"
    return temp_dir() / fle_name


def mol_to_xyzstring(
    mol: "Chem.Mol",
    conf_id=0,
) -> "tuple[Chem.Mol, str]":
    """
    Embeds the given molecule in 3D and returns the
    resulting molecule with 3D coordinates and the
    corresponding xyz string.

    >>> mol = embed_molecule(Chem.MolFromSmiles('CCCO'))
    >>> mol, xyz = mol_to_xyzstring(mol)
    >>> print(xyz) # doctest: +SKIP
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
    rand_fle = random_fle("xyz")
    Chem.MolToXYZFile(mol, str(rand_fle), confId=conf_id)
    if rand_fle.exists():
        xyz_string = rand_fle.read_text()
        rand_fle.unlink()
        return mol, xyz_string
    else:
        return None, None
        

"""
A simple eventually consistent cache utility that
supports asynchronous access.

Uses the browser temporary file hack to handle
collisions during concurrent write access.


>>> cache = Path(f"/tmp/test/{random.randint(1,123)}")
>>> obj = {"my-key":123, "another-key":456,}
>>> cache_store(obj,"my-obj-id",cache)
>>> cache_load("my-obj-id",cache)
{'my-key': 123, 'another-key': 456}


"""

import hashlib
import joblib
import shutil

def cache_store(obj, obj_id:str, cache:Path):
    if isinstance(cache,str):
        cache = Path(cache)
    fle = _obj_id_to_fle(obj_id,cache)
    tmp_fle = fle.with_suffix(f".{random.randint(1,1000000000)}tmp")
    tmp_fle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj,tmp_fle)
    shutil.move(tmp_fle,fle)

def cache_load(obj_id:str, cache:Path):
    if isinstance(cache,str):
        cache = Path(cache)
    fle = _obj_id_to_fle(obj_id,cache)
    try:
        return joblib.load(fle)
    except:
        return None

def _obj_id_to_fle(obj_id:str, cache:Path,) -> Path:
    try:
        obj_hash = hashlib.sha1(obj_id.encode("utf-8"), usedforsecurity=False,).hexdigest()
    except:
        obj_hash = hashlib.sha1(obj_id.encode("utf-8"),).hexdigest()
    sub_dir_0 = obj_hash[0:3]
    sub_dir_1 = obj_hash[3:6]
    obj_fle = cache / sub_dir_0 / sub_dir_1 / obj_hash
    obj_fle = obj_fle.with_suffix(".pkl")
    return obj_fle

def run_parallel(
    n_jobs:int,
    fun:"callable", elts:list,
    do_tqdm:bool,
) -> None:
    if n_jobs > 1:
        rslt = Parallel(n_jobs=n_jobs)(
            delayed(fun)(elt)
            for elt in (tqdm.tqdm(elts) if do_tqdm else elts)
        )
    else:
        rslt = [fun(elt) for elt in (tqdm.tqdm(elts) if do_tqdm else elts)] 
    return rslt

def embed_molecule(mol: "Chem.Mol", enforceChirality:int=1, maxAttempts=100) -> "Chem.Mol":
    mol = Chem.AddHs(mol)
    ps = rdDistGeom.ETKDGv3()
    ps.enforceChirality = bool(enforceChirality)
    AllChem.EmbedMolecule(
        mol,
        enforceChirality=enforceChirality,
        maxAttempts=maxAttempts,
    )
    return mol

def embed_multi(
    mol: "Chem.Mol",
    n_confs: int,
) -> "Chem.Mol":
    mol = Chem.AddHs(mol)
    ps = rdDistGeom.ETKDGv3()
    cids = rdDistGeom.EmbedMultipleConfs(mol, n_confs, ps)

    return cids

def embed_multi_keep_lowest(
    mol: "Chem.Mol",
    n_confs: int,
) -> "Chem.Mol":
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMultipleConfs(
            mol, numConfs=n_confs, enforceChirality=False,
            randomSeed=0xf00d,)
    mmff_optimized = Chem.AllChem.MMFFOptimizeMoleculeConfs(
        mol, maxIters=2000)
    ens = np.array(mmff_optimized)[:,1]
    for en,conf in zip(ens,mol.GetConformers()):
        conf.SetDoubleProp("energy",en)
    return mol


def _determine_min_conf_id(mol):
    min_conf_id = None
    min_en = None
    for conf in mol.GetConformers():
        en = conf.GetDoubleProp("energy")
        if min_en is None or en < min_en :
            min_en = en
            min_conf_id = conf.GetId()
    if min_conf_id is None:
        raise ValueError("no conformer (with energy) found")

    return min_conf_id


def run_xtb(xtb_command:str, mol:Chem.Mol, multiplicity:int, conf_id:int="lowest", cache=None, store_failures=True, charge:int=0,):
    """

    >>> mol = Chem.MolFromSmiles("CCCOCCC")
    >>> mol = embed_molecule(mol)
    >>> success,rslt = run_xtb("--opt", mol, 1)
    >>> success
    True

    The xtb command can also employ a cache to avoid recomputation of results:

    >>> mol = Chem.MolFromSmiles("CCCOCCC")
    >>> mol = embed_molecule(mol)
    >>> start = time.time()
    >>> cache = f"/tmp/xtb_test{random.randint(1,1234)}/"
    >>> success,rslt = run_xtb("--opt", mol, 1, cache=cache)
    >>> success
    True
    >>> end = time.time()
    >>> delta1 = (end-start)
    >>> start = time.time()
    >>> success,rslt = run_xtb("--opt", mol, 1, cache=cache)
    >>> success
    True
    >>> end = time.time()
    >>> delta2 = (end-start)
    
    Then, caching made the whole process faster:
    >>> print((delta1 / delta2) > 1)
    True

    We can also perform a conformer search and keep only the lowest
    conformer:
    >>> mol = Chem.MolFromSmiles("CCCOCCC")
    >>> mol = embed_multi_keep_lowest(mol, 100)
    >>> success,rslt = run_xtb("--opt --alpb water", mol, 1, conf_id="lowest", )
    >>> success
    True

    """
    assert conf_id == "lowest"

    smi = Chem.MolToSmiles(mol)
    obj_id = "__".join(["v1",str(smi),str(xtb_command)])

    if cache is None:
        if conf_id == "lowest":
            mol = embed_multi_keep_lowest(mol,100)
            conf_id = _determine_min_conf_id(mol)
        mol, xyz_string = mol_to_xyzstring(mol,conf_id=conf_id)
        return run_xtb_xyz(xtb_command=xtb_command,xyz_string=xyz_string, multiplicity=multiplicity, )
    else:
        stored_result = cache_load(obj_id=obj_id,cache=cache)
        if stored_result is not None:
            return stored_result
        else:
            if str(os.environ.get("XTBF_SKIP_CALCS", None)) in ("1","t","T","True","true"):
                return None
            
            if conf_id == "lowest":
                mol = embed_multi_keep_lowest(mol,100)
                conf_id = _determine_min_conf_id(mol)
            mol, xyz_string = mol_to_xyzstring(mol,conf_id=conf_id)
            try:
                rslt = run_xtb_xyz(xtb_command=xtb_command,xyz_string=xyz_string, multiplicity=multiplicity, charge=charge, )
            except:
                if store_failures:
                    rslt = None
                else:
                    raise
            cache_store(rslt,obj_id=obj_id,cache=cache)
            return rslt

def run_xtb_xyz(xtb_command:str, xyz_string:str, multiplicity:int, charge:int=0,):
    """
    Runs the given xtb job as identified by
    the following components:
     - The command <xtb>, 
     - the xyz string of the molecule <xyz_string>

    """
    assert XTB_TMP_DIR.parent.exists()
    XTB_TMP_DIR.mkdir(exist_ok=True)
    assert XTB_TMP_DIR.exists()
    job_dir = XTB_TMP_DIR / f"job_{random.randint(1,10**9)}"
    assert not job_dir.exists(), "internal logic error"
    cwd_before = os.getcwd()

    if multiplicity != 1:
        uhf_arg = f"--uhf {multiplicity} "
    else:
        uhf_arg = ""
    if charge != 0:
        chrg_arg = f"--chrg {charge} "
    else:
        chrg_arg = ""
    try:
        os.mkdir(job_dir)
        os.chdir(job_dir)
        (job_dir / "input.xyz").write_text(xyz_string)
        output_fle = job_dir / "output.out"
        assert not output_fle.exists()
        with Silencer() as s:
            # alpb_str = ""
            cmd = f"{BIN_XTB} input.xyz --parallel {XTB_INNER_JOBS} {xtb_command} {uhf_arg} {chrg_arg}> output.out 2> err.out"

            # normal approach:
            # subprocess.check_output(cmd,shell=True,timeout=XTB_TIMEOUT_SECONDS)
            # ^
            # \__ sadly, this approach does not work because the timeout isnt applied
            #
            # As suggested in https://stackoverflow.com/questions/48763362/python-subprocess-kill-with-timeout
            # we apply the following workaround:
            parent = subprocess.Popen(cmd, shell=True) # nosec
            for _ in range(XTB_TIMEOUT_SECONDS):
                if parent.poll() is not None:  # process just ended
                    break
                time.sleep(1)
            else:
                # the for loop ended without break: timeout
                parent = psutil.Process(parent.pid)
                for child in parent.children(
                    recursive=True
                ):  # or parent.children() for recursive=False
                    child.kill()
                parent.kill()

        if output_fle.exists():
            try:
                rslt = (True, output_fle.read_text())
            except UnicodeDecodeError:
                rslt = (True, output_fle.read_text(encoding='latin-1'))
            return rslt
        else:
            return False, ""
    finally:
        os.chdir(cwd_before)
        # Garbage Collection
        if job_dir.exists():
            shutil.rmtree(job_dir)


