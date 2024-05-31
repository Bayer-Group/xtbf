from xtbf import *
from smal.all import *
import subprocess
 
def atom_access_score(mol,atm_idx:int,cutoff=0.5,cache=None,):
    """
    >>> for smi in ["CC","CCC","CC(C)(C)(C)","C1C(CC)C(CC)C(CCC)C1"]: # doctest: +SKIP
    ...     print(smi)
    ...     print(atom_access_score(from_smi(smi),2))
    """
    if cache:
        smi = to_smi(mol)
        result = cache_load(smi,cache)
        if result is not None:
            return result

    fle = random_fle(".xyz")
    
    xyz = mol_to_xyzstring(mol)[1]
    xyz_no_h = [
        lne for lne in xyz.split("\n")
        if not "H" in lne.split()
    ]
    hs = [lne for lne in xyz.split("\n") if "H" in lne.split()]
    xyz_no_h[0] = str(int(xyz.split("\n")[0]) - len(hs))
    xyz_no_h = "\n".join(xyz_no_h)
    fle.write_text(xyz_no_h)
    result = subprocess.run(['atom_access', str(fle), "-a", str(atm_idx),"-c",str(cutoff),], stdout=subprocess.PIPE)
    result = result.stdout.decode("utf-8")
    result = [lne for lne in result.split("\n") if "of solid angle is visible" in lne]
    if result:
        assert len(result) == 1
        result = float(result[0].split()[0].replace("\x1b[32m",""))
        if cache:
            smi = to_smi(mol)
            cache_store(result,smi,cache)
        return result
    else:
        return None
