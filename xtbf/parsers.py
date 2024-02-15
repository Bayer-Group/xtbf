
import re


def _test_partial_charges() -> str:
    return """
    HL-Gap            0.4164018 Eh           11.3309 eV
    Fermi-level           -0.1811489 Eh           -4.9293 eV

     #   Z          covCN         q      C6AA      α(0)
     1   6 C        3.753    -0.106    22.589     6.778
     2   6 C        3.798    -0.064    21.737     6.638
     3   6 C        3.697     0.080    19.407     6.301
     4   8 O        1.701    -0.319    19.336     5.937
     5   6 C        3.696     0.088    19.284     6.282

    """

def partial_charges(s:str):
    """
    Reads partial charges from 
    the given xtb output string.

    >>> partial_charges(_test_partial_charges())
    [[-0.106, -0.064, 0.08, -0.319, 0.088]]
    """
    header = "#   Z          covCN         q      C6AA      α(0)"
    parsing = False
    all_rslt = []
    rslt = []
    for lne in s.split("\n"):
        lne = lne.strip()
        if lne == header:
            parsing = True
            continue
        if not lne.strip():
            parsing = False
            if rslt:
                all_rslt.append(rslt)
            rslt = []
        if parsing:
            lne = lne.strip().split()
            rslt.append(
                float(lne[4])
            )
    return all_rslt


def _test_total_energy():
    return """
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         ::                     SUMMARY                     ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         :: total energy             -24.049430339454 Eh    ::
         :: gradient norm              0.000226432010 Eh/a0 ::
         :: HOMO-LUMO gap             11.172729268103 eV    ::
         ::.................................................::
         :: SCC energy               -24.351760879200 Eh    ::
         :: -> isotropic ES            0.021341903977 Eh    ::
         :: -> anisotropic ES          0.005354357932 Eh    ::
         :: -> anisotropic XC          0.009269957035 Eh    ::
         :: -> dispersion             -0.011203867627 Eh    ::
         :: repulsion energy           0.302359352207 Eh    ::
         :: add. restraining           0.000000000000 Eh    ::
         :: total charge              -0.000000000000 e     ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
    """

def total_energy(txt:str) -> float:
    """
    
    >>> total_energy(_test_total_energy())
    -24.049430339454
    """
    rslts = []
    pat = re.compile(re.escape(":: total energy")+r"\s+"+r"(\S+)\s+"+re.escape("Eh    ::"))
    for lne in txt.split("\n"):
        lne = lne.strip()
        if re.match(pat,lne):
            rslts.append(float(re.match(pat,lne).group(1)))

    return rslts[-1]
            

def _test_fukui_indices():
    return """
     #        f(+)     f(-)     f(0)
     1B       0.015    0.268    0.142
     2F       0.335    0.243    0.289
     3F       0.319    0.244    0.282
     4F       0.330    0.244    0.287
"""

import numpy as np
def fukui_indices(txt:str) -> "np.array":
    """
    
    >>> fukui_indices(_test_fukui_indices())
    array([[0.015, 0.268, 0.142],
        [0.335, 0.243, 0.289],
        [0.319, 0.244, 0.282],
        [0.33 , 0.244, 0.287]])
    """
    enter = False
    pat = re.compile("\S+\s+(\S+)\s+(\S+)\s+(\S+)")
    fs = []
    for lne in txt.split("\n"):
        lne = lne.strip()
        if lne == "#        f(+)     f(-)     f(0)":
            enter = True
        elif enter:
            rslt = re.match(pat,lne)
            if rslt:
                fs.append([float(rslt.group(g)) for g in [1,2,3]])
            else:
                enter = False
                continue
    if fs:
        return np.array(fs)
    else:
        return None
        

    
