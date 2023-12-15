from xtbf import *
from xtbf.parsers import *
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from smal.all import all_hydrolysis_candidates,all_hydrolysis_candidates_2

def add_xtb_charges_else_gasteiger_charges(
        mol:"Chem.Mol",
        multiplicity=1,
        conf_id='lowest',
        property="xtb_partial_charge",
        cache=None,):
    """
    
    >>> smi = "O=P([O-])([O-])F.[Na+].[Na+]"
    >>> mol = Chem.MolFromSmiles(smi)
    >>> mol = embed_molecule(mol)
    >>> add_xtb_charges_else_gasteiger_charges(mol)
    >>> for atm in mol.GetAtoms():
    ...     print(atm.GetSymbol(),"<>",atm.GetDoubleProp('xtb_partial_charge')) # doctest: +SKIP
    
    """
    try:
        add_xtb_charges(mol=mol,multiplicity=multiplicity,conf_id=conf_id,property=property,cache=cache,)
    except:
        ComputeGasteigerCharges(mol)
        for idx in range(mol.GetNumAtoms()):
            atm = mol.GetAtomWithIdx(idx)
            z = atm.GetDoubleProp("_GasteigerCharge")
            atm.SetDoubleProp(property,z)

def add_xtb_charges(mol:"Chem.Mol",
                    multiplicity=1,
                    conf_id='lowest',
                    property="xtb_partial_charge",
                    cache=None,
                    ) -> bool:
    """
    
    >>> mol = Chem.MolFromSmiles("NCCCO")
    >>> mol = embed_molecule(mol)
    >>> add_xtb_charges(mol)
    True
    >>> for atm in mol.GetAtoms():
    ...     print(atm.GetSymbol(),"<>",atm.GetDoubleProp('xtb_partial_charge')) # doctest: +SKIP
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
    """
    success, out = run_xtb("--opt --alpb water", mol, multiplicity, conf_id, cache=cache,)
    if not success:
        return False
    else:
        charges = partial_charges(out)[-1]
        for idx in range(mol.GetNumAtoms()):
            atm = mol.GetAtomWithIdx(idx)
            atm.SetDoubleProp(property,charges[idx])
        return True



def hydrolysis_atom_vector(mol,cache=None)->np.ndarray:
    """

    >>> mol = Chem.MolFromSmiles("CCCCCF")
    >>> hydrolysis_atom_vector(mol) # doctest: +SKIP
    array([10.        , 10.        , 10.        , 10.        ,  4.25686951,
           10.        ])
    >>> mol = Chem.MolFromSmiles("CCCCC[Cl]")
    >>> hydrolysis_atom_vector(mol) # doctest: +SKIP
    array([10.        , 10.        , 10.        , 10.        ,  1.01675859,
           10.        ])
    >>> mol = Chem.MolFromSmiles("CCCCC[Br]")
    >>> hydrolysis_atom_vector(mol) # doctest: +SKIP
    array([10.        , 10.        , 10.        , 10.        ,  3.85704342,
           10.        ])
    >>> mol = Chem.MolFromSmiles("CCCCC[I]")
    >>> hydrolysis_atom_vector(mol) # doctest: +SKIP
    array([10.        , 10.        , 10.        , 10.        , -0.49491552,
           10.        ])
    >>> mol = Chem.MolFromSmiles("CCCCC(C)(-O-)OC")
    >>> hydrolysis_atom_vector(mol) # doctest: +SKIP 
    array([10.        , 10.        , 10.        , 10.        ,  3.74949635,
           10.        , 10.        ,  3.74949635])

    >>> mol = Chem.MolFromSmiles("CCCCCOC")
    >>> hydrolysis_atom_vector(mol) # doctest: +SKIP 
    array([10.        , 10.        , 10.        , 10.        ,  5.75256376,
           10.        ,  5.75256376])


    Example of a prodrug taken from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6273601/
    We see that the ester is correctly identified as a potential 
    position for hydrolysis:
    >>> hydrolysis_atom_vector(Chem.MolFromSmiles("NCC(=O)Oc1ccc2c3c(cccc13)OC1(C=CC(=O)c3c(O)cccc31)O2")) # doctest: +SKIP 
    array([ 1.00000000e+01,  8.94397650e+00, -1.89853302e+00,  1.00000000e+01,
        1.00000000e+01, -1.89853302e+00,  1.00000000e+01,  1.00000000e+01,
        3.11363999e+00,  1.00000000e+01,  2.85188437e+00,  1.00000000e+01,
        1.00000000e+01,  1.00000000e+01,  1.00000000e+01,  1.00000000e+01,
        2.85188437e+00,  1.00000000e+01,  1.00000000e+01,  1.00000000e+01,
        1.00000000e+01,  1.00000000e+01,  5.57331958e-13,  1.00000000e+01,
        1.00000000e+01,  1.00000000e+01,  1.00000000e+01,  1.00000000e+01,
        1.00000000e+01])
    """
    CLAMP_LOW = -100
    CLAMP_HIGH = 100
    ENERGY_WATER = -5.085021399705
    HA_TO_KCAL = 627.5
    HA_TO_EV = 27
    NO_REACTION = CLAMP_HIGH / HA_TO_KCAL + ENERGY_WATER
    parent_en_success,parent_en_output = run_xtb("--alpb water", mol, 1, cache=cache)
    if not parent_en_success:
        return np.zeros(mol.GetNumAtoms())+10
    parent_en = total_energy(parent_en_output)
    hydrolysis_ens = []
    for atm_idx in range(mol.GetNumAtoms()):
        delta_ens = []
        for fragment in all_hydrolysis_candidates(mol,only_atm_idx=atm_idx):
            fragment = Chem.MolToSmiles(fragment)
            all_fragment_en_success = True
            all_fragment_en = []
            for part in fragment.split("."):
                part = Chem.MolFromSmiles(part)
                try:
                    fragment_en_success,fragment_en_output = run_xtb("--alpb water", part, 1, cache=cache) # formerly --opt --alpb water
                except:
                    fragment_en_success = False
                    fragment_en_output = ""

                all_fragment_en_success = all_fragment_en_success and fragment_en_success
                if fragment_en_success:
                    fragment_en = total_energy(fragment_en_output)
                    all_fragment_en.append(fragment_en)

            if all_fragment_en_success:
                delta_ens.append(sum(all_fragment_en) - parent_en)
        if len(delta_ens):
            hydrolysis_ens.append(min(delta_ens))
        else:
            hydrolysis_ens.append(NO_REACTION)
    return (HA_TO_KCAL * (np.array(hydrolysis_ens) - ENERGY_WATER)).clip(CLAMP_LOW,CLAMP_HIGH)


def hydrolysis_atom_vector_2(mol,cache=None)->np.ndarray:
    """

    >>> mol = Chem.MolFromSmiles("CCCC(=O)OC")
    >>> hydrolysis_atom_vector_2(mol)# doctest: +SKIP 
    array([100.        , 100.        , 100.        ,  -5.65858484,
           100.        , 100.        , 100.        ])
    >>> mol = Chem.MolFromSmiles("CCCC(=O)NC")
    >>> hydrolysis_atom_vector_2(mol)# doctest: +SKIP 
    array([100.        , 100.        , 100.        ,   1.20517045,
           100.        , 100.        , 100.        ])

    >>> mol = Chem.MolFromSmiles("CCCC(=O)Cl")
    >>> hydrolysis_atom_vector_2(mol)# doctest: +SKIP 
    array([100.        , 100.        , 100.        , -11.33278215,
        100.        , 100.        ])

    Example of a prodrug taken from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6273601/
    We see that the ester is correctly identified as a potential 
    position for hydrolysis:
    >>> hydrolysis_atom_vector_2(Chem.MolFromSmiles("NCC(=O)Oc1ccc2c3c(cccc13)OC1(C=CC(=O)c3c(O)cccc31)O2")) # doctest: +SKIP 
    array([100.        , 100.        ,  -7.12262152, 100.        ,
           100.        , 100.        , 100.        , 100.        ,
           100.        , 100.        , 100.        , 100.        ,
           100.        , 100.        , 100.        , 100.        ,
           100.        , 100.        , 100.        ,  -7.20208318,
           100.        , 100.        , 100.        , 100.        ,
           100.        , 100.        , 100.        , 100.        ,
           100.        ])

    """
    CLAMP_LOW = -20
    CLAMP_HIGH = 20
    ENERGY_WATER = -5.085021399705
    HA_TO_KCAL = 627.5
    HA_TO_EV = 27
    NO_REACTION = CLAMP_HIGH / HA_TO_KCAL + ENERGY_WATER
    parent_en_success,parent_en_output = run_xtb("--alpb water", mol, 1, cache=cache)
    if not parent_en_success:
        return np.zeros(mol.GetNumAtoms())+10
    parent_en = total_energy(parent_en_output)
    hydrolysis_ens = []
    for atm_idx in range(mol.GetNumAtoms()):
        delta_ens = []
        for fragment in all_hydrolysis_candidates_2(mol,only_atm_idx=atm_idx):
            fragment = Chem.MolToSmiles(fragment)
            all_fragment_en_success = True
            all_fragment_en = []
            for part in fragment.split("."):
                part = Chem.MolFromSmiles(part)
                try:
                    fragment_en_success,fragment_en_output = run_xtb("--alpb water", part, 1, cache=cache) # formerly --opt --alpb water
                except:
                    fragment_en_success = False
                    fragment_en_output = ""

                all_fragment_en_success = all_fragment_en_success and fragment_en_success
                if fragment_en_success:
                    fragment_en = total_energy(fragment_en_output)
                    all_fragment_en.append(fragment_en)

            if all_fragment_en_success:
                delta_ens.append(sum(all_fragment_en) - parent_en)
        if len(delta_ens):
            hydrolysis_ens.append(min(delta_ens))
        else:
            hydrolysis_ens.append(NO_REACTION)
    return (HA_TO_KCAL * (np.array(hydrolysis_ens) - ENERGY_WATER)).clip(CLAMP_LOW,CLAMP_HIGH)