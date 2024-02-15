from xtbf import *
from xtbf.shortcuts import hydrolysis_atom_vector
from xtbf.parsers import total_energy
from smal.all import all_hydrolysis_candidates
from typing import List, Tuple

import pytest
if pytest.__version__ < "3.0.0":
  pytest.skip()
else:
  pytestmark = pytest.mark.skip

try:
    import deepchem as dc
    from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
    from deepchem.feat.base_classes import MolecularFeaturizer
    from deepchem.feat.graph_data import GraphData
    from deepchem.utils.molecule_feature_utils import one_hot_encode
    from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
    from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
    from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
    from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
    from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
    from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot
    from deepchem.utils.molecule_feature_utils import get_atom_chirality_one_hot
    from deepchem.utils.molecule_feature_utils import get_atom_formal_charge
    from deepchem.utils.molecule_feature_utils import get_atom_partial_charge
    from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
    from deepchem.utils.molecule_feature_utils import get_bond_type_one_hot
    from deepchem.utils.molecule_feature_utils import get_bond_is_in_same_ring_one_hot
    from deepchem.utils.molecule_feature_utils import get_bond_is_conjugated_one_hot
    from deepchem.utils.molecule_feature_utils import get_bond_stereo_one_hot
    #from deepchem.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
    #from deepchem.utils.molecule_feature_utils import get_atom_implicit_valence_one_hot
    #from deepchem.utils.molecule_feature_utils import get_atom_explicit_valence_one_hot
    #from deepchem.utils.rdkit_utils import compute_all_pairs_shortest_path
    #from deepchem.utils.rdkit_utils import compute_pairwise_ring_info
except:
    print("could not import deepchem")

from xtbf.shortcuts import add_xtb_charges_else_gasteiger_charges


from rdkit.Chem.MolStandardize.standardize import Standardizer
standardizer = Standardizer(max_tautomers=10)
include_stereoinfo = False


def standardize_mol(mol):
    """
    Standardizer from MELLODDY consortium
    (copied from mlr-phospholipidosis project)
    :param mol: rdkit molecule object
    :return: cleaned rdkit molecule object
    """
    if mol is None:
        return mol
    try:
        mol = standardizer.charge_parent(mol)
        mol = standardizer.isotope_parent(mol)
        if include_stereoinfo is False:
            mol = standardizer.stereo_parent(mol)
        mol = standardizer.tautomer_parent(mol)
        mol = standardizer.standardize(mol)
        return mol
    except:
        return None

def _construct_atom_feature_xtb_charge(atom: "RDKitAtom", h_bond_infos: List[Tuple[int,
                                                                      str]],
                            use_chirality: bool, force_gasteiger_charge=False,
                            ) -> np.ndarray:

    atom_type = get_atom_type_one_hot(atom)
    formal_charge = get_atom_formal_charge(atom)
    hybridization = get_atom_hybridization_one_hot(atom)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = get_atom_is_in_aromatic_one_hot(atom)
    degree = get_atom_total_degree_one_hot(atom)
    total_num_Hs = get_atom_total_num_Hs_one_hot(atom)

    if force_gasteiger_charge:
        partial_charge = atom.GetProp('_GasteigerCharge') 
    else:
        partial_charge = atom.GetDoubleProp('xtb_partial_charge')
    if str(partial_charge).lower() in ['nan','none']:
        print("warning: could not compute atom charge!")
        partial_charge = 0 # TODO: best nan imputation?

    atom_feat = np.concatenate([
        atom_type, formal_charge, hybridization, acceptor_donor, aromatic,
        degree, total_num_Hs, [partial_charge],
    ])

    if use_chirality:
        chirality = get_atom_chirality_one_hot(atom)
        atom_feat = np.concatenate([atom_feat, np.array(chirality)])

    return atom_feat


def _construct_bond_feature(bond: "RDKitBond") -> np.ndarray:
    """Construct a bond feature from a RDKit bond object.

    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
        RDKit bond object

    Returns
    -------
    np.ndarray
        A one-hot vector of the bond feature.

    """
    bond_type = get_bond_type_one_hot(bond)
    same_ring = get_bond_is_in_same_ring_one_hot(bond)
    conjugated = get_bond_is_conjugated_one_hot(bond)
    stereo = get_bond_stereo_one_hot(bond)
    return np.concatenate([bond_type, same_ring, conjugated, stereo])

class XTBChargeMolGCFeaturizer(MolecularFeaturizer):
    def __init__(self,
                 cache:Path,
                 use_edges: bool = False,
                 use_chirality: bool = False,
                 ):
        self.use_edges = use_edges
        self.use_chirality = use_chirality
        self.use_partial_charge = False
        self.cache = cache

    def featurize_in_parallel(self, mols, n_jobs:int) -> "np.ndarray[GraphData]":
        mols = [Chem.MolFromSmiles(mol) if isinstance(mol,str) else mol for mol in mols]
        return np.asarray(joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(self._featurize)(mol) for mol in mols))

    def _featurize(self, datapoint,  **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.

        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
            RDKit mol object.

        Returns
        -------
        graph: GraphData
            A molecule graph with some features.

        """
        assert datapoint.GetNumAtoms(
        ) > 1, "More than one atom should be present in the molecule for this featurizer to work."
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        add_xtb_charges_else_gasteiger_charges(datapoint,cache=self.cache)

        if self.use_partial_charge:
            try:
                datapoint.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
            except:
                # If partial charges were not computed
                try:
                    from rdkit.Chem import AllChem
                    AllChem.ComputeGasteigerCharges(datapoint)
                except ModuleNotFoundError:
                    raise ImportError(
                        "This class requires RDKit to be installed.")

        # construct atom (node) feature
        h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        atom_features = np.asarray(
            [
                _construct_atom_feature_xtb_charge(atom, h_bond_infos, self.use_chirality,)
                for atom in datapoint.GetAtoms()
            ],
            dtype=float,
        )


        # construct edge (bond) index
        src, dest = [], []
        for bond in datapoint.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = None  # deafult None
        if self.use_edges:
            features = []
            for bond in datapoint.GetBonds():
                features += 2 * [_construct_bond_feature(bond)]
            bond_features = np.asarray(features, dtype=float)

        # load_sdf_files returns pos as strings but user can also specify
        # numpy arrays for atom coordinates
        pos = []
        if 'pos_x' in kwargs and 'pos_y' in kwargs and 'pos_z' in kwargs:
            if isinstance(kwargs['pos_x'], str): 
                pos_x = eval(kwargs['pos_x']) # nosec
            elif isinstance(kwargs['pos_x'], np.ndarray): 
                pos_x = kwargs['pos_x'] 
            if isinstance(kwargs['pos_y'], str): 
                pos_y = eval(kwargs['pos_y']) # nosec
            elif isinstance(kwargs['pos_y'], np.ndarray): 
                pos_y = kwargs['pos_y'] 
            if isinstance(kwargs['pos_z'], str):
                pos_z = eval(kwargs['pos_z']) # nosec
            elif isinstance(kwargs['pos_z'], np.ndarray):
                pos_z = kwargs['pos_z']

            for x, y, z in zip(pos_x, pos_y, pos_z):
                pos.append([x, y, z])
            node_pos_features = np.asarray(pos)
        else:
            node_pos_features = None
        return GraphData(node_features=atom_features,
                         edge_index=np.asarray([src, dest], dtype=int),
                         edge_features=bond_features,
                         node_pos_features=node_pos_features)

class HydrolysisMolGCFeaturizer(MolecularFeaturizer):
    def __init__(self,
                 cache:Path,
                 use_edges: bool = False,
                 use_chirality: bool = False,
                 ):
        self.use_edges = use_edges
        self.use_chirality = use_chirality
        self.use_partial_charge = True
        self.cache = cache

    def featurize_in_parallel(self, mols, n_jobs:int) -> "np.ndarray[GraphData]":
        mols = [Chem.MolFromSmiles(mol) if isinstance(mol,str) else mol for mol in mols]
        return np.asarray(joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(self._featurize)(mol) for mol in mols))

    def _featurize(self, datapoint, **kwargs) -> GraphData:
        try:
            return self._featurize_unsafe(datapoint=datapoint,**kwargs)
        except:
            print(f"warning: featurization of the smiles '{Chem.MolToSmiles(datapoint)}' failed. filling with garbage values!")
            SENTINEL = Chem.MolFromSmiles("CCC")
            return self._featurize_unsafe(datapoint=SENTINEL,**kwargs)

    def _featurize_unsafe(self, datapoint,  **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.

        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
            RDKit mol object.

        Returns
        -------
        graph: GraphData
            A molecule graph with some features.

        """
        datapoint = standardize_mol(datapoint)
        if datapoint.GetNumAtoms(
        ) < 1:
            raise ValueError("More than one atom should be present in the molecule for this featurizer to work.")

        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        add_xtb_charges_else_gasteiger_charges(datapoint,cache=self.cache)

        if self.use_partial_charge:
            try:
                datapoint.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
            except:
                # If partial charges were not computed
                try:
                    from rdkit.Chem import AllChem
                    AllChem.ComputeGasteigerCharges(datapoint)
                except ModuleNotFoundError:
                    raise ImportError(
                        "This class requires RDKit to be installed.")

        # construct atom (node) feature
        h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        atom_features = np.asarray(
            [
                _construct_atom_feature_xtb_charge(atom, h_bond_infos, self.use_chirality,force_gasteiger_charge=True)
                for atom in datapoint.GetAtoms()
            ],
            dtype=float,
        )
        hydrol_features = hydrolysis_atom_vector(datapoint, cache=self.cache)
        hydrol_features = hydrol_features / 10.0 # scale into [-1.0,1.0]
        atom_features = np.hstack([atom_features,hydrol_features.reshape(len(hydrol_features),1)])

        # construct edge (bond) index
        src, dest = [], []
        for bond in datapoint.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = None  # deafult None
        if self.use_edges:
            features = []
            for bond in datapoint.GetBonds():
                features += 2 * [_construct_bond_feature(bond)]
            bond_features = np.asarray(features, dtype=float)

        # load_sdf_files returns pos as strings but user can also specify
        # numpy arrays for atom coordinates
        pos = []
        if 'pos_x' in kwargs and 'pos_y' in kwargs and 'pos_z' in kwargs:
            if isinstance(kwargs['pos_x'], str):
                pos_x = eval(kwargs['pos_x']) # nosec
            elif isinstance(kwargs['pos_x'], np.ndarray):
                pos_x = kwargs['pos_x']
            if isinstance(kwargs['pos_y'], str):
                pos_y = eval(kwargs['pos_y']) # nosec
            elif isinstance(kwargs['pos_y'], np.ndarray):
                pos_y = kwargs['pos_y']
            if isinstance(kwargs['pos_z'], str):
                pos_z = eval(kwargs['pos_z']) # nosec
            elif isinstance(kwargs['pos_z'], np.ndarray):
                pos_z = kwargs['pos_z']

            for x, y, z in zip(pos_x, pos_y, pos_z):
                pos.append([x, y, z])
            node_pos_features = np.asarray(pos)
        else:
            node_pos_features = None
        return GraphData(node_features=atom_features,
                         edge_index=np.asarray([src, dest], dtype=int),
                         edge_features=bond_features,
                         node_pos_features=node_pos_features)
