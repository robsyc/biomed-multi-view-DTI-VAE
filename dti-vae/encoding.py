from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.utils.smiles import from_smiles, to_smiles, x_map, e_map

def smiles2pydata(smiles, with_hydrogen=False, give_description=False):
    """
    Convert a SMILES string to a torch_geometric Data instance
    with categorical node and edge features.
    Also see: 
        - https://pytorch-geometric.readthedocs.io/en/2.4.0/_modules/torch_geometric/utils/smiles.html
    """
    pydata = from_smiles(smiles, with_hydrogen=with_hydrogen)
    if give_description:
        m = Chem.MolFromSmiles(smiles)
        if with_hydrogen:
            m = Chem.AddHs(m)
        img = Draw.MolToImage(m)
        print("Number of atoms: ", m.GetNumAtoms())
        print("Number of bonds: ", m.GetNumBonds())
        display(img)
        print("Converting SMILES to torch_geometric Data instance for: ", smiles)
        print("Number of nodes: ", pydata.num_nodes)
        print("Node features: ", pydata.x.shape, '\n- example:', pydata.x[0], '\n- out of classes: ', list(map(len, [key for key in x_map.values()])))
        print("\nNumber of edges: ", pydata.num_edges)
        print("Edges have to be represented undirected as a COO tensor")
        print("Edge features: ", pydata.edge_attr.shape, '\n- example:', pydata.edge_attr[0], '\n- out of classes: ', list(map(len, [key for key in e_map.values()])))
    return pydata