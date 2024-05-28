"""
Tools for the visualization of chemical spaces.
"""

from pathlib import Path
import shutil
from smal.io import random_fle
from rdkit.Chem import Draw
from pathlib import Path
from smal.cluster import sha1_hash
from smal.config import _DIR_DOCS
from smal.io import to_smi

from pathlib import Path
from rdkit import rdBase
from rdkit import Chem

from smal.io import mol_to_image

rdBase.DisableLog("rdApp.*")

import PIL
from PIL import ImageDraw
from PIL.Image import Image

# from .helper import *

# from .config import ConfigDict, SecretDict

import math
import random
from typing import List
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image
from collections import defaultdict

def highlight_mol(mol,atom_colors:list=None,atom_radii:list=None,bond_colors:list=None,width=350,height=400) -> str:

    if atom_radii is None:
        atom_radii = [0.3 for _ in mol.GetAtoms()]
    if atom_colors is None:
        atom_colors = [(0.,0.,0.,0.) for _ in mol.GetAtoms()]
    if bond_colors is None:
        bond_colors = [(0.,0.,0.,0.) for _ in mol.GetBonds()]

    assert len(atom_colors) == mol.GetNumAtoms()
    assert len(bond_colors) == mol.GetNumBonds()
    assert all(len(col) == 4 for col in atom_colors+bond_colors)

    athighlights = defaultdict(list)
    arads = {}
    for a in mol.GetAtoms():
        aid = a.GetIdx()
        athighlights[aid].append(atom_colors[aid])
        arads[aid] = atom_radii[aid]

    bndhighlights = defaultdict(list)
    for bond in mol.GetBonds():
        aid1 = bond.GetBeginAtomIdx()
        aid2 = bond.GetEndAtomIdx()
        bid = mol.GetBondBetweenAtoms(aid1,aid2).GetIdx()
        bndhighlights[bid].append(bond_colors[bid])

    d2d = rdMolDraw2D.MolDraw2DCairo(width,height)
    d2d.DrawMoleculeWithHighlights(mol,"",dict(athighlights),dict(bndhighlights),arads,{})
    d2d.FinishDrawing()
    bio = io.BytesIO(d2d.GetDrawingText())
    return Image.open(bio)


def mol_to_svg(mol, save_as=None, width=300, height=300) -> Path:
    fle = random_fle("svg")
    fle.touch()
    Draw.MolToFile(
        mol=mol,
        filename=str(fle),
        size=(width, height),
        imageType="svg",
    )
    if save_as:
        fle = shutil.move(fle, save_as)
    return fle


def show_molecule_markup(mol, width=300, height=300) -> str:
    fle = _DIR_DOCS / "images" / sha1_hash(to_smi(mol))
    fle = fle.with_suffix(".svg")
    mol_to_svg(mol, width=width, height=height).rename(fle)
    return f"![image]({fle.resolve()})"


def tile_images_with_annots(
    imgs: List,
    annots: List[str],
    tile_w: int,
    tile_h: int,
    n_tiles_w: int,
    n_tiles_h: int,
    sampling_strategy="random",
    with_replacement=False,
    annot_color=(0, 0, 0),
) -> Image:
    assert sampling_strategy in ["random", "deterministic"]
    imgs_left = list(range(len(imgs)))
    w_total = n_tiles_w * tile_w
    h_total = n_tiles_h * tile_h
    tile_image = PIL.Image.new("RGB", (w_total, h_total))
    draw = ImageDraw.Draw(tile_image)
    x = 0
    y = 0
    while y < h_total:
        while x < w_total:
            if sampling_strategy == "random" and imgs_left:
                if sampling_strategy == "random":
                    idx = random.choice(imgs_left)
                img = imgs[idx]

            elif sampling_strategy == "deterministic":
                ix = x / tile_w
                iy = y / tile_h
                idx = int(round(iy * n_tiles_w + ix % n_tiles_w))

                if idx < len(imgs_left):
                    img = imgs[idx]
                else:
                    img = PIL.Image.new(
                        "RGB", (16, 16), color="white"
                    )  # will be resized anyways
            else:
                img = PIL.Image.new(
                    "RGB", (16, 16), color="white"
                )  # will be resized anyways
            tile_image.paste(img.resize((tile_w, tile_h)), (x, y))

            if sampling_strategy == "deterministic":
                if idx < len(annots):
                    annot = annots[idx]
                    draw.text((x, y), annot, annot_color)

            if imgs_left and sampling_strategy == "random":
                if idx < len(annots):
                    annot = annots[idx]
                else:
                    annot = ""
                draw.text((x, y), annot, annot_color)

                if not with_replacement:
                    imgs_left = [i for i in imgs_left if i != idx]

            x += tile_w
        x = 0
        y += tile_h

    return tile_image


def show_transformation(
    before: "list[str]",
    after: "list[str]",
    width: int,
    height: int,
    n_total: int = 128,
    n_batch: int = 16,
    annotations: "list[str]" = None,
    show=False,
):
    """
    Illustrates the transformation from before -> after by
    displaying a corresponding image collage.
    """
    assert len(after) == len(before)
    n_total = min(n_total, len(before))
    if annotations is None:
        annotations = ["before", "after"]

    tile_imgs = []
    for start in range(0, n_total, n_batch):
        end = start + n_batch
        imgs, annots = [], []
        for smi_before, smi_after in zip(before[start:end], after[start:end]):
            mol_before = Chem.MolFromSmiles(smi_before)
            mol_after = Chem.MolFromSmiles(smi_after)
            imgs.append(mol_to_image(mol_before, width=width, height=height))
            imgs.append(mol_to_image(mol_after, width=width, height=height))
            annots += annotations
        tile_img = tile_images_with_annots(
            imgs,
            annots,
            tile_w=width,
            tile_h=height,
            n_tiles_w=2,
            n_tiles_h=math.ceil(n_batch / 2),
            sampling_strategy="deterministic",
        )

        if show:
            tile_img.show()

        tile_imgs.append(tile_img)

    return tile_imgs
