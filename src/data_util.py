"""Module for manipulating the data"""

from omrdatasettools import Downloader, OmrDataset
from glob import glob
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np
import os
from mung.io import read_nodes_from_file
from mung.graph import NotationGraph
from mung.node import Node, bounding_box_intersection
from tqdm import tqdm
import yaml


def get_muscima_pp_v2():
    """ Downloads the complete MUSCIMA++ dataset and saves it in "/data/MUSCIMA++"
    """
    downloader = Downloader()
    downloader.download_and_extract_dataset(OmrDataset.MuscimaPlusPlus_V2, "data/MUSCIMA++")


# Load train/validation/test split from yaml file
def load_split(split_file):
    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl, Loader=yaml.BaseLoader)
    return split


# source: https://github.com/OMR-Research/MungLinker
def __load_mung(filename: str, exclude_classes: List[str]) -> NotationGraph:
    mungos = read_nodes_from_file(filename)
    mung = NotationGraph(mungos)
    objects_to_exclude = [m for m in mungos if m.class_name in exclude_classes]
    for m in objects_to_exclude:
        mung.remove_vertex(m.id)
    return mung


# source: https://github.com/OMR-Research/MungLinker
def __load_image(filename: str) -> np.ndarray:
    image = np.array(Image.open(filename).convert('1')).astype('uint8')
    return image


# source: https://github.com/OMR-Research/MungLinker
def load_mungs_images(mung_root: str, images_root: str,
                      include_names: List[str] = None,
                      max_items: int = None,
                      exclude_classes=None,
                      masks_to_bounding_boxes=False):
    """Loads the MuNGs and corresponding images from the given folders.
    All *.xml files in ``mung_root`` are considered MuNG files, all *.png
    files in ``images_root`` are considered image files.

    Use this to get data for initializing the PairwiseMungoDataPool.

    :param mung_root: Directory containing MuNG XML files.

    :param images_root: Directory containing underlying image files (png).

    :param include_names: Only load files such that their basename is in
        this list. Useful for loading train/test/validate splits.

    :param max_items: Load at most this many files.

    :param exclude_classes: When loading the MuNG, exclude notation objects
        that are labeled as one of these classes. (Most useful for excluding
        staff objects.)

    :param masks_to_bounding_boxes: If set, will replace the masks of the
        loaded MuNGOs with everything in the corresponding bounding box
        of the image. This is to make the training data compatible with
        the runtime outputs of RCNN-based detectors, which only output
        the bounding box, not the mask.

    :returns: mungs, images  -- a tuple of lists.
    """
    if exclude_classes is None:
        exclude_classes = {}

    all_mung_files = glob(mung_root + "/**/*.xml", recursive=True)
    mung_files_in_this_split = sorted([f for f in all_mung_files if os.path.splitext(os.path.basename(f))[0] in include_names])

    all_image_files = glob(images_root + "/**/*.png", recursive=True)
    image_files_in_this_split = sorted([f for f in all_image_files if
                                 os.path.splitext(os.path.basename(f))[0] in include_names])

    mungs = []
    images = []
    for mung_file, image_file in tqdm(zip(mung_files_in_this_split, image_files_in_this_split),
                                      desc="Loading mung/image pairs from disk",
                                      total=len(mung_files_in_this_split)):
        mung = __load_mung(mung_file, exclude_classes)
        mungs.append(mung)

        image = __load_image(image_file)
        images.append(image)

        # This is for training on bounding boxes,
        # which needs to be done in order to then process
        # R-CNN detection outputs with Munglinker trained on ground truth
        if masks_to_bounding_boxes:
            for mungo in mung.vertices:
                t, l, b, r = mungo.bounding_box
                image_mask = image[t:b, l:r]
                mungo.set_mask(image_mask)

        if max_items is not None:
            if len(mungs) >= max_items:
                break

    return mungs, images
