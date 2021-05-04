"""Module for loading MUSCIMA++ data"""

from glob import glob
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np
import os
from mung.io import read_nodes_from_file
from mung.graph import NotationGraph
from mung.node import Node, bounding_box_intersection
from tqdm import tqdm


def load_all_muscima_annotations(annotations_directory) -> Dict[str, List[Node]]:
    """
    :param annotations_directory:
    :return: Returns a dictionary of annotations with the filename as key
    """
    all_xml_files = [y for x in os.walk(annotations_directory) for y in glob(os.path.join(x[0], '*.xml'))]
    node_annotations = {}
    for xml_file in tqdm(all_xml_files, desc='Parsing annotation files'):
        nodes = read_nodes_from_file(xml_file)
        doc = nodes[0].document
        node_annotations[doc] = nodes
    return node_annotations


def load_split(split_file_path: str) -> np.ndarray:
    """ Load train/validation/test split from file

    Parameters
    ----------
    split_file_path
        path to the file containing which files to include in the split

    Returns
    -------
    numpy.ndarray
        contents of the split-file, line by line

    """
    split = np.genfromtxt(split_file_path, dtype=str, delimiter="\n")
    return split


# source: https://github.com/OMR-Research/MungLinker
def __load_mung(filename: str, exclude_classes: List[str]) -> NotationGraph:
    mungos = read_nodes_from_file(filename)
    mung = NotationGraph(mungos)
    objects_to_exclude = [m for m in mungos if m.class_name in exclude_classes]
    for m in objects_to_exclude:
        mung.remove_vertex(m.id)
    if len(mung.vertices) == 0:
        print("filename", filename)
        exit()
    return mung


# source: https://github.com/OMR-Research/MungLinker
def __load_image(filename: str) -> np.ndarray:
    image = np.array(Image.open(filename).convert('1')).astype('uint8')
    return image


# source: https://github.com/OMR-Research/MungLinker
def load_mungs_images(mung_root: str, images_root: str,
                      include_names: np.ndarray = None,
                      max_items: int = None,
                      exclude_classes=None,
                      masks_to_bounding_boxes=False):
    """ Loads the MuNGs and corresponding images from the given folders.
    All *.xml files in ``mung_root`` are considered MuNG files, all *.png
    files in ``images_root`` are considered image files.

    Use this to get data for initializing the PairwiseMungoDataPool.


    Parameters
    ----------
    mung_root
        Directory containing MuNG XML files.

    images_root
        Directory containing underlying image files (png).

    include_names
        Only load files such that their basename is in
        this list. Useful for loading train/test/validate splits.

    max_items
        Load at most this many files.

    exclude_classes
        When loading the MuNG, exclude notation objects
        that are labeled as one of these classes. (Most useful for excluding
        staff objects.)

    masks_to_bounding_boxes
        If set, will replace the masks of the
        loaded MuNGOs with everything in the corresponding bounding box
        of the image. This is to make the training data compatible with
        the runtime outputs of RCNN-based detectors, which only output
        the bounding box, not the mask.

    Returns
    -------
        mungs, images  -- a tuple of lists.
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
