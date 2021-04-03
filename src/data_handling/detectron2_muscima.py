import json
import pickle
import numpy as np
from data_handling import muscima_loader
from pycocotools import mask
from skimage import measure
from detectron2.structures import BoxMode
from tqdm import tqdm

def load_muscima_detectron_dataset(split_location):
    file = open(split_location, 'rb')
    data = pickle.load(file)
    file.close()
    return data


# https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


# https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def get_muscima_classid_mapping():
    class_info_path = "data/MUSCIMA++/v2.0/mapping_all_classes.json"
    classes = {}
    with open(class_info_path) as json_file:
        data = json.load(json_file)
        for c in data:
            classes[c["name"]] = c["id"]
    return classes

def create_muscima_detectron_dataset(split_location):
    images_root = "data/MUSCIMA++/v2.0/data/images"
    mung_root = "data/MUSCIMA++/v2.0/data/annotations"

    # Get class id mapping
    classes = get_muscima_classid_mapping()

    # Load filenames for split
    split = muscima_loader.load_split(split_location)

    # Load the (full) images and corresponding annotations
    mungs, images = muscima_loader.load_mungs_images(
        images_root=images_root, mung_root=mung_root, include_names=split
    )

    # Convert mungs to detectron2 format
    dataset = []
    for i, mung in enumerate(tqdm(mungs, desc="Converting Mung to Detectron/Coco format")):
        img_id = mung.vertices[0].document
        height = images[i].shape[0]
        width = images[i].shape[1]
        img_instance = {
            "file_name": images_root + "/" + img_id + ".png",
            "height": height,
            "width": width,
            "image_id": img_id,
            "annotations": [],
        }

        for annotation in mung.vertices:
            # Convert bbox mask to image mask
            mask_projected = annotation.project_on(images[i])
            # Based on https://github.com/cocodataset/cocoapi/issues/131
            fortran_binary_mask = np.asfortranarray(mask_projected)
            encoded_mask = mask.encode(fortran_binary_mask)
            # area = mask.area(encoded_mask)
            # bounding_box = mask.toBbox(encoded_mask).astype(int)
            # bounding_box = annotation["bounding_box"]
            bounding_box = [annotation.left, annotation.top, annotation.right, annotation.bottom]
            segmentations = binary_mask_to_polygon(mask_projected, tolerance=2)
            img_instance["annotations"].append(
                {
                    "bbox": bounding_box,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": classes[annotation.class_name],
                    "segmentation": segmentations,
                }
            )
        dataset.append(img_instance)

    # Dump dataset to disk
    filename = split_location.split("/")[-1].split(".")[0] + ".pickle"
    with open("data/" + filename, "wb") as dumpfile:
        pickle.dump(dataset, dumpfile)
