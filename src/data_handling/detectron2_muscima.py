import json
import numpy as np
from data_handling import muscima_loader
from pycocotools import mask
from skimage import measure
from detectron2.structures import BoxMode

def muscima_detectron_dataset(split_location):
    images_root = "data/MUSCIMA++/v2.0/data/images"
    mung_root = "data/MUSCIMA++/v2.0/data/annotations"
    
    # Get class id mapping
    class_info_path = "data/MUSCIMA++/v2.0/mapping_all_classes.json"
    classes = {}
    with open(class_info_path) as json_file:
        data = json.load(json_file)
        for c in data:
            classes[c["name"]] = c["id"]

    # Load filenames for split
    split = muscima_loader.load_split(split_location)

    # Load the (full) images and corresponding annotations
    mungs, images = muscima_loader.load_mungs_images(
        images_root=images_root,
        mung_root=mung_root,
        include_names=split)

    # Convert mungs to detectron2 format
    dataset = []
    for i, mung in enumerate(mungs):
        img_id = mung.vertices[0].document
        height = images[i].shape[0]
        width = images[i].shape[1]
        img_instance = {
            "file_name": images_root + img_id,
            "height": height,
            "width": width,
            "image_id": img_id,
            "annotations": []
        }

        for annotation in mung.vertices:
            # Convert bbox mask to image mask
            mask_projected = annotation.project_on(images[i])
            # Based on https://github.com/cocodataset/cocoapi/issues/131
            fortran_binary_mask = np.asfortranarray(mask_projected)
            encoded_mask = mask.encode(fortran_binary_mask)
            # area = mask.area(encoded_mask)
            bounding_box = mask.toBbox(encoded_mask)
            contours = measure.find_contours(mask_projected, 0.5)
            segmentations = []
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                segmentations.append(segmentation)
            img_instance["annotations"].append(
                {
                    "bbox": bounding_box,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": classes[annotation.class_name],
                    "segmentation": segmentations,
                }
            )
        
        dataset.append(img_instance)

    return dataset

