# Source: https://github.com/apacha/MusicObjectDetector-TF

import os
import shutil
import argparse
from glob import glob

from PIL import Image
from tqdm import tqdm
from mung.io import read_nodes_from_file


from nodes_to_pascal_voc_xml import create_annotations_in_pascal_voc_format_from_nodes


def prepare_annotations(muscima_pp_dataset_directory: str,
                        exported_annotations_file_path: str,
                        annotations_path: str):
    muscima_image_directory = os.path.join(muscima_pp_dataset_directory, "v2.0", "data", "images", "*.png")
    image_paths = glob(muscima_image_directory)

    xml_annotations_directory = os.path.join(muscima_pp_dataset_directory, "v2.0", "data", "annotations")
    all_xml_files = [y for x in os.walk(xml_annotations_directory) for y in glob(os.path.join(x[0], '*.xml'))]

    if os.path.exists(exported_annotations_file_path):
        os.remove(exported_annotations_file_path)

    shutil.rmtree(annotations_path, ignore_errors=True)

    for xml_file in tqdm(all_xml_files, desc='Parsing annotation files'):
        nodes = read_nodes_from_file(xml_file)
        doc = nodes[0].document

        image_path = None
        for path in image_paths:
            if doc in path:
                image_path = path
                break

        image = Image.open(image_path, "r")  # type: Image.Image
        image_width = image.width
        image_height = image.height
        create_annotations_in_pascal_voc_format_from_nodes(annotations_path,
                                                           os.path.basename(image_path),
                                                           nodes,
                                                           image_width,
                                                           image_height,
                                                           3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/MUSCIMA++",
        help="The directory where MUSCIMA++ is located")
    parser.add_argument(
        "--destination_dir",
        type=str,
        default="data/MUSCIMA++/full_page/annotations",
        help="The directory where the Pascal VOC XML files should be placed")

    flags, unparsed = parser.parse_known_args()

    prepare_annotations(flags.data_dir,
                        "data/full_page_annotations.csv",
                        flags.destination_dir)
