# Source: https://github.com/apacha/MusicObjectDetector-TF/blob/master/MusicObjectDetector/draw_bounding_boxes.py

"""Given MUSCIMA++ v2.0 images and annotations, draws bounding boxes and
   class name labels for each object in the image"""

import os
from glob import glob

import cv2
import matplotlib as mpl
import argparse

from PIL import ImageColor
from mung.io import read_nodes_from_file
from tqdm import tqdm
from typing import List

from src.data_handling.detectron2_muscima import get_muscima_classid_mapping

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_bounding_boxes_into_image(image_path: str,
                                   ground_truth_annotations_path: str,
                                   destination_path: str,
                                   classes_mapping: dict,
                                   classes: List[str],
                                   draw_links: bool):
    music_objects = read_nodes_from_file(ground_truth_annotations_path)
    img = cv2.imread(image_path)
    ignore_classes = [
        "staffLine",
        "staffSpace"
    ]
    for index, music_object in enumerate(music_objects):
        if music_object.class_name in ignore_classes:
            continue
        # If classes to draw are specified
        if classes:
            if music_object.class_name not in classes:
                continue
        # String to float, float to int
        x1 = music_object.left
        y1 = music_object.top
        x2 = music_object.right
        y2 = music_object.bottom

        color_name = STANDARD_COLORS[classes_mapping[music_object.class_name] % len(STANDARD_COLORS)]
        color = ImageColor.getrgb(color_name)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # cv2.putText(img=img,
        #             text=music_object.class_name + '/' + str(index + 1),
        #             org=(x1, y1), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
        #             color=color, thickness=1)
        if draw_links:
            # Draw links to connected nodes
            connected_nodes = list(filter(lambda node: node.id in music_object.outlinks, music_objects))
            for node in connected_nodes:
                cv2.line(img, tuple(reversed(music_object.middle)), tuple(reversed(node.middle)), color=color, thickness=2)
    cv2.imwrite(destination_path, cv2.bitwise_not(img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw the bounding boxes from the ground-truth data.')
    parser.add_argument('--images', dest='image', type=str, help='Path to the image (file or directory).',
                        default="data/MUSCIMA++/v2.0/data/images")
    parser.add_argument('--annotations', dest='annotations', type=str,
                        help='Path to the annotations (file or directory).',
                        default="data/MUSCIMA++/v2.0/data/annotations")
    # Defaults to args.image/with_bboxes
    parser.add_argument('--save_directory', dest='save_directory', type=str,
                        help='Directory, where to save the processed image.')
    parser.add_argument('--label_map', dest='label_map', type=str,
                        default="data/MUSCIMA++/v2.0/mapping_all_classes.json",
                        help='Path to the label map, which is json-file that'
                             'maps each category name to a unique number.')
    parser.add_argument('-c', '--classes', dest='classes', action='append', default=[],
                        help="The classes to draw bboxes for."
                             "For each class, supply to new -c argument,"
                             "e.g. \"-c noteheadFull -c barline\"")
    parser.add_argument('-l', '--links', dest='draw_links', action='store_true', default=False,
                        help="Whether or not to draw links between connected nodes")

    args = parser.parse_args()

    if args.save_directory is None:
        args.save_directory = os.path.join(args.image, "with_bboxes")

    # Create a dict of the classes with key-value pair: name, id
    classes_mapping = get_muscima_classid_mapping()

    image_file_names = []
    if os.path.isfile(args.image):
        image_file_names.append(args.image)
    elif os.path.isdir(args.image):
        image_file_names.extend(glob(args.image + "/*.png"))

    annotation_file_names = []
    if os.path.isfile(args.annotations):
        annotation_file_names.append(args.annotations)
    elif os.path.isdir(args.annotations):
        annotation_file_names.extend(glob(args.annotations + "/*.xml"))

    # Make images and annotations follow the same order
    image_file_names.sort()
    annotation_file_names.sort()

    output_files = [os.path.join(args.save_directory, os.path.basename(f)) for
                    f in image_file_names]
    os.makedirs(args.save_directory, exist_ok=True)
    print(f"Saving all images with bounding boxes in {args.save_directory}")
    for image, annotation, output in tqdm(
            zip(image_file_names, annotation_file_names, output_files),
            total=len(output_files),
            desc="Drawing annotations"):
        draw_bounding_boxes_into_image(image, annotation, output,
                                       classes_mapping, args.classes, args.draw_links)
