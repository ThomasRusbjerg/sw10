"""Given MUSCIMA++ v2.0 images and annotations, draws bounding boxes and
   class name labels for each object in the image"""

import os
from glob import glob

import cv2
import argparse
import json

from PIL import ImageColor
from mung.io import read_nodes_from_file
from tqdm import tqdm

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
                                   classes_mapping : dict):
    music_objects = read_nodes_from_file(ground_truth_annotations_path)
    img = cv2.imread(image_path)

    for index, music_object in enumerate(music_objects):
        # String to float, float to int
        x1 = music_object.left
        y1 = music_object.top
        x2 = music_object.right
        y2 = music_object.bottom

        color_name = STANDARD_COLORS[classes_mapping[music_object.class_name] % len(STANDARD_COLORS)]
        color = ImageColor.getrgb(color_name)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img=img,
                    text=music_object.class_name + '/' + str(index + 1),
                    org=(x1, y1), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=color, thickness=1)
    cv2.imwrite(destination_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw the bounding boxes from the ground-truth data.')
    parser.add_argument('--image', dest='image', type=str, help='Path to the image (file or directory).',
                        default="data/MUSCIMA++/v2.0/data/images")
    parser.add_argument('--annotations', dest='annotations', type=str,
                        help='Path to the annotations (file or directory).',
                        default="data/MUSCIMA++/v2.0/data/annotations")
    parser.add_argument('--save_directory', dest='save_directory', type=str,
                        help='Directory, where to save the processed image.',
                        default="data/drawn_bboxes")
    parser.add_argument('--label_map', dest='label_map', type=str,
                        default="data/MUSCIMA++/v2.0/mapping_all_classes.json",
                        help='Path to the label map, which is json-file that'
                             'maps each category name to a unique number.')
    args = parser.parse_args()

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
                                       classes_mapping)
