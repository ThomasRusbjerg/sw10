# Source: https://github.com/apacha/MusicObjectDetector-TF/blob/master/MusicObjectDetector/prepare_muscima_stavewise_annotations.py
import os
import argparse
from glob import glob
from typing import Tuple, List, Dict

from PIL import Image
from tqdm import tqdm
from mung.node import Node
from mung.io import read_nodes_from_file

from lxml.etree import Element, SubElement, tostring


def cut_images(image_paths, annotations_dictionary, output_path: str,
               method: str):
    os.makedirs(output_path, exist_ok=True)

    for image_path in tqdm(image_paths, desc=f"Cutting images and saving to {output_path}", total=len(image_paths)):
        file_name = os.path.basename(image_path)[:-4]  # cut away the extension .png
        image = Image.open(image_path, "r")  # type: Image.Image
        image_width = image.width
        image_height = image.height
        objects_appearing_in_image: List[Node] = None

        for document, nodes in annotations_dictionary.items():
            if file_name in document:
                objects_appearing_in_image = nodes
                break

        if objects_appearing_in_image is None:
            # Image has annotated staff-lines, but does not have corresponding node annotations, so skip it
            continue

        bboxes_to_crop_to = []

        if method == "staves":
            bboxes_to_crop_to = get_staff_bboxes(objects_appearing_in_image, image_path,
                                                 image_width, image_height)
        elif method == "measures":
            bboxes_to_crop_to = get_measure_bboxes(image, objects_appearing_in_image)

        # Crop to bounding boxes and save cropped images with annotations
        for i, bbox in enumerate(bboxes_to_crop_to):

            output_file_name = f"{file_name}_{i+1}"

            nodes_appearing_in_cropped_image =\
                get_objects_in_cropped_image(bbox, objects_appearing_in_image)

            cropped_image = image.crop(bbox).convert('RGB')

            create_muscima_annotations(output_path + "annotations",
                                       output_file_name,
                                       nodes_appearing_in_cropped_image)

            image_output_path = output_path + "images/"
            os.makedirs(image_output_path, exist_ok=True)
            output_file = os.path.join(image_output_path, output_file_name + ".png")
            cropped_image.save(output_file, "png")


def get_measure_bboxes(image, objects_appearing_in_image):
    staves = [x for x in objects_appearing_in_image if x.class_name == "staff"]
    mss = [x for x in objects_appearing_in_image if x.class_name == "measureSeparator"]

    # If no staves are in this image
    if staves is None:
        return

    # Number of pixels above and below each measure to include in cropped image
    vertical_padding = 120

    # Go through each staff and crop to measure
    measure_bboxes = []
    for staff_index, staff in enumerate(staves):
        measure_begin = staff.left
        ms_in_staff = list(filter(lambda b: staff.overlaps(b), mss))
        ms_in_staff = sorted(ms_in_staff, key=lambda x: x.left)
        for measure_index, ms in enumerate(ms_in_staff):
            # If end of staff is reached, go to next staff. 20 is arbitrary
            if abs(measure_begin - staff.right) < 20:
                break
            # First and last staff crops should include top and bottom of image, respectively
            measure_bbox_ltrb = (measure_begin,
                                 0 if staff_index == 0 else staff.top - vertical_padding,
                                 ms.right,
                                 image.height if staff_index == len(staves) - 1 else staff.bottom + vertical_padding)

            measure_bboxes.append(measure_bbox_ltrb)

            # Next measure begins at the current measure separator
            measure_begin = ms.left

    return measure_bboxes


def get_staff_bboxes(objects_appearing_in_image, image_path,
                     image_width, image_height):
    staff_objects = [x for x in objects_appearing_in_image if x.class_name == "staff"]
    max_offset_before_first_and_after_last_staff = 120

    if staff_objects is None:
        # Image has no staff lines -> Report error
        print("Error: Image {0} has no annotated staff lines".format(image_path))
        return

    next_y_top = max(0, staff_objects[0].top - max_offset_before_first_and_after_last_staff)
    last_bottom = min(staff_objects[len(staff_objects) - 1].bottom + max_offset_before_first_and_after_last_staff,
                      image_height)

    staff_bboxes = []
    for staff_index in range(len(staff_objects)):
        staff = staff_objects[staff_index]
        if staff_index < len(staff_objects) - 1:
            y_bottom = staff_objects[staff_index + 1].top
        else:
            y_bottom = last_bottom
        top_offset = next_y_top
        next_y_top = staff.bottom
        left_offset = 0

        staff_bbox_ltrb = (left_offset, top_offset, image_width, y_bottom)
        staff_bboxes.append(staff_bbox_ltrb)

    return staff_bboxes


# Does not create mask annotations
# WARNING: This will create invalid Mung Graph structures
def create_muscima_annotations(output_path: str,
                               output_file_name: str,
                               nodes_appearing_in_image: List[Node]):
    os.makedirs(output_path, exist_ok=True)

    nodes = Element("Nodes",
                    dataset="MUSCIMA-pp_2.0",
                    document=output_file_name)
    for music_object in nodes_appearing_in_image:
        node = SubElement(nodes, "Node")
        identifier = SubElement(node, "Id")
        identifier.text = music_object.id.__str__()
        class_name = SubElement(node, "ClassName")
        class_name.text = music_object.class_name
        top = SubElement(node, "Top")
        top.text = music_object.top.__str__()
        left = SubElement(node, "Left")
        left.text = music_object.left.__str__()
        width = SubElement(node, "Width")
        width.text = music_object.width.__str__()
        height = SubElement(node, "Height")
        height.text = music_object.height.__str__()

        outlinks = SubElement(node, "Outlinks")
        # Only get outlinks to nodes that are in this image
        ids_of_nodes_in_this_image = [node.id for node in nodes_appearing_in_image]
        valid_outlinks = list(set(music_object.outlinks).intersection(ids_of_nodes_in_this_image))
        outlinks.text = valid_outlinks.__str__()[1:-1].replace(',', '')

    xml_file_path = os.path.join(output_path,
                                 os.path.splitext(output_file_name)[0] + ".xml")
    pretty_xml_string = tostring(nodes, pretty_print=True)

    with open(xml_file_path, "wb") as xml_file:
        xml_file.write(pretty_xml_string)


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


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def area(a):
    top, left, bottom, right = a
    return (bottom - top) * (right - left)


def get_objects_in_cropped_image(image_crop_bbox_ltrb: Tuple[int, int, int, int],
                                 all_music_objects_appearing_in_image: List[Node],
                                 intersection_over_area_threshold_for_inclusion=0.8) \
        -> List[Node]:
    x_translation_for_cropped_image = image_crop_bbox_ltrb[0]
    y_translation_for_cropped_image = image_crop_bbox_ltrb[1]

    nodes_appearing_in_cropped_image: List[Node] = []
    for music_object in all_music_objects_appearing_in_image:
        if music_object.class_name in ["staff", "staff_line", "staff_space"]:
            continue

        img_left, img_top, img_right, img_bottom = image_crop_bbox_ltrb
        intersection_over_area = intersection([img_top, img_left, img_bottom, img_right],
                                              music_object.bounding_box) / area(music_object.bounding_box)
        if intersection_over_area > intersection_over_area_threshold_for_inclusion:
            top, left, bottom, right = music_object.bounding_box
            img_width = img_right - img_left - 1
            img_height = img_bottom - img_top - 1

            translated_bounding_box = (
                max(0, top - y_translation_for_cropped_image),  # Top
                max(0, left - x_translation_for_cropped_image),  # Left
                min(img_height, bottom - y_translation_for_cropped_image),  # Bottom
                min(img_width, right - x_translation_for_cropped_image))  # Right

            nodes_appearing_in_cropped_image.append(
                Node(music_object.id,
                     music_object.class_name,
                     top=translated_bounding_box[0],
                     left=translated_bounding_box[1],
                     height=translated_bounding_box[2] - translated_bounding_box[0],
                     width=translated_bounding_box[3] - translated_bounding_box[1],
                     outlinks=music_object.outlinks))

    return nodes_appearing_in_cropped_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop the MUSCIMA++ images.')
    parser.add_argument('--images', dest='image_directory', type=str, help='Path to the images directory.',
                        default="data/MUSCIMA++/v2.0/data/images")
    parser.add_argument('--annotations', dest='annotation_directory', type=str,
                        help='Path to the annotations directory.',
                        default="data/MUSCIMA++/v2.0/data/annotations")
    # Defaults to parent directory of images
    parser.add_argument('--save_directory', dest='save_directory', type=str,
                        help='Directory, where to save the cropped images and annotations.')
    parser.add_argument('--method', dest='method', type=str,
                        help='Crop to staves or measures.',
                        default="staves",
                        choices=["staves", "measures"])

    args = parser.parse_args()

    # Save cropped images in the parent folder of the images
    if args.save_directory is None:
        args.save_directory = os.path.dirname(args.image_directory) + "/" + args.method + "/"

    muscima_image_directory = os.path.join(args.image_directory, "*.png")
    image_paths = glob(muscima_image_directory)

    annotations_dictionary = load_all_muscima_annotations(args.annotation_directory)

    cut_images(image_paths, annotations_dictionary, args.save_directory, args.method)
