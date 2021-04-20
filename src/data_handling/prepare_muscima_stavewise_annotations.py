# Source: https://github.com/apacha/MusicObjectDetector-TF/blob/master/MusicObjectDetector/prepare_muscima_stavewise_annotations.py
import os
import shutil
from glob import glob
from typing import Tuple, List, Dict

from PIL import Image
from tqdm import tqdm
from mung.node import Node
from mung.io import read_nodes_from_file

from lxml.etree import Element, SubElement, tostring


def cut_images(muscima_pp_dataset_directory: str, output_path: str,
               exported_annotations_file_path: str, annotations_path: str):
    muscima_image_directory = os.path.join(muscima_pp_dataset_directory, "v2.0", "data", "images", "*.png")
    os.makedirs(output_path, exist_ok=True)
    if os.path.exists(exported_annotations_file_path):
        os.remove(exported_annotations_file_path)
    shutil.rmtree(annotations_path, ignore_errors=True)

    annotations_dictionary = load_all_muscima_annotations(muscima_pp_dataset_directory)
    crop_annotations = []

    image_paths = glob(muscima_image_directory)
    for image_path in tqdm(image_paths, desc="Cutting images"):
        image_name = os.path.basename(image_path)[:-4]  # cut away the extension .png
        image = Image.open(image_path, "r")  # type: Image.Image
        image_width = image.width
        image_height = image.height
        objects_appearing_in_image: List[Node] = None

        for document, nodes in annotations_dictionary.items():
            if image_name in document:
                objects_appearing_in_image = nodes
                break

        if objects_appearing_in_image is None:
            # Image has annotated staff-lines, but does not have corresponding node annotations, so skip it
            continue

        staff_objects = [x for x in objects_appearing_in_image if x.class_name == "staff"]
        max_offset_before_first_and_after_last_staff = 120

        if staff_objects is None:
            # Image has no staff lines -> Report error
            print("Error: Image {0} has no annotated staff lines".format(image_path))
            continue

        next_y_top = max(0, staff_objects[0].top - max_offset_before_first_and_after_last_staff)
        last_bottom = min(staff_objects[len(staff_objects) - 1].bottom + max_offset_before_first_and_after_last_staff,
                          image_height)

        output_image_counter = 1
        for staff_index in range(len(staff_objects)):
            staff = staff_objects[staff_index]
            if staff_index < len(staff_objects) - 1:
                y_bottom = staff_objects[staff_index + 1].top
            else:
                y_bottom = last_bottom
            top_offset = next_y_top
            next_y_top = staff.bottom
            left_offset = 0

            image_crop_bounding_box_left_top_bottom_right = (left_offset, top_offset, image_width, y_bottom)
            image_crop_bounding_box_top_left_bottom_right = (top_offset, left_offset, y_bottom, image_width)

            file_name = "{0}_{1}.png".format(image_name, output_image_counter)
            output_image_counter += 1

            objects_appearing_in_cropped_image, nodes_appearing_in_cropped_image = \
                compute_objects_appearing_in_cropped_image(file_name,
                                                           image_crop_bounding_box_top_left_bottom_right,
                                                           objects_appearing_in_image)

            cropped_image = image.crop(image_crop_bounding_box_left_top_bottom_right).convert('RGB')

            for object_appearing_in_cropped_image in objects_appearing_in_cropped_image:
                file_name = object_appearing_in_cropped_image[0]
                class_name = object_appearing_in_cropped_image[1]
                translated_bounding_box = object_appearing_in_cropped_image[2]
                trans_top, trans_left, trans_bottom, trans_right = translated_bounding_box
                crop_annotations.append([file_name, trans_left, trans_top, trans_right, trans_bottom, class_name])

            create_muscima_annotations(annotations_path,
                                       file_name,
                                       nodes_appearing_in_cropped_image)

            output_file = os.path.join(output_path, file_name)
            cropped_image.save(output_file, "png")


# Does not create mask and outlink annotations
def create_muscima_annotations(annotations_folder: str,
                               file_name: str,
                               nodes_appearing_in_image: List[Node]):
    os.makedirs(annotations_folder, exist_ok=True)

    file_name = os.path.basename(file_name)[:-4]  # cut away the extension .png

    nodes = Element("Nodes",
                    dataset="MUSCIMA-pp_2.0",
                    document=file_name)
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

        # Outlinks will not be correct because of cropping

        # outlinks = SubElement(node, "Outlinks")
        # outlinks.text = music_object.outlinks.__str__()[1:-1].replace(',', '')

    xml_file_path = os.path.join(annotations_folder,
                                 os.path.splitext(file_name)[0] + ".xml")
    pretty_xml_string = tostring(nodes, pretty_print=True)

    with open(xml_file_path, "wb") as xml_file:
        xml_file.write(pretty_xml_string)


def load_all_muscima_annotations(muscima_pp_dataset_directory) -> Dict[str, List[Node]]:
    """
    :param muscima_pp_dataset_directory:
    :return: Returns a dictionary of annotations with the filename as key
    """
    raw_data_directory = os.path.join(muscima_pp_dataset_directory, "v2.0", "data", "annotations")
    all_xml_files = [y for x in os.walk(raw_data_directory) for y in glob(os.path.join(x[0], '*.xml'))]
    all_xml_files = all_xml_files
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


def compute_objects_appearing_in_cropped_image(file_name: str,
                                               image_crop_bounding_box_top_left_bottom_right: Tuple[int, int, int, int],
                                               all_music_objects_appearing_in_image: List[Node],
                                               intersection_over_area_threshold_for_inclusion=0.8) \
        -> Tuple[List[Tuple[str, str, Tuple[int, int, int, int]]], List[Node]]:
    x_translation_for_cropped_image = image_crop_bounding_box_top_left_bottom_right[1]
    y_translation_for_cropped_image = image_crop_bounding_box_top_left_bottom_right[0]

    objects_appearing_in_cropped_image: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
    nodes_appearing_in_cropped_image: List[Node] = []
    for music_object in all_music_objects_appearing_in_image:
        if music_object.class_name in ["staff", "staff_line", "staff_space"]:
            continue

        intersection_over_area = intersection(image_crop_bounding_box_top_left_bottom_right,
                                              music_object.bounding_box) / area(music_object.bounding_box)
        if intersection_over_area > intersection_over_area_threshold_for_inclusion:
            top, left, bottom, right = music_object.bounding_box
            img_top, img_left, img_bottom, img_right = image_crop_bounding_box_top_left_bottom_right
            img_width = img_right - img_left - 1
            img_height = img_bottom - img_top - 1

            translated_bounding_box = (
                max(0, top - y_translation_for_cropped_image),              # Top
                max(0, left - x_translation_for_cropped_image),             # Left
                min(img_height, bottom - y_translation_for_cropped_image),  # Bottom
                min(img_width, right - x_translation_for_cropped_image))    # Right

            objects_appearing_in_cropped_image.append((file_name,
                                                       music_object.class_name,
                                                       translated_bounding_box))
            nodes_appearing_in_cropped_image.append(
                Node(music_object.id,
                     music_object.class_name,
                     top=translated_bounding_box[0],
                     left=translated_bounding_box[1],
                     height=translated_bounding_box[2]-translated_bounding_box[0],
                     width=translated_bounding_box[3]-translated_bounding_box[1],
                     outlinks=music_object.outlinks))
    return objects_appearing_in_cropped_image, nodes_appearing_in_cropped_image


if __name__ == "__main__":
    muscima_pp_dataset_directory = os.path.join("data", "MUSCIMA++")

    annotations_csv = "data/MUSCIMA++/stavewise_Annotations.csv"
    cut_images(muscima_pp_dataset_directory,
               "data/MUSCIMA++/muscima_pp_cropped_images_with_stafflines",
               annotations_csv,
               "data/MUSCIMA++/stavewise_annotations")

    print("Cropped images saved in: \"data/MUSCIMA++/muscima_pp_cropped_images_with_stafflines\"")
    print("Annotations for cropped images saved in: \"data/MUSCIMA++/stavewise_annotations\"")
