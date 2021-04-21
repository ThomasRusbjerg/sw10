# Source: https://github.com/apacha/MusicObjectDetector-TF/blob/master/MusicObjectDetector/prepare_muscima_stavewise_annotations.py
import os
import argparse
import shutil
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

    crop_annotations = []
    for image_path in tqdm(image_paths, desc=f"Cutting images and saving to {output_path}", total=len(image_paths)):
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

        if method == "staves":
            crop_to_staves(image, objects_appearing_in_image, image_path,
                           image_width, image_height, image_name, crop_annotations,
                           output_path)
        elif method == "measures":
            crop_to_measures(image, objects_appearing_in_image,
                             image_name, crop_annotations, output_path)


def barlines_in_each_staff(barlines: List[Node]):
    # Order barlines in correct order
    first_barline_of_every_staff = []

    # todo: This may be wrong, it is possible for the second staff first barline to be closest
    # The first barline is the one closest to the top left corner
    node_top_left_corner = Node(0, "", 0, 0, 0, 0, document=barlines[0].document)
    first_barline = sorted(barlines, key=lambda barline: barline.distance_to(node_top_left_corner))[0]

    # todo: This is flawed, barlines are not vertically aligned
    # Add the rest of the leftmost barlines
    first_barline_of_every_staff += list(filter(
        lambda barline: abs(barline.left - first_barline.left < 10), barlines))

    # Add barlines in the order from left to right for each leftmost barline
    result = []
    for leftmost_barline in first_barline_of_every_staff:
        def same_height(barline): return abs(barline.top - leftmost_barline.top) < 10

        barlines_at_this_height = list(filter(same_height, barlines))
        result.append(sorted(barlines_at_this_height,
                                   key=lambda barline: barline.distance_to(leftmost_barline)))
    return result


def crop_to_measures_old(image, objects_appearing_in_image,
                     image_name, crop_annotations, output_path):
    barlines = [x for x in objects_appearing_in_image if x.class_name == "staff"]
    barlines.sort(key=lambda b: b.id)
    for i, barline in enumerate(barlines):
        print(barline.id)
        from PIL import ImageDraw, ImageFont
        image = image.convert('RGB')
        imagedraw = ImageDraw.Draw(image)
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
        imagedraw.text((barline.left, barline.top), text=barline.id.__str__(), fill='red', font=fnt)
        image.save(image_name + "bbox_test.png")
    exit()

    # If no barlines are in this image
    if barlines is None:
        return

    # Barlines are read in no particular order, so get them for each staff
    staves = barlines_in_each_staff(barlines)

    # todo: Sometimes the barlines span ~5 staves, so take only one staff, one measure
    #       But the same barline has to be used to crop different measures in different staves, then!
    # todo: I wrongly assumed each staff starts with a barline - but many do not

    # todo: Handle case where barlines basically touch (double barline)

    vertical_padding = 120

    # Go through each staff and crop to measure
    for staff_index in range(len(staves)):
        for barline_index in range(len(staves[staff_index])):
            # If the last barline is reached, you're finished
            if barline_index == len(staves[staff_index]) - 1:
                break

            left_barline = staves[staff_index][barline_index]
            right_barline = staves[staff_index][barline_index + 1]

            measure_bbox_tlbr = (left_barline.top - vertical_padding,
                                 left_barline.left,
                                 left_barline.bottom + vertical_padding,
                                 right_barline.right)
            measure_bbox_ltrb = (left_barline.left,
                                 left_barline.top - vertical_padding,
                                 right_barline.right,
                                 left_barline.bottom + vertical_padding)

            file_name = f"{image_name}_{staff_index+1}_{barline_index+1}.png"

            objects_in_cropped_image, nodes_appearing_in_cropped_image = \
                get_objects_in_cropped_image(file_name,
                                             measure_bbox_tlbr,
                                             objects_appearing_in_image)

            cropped_image = image.crop(measure_bbox_ltrb).convert('RGB')

            for object_in_cropped_image in objects_in_cropped_image:
                file_name = object_in_cropped_image[0]
                class_name = object_in_cropped_image[1]
                translated_bounding_box = object_in_cropped_image[2]
                trans_top, trans_left, trans_bottom, trans_right = translated_bounding_box
                crop_annotations.append([file_name, trans_left, trans_top, trans_right, trans_bottom, class_name])

            create_muscima_annotations(output_path + "annotations",
                                       file_name,
                                       nodes_appearing_in_cropped_image)

            image_output_path = output_path + "images/"
            os.makedirs(image_output_path, exist_ok=True)

            output_file = os.path.join(image_output_path, file_name)
            try:
                cropped_image.save(output_file, "png")
            except SystemError:
                print(measure_bbox_ltrb)
                print(image.height)
                print(image.width)
                from PIL import ImageDraw
                image = image.convert('RGB')
                imagedraw = ImageDraw.Draw(image)
                imagedraw.rectangle(measure_bbox_ltrb, outline='red', width=3)
                image.save(image_name + "bbox_test.png")
                exit()


def crop_to_measures(image, objects_appearing_in_image,
                     image_name, crop_annotations, output_path):
    staves = [x for x in objects_appearing_in_image if x.class_name == "staff"]
    mss = [x for x in objects_appearing_in_image if x.class_name == "measureSeparator"]

    # If no staves are in this image
    if staves is None:
        return

    vertical_padding = 120

    # Go through each staff and crop to measure
    measure_bboxes = []
    for staff_index, staff in enumerate(staves):
        measure_begin = staff.left
        ms_in_staff = list(filter(lambda b: staff.overlaps(b), mss))
        ms_in_staff = sorted(ms_in_staff, key=lambda x: x.left)
        for measure_index, ms in enumerate(ms_in_staff):
            # If end of staff is reached, go to next staff
            if abs(measure_begin - staff.right) < 20:
                break
            # todo: Check if the 0 and image.height works or not
            measure_bbox_tlbr = (0 if staff_index == 0 else staff.top - vertical_padding,
                                 measure_begin,
                                 image.height if staff_index == len(staves) - 1 else staff.bottom + vertical_padding,
                                 ms.right)
            measure_bbox_ltrb = (measure_begin,
                                 0 if staff_index == 0 else staff.top - vertical_padding,
                                 ms.right,
                                 image.height if staff_index == len(staves) - 1 else staff.bottom + vertical_padding)

            # from PIL import 2ImageDraw
            # image = image.convert('RGB')
            # imagedraw = ImageDraw.Draw(image)
            # imagedraw.rectangle(measure_bbox_ltrb, outline='red', width=3)
            # image.save(image_name + "bbox_test.png")

            # Next measure begins at the current measure separator
            measure_begin = ms.left

            file_name = f"{image_name}_{staff_index+1}_{measure_index+1}.png"

            objects_in_cropped_image, nodes_appearing_in_cropped_image = \
                get_objects_in_cropped_image(file_name,
                                             measure_bbox_tlbr,
                                             objects_appearing_in_image)

            cropped_image = image.crop(measure_bbox_ltrb).convert('RGB')

            for object_in_cropped_image in objects_in_cropped_image:
                file_name = object_in_cropped_image[0]
                class_name = object_in_cropped_image[1]
                translated_bounding_box = object_in_cropped_image[2]
                trans_top, trans_left, trans_bottom, trans_right = translated_bounding_box
                crop_annotations.append([file_name, trans_left, trans_top, trans_right, trans_bottom, class_name])

            create_muscima_annotations(output_path + "annotations",
                                       file_name,
                                       nodes_appearing_in_cropped_image)

            image_output_path = output_path + "images/"
            os.makedirs(image_output_path, exist_ok=True)

            output_file = os.path.join(image_output_path, file_name)

            cropped_image.save(output_file, "png")
            # try:
            # except SystemError:
            #     print(measure_bbox_ltrb)
            #     print(image.height)
            #     print(image.width)
            #     from PIL import ImageDraw
            #     image = image.convert('RGB')
            #     imagedraw = ImageDraw.Draw(image)
            #     imagedraw.rectangle(measure_bbox_ltrb, outline='red', width=3)
            #     image.save(image_name + "bbox_test.png")
            #     exit()


def crop_to_staves(image, objects_appearing_in_image, image_path,
                   image_width, image_height, image_name, crop_annotations,
                   output_path):
    staff_objects = [x for x in objects_appearing_in_image if x.class_name == "staff"]
    max_offset_before_first_and_after_last_staff = 120

    if staff_objects is None:
        # Image has no staff lines -> Report error
        print("Error: Image {0} has no annotated staff lines".format(image_path))
        return

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

        objects_in_cropped_image, nodes_appearing_in_cropped_image = \
            get_objects_in_cropped_image(file_name,
                                         image_crop_bounding_box_top_left_bottom_right,
                                         objects_appearing_in_image)

        cropped_image = image.crop(image_crop_bounding_box_left_top_bottom_right).convert('RGB')

        for object_in_cropped_image in objects_in_cropped_image:
            file_name = object_in_cropped_image[0]
            class_name = object_in_cropped_image[1]
            translated_bounding_box = object_in_cropped_image[2]
            trans_top, trans_left, trans_bottom, trans_right = translated_bounding_box
            crop_annotations.append([file_name, trans_left, trans_top, trans_right, trans_bottom, class_name])

        create_muscima_annotations(output_path + "annotations",
                                   file_name,
                                   nodes_appearing_in_cropped_image)

        image_output_path = output_path + "images/"
        os.makedirs(image_output_path, exist_ok=True)

        output_file = os.path.join(image_output_path, file_name)
        cropped_image.save(output_file, "png")


# Does not create mask and outlink annotations
def create_muscima_annotations(output_path: str,
                               file_name: str,
                               nodes_appearing_in_image: List[Node]):
    os.makedirs(output_path, exist_ok=True)

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

    xml_file_path = os.path.join(output_path,
                                 os.path.splitext(file_name)[0] + ".xml")
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


def get_objects_in_cropped_image(file_name: str,
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
                max(0, top - y_translation_for_cropped_image),  # Top
                max(0, left - x_translation_for_cropped_image),  # Left
                min(img_height, bottom - y_translation_for_cropped_image),  # Bottom
                min(img_width, right - x_translation_for_cropped_image))  # Right

            objects_appearing_in_cropped_image.append((file_name,
                                                       music_object.class_name,
                                                       translated_bounding_box))
            nodes_appearing_in_cropped_image.append(
                Node(music_object.id,
                     music_object.class_name,
                     top=translated_bounding_box[0],
                     left=translated_bounding_box[1],
                     height=translated_bounding_box[2] - translated_bounding_box[0],
                     width=translated_bounding_box[3] - translated_bounding_box[1],
                     outlinks=music_object.outlinks))
    return objects_appearing_in_cropped_image, nodes_appearing_in_cropped_image


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
