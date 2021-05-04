# Source: https://github.com/apacha/MusicObjectDetector-TF

""" Generates mappings for images and classes in muscima++ """

import os
import argparse
import json
from collections import Counter
from muscima_loader import load_all_muscima_annotations


def image_mappings(images_dir, output_path):
    img_mappings = []
    for i, img in enumerate(os.listdir(images_dir)):
        img_name = os.path.splitext(img)[0]  # cut away the file extension
        img_mappings.append({"name": img_name, "id": i})
    with open(output_path + '/mapping_img.json', 'w') as f:
        json.dump(img_mappings, f)


def class_mappings(annotation_dictionary, output_path, remove_line_shaped_or_construct=False):

    datas = []
    for value in annotation_dictionary.values():
        for val in value:
            datas.append(val)

    c = Counter([x.class_name for x in datas])

    ignored_classes = []  # ["double_sharp", "numeral_2", "numeral_5", "numeral_6", "numeral_7", "numeral_8"]
    line_shaped_or_construct = ["stem",
                                "beam",
                                "thin_barline",
                                "measure_separator",
                                "slur",
                                "tie",
                                "key_signature",
                                "dynamics_text",
                                "hairpin-decr.",
                                "other_text",
                                "tuple",
                                "hairpin-cresc.",
                                "time_signature",
                                "staff_grouping",
                                "trill",
                                "tenuto",
                                "tempo_text",
                                "multi-staff_brace",
                                "multiple-note_tremolo",
                                "repeat",
                                "multi-staff_bracket",
                                "tuple_bracket/line"]
    filtered_class_id = []
    for key, value in c.items():
        if key in ignored_classes:
            continue
        if remove_line_shaped_or_construct:
            if key not in line_shaped_or_construct:
                filtered_class_id.append(key)
        else:
            filtered_class_id.append(key)
    filtered_class_id.sort()
    with open(output_path + "/mapping_all_classes.json", "w") as f:
        f.write("[")
        for i, classname in enumerate(filtered_class_id):
            f.write("""{{
"id": {0},
"name": "{1}"
}}, 
""".format(i + 1, classname))
        f.write("]")
    print("Remember to remove last comma in class mappings json file!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/MUSCIMA++/v2.0/data/images/")
    parser.add_argument("--annotations", default="data/MUSCIMA++/v2.0/data/annotations/")
    parser.add_argument("--output_path", default="data/MUSCIMA++/v2.0/")
    parser.add_argument("--remove_line_shaped_or_construct", type=bool, default=False)
    args = parser.parse_args()

    image_mappings(args.images, args.output_path)
    class_mappings(args.annotations, args.output_path, args.remove_line_shaped_or_construct)
