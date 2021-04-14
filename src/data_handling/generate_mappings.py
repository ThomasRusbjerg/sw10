# Source: https://github.com/apacha/MusicObjectDetector-TF

""" Generates mappings for images and classes in muscima++ """

import os
import json
import muscima.io
import glob
from collections import Counter
import argparse

from mung.io import read_nodes_from_file
from tqdm import tqdm

def image_mappings():
    img_mappings = []
    for i, img in enumerate(os.listdir("data/MUSCIMA++/v2.0/data/images")):
        img_name = img[:-4]
        img_mappings.append({"name": img_name, "id": i})
    with open('data/MUSCIMA++/v2.0/mapping_img.json', 'w') as f:
        json.dump(img_mappings, f)

def class_mappings():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/MUSCIMA++/v2.0/data/annotations/")
    parser.add_argument("--mapping_output_path", default="mapping_all_classes.json")
    parser.add_argument("--remove_line_shaped_or_construct", type=bool, default=False)
    args = parser.parse_args()

    names = glob.glob(os.path.join(args.dataset_dir, "*.xml"))
    data = {}

    for name in tqdm(names, desc="Reading all objects from MUSCIMA++ annotations"):
        data[name] = read_nodes_from_file(name)
    datas = []

    for value in data.values():
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
        if args.remove_line_shaped_or_construct:
            if key not in line_shaped_or_construct:
                filtered_class_id.append(key)
        else:
            filtered_class_id.append(key)
    filtered_class_id.sort()
    with open(args.mapping_output_path, "w") as f:
        f.write("[")
        for i, classname in enumerate(filtered_class_id):
            f.write("""{{
"id": {0},
"name": "{1}"
}}, 
""".format(i + 1, classname))
        f.write("]")

if __name__ == "__main__":
    image_mappings()