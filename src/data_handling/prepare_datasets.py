import argparse
import os
from glob import glob

from data_handling.download_muscima import download_muscima_pp_v2
from data_handling.generate_mappings import image_mappings, class_mappings
from data_handling.muscima_crop import cut_images
from data_handling.dataset_splitter import DatasetSplitter
from data_handling.detectron2_muscima import create_muscima_detectron_dataset
from data_handling.muscima_loader import load_all_muscima_annotations


def main():
    parser = argparse.ArgumentParser(description='Prepare MUSCIMA++ datasets.')

    parser.add_argument('--save_dir', '-o', dest='save_dir',
                        type=str,
                        help='Directory to put MUSCIMA++ in.',
                        default="data/MUSCIMA++")
                        # default="/user/student.aau.dk/trusbj16/data/MUSCIMA++")
    parser.add_argument('-d', dest='download',
                        action='store_true',
                        help='Download MUSCIMA++ (True) or use existing (False).',
                        default=False)
    parser.add_argument('-c', dest='crop',
                        action='store_true',
                        help='Crop images (True) or use existing cropped images (False).',
                        default=False)
    parser.add_argument('-gen_label_map', dest='gen_label_map',
                        action='store_true',
                        help='Generate class label map',
                        default=False)
    parser.add_argument("--independent_set",
                        type=str,
                        default="data/MUSCIMA++/v2.0/specifications/testset-independent.txt",
                        help="text file with independent writer set")

    args = parser.parse_args()

    # Download MUSCIMA++
    if args.download:
        download_muscima_pp_v2(args.save_dir)

    data_directory = "data/MUSCIMA++/v2.0/data"
    # data_directory = "/user/student.aau.dk/trusbj16/data/MUSCIMA++/v2.0/data"
    image_directory = os.path.join(data_directory, "full_page/images")
    annotation_directory = os.path.join(data_directory, "full_page/annotations")

    annotation_dictionary = {}
    if args.gen_label_map or args.crop:
        annotation_dictionary = load_all_muscima_annotations(annotation_directory)

    if args.gen_label_map:
        # Create class image label map
        class_mappings(annotation_dictionary, output_path=os.path.dirname(data_directory))

    crop_methods = ["measures", "staves"]

    if args.crop:
        # Generate measure- and stave-wise cropped images
        muscima_image_directory = os.path.join(image_directory, "*.png")
        image_paths = glob(muscima_image_directory)

        for method in crop_methods:
            output_dir = data_directory + "/" + method + "/"
            cut_images(image_paths, annotation_dictionary, output_dir, method)

    # Split datasets
    image_directories = [image_directory]
    for method in crop_methods:
        image_directories.append(data_directory + "/" + method + "/" + "images")

    independent_set = "data/MUSCIMA++/v2.0/specifications/testset-independent.txt"
    # independent_set = "/user/student.aau.dk/trusbj16/data/MUSCIMA++/v2.0/specifications/testset-independent.txt"

    # For each dataset, split into train/validation/test
    for image_dir in image_directories:
        output_dir = os.path.join(os.path.dirname(image_dir), "training_validation_test")
        dataset_splitter = DatasetSplitter(image_dir, output_dir,
                                           independent_set)
        dataset_splitter.delete_split_directories()
        dataset_splitter.split_images_into_training_validation_and_test_set()

    # Generate pickle file for Detectron2
    data_folders = glob(data_directory + "/*/")
    for folder in data_folders:
        print("Creating Detectron/Coco dataset for", os.path.dirname(folder))
        image_dir = os.path.join(folder, "images")
        image_mappings(image_dir, folder)
        splits = ["training", "validation", "test"]
        for split in splits:
            split_location = "training_validation_test/" + split + ".txt"
            create_muscima_detectron_dataset(data_dir=folder, split_location=split_location)

if __name__ == "__main__":
    main()