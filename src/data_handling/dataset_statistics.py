import os
import argparse
import yaml
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

from mung.io import read_nodes_from_file


# source: https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
def calculate_pixel_mean_std(image_file_names, channel_num=3):

    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(channel_num)
    channel_sum_squared = np.zeros(channel_num)

    for f in tqdm(image_file_names, total=len(image_file_names),
                  desc="Calculating pixel mean and std of images"):
        im = cv2.imread(f).astype(np.float)  # image in M*N*CHANNEL_NUM shape, channel in BGR order
        pixel_num += (im.size / channel_num)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return {
        "RGB channel-wise pixel mean and standard deviation of image set": {
            "Mean": np.round(rgb_mean, 3).__str__(),
            "Std": np.round(rgb_std, 3).__str__()
        }
    }


def count_objects_per_annotation(annotation_file_names):
    n_objects = []
    for annotation_file in tqdm(annotation_file_names, total=len(annotation_file_names),
                                desc="Counting number of objects per image"):
        music_objects = read_nodes_from_file(annotation_file)
        n_objects.append(len(music_objects))

    return {
        "Number of objects per image": {
            "Mean": np.mean(n_objects).__float__(),
            "Median": np.median(n_objects).__float__(),
            "Max": np.max(n_objects).__float__(),
            "Min": np.min(n_objects).__float__(),
            "Std": np.std(n_objects).__float__(),
        }
    }


def load_file_names(image_folder: str, file_extension: str):
    file_names = []
    if os.path.isfile(image_folder):
        file_names.append(image_folder)
    elif os.path.isdir(image_folder):
        file_names.extend(glob(image_folder + file_extension))

    return file_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_directory",
        dest="image_directory",
        type=str,
        default="data/MUSCIMA++/v2.0/data/images",
        help="The dataset image directory")

    parser.add_argument(
        "--annotations_directory",
        dest="annotations_directory",
        type=str,
        default="data/MUSCIMA++/v2.0/data/annotations",
        help="The dataset annotation directory")
    parser.add_argument(
        "--save_directory",
        dest="save_directory",
        type=str,
        help="Where to save the output statistics file")

    args = parser.parse_args()

    # Save statistics in the parent folder of the images
    if args.save_directory is None:
        args.save_directory = os.path.dirname(args.image_directory) + "/"

    image_file_names = load_file_names(args.image_directory, "/*.png")
    annotation_file_names = load_file_names(args.annotations_directory, "/*.xml")

    statistics = {}
    statistics.update(count_objects_per_annotation(annotation_file_names))
    statistics.update(calculate_pixel_mean_std(image_file_names))

    print(f"Dataset statistics (also saved in \"{args.save_directory}\":")

    print(yaml.dump(statistics, sort_keys=False, default_flow_style=False))
    with open(args.save_directory + "statistics.yaml", "w") as f:
        yaml.dump(statistics, f, default_flow_style=False)
