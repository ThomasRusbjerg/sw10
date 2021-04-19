import numpy as np
from glob import glob
import cv2
import os
import argparse
from tqdm import tqdm


# source: https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
def calculate_pixel_mean_std(image_folder, channel_num=3):
    image_file_names = []
    if os.path.isfile(image_folder):
        image_file_names.append(image_folder)
    elif os.path.isdir(image_folder):
        image_file_names.extend(glob(image_folder + "/*.png"))

    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(channel_num)
    channel_sum_squared = np.zeros(channel_num)

    for f in tqdm(image_file_names, desc="Calculating pixel mean and std of images", total=len(image_file_names)):
        im = cv2.imread(f)  # image in M*N*CHANNEL_NUM shape, channel in BGR order
        # im = im / 255.0
        im = im / 1.0
        pixel_num += (im.size / channel_num)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return rgb_mean, rgb_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_directory",
        dest="image_directory",
        type=str,
        default="data/MUSCIMA++/v2.0/data/images",
        help="The dataset image directory")

    args = parser.parse_args()
    mean, std = calculate_pixel_mean_std(args.image_directory)
    print("RGB channelwise pixel mean and std of image set:")
    print("     Mean:", np.round(mean, 3))
    print("     Std: ", np.round(std, 3))
