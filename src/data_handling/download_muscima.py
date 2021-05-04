import os
import shutil
import argparse

from omrdatasettools import Downloader, OmrDataset


def download_muscima_pp_v2(save_dir):
    """ Downloads the complete MUSCIMA++ dataset and saves it in "/data/MUSCIMA++"
    """
    downloader = Downloader()
    downloader.download_and_extract_dataset(OmrDataset.MuscimaPlusPlus_V2, save_dir)

    # Move images and annotations to "full_page" folder
    for name in ["images", "annotations"]:
        original_dir = r'data/MUSCIMA++/v2.0/data/' + name
        target_dir = r'data/MUSCIMA++/v2.0/data/full_page/' + name
        shutil.move(original_dir, target_dir)

    print("Deleting downloaded zip files")
    if os.path.exists("CVC_MUSCIMA_PP_Annotated-Images.zip"):
        os.remove("CVC_MUSCIMA_PP_Annotated-Images.zip")

    if os.path.exists("MUSCIMA-pp_v2.0.zip"):
        os.remove("MUSCIMA-pp_v2.0.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download MUSCIMA++ dataset.')

    parser.add_argument('--save_dir', '-o', dest='save_dir',
                        type=str,
                        help='Directory to put MUSCIMA++ in.',
                        default="data/MUSCIMA++")

    args = parser.parse_args()

    download_muscima_pp_v2(args.save_dir)
