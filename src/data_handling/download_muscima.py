import os
from omrdatasettools import Downloader, OmrDataset


def download_muscima_pp_v2():
    """ Downloads the complete MUSCIMA++ dataset and saves it in "/data/MUSCIMA++"
    """
    downloader = Downloader()
    downloader.download_and_extract_dataset(OmrDataset.MuscimaPlusPlus_V2, "data/MUSCIMA++")

    print("Deleting downloaded zip files")
    if os.path.exists("CVC_MUSCIMA_PP_Annotated-Images.zip"):
        os.remove("CVC_MUSCIMA_PP_Annotated-Images.zip")

    if os.path.exists("MUSCIMA-pp_v2.0.zip"):
        os.remove("MUSCIMA-pp_v2.0.zip")


if __name__ == "__main__":
    download_muscima_pp_v2()
