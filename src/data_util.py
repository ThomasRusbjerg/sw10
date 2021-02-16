"""Module for manipulating the data"""

from omrdatasettools import Downloader, OmrDataset


def get_muscima_pp_v2():
    """ Downloads the complete MUSCIMA++ dataset and saves it in "/data/MUSCIMA++"
    """
    downloader = Downloader()
    downloader.download_and_extract_dataset(OmrDataset.MuscimaPlusPlus_V2, "data/MUSCIMA++")
