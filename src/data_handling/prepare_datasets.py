import argparse
from download_muscima import download_muscima_pp_v2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare MUSCIMA++ datasets.')
    args = parser.parse_args()

    download_muscima_pp_v2()






