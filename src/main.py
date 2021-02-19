import data_util
from dataset_splitter import DatasetSplitter


def main():
    images_root = "data/MUSCIMA++/v2.0/data/images"
    mung_root = "data/MUSCIMA++/v2.0/data/annotations"
    split_file = "src/test_split.yaml"
    split = data_util.load_split(split_file)

    # Load the (full) images and corresponding annotations
    training_mungs, training_images = data_util.load_mungs_images(
        images_root=images_root,
        mung_root=mung_root,
        include_names=split)
    print(len(training_images))


if __name__ == "__main__":
    main()
