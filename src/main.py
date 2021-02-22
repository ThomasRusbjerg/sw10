import data_util
from dataset_splitter import DatasetSplitter


def main():
    images_root = "data/MUSCIMA++/v2.0/data/images"
    mung_root = "data/MUSCIMA++/v2.0/data/annotations"

    training_split_file_path = "data/training_validation_test/training.txt"
    training_split = data_util.load_split(training_split_file_path)

    # Load the (full) images and corresponding annotations
    training_mungs, training_images = data_util.load_mungs_images(
        images_root=images_root,
        mung_root=mung_root,
        include_names=training_split)
    print("Number of training examples:", len(training_images))


if __name__ == "__main__":
    main()
