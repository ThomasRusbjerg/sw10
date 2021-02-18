import data_util
from dataset_splitter import DatasetSplitter


def main():
    images_root = "data/MUSCIMA++/v2.0/data/images"
    mung_root = "data/MUSCIMA++/v2.0/data/annotations"
    split_file = "src/test_split.yaml"
    split = data_util.load_split(split_file)

    # Load the (full) images and corresponding annotations
    mungs, images = data_util.load_munglinker_data(images_root=images_root,
                                                   mung_root= mung_root,
                                                   include_names=split["train"])

    # Split the data into train, validation and testing
    # To comply with the code above, we could modify it to create a yaml file
    # instead of 3 different text files
    source_directory = "data/MUSCIMA++/v2.0/data/images"
    destination_directory = "data/splitdata"
    independent_set = "data/MUSCIMA++/v2.0/specifications/testset-independent.txt"
    datasest = DatasetSplitter(source_directory,
                               destination_directory,
                               independent_set)
    datasest.delete_split_directories()
    datasest.split_images_into_training_validation_and_test_set()


if __name__ == "__main__":
    main()
