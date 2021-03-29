from data_handling import muscima_loader
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import detectron2
print(f"Detectron2 version is {detectron2.__version__}")
def main():

    training_split_file_path = "data/training_validation_test/training.txt"
    val_split_file_path = "data/training_validation_test/validation.txt"
    test_split_file_path = "data/training_validation_test/test.txt"

    images_root = "data/MUSCIMA++/v2.0/data/images"
    mung_root = "data/MUSCIMA++/v2.0/data/annotations"

    training_split_file_path = "data/training_validation_test/training.txt"
    training_split = muscima_loader.load_split(training_split_file_path)

    # Load the (full) images and corresponding annotations
    training_mungs, training_images = muscima_loader.load_mungs_images(
        images_root=images_root,
        mung_root=mung_root,
        include_names=training_split)
    print("Number of training examples:", len(training_images))


if __name__ == "__main__":
    main()
