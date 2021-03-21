from data_handling import muscima_loader
from PIL import Image, ImageDraw, ImageFont
from models.muscimapp import MuscimaPPDataset, MuscimaPPConfig
from models.mask_rcnn import MaskRCNN

def load_data(split_file_path):
    images_root = "data/MUSCIMA++/v2.0/data/images"
    mung_root = "data/MUSCIMA++/v2.0/data/annotations"

    data_split = muscima_loader.load_split(split_file_path)

    # Load the (full) images and corresponding annotations
    mungs, images = muscima_loader.load_mungs_images(
        images_root=images_root,
        mung_root=mung_root,
        include_names=data_split)

    dataset = MuscimaPPDataset()
    dataset.load_muscimapp(mungs, images)
    dataset.prepare()
    return dataset

def train(model, training_data, validation_data, config):
    """Train the model."""

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(training_data, validation_data,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')

    print("Train all layers")
    model.train(training_data, validation_data,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='all')



def main():
    training_split_file_path = "data/training_validation_test/training.txt"
    val_split_file_path = "data/training_validation_test/validation.txt"
    test_split_file_path = "data/training_validation_test/test.txt"
        
    train_data = load_data(training_split_file_path)
    val_data = load_data(val_split_file_path)
    test_data = load_data(test_split_file_path)

    config = MuscimaPPConfig()
    model = MaskRCNN(mode="training", config=config, model_dir="ayyy")

    train(model, train_data, val_data, config)
    


    # img = Image.fromarray(training_images[0]*255).convert("RGB")
    # draw = ImageDraw.Draw(img)
    # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 10)
    # for v in training_mungs[0].vertices:
    #     bbox = [(v.left, v.top), (v.right, v.bottom)]
    #     draw.rectangle(bbox, outline ="red") 
    #     draw.text((v.left, v.top), v.class_name, font=fnt, fill="red")
    # img.save("123.png")

if __name__ == "__main__":
    main()
