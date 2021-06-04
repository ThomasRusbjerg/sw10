import cv2, random
import torch
import detectron2
import numpy as np
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import default_argument_parser, launch

from data_handling.detectron2_muscima import (
    create_muscima_detectron_dataset,
    load_muscima_detectron_dataset,
    get_muscima_classid_mapping,
)
import models.detr.train_net as detr_train

def detr(args):
    print("Command Line Args:", args)
    launch(
        detr_train.main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


def visualise(cfg, data, metadata, n_samples):
    pred = DefaultPredictor(cfg)
    for d in random.sample(data, n_samples):
        im = cv2.imread(d["file_name"])
        outputs = pred(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata, 
                    scale=0.5
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(
            "image", out.get_image()[:, :, ::-1]
        )  # ::-1 converts BGR to RGB
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # from data_handling.prepare_datasets import main
    # main()
    # exit()
    # Create detectron format datasets
    # training_dir = "data/MUSCIMA++/v2.0/data/full_page"
    # training_split = "/training_validation_test/training.txt"
    # val_split_file_path = "data/training_validation_test/validation.txt"
    # test_split_file_path = "data/training_validation_test/test.txt"
    # create_muscima_detectron_dataset(training_dir, training_split)
    # exit()

    # Register datasets in detectron
    basepath = "data/MUSCIMA++/v2.0/data/staves/training_validation_test/"
    for dataset in ["training", "validation", "test"]:
        DatasetCatalog.register(
            "muscima_" + dataset,
            lambda dataset=dataset: load_muscima_detectron_dataset(
                basepath + dataset + ".pickle"
            ),
        )
        MetadataCatalog.get("muscima_" + dataset).set(
            thing_classes=[classname for classname in get_muscima_classid_mapping()]
        )

    # Setup DETR config
    args = default_argument_parser().parse_args()
    setattr(
        args, "config_file", "src/models/detr/configs/detr_256_6_6_torchvision.yaml"
    )
    setattr(args, "num_gpus", 1)

    # Predict and visualise
    setattr(args, "opts", ['MODEL.WEIGHTS', 'data/omr_jobs_20210602-180916_model_0110879.pth'])
    cfg = detr_train.setup(args)
        
    # Training
    # pred = DefaultPredictor(cfg)
    # exit()
    detr(args)

    ## Count object instances
    # muscima_metadata = MetadataCatalog.get("muscima_training")
    # data = load_muscima_detectron_dataset("data/MUSCIMA++/v2.0/data/measures/training_validation_test/training.pickle")
    # count_object_instances(data, muscima_metadata)

    # visualise_dataset(cfg, data, muscima_metadata)


def count_object_instances(data, metadata):
    category_instance_count = np.zeros(len(metadata.thing_classes))
    for instance in data:
        for annotaion in instance["annotations"]:
            id = annotaion["category_id"]
            category_instance_count[id] = category_instance_count[id] + 1
    combine = zip(metadata.thing_classes, category_instance_count)
    res = sorted(combine, key = lambda x: x[1])
    for obj in res:
        print(obj)

def visualise_dataset(cfg, data, metadata):
    visualise(cfg, data, metadata, 3)
    for d in random.sample(data, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(
            "image", vis.get_image()[:, :, ::-1]
        )  # ::-1 converts BGR to RGB
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.is_available())
    print(f"Detectron2 version is {detectron2.__version__}")
    setup_logger()
    main()
