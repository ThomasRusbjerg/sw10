import os, json, cv2, random
import numpy
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import default_argument_parser, launch

from detectron2.data.datasets import register_coco_instances

from datetime import date, datetime
from data_handling.detectron2_muscima import (
    create_muscima_detectron_dataset,
    load_muscima_detectron_dataset,
    get_muscima_classid_mapping,
)
import models.detr.train_net as detr_train


def detr():
    args = default_argument_parser().parse_args()
    setattr(
        args, "config_file", "src/models/detr/configs/detr_256_6_6_torchvision.yaml"
    )
    setattr(args, "num_gpus", 1)

    print("Command Line Args:", args)
    launch(
        detr_train.main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


def visualise_examples():
    muscima_metadata = MetadataCatalog.get("muscima_training")
    muscima = DatasetCatalog.get("muscima_training")
    for d in random.sample(muscima, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=muscima_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(
            "image", cv2.resize(out.get_image()[:, :, ::-1], (960, 540))
        )  # ::-1 converts BGR to RGB
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    training_split_file_path = "data/training_validation_test/training.txt"
    val_split_file_path = "data/training_validation_test/validation.txt"
    test_split_file_path = "data/training_validation_test/test.txt"

    data = load_muscima_detectron_dataset("data/validation.pickle")
    for dataset in ["training", "validation"]:
        DatasetCatalog.register(
            "muscima_" + dataset,
            lambda dataset=dataset: load_muscima_detectron_dataset(
                "data/" + dataset + ".pickle"
            ),
        )
        MetadataCatalog.get("muscima_" + dataset).set(
            thing_classes=[classname for classname in get_muscima_classid_mapping()]
        )

    detr()


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.is_available())
    print(f"Detectron2 version is {detectron2.__version__}")
    setup_logger()
    main()
