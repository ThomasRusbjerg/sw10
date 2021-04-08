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

from data_handling.detectron2_muscima import create_muscima_detectron_dataset, load_muscima_detectron_dataset, get_muscima_classid_mapping
from models.mask_rcnn import mask_rcnn

def main():
    # print("Hello world!")
    # exit()
    training_split_file_path = "data/training_validation_test/training.txt"
    val_split_file_path = "data/training_validation_test/validation.txt"
    test_split_file_path = "data/training_validation_test/test.txt"

    # create_muscima_detectron_dataset(val_split_file_path)

    data = load_muscima_detectron_dataset("data/validation.pickle")
    for dataset in ["training", "validation"]:
        DatasetCatalog.register("muscima_" + dataset, lambda dataset=dataset: load_muscima_detectron_dataset("data/" + dataset + ".pickle"))
        MetadataCatalog.get("muscima_" + dataset).set(thing_classes=[classname for classname in get_muscima_classid_mapping()])
        # MetadataCatalog.get("muscima_" + dataset).set(box_mode=BoxMode.XYXY_ABS)
    muscima_metadata = MetadataCatalog.get("muscima_training")
    # print(muscima_metadata)
    # exit()
    # mask_rcnn.train()
    # exit()
    stuff()

def stuff():
    im_gray = cv2.imread("data/MUSCIMA++/v2.0/data/images/CVC-MUSCIMA_W-01_N-10_D-ideal.png", cv2.IMREAD_GRAYSCALE)
    im_rgb = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2RGB)
    # Inference
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im_rgb)

    # Visualise
    v = Visualizer(im_rgb[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    cv2.imshow('image', cv2.resize(out.get_image()[:, :, ::-1], (960, 540))) # ::-1 converts BGR to RGB
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    print(torch.__version__, torch.cuda.is_available())
    print(f"Detectron2 version is {detectron2.__version__}")
    setup_logger()
    main()
