import cv2, random
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import default_argument_parser, launch

from visualisation.attention_weights import visualise_attention_weights

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


def visualise_ground_truth():
    muscima_metadata = MetadataCatalog.get("muscima_test")
    data = DatasetCatalog.get("muscima_test")
    img = cv2.imread(data[89]["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=muscima_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(data[89])
    cv2.imwrite("data/MUSCIMA++/v2.0/visualisations/ground-truth-output.jpg",
                cv2.bitwise_not(vis.get_image()[:, :, ::-1]))  # ::-1 converts BGR to RGB


def visualise(cfg, data, metadata, n_samples):
    pred = DefaultPredictor(cfg)
    # for i, d in enumerate(data):
    #     print(i, d['file_name'])
    # exit()
    im = cv2.imread(data[89]["file_name"])
    outputs = pred(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=0.5
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("data/MUSCIMA++/v2.0/visualisations/predictions-output.jpg",
                cv2.bitwise_not(out.get_image()[:, :, ::-1]))  # ::-1 converts BGR to RGB


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
    # setattr(args, "opts", ['MODEL.WEIGHTS', 'models/model_final.pth'])
    # setattr(args, "opts", ['MODEL.WEIGHTS', 'src/models/detr/models/orig_detr/omr_jobs_20210604-121910_model_0942479.pth'])
    setattr(args, "opts", ['MODEL.WEIGHTS', 'src/models/detr/models/rel_detr/omr_jobs_20210525-153853_model_0110879.pth'])
    cfg = detr_train.setup(args)
    muscima_metadata = MetadataCatalog.get("muscima_test")
    data = load_muscima_detectron_dataset("data/MUSCIMA++/v2.0/data/staves/training_validation_test/test.pickle")
    visualise(cfg, data, muscima_metadata, 1)
    # visualise_attention_weights(cfg, data)
    visualise_ground_truth()

    exit()
    # Training
    # detr(args)


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.is_available())
    print(f"Detectron2 version is {detectron2.__version__}")
    setup_logger()
    main()
