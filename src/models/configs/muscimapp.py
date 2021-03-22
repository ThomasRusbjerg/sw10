"""
Mask R-CNN
Common utility functions and classes.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import json
import numpy as np
from PIL import Image
from models.configs.config import Config
from models.dataset import Dataset

import warnings
warnings.filterwarnings('ignore')

############################################################
#  Configurations
############################################################


class MuscimaPPConfig(Config):
    """Configuration for training on Muscima++.
    Derives from the base Config class and overrides values specific
    to the Muscima++ dataset.
    """

    # Give the configuration a recognizable name
    NAME = "muscimapp"

    BACKBONE = "resnet101"

    MEAN_PIXEL = np.array([120.0, 120.0, 120.0])


    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 115  # Muscima++ has 115 classes

    USE_MINI_MASK = True
    # IMAGE_RESIZE_MODE = "pad64"
    IMAGE_RESIZE_MODE = "square"
    
    IMAGES_PER_GPU = 2
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 20


############################################################
#  Dataset
############################################################


class MuscimaPPDataset(Dataset):
    def load_muscimapp(self, mung, images):
        """Load the Muscima++ dataset in expected model format."""
        self.mung = mung
        self.images = images
        data_folder = "data/MUSCIMA++/v2.0/data/"
        class_info_path = "data/MUSCIMA++/v2.0/mapping_all_classes.json"
        image_ids = []
        with open(class_info_path) as json_file:
            data = json.load(json_file)
            for c in data:
                self.add_class("muscimapp", c["id"], c["name"])
        
        for img in self.mung:
            image_ids.append(img.vertices[0].document)

        # Add images
        for i, image_id in enumerate(image_ids):
            img_path = data_folder + "images/" + image_id + ".png"
            im = Image.open(img_path)
            width, height = im.size
            self.add_image(
                "muscimapp",
                image_id=image_id,
                path=img_path,
                width=width,
                height=height,
            )

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        class_ids = []
        [idx, mung] = [
            [idx, m]
            for idx, m in enumerate(self.mung)
            if m.vertices[0].document == info['id']
        ][0]
        masks = np.zeros([info["height"], info["width"], len(mung.vertices)], dtype=np.uint8)
        for idx, v in enumerate(mung.vertices):
            class_ids.append(
                [c["id"] for c in self.class_info if c["name"] == v.class_name][0]
            )
            mask = v.project_on(self.images[image_id])
            masks[mask[idx][:], mask[:][idx], idx] = 1

        return masks.astype(np.bool), np.array(class_ids)
