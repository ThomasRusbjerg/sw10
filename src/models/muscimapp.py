"""
Mask R-CNN
Common utility functions and classes.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np
from PIL import Image
from models.configs.config import Config
from models.dataset import Dataset

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

    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 114  # Muscima++ has 107 classes... but not?


    # IMAGE_RESIZE_MODE = "pad64"
    IMAGE_RESIZE_MODE = "square"
    
    STEPS_PER_EPOCH = 1
    VALIDATION_STEPS = 1


############################################################
#  Dataset
############################################################


class MuscimaPPDataset(Dataset):
    def load_muscimapp(self, mung, images):
        """Load the Muscima++ dataset in expected model format."""
        self.mung = mung
        self.images = images
        data_folder = "data/MUSCIMA++/v2.0/data/"
        image_ids = []
        class_names = []
        # TODO: Better way to get all unique classes
        for img in self.mung:
            image_ids.append(img.vertices[0].document)
            for v in img.vertices:
                class_name = v.class_name
                if class_name not in class_names:
                    class_names.append(class_name)

        # Add classes
        for i, class_name in enumerate(class_names):
            self.add_class("muscimapp", i, class_name)

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
        masks = []
        class_ids = []
        [idx, mung] = [
            [idx, m]
            for idx, m in enumerate(self.mung)
            if m.vertices[0].document == info['id']
        ][0]
        for v in mung.vertices:
            class_ids.append(
                [c["id"] for c in self.class_info if c["name"] == v.class_name][0]
            )
            masks.append(np.array(v.project_on(self.images[idx]), dtype=bool))
            
        m = np.transpose(np.array(masks), (2,0,1))
        c = np.array(class_ids)
        return c,m
