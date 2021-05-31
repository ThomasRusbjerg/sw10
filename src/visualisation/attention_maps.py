import math

from PIL import Image
import argparse
import matplotlib.pyplot as plt
import torch
from models.detr.d2.detr import Detr
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import models.detr.train_net as detr_train
torch.set_grad_enabled(False)


if __name__ == "__main__":
    model_path = "src/models/detr/models/detr_relations/omr_jobs_20210525-153853_model_0055439.pth"
    image_path = "data/MUSCIMA++/v2.0/data/full_page/images/CVC-MUSCIMA_W-01_N-10_D-ideal.png"

    # Setup DETR config
    args = argparse.ArgumentParser().parse_args()
    setattr(args, "config_file", "src/models/detr/configs/detr_256_6_6_torchvision.yaml")
    setattr(args, "opts", ['MODEL.WEIGHTS', 'data/omr_jobs_20210525-153853_model_0055439.pth'])
    setattr(args, "num_gpus", 1)
    cfg = detr_train.setup(args)

    checkpoint = torch.load(model_path, map_location="cpu")
    print(checkpoint["model"].keys())
    model = Detr(cfg)
    model.detr.load_state_dict(checkpoint["state_dict"])
    img = Image.open(image_path)

    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]



    # propagate through the model
    outputs = model(img)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]
    print("Hey")
