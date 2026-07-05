'''
Model builders and checkpoint loading for SPL experiments.
'''

from pathlib import Path

import torch
import tltorch

from src.model import VGG16_FCN32s
from src.smallnet.factorization import infer_cp_rank_from_state_dict


def build_vgg16_fcn32s(num_classes=32, pretrained=False):
    return VGG16_FCN32s(num_classes=num_classes, pretrained=pretrained)


def load_vgg16_fcn32s_checkpoint(checkpoint_path, num_classes=32, cp_layer="classifier.0"):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    rank = infer_cp_rank_from_state_dict(state_dict, cp_layer)
    model = build_vgg16_fcn32s(num_classes=num_classes, pretrained=False)
    if rank is not None:
        model.classifier[0] = tltorch.FactorizedConv.from_conv(
            model.classifier[0],
            rank=rank,
            factorization="cp",
            decomposition_kwargs={"init": "random", "n_iter_max": 0},
        )
    model.load_state_dict(state_dict)
    return model, rank


def build_deeplabv3_resnet50(num_classes=21, pretrained=True):
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

    if pretrained:
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        return deeplabv3_resnet50(weights=weights)
    return deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)


def load_model_from_config(config):
    family = config["family"]
    if family == "vgg16_fcn32s":
        checkpoint = config.get("checkpoint")
        if checkpoint:
            return load_vgg16_fcn32s_checkpoint(checkpoint, num_classes=config.get("num_classes", 32))[0]
        return build_vgg16_fcn32s(
            num_classes=config.get("num_classes", 32),
            pretrained=config.get("pretrained", False),
        )
    if family == "deeplabv3_resnet50":
        return build_deeplabv3_resnet50(
            num_classes=config.get("num_classes", 21),
            pretrained=config.get("pretrained", True),
        )
    raise ValueError(f"Unsupported model family: {family}")


def checkpoint_name(path):
    return Path(path).stem if path else "uncheckpointed"
