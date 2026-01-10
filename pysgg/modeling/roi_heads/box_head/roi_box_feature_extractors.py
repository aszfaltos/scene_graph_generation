# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from pysgg.modeling import registry
from pysgg.modeling.backbone import resnet
from pysgg.modeling.make_layers import group_norm
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.poolers import Pooler


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=cfg.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

        if cfg.MODEL.RELATION_ON:
            # for the following relation head, the features need to be flattened
            pooling_size = 2
            self.adptive_pool = nn.AdaptiveAvgPool2d((pooling_size, pooling_size))
            input_size = self.out_channels * pooling_size ** 2
            representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
            use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN

            if half_out:
                out_dim = int(representation_size / 2)
            else:
                out_dim = representation_size

            self.fc7 = make_fc(input_size, out_dim, use_gn)
            self.resize_channels = input_size
            self.flatten_out_channels = out_dim

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x

    def forward_without_pool(self, x):
        x = self.head(x)
        return self.flatten_roi_features(x)

    def flatten_roi_features(self, x):
        x = self.adptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc7(x))
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler

        self.fc6 = make_fc(input_size, representation_size, use_gn)
        if half_out:
            out_dim = int(representation_size / 2)
        else:
            out_dim = representation_size

        self.fc7 = make_fc(representation_size, out_dim, use_gn)
        self.resize_channels = input_size
        self.out_channels = out_dim

        self.out_channels = out_dim

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

    def forward_without_pool(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs, ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("MultiscaleROIFeatureExtractor")
class MultiscaleROIFeatureExtractor(nn.Module):
    """
    Multiscale ROI Feature Extractor that wraps existing feature extractors
    at multiple scales and combines their outputs.

    This allows reusing any registered extractor (FPN2MLPFeatureExtractor,
    FPNXconv1fcFeatureExtractor, etc.) at different scales.

    Config options:
        cfg.MODEL.ROI_BOX_HEAD.MULTISCALE_POOLER_SCALES: list of scale tuples
        cfg.MODEL.ROI_BOX_HEAD.MULTISCALE_BASE_EXTRACTOR: base extractor name
        cfg.MODEL.ROI_BOX_HEAD.MULTISCALE_FUSION: fusion method
    """

    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False):
        """
        Args:
            cfg: Config object (reads MULTISCALE_* options)
            in_channels: Number of input channels
            half_out: Whether to halve output dimension
            cat_all_levels: Whether to concatenate all FPN levels
            for_relation: Whether this is for relation head
        """
        super(MultiscaleROIFeatureExtractor, self).__init__()

        # Read multiscale config
        scales_list = cfg.MODEL.ROI_BOX_HEAD.MULTISCALE_POOLER_SCALES
        base_extractor = cfg.MODEL.ROI_BOX_HEAD.MULTISCALE_BASE_EXTRACTOR
        fusion = cfg.MODEL.ROI_BOX_HEAD.MULTISCALE_FUSION

        self.fusion_method = fusion
        self.num_scales = len(scales_list)

        # Create an extractor for each scale by modifying the config
        self.extractors = nn.ModuleList()
        extractor_class = registry.ROI_BOX_FEATURE_EXTRACTORS[base_extractor]

        for scales in scales_list:
            # Clone and modify config for this scale
            scale_cfg = cfg.clone()
            scale_cfg.defrost()
            scale_cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES = scales
            scale_cfg.freeze()

            extractor = extractor_class(
                scale_cfg, in_channels, half_out=half_out,
                cat_all_levels=cat_all_levels, for_relation=for_relation
            )
            self.extractors.append(extractor)

        # Get output dimension from first extractor
        single_out_channels: torch.Tensor = self.extractors[0].out_channels # type: ignore

        # Attention-based fusion
        if self.fusion_method == "attention":
            self.scale_attention = nn.Sequential(
                make_fc(single_out_channels, single_out_channels // 4),
                nn.ReLU(inplace=True),
                make_fc(single_out_channels // 4, 1),
            )

        # Set output channels based on fusion method
        if self.fusion_method == "concat":
            self.out_channels = single_out_channels * self.num_scales
            # Projection to match expected output size
            self.concat_proj = make_fc(self.out_channels, single_out_channels)
            self.out_channels = single_out_channels
        else:
            self.out_channels = single_out_channels

        # Copy resize_channels from base extractor
        self.resize_channels = self.extractors[0].resize_channels

    def forward(self, x, proposals):
        # Extract features at each scale
        scale_features = []
        for extractor in self.extractors:
            feat = extractor(x, proposals)
            scale_features.append(feat)

        # Fuse features from different scales
        if self.fusion_method == "sum":
            fused = torch.stack(scale_features, dim=0).sum(dim=0)
        elif self.fusion_method == "mean":
            fused = torch.stack(scale_features, dim=0).mean(dim=0)
        elif self.fusion_method == "concat":
            fused = torch.cat(scale_features, dim=-1)
            fused = F.relu(self.concat_proj(fused))
        elif self.fusion_method == "attention":
            # Compute attention weights for each scale
            attention_weights = []
            for feat in scale_features:
                weight = self.scale_attention(feat)
                attention_weights.append(weight)
            attention_weights = torch.softmax(torch.cat(attention_weights, dim=-1), dim=-1)

            # Weighted sum
            stacked = torch.stack(scale_features, dim=-1)  # (N, C, num_scales)
            fused = (stacked * attention_weights.unsqueeze(1)).sum(dim=-1)
        elif self.fusion_method == "max":
            fused = torch.stack(scale_features, dim=0).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused

    def forward_without_pool(self, x):
        return self.extractors[0](x)


def make_roi_box_feature_extractor(cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels, half_out, cat_all_levels, for_relation)
