"""Minimal YOLOX-S inference runtime derived from Megvii YOLOX.

Copyright (c) 2014-2021 Megvii Inc. Licensed under Apache-2.0. Training-only
code and the optional C++ extension are intentionally omitted. Module names
match the official checkpoint so its state dictionary remains compatible.
"""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.ops import batched_nms, nms


def _activation(name: str = "silu", *, inplace: bool = True) -> nn.Module:
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    raise ValueError(f"Unsupported activation: {name}")


class BaseConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
        act: str = "silu",
    ) -> None:
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _activation(act)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.act(self.bn(self.conv(inputs))))


class DWConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, ksize: int, stride: int, act: str
    ) -> None:
        super().__init__()
        self.dconv = BaseConv(
            in_channels, in_channels, ksize, stride, groups=in_channels, act=act
        )
        self.pconv = BaseConv(in_channels, out_channels, 1, 1, act=act)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.pconv(self.dconv(inputs)))


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: str = "silu",
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden, 1, 1, act=act)
        self.conv2 = conv(hidden, out_channels, 3, 1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.conv2(self.conv1(inputs))
        return cast(torch.Tensor, output + inputs if self.use_add else output)


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = "silu") -> None:
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden, 1, 1, act=act)
        self.m = nn.ModuleList(
            nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2)
            for size in (5, 9, 13)
        )
        self.conv2 = BaseConv(hidden * 4, out_channels, 1, 1, act=act)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv1(inputs)
        return cast(
            torch.Tensor,
            self.conv2(torch.cat([inputs, *(layer(inputs) for layer in self.m)], dim=1)),
        )


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: str = "silu",
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1, act=act)
        self.conv2 = BaseConv(in_channels, hidden, 1, 1, act=act)
        self.conv3 = BaseConv(hidden * 2, out_channels, 1, 1, act=act)
        self.m = nn.Sequential(
            *(
                Bottleneck(hidden, hidden, shortcut, 1.0, depthwise, act)
                for _ in range(n)
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.conv3(torch.cat((self.m(self.conv1(inputs)), self.conv2(inputs)), dim=1)),
        )


class Focus(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: str = "silu") -> None:
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, 3, 1, act=act)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.conv(
                torch.cat(
                    (
                        inputs[..., ::2, ::2],
                        inputs[..., 1::2, ::2],
                        inputs[..., ::2, 1::2],
                        inputs[..., 1::2, 1::2],
                    ),
                    dim=1,
                )
            )
        )


class CSPDarknet(nn.Module):
    def __init__(self, depth: float, width: float, act: str = "silu") -> None:
        super().__init__()
        base_channels = int(width * 64)
        base_depth = max(round(depth * 3), 1)
        self.stem = Focus(3, base_channels, act)
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, base_depth, act=act),
        )
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, base_depth * 3, act=act),
        )
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, base_depth * 3, act=act),
        )
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                base_depth,
                shortcut=False,
                act=act,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        inputs = self.stem(inputs)
        inputs = self.dark2(inputs)
        inputs = self.dark3(inputs)
        outputs["dark3"] = inputs
        inputs = self.dark4(inputs)
        outputs["dark4"] = inputs
        inputs = self.dark5(inputs)
        outputs["dark5"] = inputs
        return outputs


class YOLOPAFPN(nn.Module):
    def __init__(self, depth: float, width: float, act: str = "silu") -> None:
        super().__init__()
        channels = (256, 512, 1024)
        self.backbone = CSPDarknet(depth, width, act)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(channels[2] * width), int(channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * channels[1] * width),
            int(channels[1] * width),
            round(3 * depth),
            False,
            act=act,
        )
        self.reduce_conv1 = BaseConv(
            int(channels[1] * width), int(channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * channels[0] * width),
            int(channels[0] * width),
            round(3 * depth),
            False,
            act=act,
        )
        self.bu_conv2 = BaseConv(
            int(channels[0] * width), int(channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * channels[0] * width),
            int(channels[1] * width),
            round(3 * depth),
            False,
            act=act,
        )
        self.bu_conv1 = BaseConv(
            int(channels[1] * width), int(channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * channels[1] * width),
            int(channels[2] * width),
            round(3 * depth),
            False,
            act=act,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.backbone(inputs)
        x2, x1, x0 = (features[name] for name in ("dark3", "dark4", "dark5"))
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.C3_p4(torch.cat((self.upsample(fpn_out0), x1), dim=1))
        fpn_out1 = self.reduce_conv1(f_out0)
        pan_out2 = self.C3_p3(torch.cat((self.upsample(fpn_out1), x2), dim=1))
        pan_out1 = self.C3_n3(torch.cat((self.bu_conv2(pan_out2), fpn_out1), dim=1))
        pan_out0 = self.C3_n4(torch.cat((self.bu_conv1(pan_out1), fpn_out0), dim=1))
        return pan_out2, pan_out1, pan_out0


class YOLOXHead(nn.Module):
    def __init__(self, num_classes: int, width: float, act: str = "silu") -> None:
        super().__init__()
        channels = (256, 512, 1024)
        self.num_classes = num_classes
        self.strides = (8, 16, 32)
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        for channel in channels:
            hidden = int(256 * width)
            self.stems.append(BaseConv(int(channel * width), hidden, 1, 1, act=act))
            self.cls_convs.append(
                nn.Sequential(
                    BaseConv(hidden, hidden, 3, 1, act=act),
                    BaseConv(hidden, hidden, 3, 1, act=act),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    BaseConv(hidden, hidden, 3, 1, act=act),
                    BaseConv(hidden, hidden, 3, 1, act=act),
                )
            )
            self.cls_preds.append(nn.Conv2d(hidden, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(hidden, 4, 1))
            self.obj_preds.append(nn.Conv2d(hidden, 1, 1))

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        shapes: list[tuple[int, int]] = []
        for index, feature in enumerate(inputs):
            stem = self.stems[index](feature)
            cls_output = self.cls_preds[index](self.cls_convs[index](stem)).sigmoid()
            reg_feature = self.reg_convs[index](stem)
            reg_output = self.reg_preds[index](reg_feature)
            obj_output = self.obj_preds[index](reg_feature).sigmoid()
            output = torch.cat((reg_output, obj_output, cls_output), dim=1)
            shapes.append((output.shape[-2], output.shape[-1]))
            outputs.append(output.flatten(start_dim=2))
        decoded = torch.cat(outputs, dim=2).permute(0, 2, 1)
        grids: list[torch.Tensor] = []
        strides: list[torch.Tensor] = []
        for (height, width), stride in zip(shapes, self.strides, strict=True):
            y_grid, x_grid = torch.meshgrid(
                torch.arange(height, device=decoded.device),
                torch.arange(width, device=decoded.device),
                indexing="ij",
            )
            grid = torch.stack((x_grid, y_grid), dim=2).view(1, -1, 2)
            grids.append(grid)
            strides.append(torch.full((*grid.shape[:2], 1), stride, device=decoded.device))
        grid_tensor = torch.cat(grids, dim=1).type_as(decoded)
        stride_tensor = torch.cat(strides, dim=1).type_as(decoded)
        decoded[..., :2] = (decoded[..., :2] + grid_tensor) * stride_tensor
        decoded[..., 2:4] = torch.exp(decoded[..., 2:4]) * stride_tensor
        return decoded


class YOLOX(nn.Module):
    def __init__(self, depth: float = 0.33, width: float = 0.5) -> None:
        super().__init__()
        self.backbone = YOLOPAFPN(depth, width)
        self.head = YOLOXHead(80, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(self.backbone(inputs)))


def build_yolox_s() -> YOLOX:
    return YOLOX(depth=0.33, width=0.5)


def preproc(image: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    padded = np.full((*input_size, 3), 114, dtype=np.uint8)
    ratio = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


def postprocess(
    prediction: torch.Tensor,
    num_classes: int,
    confidence: float,
    nms_iou: float,
    *,
    class_agnostic: bool,
) -> list[torch.Tensor | None]:
    corners = prediction.new_empty(prediction.shape)
    corners[..., 0] = prediction[..., 0] - prediction[..., 2] / 2
    corners[..., 1] = prediction[..., 1] - prediction[..., 3] / 2
    corners[..., 2] = prediction[..., 0] + prediction[..., 2] / 2
    corners[..., 3] = prediction[..., 1] + prediction[..., 3] / 2
    prediction[..., :4] = corners[..., :4]
    results: list[torch.Tensor | None] = []
    for image_prediction in prediction:
        class_confidence, class_prediction = torch.max(
            image_prediction[:, 5 : 5 + num_classes], dim=1, keepdim=True
        )
        keep = image_prediction[:, 4] * class_confidence.squeeze(1) >= confidence
        detections = torch.cat(
            (
                image_prediction[:, :5],
                class_confidence,
                class_prediction.to(image_prediction.dtype),
            ),
            dim=1,
        )[keep]
        if detections.shape[0] == 0:
            results.append(None)
            continue
        scores = detections[:, 4] * detections[:, 5]
        indices = (
            nms(detections[:, :4], scores, nms_iou)
            if class_agnostic
            else batched_nms(detections[:, :4], scores, detections[:, 6], nms_iou)
        )
        results.append(detections[indices])
    return results


def checkpoint_model(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return the official YOLOX state dictionary from its release checkpoint."""
    state = checkpoint.get("model", checkpoint)
    return dict(state)
