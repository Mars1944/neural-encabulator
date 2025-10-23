import torch
import torch.nn as nn
from typing import List, Sequence


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        dilation = max(int(dilation), 1)
        padding = (kernel_size // 2) * dilation
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=not use_batchnorm,
            ),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers += [
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    A simple, configurable CNN.

    Config parameters (dict):
    - input_channels: int
    - num_classes: int
    - conv_channels: List[int]  (e.g., [32, 64, 128])
    - kernel_size: int          (default 3)
    - pool_every: int           (apply MaxPool after this many conv blocks, default 1)
    - dropout: float            (dropout probability in conv blocks, default 0.0)
    - use_batchnorm: bool       (default True)
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        in_ch: int = int(config.get("input_channels", 3))
        num_classes: int = int(config.get("num_classes", 10))
        conv_channels: List[int] = list(config.get("conv_channels", [32, 64]))
        kernel_size: int = int(config.get("kernel_size", 3))
        pool_every: int = int(config.get("pool_every", 1))
        dropout: float = float(config.get("dropout", 0.0))
        use_batchnorm: bool = bool(config.get("use_batchnorm", True))
        # Optional dilation(s) to enlarge receptive field without aggressive pooling
        dilations_cfg = config.get("dilations", 1)
        if isinstance(dilations_cfg, Sequence) and not isinstance(dilations_cfg, (str, bytes)):
            dilations: List[int] = [max(int(d), 1) for d in dilations_cfg]
        else:
            dilations = [max(int(dilations_cfg), 1)] * len(conv_channels)

        features: List[nn.Module] = []
        last_ch = in_ch
        blocks_since_pool = 0
        for idx, out_ch in enumerate(conv_channels):
            dilation = dilations[idx] if idx < len(dilations) else dilations[-1]
            features.append(
                ConvBlock(
                    in_ch=last_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
            )
            last_ch = out_ch
            blocks_since_pool += 1
            if pool_every > 0 and blocks_since_pool >= pool_every:
                features.append(nn.MaxPool2d(kernel_size=2, stride=2))
                blocks_since_pool = 0

        # Ensure some spatial normalization regardless of conv stack
        features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_ch, max(num_classes, 1)),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
