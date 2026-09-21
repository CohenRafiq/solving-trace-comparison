from typing import cast

from torch import Tensor, nn

from neural_models_old.models.base import BaseModel


class ConvNeuralNetwork(BaseModel):
    def __init__(
        self,
        input_channels: int,
        output_size: int,
        hidden_channels: list[int],
        kernel_size: int | tuple[int, int],
        dims: int = 2,
        fc_hidden_size: int = 256,
        fc_num_layers: int = 2,
        input_shape: tuple[int, ...] | None = None,
        pool_every: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__(output_size=output_size)
        assert dims in (1, 2), "dims must be 1 or 2"

        conv_blocks: list[nn.Module] = []
        in_ch = input_channels

        for i, out_ch in enumerate(hidden_channels):
            if dims == 1:
                k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
                p = k // 2
                conv_blocks += [
                    nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                ]
                if pool_every and (i + 1) % pool_every == 0:
                    conv_blocks.append(nn.MaxPool1d(kernel_size=2))
            else:
                pad2 = cast(
                    int | tuple[int, int],
                    (
                        kernel_size
                        if isinstance(kernel_size, int)
                        else tuple(x // 2 for x in kernel_size)
                    ),
                )
                conv_blocks += [
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
                if pool_every and (i + 1) % pool_every == 0:
                    conv_blocks.append(nn.MaxPool2d(kernel_size=2))
            in_ch = out_ch

        conv_blocks.append(nn.AdaptiveAvgPool1d(1) if dims == 1 else nn.AdaptiveAvgPool2d((1, 1)))
        self.conv = nn.Sequential(*conv_blocks)

        fc_in = in_ch
        fc_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(fc_in, fc_hidden_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            fc_layers.append(nn.Dropout(dropout))

        for _ in range(fc_num_layers - 1):
            fc_layers += [
                nn.Linear(fc_hidden_size, fc_hidden_size),
                nn.BatchNorm1d(fc_hidden_size),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                fc_layers.append(nn.Dropout(dropout))

        fc_layers += [nn.Linear(fc_hidden_size, output_size), nn.Sigmoid()]
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.fc(out)
        return out
