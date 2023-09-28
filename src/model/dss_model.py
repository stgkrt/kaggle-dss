# 1D CNNのtime series segmentationモデル

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 32,
        conv_kernel_size: int = 5,
        conv_padding: str = "same",
        pool_kernel_size: int = 10,
        pool_stride: int = 10,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, conv_kernel_size, padding=conv_padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, conv_kernel_size, padding=conv_padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        pooled = self.maxpool(x)
        return x, pooled


# 1DCNNのエンコーダモデル
class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 16,
    ) -> None:
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            EncoderBlock(
                input_channels, input_channels * 2, pool_kernel_size=10, pool_stride=10
            ),
            EncoderBlock(
                input_channels * 2,
                input_channels * 4,
                pool_kernel_size=8,
                pool_stride=8,
            ),
            EncoderBlock(
                input_channels * 4,
                input_channels * 8,
                pool_kernel_size=6,
                pool_stride=6,
            ),
            EncoderBlock(
                input_channels * 8,
                input_channels * 16,
                pool_kernel_size=4,
                pool_stride=4,
            ),
        )

    def forward(self, x):
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            skip_connection, x = encoder_block(x)
            skip_connections.append(skip_connection)
        return x, skip_connections


class DSS_UTime_Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.conv0 = nn.Conv1d(config.input_channels, 32, 3, padding="same")
        self.encoder = Encoder()

    def forward(self, x):
        x = self.conv0(x)
        x = self.encoder(x)
        return x


if __name__ == "__main__":
    input_channels = 16

    encoder = Encoder(input_channels)
    # encoder = EncoderBlock(input_channels)
    x = torch.randn(1, input_channels, 100000)  # (batch_size, input_channels, seq_len)
    # print(x.shape)
    output, skip_connetctions = encoder(x)
    print(output.shape)
    for idx, sk in enumerate(skip_connetctions):
        print(f"skip connection[{idx}]", sk.shape)

    # mp = nn.MaxPool1d(10, stride=10)
    # y = mp(x)
    # print(y.shape)
