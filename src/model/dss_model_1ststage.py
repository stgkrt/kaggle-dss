# 1D CNNのtime series segmentationモデル

import torch
import torch.nn as nn
from dss_model import LSTMFeatureExtractor


class DoubleConvBlock(nn.Module):
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
        x = self.maxpool(x)
        return x


# 1DCNNのエンコーダモデル
class DenseLSTM(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv1d(
                config.input_channels,
                config.embedding_base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(config.embedding_base_channels),
            nn.ReLU(),
        )

        self.dense_conv = nn.ModuleList()
        self.dense_conv_kernel_size_list = config.enc_kernelsize_list
        for kernel_size in self.dense_conv_kernel_size_list:
            # 全ての出力サイズが1/12になるようにpaddingを調整
            padding_size = int((kernel_size - 1) / 2)
            self.dense_conv.append(
                nn.Sequential(
                    nn.Conv1d(
                        config.input_channels,
                        config.embedding_base_channels,
                        kernel_size=kernel_size,
                        stride=12,
                        padding=padding_size,
                    ),
                    nn.BatchNorm1d(config.embedding_base_channels),
                    nn.ReLU(),
                )
            )
        self.lstm = nn.Sequential(
            LSTMFeatureExtractor(config),
            nn.Conv1d(
                config.embedding_base_channels * 2,
                config.embedding_base_channels,
                kernel_size=12,
                stride=12,
            ),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                config.embedding_base_channels
                * (len(self.dense_conv_kernel_size_list) + 1),
                config.embedding_base_channels,
                kernel_size=1,
                stride=1,
            ),
        )
        self.encoder_blocks = nn.Sequential(
            DoubleConvBlock(
                config.embedding_base_channels,
                config.embedding_base_channels * 2,
                pool_kernel_size=8,
                pool_stride=8,
            ),
            DoubleConvBlock(
                config.embedding_base_channels * 2,
                config.embedding_base_channels * 4,
                pool_kernel_size=6,
                pool_stride=6,
            ),
            DoubleConvBlock(
                config.embedding_base_channels * 4,
                config.embedding_base_channels * 8,
                pool_kernel_size=4,
                pool_stride=4,
            ),
        )

    def forward(self, x):
        lstm_emb = self.lstm(x)
        x = torch.cat([conv(x) for conv in self.dense_conv], dim=1)
        x = torch.cat([x, lstm_emb], dim=1)
        x = self.conv1(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x


class DSS1stUTimeModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.backbone = DenseLSTM(config)
        self.head = nn.Sequential(
            nn.Conv1d(
                config.embedding_base_channels * 8,
                config.embedding_base_channels * 8,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(config.embedding_base_channels * 8),
            nn.ReLU(),
            nn.Conv1d(
                config.embedding_base_channels * 8,
                config.class_output_channels,
                kernel_size=1,
                padding=0,
            ),
        )
        # [batch, 1]出力のclassifier
        self.cls = nn.Sequential(
            nn.Linear(7, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        class_output = self.cls(x.squeeze(1))
        return class_output


def get_model(config):
    model = DSS1stUTimeModel(config)
    return model


if __name__ == "__main__":

    class config:
        input_channels = 2
        embedding_base_channels = 16
        enc_kernelsize_list = [12, 24, 48, 96, 192, 384]
        class_output_channels = 1
        output_channels = 1
        lstm_num_layers = 2
        ave_kernel_size = 301
        maxpool_kernel_size = 11
        batch_size = 32

    x = torch.randn(
        config.batch_size, config.input_channels, 17280
    )  # (batch_size, input_channels, seq_len)

    print("input shape: ", x.shape)
    model = get_model(config)
    output = model(x)
    print("output shape", output.shape)
