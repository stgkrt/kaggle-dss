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
        input_channels: int = 1,
        embedding_base_channel: int = 16,
    ) -> None:
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            EncoderBlock(
                input_channels,
                embedding_base_channel,
                pool_kernel_size=10,
                pool_stride=10,
            ),
            EncoderBlock(
                embedding_base_channel,
                embedding_base_channel * 2,
                pool_kernel_size=8,
                pool_stride=8,
            ),
            EncoderBlock(
                embedding_base_channel * 2,
                embedding_base_channel * 4,
                pool_kernel_size=6,
                pool_stride=6,
            ),
            EncoderBlock(
                embedding_base_channel * 4,
                embedding_base_channel * 8,
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


class NeckBlock(nn.Module):
    def __init__(self, input_channels) -> None:
        super().__init__()
        self.neck_blocks = nn.Sequential(
            nn.Conv1d(input_channels, input_channels * 2, 3, padding="same"),
            nn.BatchNorm1d(input_channels * 2),
            nn.ReLU(),
            nn.Conv1d(input_channels * 2, input_channels * 2, 3, padding="same"),
            nn.BatchNorm1d(input_channels * 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.neck_blocks(x)
        return x


class CropLayer(nn.Module):
    def __init__(self, crop_rate=2):
        super().__init__()

    def forward(self, x, skip_connection):
        crop_len = int(skip_connection.shape[-1] * 0.8)  # 長さはお気持ち
        if crop_len < x.shape[-1]:
            x = x[:, :, crop_len:-crop_len]
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 16,
        conv_kernel_size: int = 5,
        conv_padding: str = "same",
        upsample_kernel_size: int = 10,
        upsample_size: int = 10,
    ) -> None:
        super().__init__()
        self.crop_layer = CropLayer(upsample_kernel_size)
        self.upsample_conv = nn.Sequential(
            nn.Upsample(size=upsample_size, mode="nearest"),
            nn.Conv1d(
                in_channels,
                in_channels // 2,
                conv_kernel_size,
                padding=conv_padding,
            ),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, conv_kernel_size, padding=conv_padding
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                out_channels, out_channels, conv_kernel_size, padding=conv_padding
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, skip_connection):
        x = self.crop_layer(x, skip_connection)
        x = self.upsample_conv(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.decoder_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 16,
        skip_connections_length: list = [20, 125, 1000, 10000],
    ) -> None:
        super().__init__()
        self.decoder_blocks = nn.Sequential(
            DecoderBlock(
                input_channels * 16,
                input_channels * 8,
                conv_kernel_size=4,
                upsample_kernel_size=4,
                upsample_size=skip_connections_length[0],
            ),
            DecoderBlock(
                input_channels * 8,
                input_channels * 4,
                conv_kernel_size=5,
                upsample_kernel_size=5,
                upsample_size=skip_connections_length[1],
            ),
            DecoderBlock(
                input_channels * 4,
                input_channels * 2,
                conv_kernel_size=5,
                upsample_kernel_size=8,
                upsample_size=skip_connections_length[2],
            ),
            DecoderBlock(
                input_channels * 2,
                input_channels,
                conv_kernel_size=5,
                upsample_kernel_size=10,
                upsample_size=skip_connections_length[3],
            ),
        )

    def forward(self, x, skip_connections):
        for idx, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, skip_connections[-idx - 1])
        return x


class DetectPeak(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        maxpool_kernel_size = 11  # 奇数じゃないと同じ長さで出力できない
        maxpool_stride = 1
        # 入力サイズと出力サイズが一致するようにpaddingを調整
        maxpool_padding = int((maxpool_kernel_size - maxpool_stride) / 2)
        self.max_pool = nn.MaxPool1d(
            kernel_size=maxpool_kernel_size,
            stride=maxpool_stride,
            padding=maxpool_padding,
        )

    def forward(self, x):
        max_pooled = self.max_pool(x)
        peaks_idx = (x == max_pooled).float()
        peaks_value = x * peaks_idx
        return peaks_value


class DSSUTimeModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = Encoder(config.input_channels, config.embedding_base_channels)
        self.neck_conv = NeckBlock(config.embedding_base_channels * 8)
        skip_connections_length = self._get_skip_connections_length()
        self.decoder = Decoder(config.embedding_base_channels, skip_connections_length)
        self.head = nn.Sequential(
            nn.Conv1d(
                config.embedding_base_channels,
                config.class_output_channels,
                kernel_size=5,
                padding="same",
            ),
            # nn.Softmax(dim=1),
            nn.Sigmoid(),
        )
        self.class_avg_pool = nn.AvgPool1d(
            kernel_size=11,
            stride=1,
            padding=5,
        )

    def _get_skip_connections_length(self):
        # return [20, 125, 1000, 10000]
        return [36, 216, 1728, 17280]

    def forward(self, x):
        x, skip_connetctions = self.encoder(x)
        x = self.neck_conv(x)
        x = self.decoder(x, skip_connetctions)
        class_output = self.head(x)
        class_output = self.class_avg_pool(class_output)
        return class_output


class DSSEventoutUTimeModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = Encoder(config.input_channels, config.embedding_base_channels)
        self.neck_conv = NeckBlock(config.embedding_base_channels * 8)
        skip_connections_length = self._get_skip_connections_length()
        self.decoder = Decoder(config.embedding_base_channels, skip_connections_length)
        self.head = nn.Sequential(
            nn.Conv1d(
                config.embedding_base_channels,
                config.class_output_channels,
                kernel_size=5,
                padding="same",
            ),
            # nn.Softmax(dim=1),
            nn.Sigmoid(),
        )
        # self.event_detect_conv = nn.Conv1d(
        #     config.class_output_channels,
        #     config.embedding_base_channels,
        #     kernel_size=10,
        #     padding="same",
        # )

        # event detectorから前の層にbackpropagationしないようにする
        # for param in self.event_detect_conv.parameters():
        #     param.requires_grad = False
        self.class_avg_pool = nn.AvgPool1d(
            kernel_size=11,
            stride=1,
            padding=5,
        )

        self.event_detector = nn.Sequential(
            # nn.Conv1d(
            #     config.class_output_channels,
            #     config.event_output_channels,
            #     kernel_size=5,
            #     padding="same",
            # ),
            # あとでこういうのも試す
            nn.Conv1d(
                config.class_output_channels,
                16,
                kernel_size=3,
                padding="same",
            ),
            nn.Conv1d(
                16,
                32,
                kernel_size=3,
                padding="same",
            ),
            nn.Conv1d(
                32,
                16,
                kernel_size=3,
                padding="same",
            ),
            nn.Conv1d(
                16,
                1,
                kernel_size=3,
                padding="same",
            ),
            # [batch_size, 2, seq_len]. seq_lenの方向でsoftmaxをとる
            # nn.Softmax(dim=2),
            # seires内に複数のイベントがある可能性があるためsigmoidにしておく
            # nn.Sigmoid(),
        )
        self.event_onset_head = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )
        self.event_wakeup_head = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )
        self.detect_peak = DetectPeak()

    def _get_skip_connections_length(self):
        # return [20, 125, 1000, 10000]
        return [36, 216, 1728, 17280]

    def forward(self, x):
        x, skip_connetctions = self.encoder(x)
        x = self.neck_conv(x)
        x = self.decoder(x, skip_connetctions)
        class_output = self.head(x)
        # class_output = self.class_avg_pool(class_output)
        # event = self.event_detect_conv(x)
        # class_output_detach = (
        #     class_output.detach()
        # )  # event detectorから前の層にbackpropagationしないようにする
        with torch.no_grad():
            class_output_detach = class_output.detach()
            class_output_detach = self.class_avg_pool(class_output_detach)
        # class_output_detach = class_output
        # class_output_detach = self.class_avg_pool(class_output_detach)
        shifted_class_output = torch.roll(class_output_detach, 1, dims=2)
        invshifted_class_output = torch.roll(class_output_detach, -1, dims=2)
        diff_class = shifted_class_output - invshifted_class_output
        # event_emb = diff_class
        event_emb = self.event_detector(diff_class)
        event_onset = self.event_onset_head(event_emb.view(-1, 1))
        event_wakeup = self.event_wakeup_head(event_emb.view(-1, 1))
        event_output = torch.cat(
            [event_onset.view(-1, 1, 17280), event_wakeup.view(-1, 1, 17280)], dim=1
        )
        # event_output = self.event_detector(diff_class)
        # event_output = self.event_detector(class_output_detach)
        # event = self.detect_peak(event)
        return class_output, event_output


def get_model(config):
    if config.model_type == "event_output":
        model = DSSEventoutUTimeModel(config)
    else:
        model = DSSUTimeModel(config)
    return model


if __name__ == "__main__":
    input_channels = 16

    class config:
        input_channels = 1
        embedding_base_channels = 16
        class_output_channels = 1
        event_output_channels = 2

    # neck_input = torch.randn(1, config.embedding_base_channel * 16, 20)
    # print("input shape", neck_input.shape)
    # neck = NeckBlock(config.embedding_base_channel * 16)
    # y = neck(neck_input)

    x = torch.randn(
        1, config.input_channels, 17280
    )  # (batch_size, input_channels, seq_len)

    print("encoder")
    encoder = Encoder(config.input_channels, config.embedding_base_channels)
    y, skip_connetctions = encoder(x)
    print(y.shape)
    for skip_connetction in skip_connetctions:
        print(skip_connetction.shape)

    print("dss model")
    model = DSSUTimeModel(config)
    class_out, event_out = model(x)
    print("class_out", class_out.shape)
    print("event_out", event_out.shape)
