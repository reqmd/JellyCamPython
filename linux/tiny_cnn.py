"""Компактная CNN для бинарной (и не только) классификации низкоразрешённых объектов.

Структура намеренно ограничена операторами, которые хорошо ложатся на NPU
(в частности RKNPU): Conv2d 3x3 -> BatchNorm2d -> ReLU -> MaxPool2d,
глобальный average pooling и один Linear. Никаких SE-блоков, hardswish,
channel shuffle, LayerNorm и depthwise-свёрток.

Пример:
    from tiny_cnn import TinyCNN, tiny_cnn_s

    model = tiny_cnn_s(num_classes=2)
    logits = model(torch.randn(1, 3, 48, 48))
"""

from typing import Sequence

import torch
import torch.nn as nn

__all__ = ["ConvBlock", "TinyCNN", "tiny_cnn_xs", "tiny_cnn_s", "tiny_cnn_m"]

class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=1 if stride == 1 else 0,
                groups=in_channels
            ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU()
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)

class MobileNet(nn.Module):
    def __init__(self, alpha=1, num_classes=2):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.layer_0 = nn.Conv2d(
            3,
            int(32 * self.alpha),
            stride=2,
            kernel_size=(3, 3),
            padding=1
        )
        self.layer_1 = self.__make_layer(int(32 * self.alpha), int(64 * self.alpha), 1)
        self.layer_2 = self.__make_layer(int(64 * self.alpha), int(128 * self.alpha), 1)
        self.layer_3 = self.__make_layer(int(128 * self.alpha), int(128 * self.alpha), 2)  # Даунсэмплинг
        self.layer_4 = self.__make_layer(int(128 * self.alpha), int(256 * self.alpha), 1)
        self.layer_5 = self.__make_layer(int(256 * self.alpha), int(256 * self.alpha), 2)  # Даунсэмплинг
        self.layer_6 = self.__make_layer(int(256 * self.alpha), int(512 * self.alpha), 1)
        self.layer_7 = self.__make_layer(int(512 * self.alpha), int(512 * self.alpha), 2)  # Даунсэмплинг
        self.layer_8 = self.__make_layer(int(512 * self.alpha), int(512 * self.alpha), 1)
        self.layer_9 = self.__make_layer(int(512 * self.alpha), int(512 * self.alpha), 1)
        self.layer_10 = self.__make_layer(int(512 * self.alpha), int(512 * self.alpha), 1)
        self.layer_11 = self.__make_layer(int(512 * self.alpha), int(512 * self.alpha), 1)
        self.layer_12 = self.__make_layer(int(512 * self.alpha), int(1024 * self.alpha), 1)
        self.layer_13 = self.__make_layer(int(1024 * self.alpha), int(1024 * self.alpha), 2)  # Даунсэмплинг
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(1024 * self.alpha), num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        return MobileNetBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)

    def forward(self, X):
        out = self.layer_0(X)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        out = self.layer_7(out)
        out = self.layer_8(out)
        out = self.layer_9(out)
        out = self.layer_10(out)
        out = self.layer_11(out)
        out = self.layer_12(out)
        out = self.layer_13(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        return self.fc(out)


class ConvBlock(nn.Module):
    """Conv3x3 -> BN -> ReLU (-> MaxPool2d(2), опционально).

    BatchNorm сворачивается в свёртку при экспорте, поэтому в инференсе
    блок превращается в один conv + relu.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class TinyCNN(nn.Module):
    """Небольшая свёрточная сеть с настраиваемой шириной и глубиной.

    Args:
        num_classes: число классов на выходе. Для бинарной задачи
            рекомендуется 2 (CrossEntropyLoss); 1 даст один логит под
            BCEWithLogitsLoss.
        in_channels: каналов на входе (3 для RGB, 1 для grayscale).
        widths: число каналов в каждом conv-блоке. Длина задаёт глубину.
        dropout: вероятность dropout перед классификатором. 0.0 — выключен.
        pool_last: применять ли MaxPool в последнем блоке. False экономит
            немного пространственного разрешения при малом входе.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        widths: Sequence[int] = (16, 32, 64),
        dropout: float = 0.0,
        pool_last: bool = False,
    ) -> None:
        super().__init__()
        if len(widths) == 0:
            raise ValueError("widths не может быть пустым")
        if num_classes < 1:
            raise ValueError("num_classes должно быть >= 1")

        self.num_classes = num_classes

        blocks = []
        ch = in_channels
        for i, w in enumerate(widths):
            is_last = i == len(widths) - 1
            blocks.append(ConvBlock(ch, w, pool=pool_last or not is_last))
            ch = w
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(ch, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Эмбеддинг до классификатора — удобно для t-SNE или kNN поверх."""
        return self.flatten(self.pool(self.features(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.forward_features(x)))

    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )


def tiny_cnn_xs(num_classes: int = 2, **kwargs) -> TinyCNN:
    """~7K параметров. Для входа 32x32 и очень простого сигнала."""
    kwargs.setdefault("widths", (8, 16, 32))
    return TinyCNN(num_classes=num_classes, **kwargs)


def tiny_cnn_s(num_classes: int = 2, **kwargs) -> TinyCNN:
    """~25K параметров. Разумный дефолт для входа 48x48."""
    kwargs.setdefault("widths", (16, 32, 64))
    return TinyCNN(num_classes=num_classes, **kwargs)


def tiny_cnn_m(num_classes: int = 2, **kwargs) -> TinyCNN:
    """~100K параметров. Если классы различаются не только цветом."""
    kwargs.setdefault("widths", (32, 64, 128))
    return TinyCNN(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    for name, fn in [("xs", tiny_cnn_xs), ("s", tiny_cnn_s), ("m", tiny_cnn_m)]:
        model = fn(num_classes=2).eval()
        x = torch.randn(2, 3, 48, 48)
        y = model(x)
        print(f"tiny_cnn_{name}: params={model.num_parameters():>7,}  out={tuple(y.shape)}")