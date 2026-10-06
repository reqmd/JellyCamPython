import torch.nn as nn

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