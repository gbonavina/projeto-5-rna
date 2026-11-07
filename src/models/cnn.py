import torch.nn as nn

# cnn for the cifar-10 dataset
class CNN(nn.Module):
    def __init__(self, input_channels=3, num_conv_layers=3, kernel_size=3,
                 padding=None, stride=1, num_classes=10, dropout=0.5, base_channels=16):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # "same" padding

        layers = []
        in_ch = input_channels
        for i in range(num_conv_layers):
            out_ch = base_channels * (2 ** i)  # double the number of channels with each layer
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)  # always 1x1, regardless of depth
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_ch // 2, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)  
