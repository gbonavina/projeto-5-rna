import torch.nn as nn


# mlp for the cifar-10 dataset
class MLP(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, num_layers=3, hidden_size=None,
                 num_classes=10, dropout=0.5):
        super().__init__()
        if not hidden_size:
            # halve the width at each layer: 1024, 512, 256, ... (min 64)
            hidden_size = [max(1024 // (2 ** i), 64) for i in range(num_layers)]
        elif len(hidden_size) != num_layers:
            raise ValueError("Length of hidden_size must match num_layers")

        layers = [nn.Flatten()]  # (N, 3, 32, 32) -> (N, 3072)
        in_features = input_size
        for out_features in hidden_size:
            layers += [
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_features = out_features

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)  # logits -> use with nn.CrossEntropyLoss
