import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Część “feature extractor”:
        # kolejne Conv + ReLU + Pool zmniejszają rozdzielczość i zwiększają liczbę kanałów.
        self.features = nn.Sequential(
            # 3 x 224 x 224
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 x 112 x 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 56 x 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 x 28 x 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # AdaptiveAvgPool2d((1,1)) sprowadza mapę cech do stałego rozmiaru 1x1,
            nn.AdaptiveAvgPool2d((1, 1)),  # 256 x 1 x 1
        )

        # Część klasyfikacyjna:
        # Flatten zamienia 256x1x1 na wektor 256.
        # MLP klasyfikację na num_classes.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # Dropout pomaga w regularizacji (mniej overfitu).
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward: obraz -> cechy -> logits
        x = self.features(x)
        return self.classifier(x)