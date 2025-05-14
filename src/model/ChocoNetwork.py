# Contains our model for the chocolate recognition task.
import torch.nn as nn

class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        # first conv layer, downsampling if stride > 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)

        return x

class CountHead(nn.Module):
    """
    in_channels : #channels coming from the encoder
    hidden      : size of the intermediate layer (default 512)
    n_classes   : how many categories we count
    """
    def __init__(self, in_channels=512, hidden=512, n_classes=3, p_drop=0.2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)           # (B, C, H, W) → (B, C, 1, 1)

        self.regressor = nn.Sequential(              # (B, C) → (B, n_classes)
            nn.Flatten(1),                           # (B, C, 1, 1) → (B, C)
            nn.Linear(in_channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes, bias=True)  # final counts (float)
        )

    def forward(self, x):
        x = self.gap(x)
        return self.regressor(x)                     # shape (B, n_classes)


class ChocoNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, 2, stride=2)

        self.head = CountHead(in_channels=512, n_classes=13)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.head(x)

        return x