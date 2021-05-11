import torch
import torch.nn as nn
from math import ceil

base_model = [
    # using table 1 from the official paper
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2), # alpha, beta, gamma, depth = alpha** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride , padding, groups= 1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.silu = nn.SiLU() # SiLU same as swish

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class SqueezeExecution(nn.Module): # to compute attenstion score for each of the channel
    def __init__(self, input_channels, reduced_dim):
        super(SqueezeExecution, self).__init__()
        self.att_sc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x W x H = C x 1 x 1
            nn.Conv2d(input_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, input_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att_sc(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride,
                 padding, expand_ratio, reduction=4, survival_prob=0.8):
        # reduction for squeezexcitation and survival for stochastic depth
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = input_channels == output_channels and stride == 1
        hidden_dim = input_channels * expand_ratio
        self.expand = input_channels != hidden_dim
        reduced_dim = int(input_channels/reduction)

        if self.expand:
            self.expand_conv = CNNBlock(input_channels, hidden_dim, kernel=3, stride=1, padding=1,)

        self.conv = nn.Sequential(
                CNNBlock(hidden_dim, hidden_dim, kernel, stride, padding, groups=hidden_dim),
                SqueezeExecution(hidden_dim, reduced_dim),
                nn.Conv2d(hidden_dim, output_channels, 1,  bias=False),
                nn.BatchNorm2d(output_channels),
            )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channel = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channel)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channel, num_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channel):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(in_channels, out_channels, expand_ratio=expand_ratio,
                                          stride=stride if layer == 0 else 1,
                                          kernel=kernel_size,
                                          padding=kernel_size//2) # if k=1, p=0 or k=3, p=1
                )
                in_channels = out_channels
        features.append(
            CNNBlock(in_channels, last_channel, kernel=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


def testing():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "b1"
    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes = 8, 12
    x = torch.randn((num_examples, 3, res, res)).to(device)
    model = EfficientNet(
        version=version,
        num_classes=num_classes
    ).to(device)

    print(model(x).shape)


testing()

# ref https://github.com/aladdinpersson/Machine-Learning-Collections