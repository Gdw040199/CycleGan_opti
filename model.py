# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision.transforms import functional as F_vision

__all__ = [
    "PathDiscriminator", "CycleNet",
    "path_discriminator", "cyclenet",
]


class PathDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            image_size: int = 70,
    ) -> None:
        super(PathDiscriminator, self).__init__()
        self.image_size = image_size

        # Initial convolution block similar to conv1 in the Discriminator
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            self.conv_block(channels, channels)  # Using conv_block style from Discriminator
        )

        # Progressively deeper convolution layers
        self.conv2 = self.conv_block(channels, channels * 2)
        self.conv3 = self.conv_block(channels * 2, channels * 3)
        self.conv4 = self.conv_block(channels * 3, channels * 4)
        self.conv5 = self.conv_block(channels * 4, channels * 5)

        # Final convolution layer similar to conv6
        self.conv6 = nn.Sequential(
            nn.Conv2d(channels * 5, channels * 5, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(channels * 5, channels * 5, kernel_size=3, stride=1, padding=1),
            nn.ELU(True)
        )

        # Embedding layers similar to Discriminator's structure
        self.embed1 = nn.Linear(channels * 5 * 8 * 8, 64)
        self.embed2 = nn.Linear(64, channels * 8 * 8)

        # Deconvolution layers for up-sampling
        self.deconv1 = self.deconv_block(channels, channels)
        self.deconv2 = self.deconv_block(channels, channels)
        self.deconv3 = self.deconv_block(channels, channels)
        self.deconv4 = self.deconv_block(channels, channels)
        self.deconv5 = self.deconv_block(channels, channels)

        self.deconv6 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F_vision.center_crop(x, [self.image_size, self.image_size])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # Flatten and apply embedding layers
        x = x.view(x.size(0), -1)
        x = self.embed1(x)
        x = self.embed2(x)
        x = x.view(x.size(0), self.ndf, 8, 8)

        # Deconvolution for up-sampling
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()

        # Flatten height and width to create a sequence-like input for attention
        x = x.view(batch_size, channels, height * width).permute(2, 0, 1)  # (sequence_length, batch_size, channels)

        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(x, x, x)

        # Reshape the output back to (batch_size, channels, height, width)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, channels, height, width)

        return attn_output

class CycleNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
    ) -> None:
        super(CycleNet, self).__init__()

        use_bias = True  # 因为 nn.InstanceNorm2d 通常在没有偏置时表现更好

        self.pad = nn.ReflectionPad2d(3)

        # 下采样部分
        self.Down_conv1 = nn.Conv2d(in_channels, channels, kernel_size=7, padding=0, bias=use_bias)
        self.conv_norm = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(True)

        self.Down_conv2 = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.SA = SelfAttention(channels * 2, 3)

        self.Down_conv3 = nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1, bias=use_bias)
        self.Sa_block_3 = _ResidualBlock(channels * 4)

        # 残差块部分
        self.resnet = _ResidualBlock(channels * 4)

        # 上采样部分
        self.Up_conv1 = nn.ConvTranspose2d(channels * 4 * 2, channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)
        self.Up_conv2 = nn.ConvTranspose2d(channels * 2 * 2, channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)

        self.Up_conv3 = nn.Conv2d(channels * 2, out_channels, kernel_size=7, padding=0)
        self.tan = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        # 下采样
        x1 = self.relu(self.conv_norm(self.Down_conv1(self.pad(x))))
        x2 = self.relu(self.conv_norm(self.Down_conv2(x1)))
        x3 = self.relu(self.conv_norm(self.Down_conv3(x2)))

        # 残差块
        # 直接在 forward 中手动实现残差块
        residual = x3  # 保存输入

        # 第一个卷积层
        out = nn.Conv2d(x3.size(1), x3.size(1), kernel_size=3, padding=1, bias=False)(x3)
        out = nn.InstanceNorm2d(x3.size(1))(out)
        out = nn.ReLU(inplace=True)(out)

        # 第二个卷积层
        out = nn.Conv2d(out.size(1), out.size(1), kernel_size=3, padding=1, bias=False)(out)
        out = nn.InstanceNorm2d(out.size(1))(out)

        # 残差连接
        out += residual
        x4 = nn.ReLU(inplace=True)(out)

        # 上采样并进行跳跃连接
        x = torch.cat([x4, x3], dim=1)
        x = self.relu(self.conv_norm(self.Up_conv1(x)))

        x = torch.cat([x, x2], dim=1)
        x = self.relu(self.conv_norm(self.Up_conv2(x)))

        x = torch.cat([x, x1], dim=1)
        x = self.tan(self.Up_conv3(self.pad(x)))

        return x


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(_ResidualBlock, self).__init__()

        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (0, 0)),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.res(x)

        x = torch.add(x, identity)

        return x


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def path_discriminator() -> PathDiscriminator:
    model = PathDiscriminator(3, 3, 64, 70)
    model.apply(_weights_init)

    return model


def cyclenet() -> CycleNet:
    model = CycleNet(3, 3, 64)
    model.apply(_weights_init)

    return model
