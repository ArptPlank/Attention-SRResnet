from torch import nn
import torch
import math
class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.attention = CBAM(in_channels=64)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        output = self.attention(output)
        return output


class _NetG(nn.Module):
    def __init__(self):
        super(_NetG, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.Leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmid = nn.Sigmoid()
        self.residual = self.make_layer(_Residual_Block, 16)
        #self.attention_residual = self.make_layer_attention(_Residual_Block, 8)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(2),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.Leakyrelu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        #out = self.attention_residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        out = self.sigmid(out)
        #out = torch.clamp(out, 0, 1)
        return out

    def make_layer_attention(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


class ChannelAttention(nn.Module):
    """
    CBAM混合注意力机制的通道注意力
    """
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        channel_inter = max(in_channels // ratio, 1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, channel_inter, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_inter, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x

class SpatialAttention(nn.Module):
    """
    CBAM混合注意力机制的空间注意力
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x

class CBAM(nn.Module):
    """
    CBAM混合注意力机制
    """

    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        value = self.value(x).view(batch, -1, height * width)

        attention = self.softmax(torch.bmm(query, key.transpose(1, 2)))
        out = torch.bmm(value, attention).view(batch, channels, height, width)

        return out + x

class AttentionResNet(nn.Module):
    def __init__(self, channels):
        super(AttentionResNet, self).__init__()
        self.layer1 = _Residual_Block()
        self.attention = SelfAttention(channels)
        self.layer2 = _Residual_Block()

    def forward(self, x):
        x = self.layer1(x)
        x = self.attention(x)
        x = self.layer2(x)
        return x


class _NetD(nn.Module):
    def __init__(self):
        super(_NetD, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )

        # Calculate the size of the flattened features after all conv and pooling layers
        # Here, we calculate it for an input of size 512x320
        final_layer_output_size = 256  # This is the number of output channels from the last conv layer
        h, w = 320 // 16, 512 // 16  # We divide by 16 because we applied MaxPool2d four times

        fc_input_features = final_layer_output_size * h * w
        self.fc1 = nn.Linear(fc_input_features, 1024)
        self.fc2 = nn.Linear(1024, 2)
        self.sigmoid = nn.Sigmoid()
        self.LeakyReLU = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        #x = x.view(-1, 1).squeeze(1)
        return x

if __name__ == "__main__":
    # 创建模型实例
    model = _NetD()
    print(sum(p.numel() for p in model.parameters()))
    # 创建一个模拟输入，大小为 [1, 3, 512, 320]，代表批大小为1，通道数为3，高宽为96的图像
    input_tensor = torch.randn(5, 3, 512, 320)
    # 将模型和输入移至同一设备（如有GPU，则使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # 执行前向传播
    output = model(input_tensor)

    # 打印输出形状和值
    print("Output shape:", output.shape)
    print("Output value:", output)