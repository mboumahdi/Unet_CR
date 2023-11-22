import torch
import torch.nn as nn
import timm


class Sentinel12UNet(nn.Module):
    def __init__(self, include_s1=True, include_s2_20m=True, include_dem_20m=True):
        super(Sentinel12UNet, self).__init__()

        s2_10m_channels = 4
        s1_10m_channels = 2 if include_s1 else 0
        s2_20m_in_channels = 6 if include_s2_20m else 0
        dem_20m_in_channels = 1 if include_dem_20m else 0

        in_channels = s2_10m_channels + s1_10m_channels + s2_20m_in_channels
        out_channels = s2_10m_channels + s2_20m_in_channels

        # U-Net encoder
        self.conv1 = self._make_conv_block(in_channels, 64)
        self.conv2 = self._make_conv_block(64 + dem_20m_in_channels, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)
        self.conv5 = self._make_conv_block(512, 1024)

        # U-Net decoder
        self.upconv1 = self._make_upconv(1024, 512)
        self.upconv2 = self._make_upconv(512 + 512, 256)
        self.upconv3 = self._make_upconv(256 + 256, 128)
        self.upconv4 = self._make_upconv(128 + 128, 64)

        # Final output
        self.out = nn.Conv2d(64 + 64, out_channels, kernel_size=1)

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _make_upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, s2_10, s2_20, s1_10, dem_20):
        # ResNet encoder
        x = torch.cat([s2_10, s2_20, s1_10], 1)

        # Encoder
        conv1 = self.conv1(x)
        pool1 = nn.MaxPool2d(kernel_size=2)(conv1)
        
        conv2 = self.conv2(torch.cat([pool1, dem_20], 1))
        pool2 = nn.MaxPool2d(kernel_size=2)(conv2)
        conv3 = self.conv3(pool2)
        pool3 = nn.MaxPool2d(kernel_size=2)(conv3)
        conv4 = self.conv4(pool3)
        pool4 = nn.MaxPool2d(kernel_size=2)(conv4)
        conv5 = self.conv5(pool4)

        # Decoder
        upconv1 = self.upconv1(conv5)
        concat1 = torch.cat([upconv1, conv4], dim=1)
        upconv2 = self.upconv2(concat1)
        concat2 = torch.cat([upconv2, conv3], dim=1)
        upconv3 = self.upconv3(concat2)
        concat3 = torch.cat([upconv3, conv2], dim=1)
        upconv4 = self.upconv4(concat3)
        concat4 = torch.cat([upconv4, conv1], dim=1)
        # Output
        out = self.out(concat4)
        return out

class Sentinel12UNetTopo(nn.Module):
    def __init__(self, include_s1=True, include_s2_20m=True, include_dem_20m=True):
        super(Sentinel12UNetTopo, self).__init__()

        s2_10m_channels = 4
        s1_10m_channels = 2 if include_s1 else 0
        s2_20m_in_channels = 6 if include_s2_20m else 0
        dem_20m_in_channels = 1 if include_dem_20m else 0
        aspect_20m_in_channels =1
        slope_20m_in_channels = 1
        hillshade_20m_in_channels = 1
        in_channels = s2_10m_channels + s1_10m_channels + s2_20m_in_channels + aspect_20m_in_channels +slope_20m_in_channels +hillshade_20m_in_channels # add aspect, slope, aspect
        out_channels = s2_10m_channels + s2_20m_in_channels

        # U-Net encoder
        self.conv1 = self._make_conv_block(in_channels, 64)
        self.conv2 = self._make_conv_block(64 + dem_20m_in_channels, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)
        self.conv5 = self._make_conv_block(512, 1024)

        # U-Net decoder
        self.upconv1 = self._make_upconv(1024, 512)
        self.upconv2 = self._make_upconv(512 + 512, 256)
        self.upconv3 = self._make_upconv(256 + 256, 128)
        self.upconv4 = self._make_upconv(128 + 128, 64)

        # Final output
        self.out = nn.Conv2d(64 + 64, out_channels, kernel_size=1)

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _make_upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, s2_10, s2_20, s1_10, aspect_20, slope_20, hillshade_20, dem_20):
        # ResNet encoder
        x = torch.cat([s2_10, s2_20, s1_10 ,aspect_20, slope_20, hillshade_20], 1)

        # Encoder
        conv1 = self.conv1(x)
        pool1 = nn.MaxPool2d(kernel_size=2)(conv1)
        
        conv2 = self.conv2(torch.cat([pool1, dem_20], 1))
        pool2 = nn.MaxPool2d(kernel_size=2)(conv2)
        conv3 = self.conv3(pool2)
        pool3 = nn.MaxPool2d(kernel_size=2)(conv3)
        conv4 = self.conv4(pool3)
        pool4 = nn.MaxPool2d(kernel_size=2)(conv4)
        conv5 = self.conv5(pool4)

        # Decoder
        upconv1 = self.upconv1(conv5)
        concat1 = torch.cat([upconv1, conv4], dim=1)
        upconv2 = self.upconv2(concat1)
        concat2 = torch.cat([upconv2, conv3], dim=1)
        upconv3 = self.upconv3(concat2)
        concat3 = torch.cat([upconv3, conv2], dim=1)
        upconv4 = self.upconv4(concat3)
        concat4 = torch.cat([upconv4, conv1], dim=1)
        # Output
        out = self.out(concat4)
        return out
    
    
class Sentinel12UNetMultiple(nn.Module):
    def __init__(self, include_s1=True, include_s2_20m=True, include_dem_20m=True):
        super(Sentinel12UNetMultiple, self).__init__()

        s2_10m_channels = 4
        s1_10m_channels = 2 if include_s1 else 0
        s2_20m_in_channels = 6 if include_s2_20m else 0
        dem_20m_in_channels = 1 if include_dem_20m else 0

        in_channels = 3*s2_10m_channels + 3*s1_10m_channels + 3*s2_20m_in_channels
        out_channels = s2_10m_channels + s2_20m_in_channels
        '''
        # U-Net encoder
        self.conv1 = self._make_first_block(in_channels, 64)
        self.conv2 = self._make_conv_block(64 + dem_20m_in_channels, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)
        self.conv5 = self._make_conv_block(512, 512)
        self.conv6 = self._make_conv_block(512, 512)

        # U-Net decoder
        self.upconv1 = self._make_upconv(512+512, 512)
        self.upconv2 = self._make_upconv(512+512, 512)
        self.upconv3 = self._make_upconv(512 + 512, 256)
        self.upconv4 = self._make_upconv(256 + 256, 128)
        self.upconv5 = self._make_upconv(128 + 128, 64)
        # Final output
        self.out = nn.Conv2d(64 + 64, out_channels, kernel_size=5)
        '''
        # U-Net encoder
        self.conv1 = self._make_conv_block(in_channels, 64)
        self.conv2 = self._make_conv_block(64 + dem_20m_in_channels, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)
        self.conv5 = self._make_conv_block(512, 1024)

        # U-Net decoder
        self.upconv1 = self._make_upconv(1024, 512)
        self.upconv2 = self._make_upconv(512 + 512, 256)
        self.upconv3 = self._make_upconv(256 + 256, 128)
        self.upconv4 = self._make_upconv(128 + 128, 64)

        # Final output
        self.out = nn.Conv2d(64 + 64, out_channels, kernel_size=1)

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _make_upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, s2_10, s2_20, s1_10,s2_10_before, s2_20_before, s1_10_before,
                s2_10_after, s2_20_after, s1_10_after, dem_20):
        # ResNet encoder
        x = torch.cat([s2_10, s2_20, s1_10, s2_10_before, s2_20_before, s1_10_before,
                s2_10_after, s2_20_after, s1_10_after], 1)
        
        
        
        '''
        # Encoder
        print(x.shape)
        conv1 = self.conv1(x)
        print(conv1.shape)
        pool1 = nn.MaxPool2d(kernel_size=2)(conv1)
        print(pool1.shape)
        print(dem_20.shape)
        conv2 = self.conv2(torch.cat([pool1, dem_20], 1))
        pool2 = nn.MaxPool2d(kernel_size=2)(conv2)
        conv3 = self.conv3(pool2)
        pool3 = nn.MaxPool2d(kernel_size=2)(conv3)
        conv4 = self.conv4(pool3)
        pool4 = nn.MaxPool2d(kernel_size=2)(conv4)
        conv5 = self.conv5(pool4)
        pool5 = nn.MaxPool2d(kernel_size=2)(conv5)
        conv6 = self.conv5(pool5)

        # Decoder
        upconv1 = self.upconv1(conv6)
        concat1 = torch.cat([upconv1, conv5], dim=1)
        upconv2 = self.upconv2(concat1)
        concat2 = torch.cat([upconv2, conv4], dim=1)
        upconv3 = self.upconv3(concat2)
        concat3 = torch.cat([upconv3, conv3], dim=1)
        upconv4 = self.upconv4(concat3)
        concat4 = torch.cat([upconv4, conv2], dim=1)
        upconv5 = self.upconv5(concat3)
        concat5 = torch.cat([upconv5, conv1], dim=1)
        # Output
        out = self.out(concat5)
        return out
        '''
        
        conv1 = self.conv1(x)
        pool1 = nn.MaxPool2d(kernel_size=2)(conv1)
        conv2 = self.conv2(torch.cat([pool1, dem_20], 1))
        pool2 = nn.MaxPool2d(kernel_size=2)(conv2)
        conv3 = self.conv3(pool2)
        pool3 = nn.MaxPool2d(kernel_size=2)(conv3)
        conv4 = self.conv4(pool3)
        pool4 = nn.MaxPool2d(kernel_size=2)(conv4)
        conv5 = self.conv5(pool4)

        # Decoder
        upconv1 = self.upconv1(conv5)
        concat1 = torch.cat([upconv1, conv4], dim=1)
        upconv2 = self.upconv2(concat1)
        concat2 = torch.cat([upconv2, conv3], dim=1)
        upconv3 = self.upconv3(concat2)
        concat3 = torch.cat([upconv3, conv2], dim=1)
        upconv4 = self.upconv4(concat3)
        concat4 = torch.cat([upconv4, conv1], dim=1)
        # Output
        out = self.out(concat4)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # U-Net encoder
        self.conv1 = self._make_conv_block(in_channels, 64)
        self.conv2 = self._make_conv_block(64, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)
        self.conv5 = self._make_conv_block(512, 1024)

        # U-Net decoder
        self.upconv1 = self._make_upconv(1024, 512)
        self.upconv2 = self._make_upconv(512, 256)
        self.upconv3 = self._make_upconv(256, 128)
        self.upconv4 = self._make_upconv(128, 64)

        # Final output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _make_upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )


def forward(self, s2_10, s2_20, s1_10, dem_20):
        # ResNet encoder
        x = torch.cat([s2_10, s2_20, s1_10], 1)

        # Encoder
        conv1 = self.conv1(x)
        pool1 = nn.MaxPool2d(kernel_size=2)(conv1)
        conv2 = self.conv2(torch.cat([pool1, dem_20], 1))
        pool2 = nn.MaxPool2d(kernel_size=2)(conv2)
        conv3 = self.conv3(pool2)
        pool3 = nn.MaxPool2d(kernel_size=2)(conv3)
        conv4 = self.conv4(pool3)
        pool4 = nn.MaxPool2d(kernel_size=2)(conv4)
        conv5 = self.conv5(pool4)

        # Decoder
        upconv1 = self.upconv1(conv5)
        concat1 = torch.cat([upconv1, conv4], dim=1)
        upconv2 = self.upconv2(concat1)
        concat2 = torch.cat([upconv2, conv3], dim=1)
        upconv3 = self.upconv3(concat2)
        concat3 = torch.cat([upconv3, conv2], dim=1)
        upconv4 = self.upconv4(concat3)
        concat4 = torch.cat([upconv4, conv1], dim=1)
        # Output
        out = self.out(concat4)
        return out


class UNetResNet(nn.Module):
    def __init__(self, in_channels, out_channels, resnet_arch='resnet50',
                 pretrained=True):
        super(UNetResNet, self).__init__()

        # Load the ResNet encoder
        self.encoder = timm.create_model(resnet_arch, pretrained=pretrained,
                                         in_chans=in_channels)
        encoder_features = list(self.encoder.children())
        seq_layers = [f for f in encoder_features if isinstance(f, nn.Sequential)]

        self.encoder_features = [nn.Sequential(*encoder_features[:3]),
                                 nn.Sequential(encoder_features[3], seq_layers[0]),
                                 seq_layers[1], seq_layers[2], seq_layers[3]]

        # U-Net decoder
        self.upconv1 = self._make_upconv(2048, 1024)
        self.upconv2 = self._make_upconv(1024 + 1024, 512)
        self.upconv3 = self._make_upconv(512 + 512, 256)
        self.upconv4 = self._make_upconv(256 + 256, 64)
        self.upconv5 = self._make_upconv(64 + 64, 64)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)

    def _make_upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # ResNet encoder
        x1 = self.encoder_features[0](x)
        x2 = self.encoder_features[1](x1)
        x3 = self.encoder_features[2](x2)
        x4 = self.encoder_features[3](x3)
        x5 = self.encoder_features[4](x4)

        # U-Net decoder
        x = self.upconv1(x5)
        x = self.upconv2(torch.cat([x, x4], 1))
        x = self.upconv3(torch.cat([x, x3], 1))
        x = self.upconv4(torch.cat([x, x2], 1))
        x = self.upconv5(torch.cat([x, x1], 1))

        # Output
        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    # Example usage
    in_channels = 5  # Number of input bands
    out_channels = 3  # Number of output channels
    model = UNetResNet(in_channels, out_channels)

    # Create dummy input (batch size of 1, 5 input channels, 256x256 resolution)
    dummy_input = torch.randn(1, in_channels, 256, 256)
    # Forward pass
    output = model(dummy_input)
    print(output.shape)  # Check the shape of the output


class UNet_Cloud(nn.Module):
    def __init__(self, include_s1=True, include_s2_20m=True, include_dem_20m = True , include_cloud_10=True):
        super(UNet_Cloud, self).__init__()

        s2_10m_channels = 4
        s1_10m_channels = 2 if include_s1 else 0
        s2_20m_in_channels = 6 if include_s2_20m else 0
        dem_20m_in_channels = 1 if include_dem_20m else 0
        cloud_10_in_channels = 1 if include_cloud_10 else 0

        in_channels = s2_10m_channels + s1_10m_channels + s2_20m_in_channels + cloud_10_in_channels
        out_channels = s2_10m_channels + s2_20m_in_channels

        # U-Net encoder
        self.conv1 = self._make_conv_block(in_channels, 64)
        self.conv2 = self._make_conv_block(64 + dem_20m_in_channels, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)
        self.conv5 = self._make_conv_block(512, 1024)

        # U-Net decoder
        self.upconv1 = self._make_upconv(1024, 512)
        self.upconv2 = self._make_upconv(512 + 512, 256)
        self.upconv3 = self._make_upconv(256 + 256, 128)
        self.upconv4 = self._make_upconv(128 + 128, 64)

        # Final output
        self.out = nn.Conv2d(64 + 64, out_channels, kernel_size=1)

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _make_upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, s2_10, s2_20, s1_10, cloud_10, dem_20):
        # ResNet encoder
        x = torch.cat([s2_10, s2_20, s1_10, cloud_10], 1)
        # Encoder
        conv1 = self.conv1(x)
        pool1 = nn.MaxPool2d(kernel_size=2)(conv1)
        conv2 = self.conv2(torch.cat([pool1, dem_20], 1))
        pool2 = nn.MaxPool2d(kernel_size=2)(conv2)
        conv3 = self.conv3(pool2)
        pool3 = nn.MaxPool2d(kernel_size=2)(conv3)
        conv4 = self.conv4(pool3)
        pool4 = nn.MaxPool2d(kernel_size=2)(conv4)
        conv5 = self.conv5(pool4)

        # Decoder
        upconv1 = self.upconv1(conv5)
        concat1 = torch.cat([upconv1, conv4], dim=1)
        upconv2 = self.upconv2(concat1)
        concat2 = torch.cat([upconv2, conv3], dim=1)
        upconv3 = self.upconv3(concat2)
        concat3 = torch.cat([upconv3, conv2], dim=1)
        upconv4 = self.upconv4(concat3)
        concat4 = torch.cat([upconv4, conv1], dim=1)
        # Output
        out = self.out(concat4)
        return out