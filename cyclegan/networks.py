import torch.nn as nn
import itertools

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, 
                kernel_size=3, stride=2, 
                padding=1, padding_mode='reflect', bias=True
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channels, out_channels=channels, 
                kernel_size=3, padding=1, padding_mode='reflect', bias=True
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channels, out_channels=channels,
                kernel_size=3, padding=1, padding_mode='reflect', bias=True
            ),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return self.conv(x) + x

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=2, 
                padding=1, output_padding=1, bias=True
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, init_features=64):
        super().__init__()

        self.model = nn.Sequential()

        self.model.add_module(
            'initial_conv', 
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=init_features,
                    kernel_size=7, padding=3, padding_mode='reflect'
                ),
            nn.InstanceNorm2d(init_features),
            nn.ReLU(inplace=True),
        ))
        channels = init_features
        
        n_down = 2
        for i in range(n_down):
            self.model.add_module(f'down_conv_{i}', DownsamplingBlock(channels, channels*2))
            channels *= 2

        n_res = 6
        for i in range(n_res):
            self.model.add_module(f'resid_conv_{i}', ResidualBlock(channels))

        n_up = n_down
        for i in range(n_up):
            self.model.add_module(f'up_conv_{i}', UpsamplingBlock(in_channels=channels, out_channels=channels//2))
            channels //= 2

        self.model.add_module(
            'final_conv',
            nn.Sequential(
                nn.Conv2d(
                    in_channels=channels, out_channels=in_channels,
                    kernel_size=7, stride=1, padding=3, padding_mode='reflect'
                ),
                nn.Tanh()
            )
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def  __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.model = nn.Sequential()

        full_features = [in_channels]
        full_features.extend(features)
        for i, (in_f, out_f) in enumerate(itertools.pairwise(full_features)):
            self.model.add_module(
                f'conv_{i}',
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_f, out_channels=out_f,
                        kernel_size=4, stride=2, padding=1, bias=True
                    ),
                    nn.InstanceNorm2d(out_f),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        self.model.add_module(
            'final_layer',
            nn.Sequential(
                nn.Conv2d(in_channels=full_features[-1], out_channels=1, kernel_size=4, padding=1),
                nn.Sigmoid()
            )
        )

    def forward(self, x):
        return self.model(x)

def denormalize(image):
    return image*0.5 + 0.5