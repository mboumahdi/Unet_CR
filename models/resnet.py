import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.resblocks_dim = 256
        self.n_resblocks = 16

        # The network
        self.conv1 = nn.Conv2d(12, self.resblocks_dim, kernel_size=3, stride=1, padding=1)
        self.resblocks = nn.ModuleList([self._resblock() for _ in range(self.n_resblocks)])
        self.conv2 = nn.Conv2d(self.resblocks_dim, 10, kernel_size=3, stride=1, padding=1)

    def _resblock(self):
        return nn.Sequential(
            nn.Conv2d(self.resblocks_dim, self.resblocks_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.resblocks_dim, self.resblocks_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, s1_t, s2_10_t, s2_20_t):
        # Concatenate the inputs along the channel dimension
        s2 = torch.cat([s2_10_t, s2_20_t], dim=1)
        net = torch.cat([s1_t, s2], dim=1)
        net = F.relu(self.conv1(net))

        for i in range(self.n_resblocks):
            net = self.resblocks[i](net) + net

        net = self.conv2(net)

        # Assuming s2_10_t and s2_20_t have the same spatial dimensions as net
        net = net + s2

        return net
