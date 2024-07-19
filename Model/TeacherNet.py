import torch
from torch import nn


class TeacherNet(nn.Module):
    def __init__(self, nf, in_channels=1, out_channels=1):
        super(TeacherNet, self).__init__()

        self.encoder_a = nn.Sequential(
            nn.Conv2d(in_channels, nf, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf * 2, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf * 2, nf * 4, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf * 4, nf * 8, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
        )

        self.encoder_b = nn.Sequential(
            nn.Conv2d(in_channels, nf, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf * 2, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf * 2, nf * 4, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf * 4, nf * 8, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(nf * 16, nf * 8, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf * 8, nf * 4, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf * 4, nf * 2, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf * 2, nf, (3, 3), (1, 1), (1, 1), bias=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        )

    def forward(self, input_a, input_b):
        en_a = self.encoder_a(input_a)
        en_b = self.encoder_b(input_b)

        fuse = torch.cat([en_a, en_b], 1)

        out = self.decoder(fuse)

        return out
