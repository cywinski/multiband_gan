import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, device):
        super(Generator, self).__init__()

        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(
                nn.Linear(latent_dim, 128 * self.init_size ** 2),
                )
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.device = device

        self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
                nn.Tanh(),
                )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_shape, device):
#         super(Generator, self).__init__()
#         self.latent_dim = latent_dim
#         self.img_shape = img_shape
#         self.device = device
#
#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#                 *block(latent_dim, 128, normalize=False),
#                 *block(128, 256),
#                 *block(256, 512),
#                 *block(512, 1024),
#                 nn.Linear(1024, int(np.prod(img_shape))),
#                 nn.Tanh()
#                 )
#
#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.size(0), *self.img_shape)
#         return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, device, is_wgan=False):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.device = device
        self.is_wgan = is_wgan

        def discriminator_block(in_filters, out_filters, bn=True):
            if not self.is_wgan:
                block = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Dropout2d(0.25)]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
            else:
                block = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
                         nn.InstanceNorm2d(out_filters, affine=True),
                         nn.LeakyReLU(0.2, inplace=True)]
            return block

        self.model = nn.Sequential(
                *discriminator_block(img_shape[0], 64, bn=False),  # 14x14x64
                *discriminator_block(64, 128),  # 7x7x128
                *discriminator_block(128, 256),  # 3x3x256
                # *discriminator_block(256, 512),  # 2x2x512
                )

        # The height and width of downsampled image
        ds_size = int(np.ceil(img_shape[1] / 2 ** 3))
        if not self.is_wgan:
            self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, 1), nn.Sigmoid())
        else:
            self.adv_layer = nn.Linear(256 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# class Discriminator(nn.Module):
#     def __init__(self, img_shape, device):
#         super(Discriminator, self).__init__()
#         self.device = device
#
#         self.model = nn.Sequential(
#                 nn.Linear(int(np.prod(img_shape)), 512),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Linear(512, 256),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Linear(256, 1),
#                 nn.Sigmoid(),
#                 )
#
#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         validity = self.model(img_flat)
#
#         return validity
