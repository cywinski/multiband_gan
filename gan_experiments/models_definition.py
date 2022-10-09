import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self, latent_dim, img_shape, device, translator, num_features, layers_type
    ):
        super(Generator, self).__init__()

        self.num_features = num_features
        self.layers_type = layers_type
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.device = device

        self.translator = translator

        self.img_shape_upsample_init_size_map = {28: 7, 32: 4, 64: 4}

        self.init_size = (
            self.img_shape_upsample_init_size_map[img_shape[1]]
            if layers_type == "upsample"
            else 2
        )
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, (num_features * 8) * self.init_size**2)
        )

        def generator_block(
            in_filters,
            out_filters,
            kernel_size,
            stride,
            padding,
            bias,
            act_fun=nn.ReLU(inplace=True),
            bn=True,
        ):
            if self.layers_type == "transpose":
                block = [
                    nn.ConvTranspose2d(
                        in_filters,
                        out_filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias,
                    )
                ]
            elif self.layers_type == "upsample":
                block = [
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(
                        in_filters,
                        out_filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias,
                    ),
                ]

            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(act_fun)
            return block

        if img_shape[1] == 28:
            if layers_type == "upsample":
                self.conv_blocks = nn.Sequential(
                    # in: 7x7
                    nn.BatchNorm2d(num_features * 8),
                    *generator_block(
                        num_features * 8,
                        num_features * 4,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    # size: 14x14
                    *generator_block(
                        num_features * 4,
                        num_features * 2,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    # size: 28x28
                    nn.Conv2d(
                        num_features * 2,
                        img_shape[0],
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    nn.Tanh(),
                )
            elif layers_type == "transpose":
                self.conv_blocks = nn.Sequential(
                    # in: 2x2
                    nn.BatchNorm2d(num_features * 8),
                    *generator_block(
                        num_features * 8,
                        num_features * 4,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 4x4
                    *generator_block(
                        num_features * 4,
                        num_features * 2,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 7x7
                    *generator_block(
                        num_features * 2,
                        num_features,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 14x14
                    *generator_block(
                        num_features,
                        img_shape[0],
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        act_fun=nn.Tanh(),
                        bn=False,
                    ),
                    # size: 28x28
                )
        elif img_shape[1] == 32:
            if layers_type == "upsample":
                self.conv_blocks = nn.Sequential(
                    # in: 4x4
                    nn.BatchNorm2d(num_features * 8),
                    *generator_block(
                        num_features * 8,
                        num_features * 4,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    # size: 8x8
                    *generator_block(
                        num_features * 4,
                        num_features * 2,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    # size: 16x16
                    *generator_block(
                        num_features * 2,
                        num_features,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    # size: 32x32
                    nn.Conv2d(
                        num_features,
                        img_shape[0],
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    nn.Tanh(),
                )
            if layers_type == "transpose":
                self.conv_blocks = nn.Sequential(
                    # in: 2x2
                    nn.BatchNorm2d(num_features * 8),
                    *generator_block(
                        num_features * 8,
                        num_features * 4,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 4x4
                    *generator_block(
                        num_features * 4,
                        num_features * 2,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 8x8
                    *generator_block(
                        num_features * 2,
                        num_features,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 16x16
                    *generator_block(
                        num_features,
                        img_shape[0],
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        act_fun=nn.Tanh(),
                        bn=False,
                    ),
                    # size: 32x32
                )
        elif img_shape[1] == 64:
            if layers_type == "upsample":
                self.conv_blocks = nn.Sequential(
                    # in: 4x4
                    nn.BatchNorm2d(num_features * 8),
                    *generator_block(
                        num_features * 8,
                        num_features * 8,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    # size: 8x8
                    *generator_block(
                        num_features * 8,
                        num_features * 4,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    # size: 16x16
                    *generator_block(
                        num_features * 4,
                        num_features * 2,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    # size: 32x32
                    *generator_block(
                        num_features * 2,
                        num_features,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    # size: 64x64
                    nn.Conv2d(
                        num_features,
                        img_shape[0],
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1,
                        bias=False,
                    ),
                    nn.Tanh(),
                )
            elif layers_type == "transpose":
                self.conv_blocks = nn.Sequential(
                    # in: 2x2
                    nn.BatchNorm2d(num_features * 8),
                    *generator_block(
                        num_features * 8,
                        num_features * 8,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 4x4
                    *generator_block(
                        num_features * 8,
                        num_features * 4,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 8x8
                    *generator_block(
                        num_features * 4,
                        num_features * 2,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 16x16
                    *generator_block(
                        num_features * 2,
                        num_features,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 32x32
                    *generator_block(
                        num_features,
                        img_shape[0],
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        act_fun=nn.Tanh(),
                        bn=False,
                    ),
                    # size: 64x64
                )

    def forward(self, z, task_id, return_emb=False):
        # Noise as input to translator, embedding as output
        translator_emb = self.translator(
            z, task_id
        )  # -> [batch_size, latent_dim] NOTE: conditioned by task_id

        out = self.l1(translator_emb)
        out = out.view(
            out.shape[0], (self.num_features * 8), self.init_size, self.init_size
        )
        img = self.conv_blocks(out)

        if return_emb:
            return img, translator_emb

        return img


class Discriminator(nn.Module):
    def __init__(
        self,
        img_shape,
        device,
        num_features,
        gen_layers_type,
        is_wgan=False,
    ):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.device = device
        self.is_wgan = is_wgan

        def discriminator_block(
            in_filters, out_filters, kernel_size, stride, padding, bias, bn=True
        ):
            if not self.is_wgan:
                block = [
                    nn.Conv2d(
                        in_filters,
                        out_filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                ]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
            else:
                block = [
                    nn.Conv2d(
                        in_filters,
                        out_filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias,
                    )
                ]
                if bn:
                    block.append(nn.InstanceNorm2d(out_filters, affine=True))
                block.append(nn.LeakyReLU(0.2, inplace=True))

            return block

        if img_shape[1] == 28:
            if gen_layers_type == "transpose":
                self.model = nn.Sequential(
                    # in: 28x28
                    *discriminator_block(
                        img_shape[0],
                        num_features,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        bn=False,
                    ),
                    # size: 14x14
                    *discriminator_block(
                        num_features,
                        num_features * 2,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 7x7
                    *discriminator_block(
                        num_features * 2,
                        num_features * 4,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 4x4
                    *discriminator_block(
                        num_features * 4,
                        num_features * 8,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    )
                    # size: 2x2
                )
            elif gen_layers_type == "upsample":
                self.model = nn.Sequential(
                    # in: 28x28
                    *discriminator_block(
                        img_shape[0],
                        num_features * 2,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        bn=False,
                    ),
                    # size: 14x14
                    *discriminator_block(
                        num_features * 2,
                        num_features * 4,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 7x7
                    *discriminator_block(
                        num_features * 4,
                        num_features * 8,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 4x4
                )

        elif img_shape[1] == 32:
            self.model = nn.Sequential(
                # in: 32x32
                *discriminator_block(
                    img_shape[0],
                    num_features,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    bn=False,
                ),
                # size: 16x16
                *discriminator_block(
                    num_features,
                    num_features * 2,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                ),
                # size: 8x8
                *discriminator_block(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                ),
                # size: 4x4
                *discriminator_block(
                    num_features * 4,
                    num_features * 8,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                ),
                # size: 2x2
            )

        elif img_shape[1] == 64:
            self.model = nn.Sequential(
                # in: 64x64
                *discriminator_block(
                    img_shape[0],
                    num_features,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    bn=False,
                ),
                # size: 32x32
                *discriminator_block(
                    num_features,
                    num_features * 2,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                ),
                # size: 16x16
                *discriminator_block(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                ),
                # size: 8x8
                *discriminator_block(
                    num_features * 4,
                    num_features * 8,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                ),
                # size: 4x4
                *discriminator_block(
                    num_features * 8,
                    num_features * 8,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                ),
                # size: 2x2
            )

        if img_shape[1] == 28 and gen_layers_type == "upsample":
            last_layers = [nn.Linear((num_features * 8) * 16, 1)]
        else:
            last_layers = [nn.Linear((num_features * 8) * 4, 1)]
        if not self.is_wgan:
            last_layers.append(nn.Sigmoid())

        self.adv_layer = nn.Sequential(*last_layers)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


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


class Translator(nn.Module):
    def __init__(self, latent_size, device, num_tasks, task_embedding_dim=5):
        super().__init__()
        self.device = device
        self.latent_size = latent_size
        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_size + self.task_embedding_dim, latent_size * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_size * 4, latent_size),
        )

        self.task_embedding = nn.Embedding(
            num_embeddings=self.num_tasks, embedding_dim=self.task_embedding_dim
        )

    def forward(self, x, task_id):
        # task_id = F.one_hot(task_id.long(), num_classes=self.num_tasks).to(self.device)
        task_id = self.task_embedding(task_id.long().to(self.device))
        x = torch.cat([x, task_id], dim=1)
        out = self.fc(x)
        return out
