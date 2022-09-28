import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, device, translator, num_features):
        super(Generator, self).__init__()

        self.init_size = (
            img_shape[1] // 4 if img_shape[1] in [28, 44] else img_shape[1] // 8
        )
        self.num_features = num_features
        self.l1 = (
            nn.Sequential(
                nn.Linear(latent_dim, (num_features * 4) * self.init_size**2),
            )
            if img_shape[1] in [28, 44]
            else nn.Sequential(
                nn.Linear(latent_dim, (num_features * 8) * self.init_size**2),
            )
        )
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.device = device

        self.translator = translator

        def generator_block(in_filters, out_filters):
            block = [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    in_filters,
                    out_filters,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                ),
                nn.BatchNorm2d(out_filters),
                nn.ReLU(inplace=True),
            ]
            return block

        self.conv_blocks = (
            nn.Sequential(
                nn.BatchNorm2d(num_features * 4),
                *generator_block(num_features * 4, num_features * 2),
                *generator_block(num_features * 2, num_features),
                nn.Conv2d(
                    num_features,
                    img_shape[0],
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                ),
                nn.Tanh(),
            )
            if img_shape[1] in [28, 44]
            else nn.Sequential(
                nn.BatchNorm2d(num_features * 8),
                *generator_block(num_features * 8, num_features * 4),
                *generator_block(num_features * 4, num_features * 2),
                *generator_block(num_features * 2, num_features),
                nn.Conv2d(
                    num_features,
                    img_shape[0],
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                ),
                nn.Tanh(),
            )
        )

    def forward(self, z, task_id, return_emb=False):
        # Noise as input to translator, embedding as output
        translator_emb = self.translator(z, task_id)  # -> [batch_size, latent_dim]

        out = self.l1(translator_emb)
        out = (
            out.view(
                out.shape[0], (self.num_features * 4), self.init_size, self.init_size
            )
            if self.img_shape[1] in [28, 44]
            else out.view(
                out.shape[0], (self.num_features * 8), self.init_size, self.init_size
            )
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
        num_tasks,
        is_wgan=False,
        task_embedding_dim=5,
    ):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.device = device
        self.is_wgan = is_wgan
        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim

        def discriminator_block(in_filters, out_filters, bn=True):
            if not self.is_wgan:
                block = [
                    nn.Conv2d(
                        in_filters,
                        out_filters,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
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
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                    ),
                    nn.InstanceNorm2d(out_filters, affine=True),
                    nn.LeakyReLU(0.2),
                ]
            return block

        self.model = nn.Sequential(
            nn.Conv2d(
                img_shape[0], num_features, kernel_size=(3, 3), stride=(2, 2), padding=1
            ),
            nn.LeakyReLU(0.2),
            *discriminator_block(num_features, num_features * 2),
            *discriminator_block(num_features * 2, num_features * 4),
            *discriminator_block(num_features * 4, num_features * 8),
        )

        self.task_embedding = nn.Embedding(
            num_embeddings=self.num_tasks, embedding_dim=self.task_embedding_dim
        )

        # The height and width of downsampled image
        ds_size = int(np.ceil(img_shape[1] / 2**4))
        if not self.is_wgan:
            self.adv_layer = nn.Sequential(
                nn.Linear(
                    ((num_features * 8) * ds_size**2) + self.task_embedding_dim, 1
                ),
                nn.Sigmoid(),
            )
        else:
            self.adv_layer = nn.Linear(
                ((num_features * 8) * ds_size**2) + self.task_embedding_dim, 1
            )

    def forward(self, img, task_id):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        # task_id = F.one_hot(task_id.long(), num_classes=self.num_tasks).to(self.device)
        task_id = self.task_embedding(task_id.long().to(self.device))
        x = torch.cat([out, task_id], dim=1)
        validity = self.adv_layer(x)

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
