import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, device, translator, num_features):
        super(Generator, self).__init__()

        self.num_features = num_features
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.device = device

        self.translator = translator

        def generator_block(
            in_filters,
            out_filters,
            kernel_size,
            stride,
            padding,
            bias,
            act_fun=nn.ReLU(inplace=True),
            bn=True,
            output_padding=0,
        ):
            block = [
                nn.ConvTranspose2d(
                    in_filters,
                    out_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    output_padding=output_padding,
                )
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(act_fun)
            return block

        if img_shape[1] == 28:
            self.scaler = 4
            self.l1 = nn.Linear(
                latent_dim, num_features * self.scaler * self.scaler * self.scaler
            )
            self.conv_blocks = nn.Sequential(
                # in: 4x4
                nn.BatchNorm2d(num_features * self.scaler),
                *generator_block(
                    num_features * self.scaler,
                    num_features * 4,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=0,
                    bias=False,
                ),
                *generator_block(
                    num_features * 4,
                    num_features * 2,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=0,
                    bias=False,
                ),
                *generator_block(
                    num_features * 2,
                    num_features,
                    kernel_size=(4, 4),
                    stride=(1, 1),
                    padding=0,
                    bias=False,
                ),
                *generator_block(
                    num_features,
                    img_shape[0],
                    kernel_size=(4, 4),
                    stride=(1, 1),
                    padding=0,
                    bias=False,
                    act_fun=nn.Tanh(),
                    bn=False,
                ),
            )
        elif img_shape[1] == 32:
            self.scaler = img_shape[1] // 16
            self.l1 = nn.Linear(
                latent_dim, num_features * self.scaler * self.scaler * self.scaler
            )
            self.conv_blocks = nn.Sequential(
                # in: 2x2
                nn.BatchNorm2d(num_features * self.scaler),
                *generator_block(
                    num_features * self.scaler,
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
                    kernel_size=(5, 5),
                    stride=(1, 1),
                    padding=2,
                    output_padding=0,
                    bias=False,
                    act_fun=nn.Tanh(),
                    bn=False,
                ),
            )

        elif img_shape[1] == 64:
            self.scaler = img_shape[1] // 8
            self.l1 = nn.Linear(
                latent_dim, num_features * self.scaler * self.scaler * self.scaler
            )
            self.conv_blocks = nn.Sequential(
                # in: 8x8
                nn.BatchNorm2d(num_features * self.scaler),
                *generator_block(
                    num_features * self.scaler,
                    num_features * 4,
                    kernel_size=(5, 5),
                    stride=(2, 2),
                    padding=2,
                    output_padding=1,
                    bias=False,
                ),
                *generator_block(
                    num_features * 4,
                    num_features * 2,
                    kernel_size=(5, 5),
                    stride=(2, 2),
                    padding=2,
                    output_padding=1,
                    bias=False,
                ),
                *generator_block(
                    num_features * 2,
                    num_features,
                    kernel_size=(5, 5),
                    stride=(2, 2),
                    padding=2,
                    output_padding=1,
                    bias=False,
                ),
                *generator_block(
                    num_features,
                    img_shape[0],
                    kernel_size=(5, 5),
                    stride=(1, 1),
                    padding=2,
                    output_padding=0,
                    bias=False,
                    act_fun=nn.Tanh(),
                    bn=False,
                ),
            )
            
        elif img_shape[1] == 44:
            self.scaler = img_shape[1] // 8
            self.l1 = nn.Linear(
                latent_dim, num_features * self.scaler * self.scaler * self.scaler
            )
            self.conv_blocks = nn.Sequential(
                # in: 5x5
                nn.BatchNorm2d(num_features * self.scaler),
                *generator_block(
                    num_features * self.scaler,
                    num_features * 4,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                ),
                *generator_block(
                    num_features * 4,
                    num_features * 2,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=0,
                    bias=False,
                ),
                *generator_block(
                    num_features * 2,
                    num_features,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                ),
                *generator_block(
                    num_features,
                    img_shape[0],
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                    bias=False,
                    act_fun=nn.Tanh(),
                    bn=False,
                ),
            )

    def forward(self, z, task_id, return_emb=False):
        # Noise as input to translator, embedding as output
        translator_emb = self.translator(
            z, task_id
        )  # -> [batch_size, latent_dim] NOTE: conditioned by task_id

        out = self.l1(translator_emb)

        out = out.view(-1, self.scaler * self.num_features, self.scaler, self.scaler)
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
        num_embeddings=0,
        embedding_dim=0
    ):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.device = device
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        def discriminator_block(
            in_filters,
            out_filters,
            kernel_size,
            stride,
            padding,
            bias,
            bn=True,
            output_img_size=None,
        ):
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
            if bn or output_img_size is not None:
                block.append(
                    nn.LayerNorm(
                        normalized_shape=[
                            out_filters,
                            output_img_size,
                            output_img_size,
                        ]
                    )
                )
            block.append(nn.LeakyReLU(0.2, inplace=True))

            return block

        if img_shape[1] == 28:
            self.model = nn.Sequential(
                # in: 28x28
                *discriminator_block(
                    img_shape[0],
                    num_features,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=14,
                ),
                # size: 14x14
                *discriminator_block(
                    num_features,
                    num_features,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=7,
                ),
                # size: 7x7
                *discriminator_block(
                    num_features,
                    num_features,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=3,
                ),
            )
            # size: 3x3
            self.conv_out_size = 3
            last_layers = [
                nn.Linear(num_features * self.conv_out_size * self.conv_out_size + self.embedding_dim, 1)
            ]

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
                    output_img_size=16,
                ),
                # size: 16x16
                *discriminator_block(
                    num_features,
                    num_features * 2,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=8,
                ),
                # size: 8x8
                *discriminator_block(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=4,
                ),
                # size 4x4
                *discriminator_block(
                    num_features * 4,
                    num_features * 8,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=2,
                ),
                # size: 2x2
                *discriminator_block(
                    num_features * 8,
                    num_features * 8,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                    bias=False,
                    output_img_size=2,
                )
                # size: 2x2
            )
            self.conv_out_size = 2
            last_layers = [
                nn.Linear(num_features * 8 * self.conv_out_size * self.conv_out_size + self.embedding_dim, 1)
            ]

        elif img_shape[1] == 64:
            self.model = nn.Sequential(
                # in: 64x64
                *discriminator_block(
                    img_shape[0],
                    num_features,
                    kernel_size=(5, 5),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=31,
                ),
                # size: 31x31
                *discriminator_block(
                    num_features,
                    num_features * 2,
                    kernel_size=(5, 5),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=15,
                ),
                # size: 15x15
                *discriminator_block(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=(5, 5),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=7,
                ),
                # size 7x7
                *discriminator_block(
                    num_features * 4,
                    num_features * 4,
                    kernel_size=(5, 5),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=3,
                ),
                # size: 3x3
            )
            self.conv_out_size = 3
            last_layers = [
                nn.Linear(num_features * 4 * self.conv_out_size * self.conv_out_size + self.embedding_dim, 1)
            ]
        elif img_shape[1] == 44:
            self.model = nn.Sequential(
                # in: 44x44
                *discriminator_block(
                    img_shape[0],
                    num_features,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=22,
                ),
                # size: 22x22
                *discriminator_block(
                    num_features,
                    num_features,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=11,
                ),
                # size: 11x11
                *discriminator_block(
                    num_features,
                    num_features * 2,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=5,
                ),
                # size: 5x5
                *discriminator_block(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                    output_img_size=2,
                ),
            )
            # size: 2x2
            self.conv_out_size = 2
            last_layers = [
                nn.Linear(num_features * 4 * self.conv_out_size * self.conv_out_size + self.embedding_dim, 1)
            ]

        self.adv_layer = nn.Sequential(*last_layers)
        
        if self.num_embeddings and self.embedding_dim:
            self.task_embedding = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim
            )

    def forward(self, img, task_id=None):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        if task_id is not None:
            task_id = self.task_embedding(task_id.long().to(self.device))
            out = torch.cat([out, task_id], dim=1)

        validity = self.adv_layer(out)

        return validity


class Translator(nn.Module):
    def __init__(self, latent_size, device, num_embeddings, embedding_dim):
        super().__init__()
        self.device = device
        self.latent_size = latent_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_size + self.embedding_dim, latent_size * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_size * 4, latent_size),
        )

        self.task_embedding = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim
        )

    def forward(self, x, task_id):
        task_id = self.task_embedding(task_id.long().to(self.device))
        x = torch.cat([x, task_id], dim=1)
        out = self.fc(x)
        return out
