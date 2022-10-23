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
            nn.Linear(latent_dim, (num_features * 8) * self.init_size**2),
            nn.BatchNorm1d((num_features * 8) * self.init_size**2),
            nn.ReLU(inplace=True),
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
            output_padding=0,
        ):
            if self.layers_type == "transpose" or self.layers_type == "vae":
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
            elif layers_type == "vae":
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
            if layers_type == "upsample":
                self.conv_blocks = nn.Sequential(
                    # in: 4x4
                    # nn.BatchNorm2d(num_features * 8),
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
                    # nn.BatchNorm2d(num_features * 8),
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
            elif layers_type == "vae":
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
            elif layers_type == "vae":
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
        gen_layers_type,
        is_wgan=False,
    ):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.device = device
        self.is_wgan = is_wgan

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
                else:
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
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        bn=False,
                    ),
                    # size: 14x14
                    *discriminator_block(
                        num_features * 2,
                        num_features * 4,
                        kernel_size=(3, 3),
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
            elif gen_layers_type == "vae":
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
                    nn.Linear(num_features * self.conv_out_size * self.conv_out_size, 1)
                ]

        elif img_shape[1] == 32:
            if gen_layers_type == "transpose":
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
                        input_img_size=32,
                    ),
                    # size: 16x16
                    *discriminator_block(
                        num_features,
                        num_features * 2,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        input_img_size=16,
                    ),
                    # size: 8x8
                    *discriminator_block(
                        num_features * 2,
                        num_features * 4,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        input_img_size=8,
                    ),
                    # size: 4x4
                    *discriminator_block(
                        num_features * 4,
                        num_features * 8,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        input_img_size=4,
                    ),
                    # size: 2x2
                )
            elif gen_layers_type == "upsample":
                self.model = nn.Sequential(
                    # in: 32x32
                    *discriminator_block(
                        img_shape[0],
                        num_features,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        bn=False,
                        input_img_size=32,
                    ),
                    # size: 16x16
                    *discriminator_block(
                        num_features,
                        num_features * 2,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        input_img_size=16,
                    ),
                    # size: 8x8
                    *discriminator_block(
                        num_features * 2,
                        num_features * 4,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        input_img_size=8,
                    ),
                    # size: 4x4
                    *discriminator_block(
                        num_features * 4,
                        num_features * 8,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        input_img_size=4,
                    ),
                )
                # size: 2x2

            elif gen_layers_type == "vae":
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
                    nn.Linear(
                        num_features * 8 * self.conv_out_size * self.conv_out_size, 1
                    )
                ]

        elif img_shape[1] == 64:
            if gen_layers_type == "transpose":
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
            elif gen_layers_type == "upsample":
                self.model = nn.Sequential(
                    # in: 64x64
                    *discriminator_block(
                        img_shape[0],
                        num_features,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                        bn=False,
                    ),
                    # size: 32x32
                    *discriminator_block(
                        num_features,
                        num_features * 2,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 16x16
                    *discriminator_block(
                        num_features * 2,
                        num_features * 4,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 8x8
                    *discriminator_block(
                        num_features * 4,
                        num_features * 8,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 4x4
                    *discriminator_block(
                        num_features * 8,
                        num_features * 8,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=1,
                        bias=False,
                    ),
                    # size: 2x2
                )
            elif gen_layers_type == "vae":
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
                    nn.Linear(
                        num_features * 4 * self.conv_out_size * self.conv_out_size, 1
                    )
                ]

        # if img_shape[1] == 28 and gen_layers_type == "upsample":
        #     last_layers = [nn.Linear((num_features * 8) * 16, 1)]
        # else:
        #     last_layers = [nn.Linear((num_features * 8) * 4, 1)]
        #     # last_layers = [nn.Linear(196, 1)]
        if not self.is_wgan:
            last_layers.append(nn.Sigmoid())

        self.adv_layer = nn.Sequential(*last_layers)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


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
