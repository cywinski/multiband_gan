import time

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch import Tensor
from torch.autograd import Variable

from gan_experiments import gan_utils

torch.autograd.set_detect_anomaly(True)


def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def train_local(
        local_generator, local_discriminator, n_epochs, task_loader, task_id, local_dis_lr, local_gen_lr,
        num_gen_images, local_scheduler_rate, gan_type, n_critic_steps, lambda_gp
        ):
    g_losses = []
    d_losses = []
    img_list = []

    # Optimizers
    optimizer_g = torch.optim.Adam(local_generator.parameters(), lr=local_gen_lr, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(local_discriminator.parameters(), lr=local_dis_lr, betas=(0.5, 0.9))
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=local_scheduler_rate)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=local_scheduler_rate)

    # Loss function
    criterion = torch.nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(num_gen_images, local_generator.latent_dim, device=local_generator.device)

    ## ZWYKLY GAN!
    if gan_type == "dcgan":
        for epoch in range(n_epochs):
            for i, batch in enumerate(task_loader):
                imgs = batch[0].to(local_generator.device)

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False).to(local_generator.device)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(local_generator.device)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor)).to(local_generator.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_d.zero_grad()

                ## Train with all-real batch
                d_output_real = local_discriminator(real_imgs)

                # Measure discriminator's ability to classify real from generated samples
                d_real_loss = criterion(d_output_real, valid)
                d_real_loss.backward()
                D_x = d_output_real.mean().item()

                ## Train with all-fake batch

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], local_generator.latent_dim)))).to(
                        local_generator.device)
                gen_imgs = local_generator(z)

                d_output_fake = local_discriminator(gen_imgs.detach())

                d_fake_loss = criterion(d_output_fake, fake)
                d_fake_loss.backward()

                D_G_z1 = d_output_fake.mean().item()
                d_loss = (d_real_loss + d_fake_loss) / 2

                optimizer_d.step()

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_g.zero_grad()

                d_output_fake = local_discriminator(gen_imgs)

                # Loss measures generator's ability to fool the discriminator
                g_loss = criterion(d_output_fake, valid)

                g_loss.backward()
                D_G_z2 = d_output_fake.mean().item()
                optimizer_g.step()
                if i % 100 == 0:
                    print(
                            f"[Epoch {epoch + 1}/{n_epochs}] [Batch {i}/{len(task_loader)}] [D loss: {d_loss.item():.3f}] [G loss: {g_loss.item():.3f}] [D(x): {D_x:.3f}] [D(G(z)): {D_G_z1:.3f} / {D_G_z2:.3f}]"
                            )

                wandb.log({
                        "d_loss": np.round(d_loss.item(), 3),
                        "g_loss": np.round(g_loss.item(), 3),
                        })

            if epoch % 10 == 0:
                fig = gan_utils.generate_images_grid(local_generator, num_gen_images, task_id, local_generator.device,
                                                     epoch=epoch, noise=fixed_noise)
                wandb.log({
                        f"generations_task_{task_id}_epoch_{epoch}": fig
                        })
                plt.close(fig)

            # scheduler_g.step()
            # scheduler_d.step()

    else:
        ## WGAN-GP
        for epoch in range(n_epochs):
            for i, batch in enumerate(task_loader):
                start = time.time()
                imgs = batch[0].to(local_generator.device)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor)).to(local_generator.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_d.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], local_generator.latent_dim)))).to(
                        local_generator.device)

                # Generate a batch of images
                fake_imgs = local_generator(z)

                # Train on real images
                d_output_real = local_discriminator(real_imgs)
                d_loss_real = -torch.mean(d_output_real)
                # Train on fake images
                d_output_fake = local_discriminator(fake_imgs)
                d_loss_fake = torch.mean(d_output_fake)
                # Gradient penalty
                gradient_penalty = gan_utils.compute_gradient_penalty(local_discriminator, real_imgs.data,
                                                                      fake_imgs.data,
                                                                      local_generator.device)
                # Wasserstein distance
                wasserstein_distance = d_loss_real + d_loss_fake

                # Adversarial loss
                d_loss = d_loss_fake + d_loss_real + lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()

                if i % 20 == 0:
                    print(
                            f"[D] [Epoch {epoch + 1}/{n_epochs}] [Batch {i}/{len(task_loader)}] [D loss: {d_loss.item():.3f}] [D loss fake: {d_loss_fake.item():.3f}] [D loss real: {d_loss_real.item():.3f}] [Gradient penalty: {gradient_penalty.item():.3f}] [Wasserstein distance: {wasserstein_distance.item():.3f}]"
                            )

                # Train the generator every n_critic steps
                if i % n_critic_steps == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_imgs = local_generator(z)

                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    d_output_fake = local_discriminator(fake_imgs)
                    g_loss = -torch.mean(d_output_fake)

                    g_loss.backward()
                    optimizer_g.step()

                    if i % 20 == 0:
                        print(
                                f"[G] [Epoch {epoch + 1}/{n_epochs}] [Batch {i}/{len(task_loader)}] [G loss: {g_loss.item():.3f}]"
                                )

                wandb.log({
                        "d_loss": np.round(d_loss.item(), 3),
                        "d_loss_fake": np.round(d_loss_fake.item(), 3),
                        "d_loss_real": np.round(d_loss_real.item(), 3),
                        "g_loss": np.round(g_loss.item(), 3),
                        "gradient_penalty": np.round(gradient_penalty.item(), 3),
                        "wasserstein_distance": np.round(wasserstein_distance.item(), 3)
                        })

            if epoch % 10 == 0:
                fig = gan_utils.generate_images_grid(local_generator, num_gen_images, task_id, local_generator.device,
                                                     epoch=epoch, noise=fixed_noise)
                wandb.log({
                        f"generations_task_{task_id}_epoch_{epoch}": fig
                        })

            # scheduler_g.step()
            # scheduler_d.step()
