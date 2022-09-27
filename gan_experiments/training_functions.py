import copy

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


def train_local_dcgan(
    n_epochs,
    task_loader,
    local_generator,
    local_discriminator,
    local_gen_lr,
    local_dis_lr,
    num_gen_images,
    task_id,
    b1,
    b2
):
    # Optimizers
    local_generator.train()
    local_discriminator.train()
    local_generator.translator.train()
    optimizer_g = torch.optim.Adam(
        local_generator.parameters(), lr=local_gen_lr, betas=(b1, b2)
    )
    optimizer_d = torch.optim.Adam(
        local_discriminator.parameters(),
        lr=local_dis_lr,
        betas=(b1, b2),
        weight_decay=1e-3,
    )
    # Loss function
    criterion = torch.nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(
        num_gen_images, local_generator.latent_dim, device=local_generator.device
    )

    for epoch in range(n_epochs):
        for i, batch in enumerate(task_loader):
            imgs = batch[0].to(local_generator.device)

            # Adversarial ground truths
            valid = Variable(
                Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False
            ).to(local_generator.device)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(
                local_generator.device
            )

            # Configure input
            real_imgs = Variable(imgs.type(Tensor)).to(local_generator.device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_d.zero_grad()

            ## Train with all-real batch
            d_output_real = local_discriminator(
                real_imgs,
                (torch.zeros([len(batch[0])]) + task_id).to(local_generator.device),
            )

            # Measure discriminator's ability to classify real from generated samples
            d_real_loss = criterion(d_output_real, valid)
            d_real_loss.backward()
            D_x = d_output_real.mean().item()

            ## Train with all-fake batch

            # Sample noise as generator input
            z = Variable(
                Tensor(
                    np.random.normal(0, 1, (imgs.shape[0], local_generator.latent_dim))
                )
            ).to(local_generator.device)

            gen_imgs = local_generator(
                z, (torch.zeros([len(batch[0])]) + task_id).to(local_generator.device)
            )

            d_output_fake = local_discriminator(
                gen_imgs.detach(),
                (torch.zeros([len(batch[0])]) + task_id).to(local_generator.device),
            )

            d_fake_loss = criterion(d_output_fake, fake)
            d_fake_loss.backward()

            D_G_z1 = d_output_fake.mean().item()
            d_loss = (d_real_loss + d_fake_loss) / 2

            optimizer_d.step()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_g.zero_grad()

            d_output_fake = local_discriminator(
                gen_imgs,
                (torch.zeros([len(batch[0])]) + task_id).to(local_generator.device),
            )

            # Loss measures generator's ability to fool the discriminator
            g_loss = criterion(d_output_fake, valid)

            g_loss.backward()
            D_G_z2 = d_output_fake.mean().item()
            optimizer_g.step()
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch + 1}/{n_epochs}] [Batch {i}/{len(task_loader)}] [D loss: {d_loss.item():.3f}] [G loss: {g_loss.item():.3f}] [D(x): {D_x:.3f}] [D(G(z)): {D_G_z1:.3f} / {D_G_z2:.3f}]"
                )

            wandb.log(
                {
                    "d_loss": np.round(d_loss.item(), 3),
                    "g_loss": np.round(g_loss.item(), 3),
                }
            )

        if epoch % 5 == 0:
            fig = gan_utils.generate_images_grid(
                local_generator,
                local_generator.device,
                (torch.zeros([len(batch[0])]) + task_id).to(local_generator.device),
            )
            wandb.log({f"generations_task_{task_id}": fig})
            plt.close(fig)

        # scheduler_g.step()
        # scheduler_d.step()


def train_local_wgan_gp(
    n_epochs,
    task_loader,
    local_generator,
    local_discriminator,
    local_gen_lr,
    local_dis_lr,
    num_gen_images,
    task_id,
    lambda_gp,
    n_critic_steps,
    local_scheduler_rate,
    b1,
    b2
):
    # Optimizers
    optimizer_g = torch.optim.Adam(
        local_generator.parameters(), lr=local_gen_lr, betas=(b1, b2)
    )
    optimizer_d = torch.optim.Adam(
        local_discriminator.parameters(), lr=local_dis_lr, betas=(b1, b2), weight_decay=1e-3
    )
    # Schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_g, gamma=local_scheduler_rate
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_d, gamma=local_scheduler_rate
    )

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(
        num_gen_images, local_generator.latent_dim, device=local_generator.device
    )

    for epoch in range(n_epochs):
        local_generator.train()
        for i, batch in enumerate(task_loader):
            # Configure input
            real_imgs = Variable(batch[0].type(Tensor)).to(local_generator.device)
            task_ids = (torch.zeros([len(batch[0])]) + task_id).to(
                local_generator.device
            )

            # ---------------------
            #  Train Discriminator (Critic)
            # ---------------------

            optimizer_d.zero_grad()

            # Sample noise as generator input
            z = Variable(
                Tensor(
                    np.random.normal(
                        0, 1, (real_imgs.shape[0], local_generator.latent_dim)
                    )
                )
            ).to(local_generator.device)

            # Generate a batch of images
            fake_imgs = local_generator(z, task_ids)

            # Add detach() so the backward() will not change the weights of generator
            fake_imgs.detach()

            # Train on real images -> compare predictions to 1
            d_output_real = local_discriminator(real_imgs, task_ids)
            d_loss_real = -torch.mean(d_output_real)

            # Train on fake images -> compare predictions to -1
            d_output_fake = local_discriminator(fake_imgs, task_ids)
            d_loss_fake = torch.mean(d_output_fake)

            # Gradient penalty
            gradient_penalty = gan_utils.compute_gradient_penalty(
                local_discriminator,
                real_imgs.data,
                fake_imgs.data,
                local_generator.device,
                task_ids,
            )
            # Wasserstein distance
            wasserstein_distance = -(d_loss_real + d_loss_fake)

            # Adversarial loss
            d_loss = d_loss_fake + d_loss_real + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            # Train the generator every n_critic steps
            if i % n_critic_steps == 0:
                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = local_generator(z, task_ids)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images -> compare predictions to 1
                d_output_fake = local_discriminator(fake_imgs, task_ids)
                g_loss = -torch.mean(d_output_fake)

                g_loss.backward()
                optimizer_g.step()

                if i % 40 == 0:
                    print(
                        f"[Local D] [Epoch {epoch + 1}/{n_epochs}] [Batch {i + 1}/{len(task_loader)}] [D loss: {d_loss.item():.3f}] [D loss fake: {d_loss_fake.item():.3f}] [D loss real: {d_loss_real.item():.3f}] [Gradient penalty: {gradient_penalty.item():.3f}] [Wasserstein distance: {wasserstein_distance.item():.3f}]"
                    )
                    print(
                        f"[Local G] [Epoch {epoch + 1}/{n_epochs}] [Batch {i + 1}/{len(task_loader)}] [G loss: {g_loss.item():.3f}]"
                    )

            wandb.log(
                {
                    f"local_d_loss_task_{task_id}": np.round(d_loss.item(), 3),
                    f"local_d_loss_fake_task_{task_id}": np.round(
                        d_loss_fake.item(), 3
                    ),
                    f"local_d_loss_real_task_{task_id}": np.round(
                        d_loss_real.item(), 3
                    ),
                    f"local_g_loss_task_{task_id}": np.round(g_loss.item(), 3),
                    f"local_gradient_penalty_task_{task_id}": np.round(
                        gradient_penalty.item(), 3
                    ),
                    f"local_wasserstein_distance_task_{task_id}": np.round(
                        wasserstein_distance.item(), 3
                    ),
                }
            )

        if epoch % 1 == 0:
            local_generator.eval()
            generations = local_generator(
                fixed_noise,
                (torch.zeros([num_gen_images]) + task_id).to(local_generator.device),
            )
            wandb.log({f"local_generations_task_{task_id}": wandb.Image(generations)})

        scheduler_g.step()
        scheduler_d.step()


def train_local(
    local_generator,
    local_discriminator,
    n_epochs,
    task_loader,
    task_id,
    local_dis_lr,
    local_gen_lr,
    num_gen_images,
    local_scheduler_rate,
    gan_type,
    n_critic_steps,
    lambda_gp,
    b1,
    b2
):
    local_generator.train()
    local_discriminator.train()
    local_generator.translator.train()

    if gan_type == "wgan":
        train_local_wgan_gp(
            n_epochs,
            task_loader,
            local_generator,
            local_discriminator,
            local_gen_lr,
            local_dis_lr,
            num_gen_images,
            task_id,
            lambda_gp,
            n_critic_steps,
            local_scheduler_rate,
            b1,
            b2
        )
    else:
        train_local_dcgan(
            n_epochs,
            task_loader,
            local_generator,
            local_discriminator,
            local_gen_lr,
            local_dis_lr,
            num_gen_images,
            task_id,
            b1,
            b2
        )


def train_global_generator(
    batch_size,
    task_id,
    limit_previous_examples,
    curr_global_generator,
    n_epochs,
    task_loader,
    curr_local_generator,
    global_gen_lr,
    warmup_rounds,
    num_epochs_noise_optim,
    num_gen_images,
    optim_noise_lr,
    global_scheduler_rate,
):
    global_generator = copy.deepcopy(curr_global_generator)
    global_generator.to(curr_global_generator.device)

    curr_global_generator.eval()
    curr_global_generator.translator.eval()

    criterion = torch.nn.MSELoss()

    optimizer_g = torch.optim.Adam(global_generator.parameters(), lr=global_gen_lr)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_g, gamma=global_scheduler_rate
    )

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(
        num_gen_images, global_generator.latent_dim, device=global_generator.device
    )

    # How many examples from all prev tasks we want to generate
    n_prev_examples = int(batch_size * min(task_id, 3) * limit_previous_examples)

    curr_noise_all = []

    for epoch in range(n_epochs):
        global_generator.train()
        for i, batch in enumerate(task_loader):

            # Generate data -> (noise, generation) pairs for each previous task
            prev_examples, prev_noise, prev_task_ids = gan_utils.generate_previous_data(
                n_prev_tasks=task_id,
                n_prev_examples=n_prev_examples,
                curr_global_generator=curr_global_generator,
            )
            # Real images and optimized noise of current task
            curr_examples = batch[0]
            if not epoch:
                curr_noise = gan_utils.optimize_noise(
                    curr_examples,
                    curr_local_generator,
                    num_epochs_noise_optim,
                    task_id,
                    lr=optim_noise_lr,
                    log=not i,  # log only first batch for readability
                )
                curr_noise_all.append(curr_noise.to("cpu"))
            else:
                curr_noise = curr_noise_all[i].to(global_generator.device)

            curr_task_ids = torch.zeros([len(curr_examples)]) + task_id

            examples_concat = torch.cat(
                [prev_examples, curr_examples.to(global_generator.device)]
            )
            noise_concat = torch.cat([prev_noise, curr_noise])
            task_ids_concat = torch.cat(
                [prev_task_ids, curr_task_ids.to(global_generator.device)]
            )

            # Randomly shuffle examples
            shuffle = torch.randperm(len(task_ids_concat))
            examples_concat = examples_concat[shuffle]
            noise_concat = noise_concat[shuffle]
            task_ids_concat = task_ids_concat[shuffle]

            optimizer_g.zero_grad()

            global_generations = global_generator(noise_concat, task_ids_concat)
            g_loss = criterion(global_generations, examples_concat)

            g_loss.backward()
            optimizer_g.step()

            if i % 20 == 0 or not epoch:
                print(
                    f"[Global G] [Epoch {epoch + 1}/{n_epochs}] [Batch {i + 1}/{len(task_loader)}] [G loss: {g_loss.item():.3f}]"
                )

            if n_prev_examples:
                wandb.log(
                    {
                        f"prev_examples_task_{task_id}": wandb.Image(prev_examples),
                        f"global_generations_task_{task_id}": wandb.Image(
                            global_generations
                        ),
                        f"global_g_loss_task_{task_id}": np.round(g_loss.item(), 3),
                    }
                )
            else:
                wandb.log(
                    {
                        f"global_generations_task_{task_id}": wandb.Image(
                            global_generations
                        ),
                        f"global_g_loss_task_{task_id}": np.round(g_loss.item(), 3),
                    }
                )

        scheduler_g.step()

        if epoch % 1 == 0:
            global_generator.eval()
            for learned_task_id in range(0, task_id + 1):
                generations = global_generator(
                    fixed_noise,
                    (torch.zeros([num_gen_images]) + learned_task_id).to(
                        global_generator.device
                    ),
                    return_emb=False,
                )
                wandb.log(
                    {
                        f"global_generations_task_{task_id}_of_task_{learned_task_id}": wandb.Image(
                            generations
                        )
                    }
                )

    return global_generator
