import copy

import torch

from gan_experiments import training_functions


def train_multiband_gan(
        task_id, local_discriminator, local_generator, task_loader, n_local_epochs, n_global_epochs, local_dis_lr,
        local_gen_lr,
        num_gen_images, local_scheduler_rate, gan_type, n_critic_steps, lambda_gp, batch_size,
        limit_previous_examples, curr_global_generator,
        global_gen_lr, num_epochs_noise_optim, optim_noise_lr
        ):
    print(f"Started training local GAN model on task nr {task_id}")
    training_functions.train_local(
            local_generator=local_generator,
            local_discriminator=local_discriminator,
            n_epochs=n_local_epochs,
            task_loader=task_loader,
            task_id=task_id,
            local_dis_lr=local_dis_lr,
            local_gen_lr=local_gen_lr,
            num_gen_images=num_gen_images,
            local_scheduler_rate=local_scheduler_rate,
            gan_type=gan_type,
            n_critic_steps=n_critic_steps,
            lambda_gp=lambda_gp,
            )
    print(f"Done training local GAN model on task nr {task_id}")
    curr_global_discriminator = copy.deepcopy(local_discriminator)
    if not task_id:
        curr_global_generator = copy.deepcopy(local_generator)
    else:
        print(f"Started training global GAN model on task nr {task_id}")
        curr_global_generator = training_functions.train_global_generator(
                batch_size=batch_size,
                task_id=task_id,
                limit_previous_examples=limit_previous_examples,
                curr_global_generator=curr_global_generator,
                n_epochs=n_global_epochs,
                task_loader=task_loader,
                curr_local_generator=local_generator,
                global_gen_lr=global_gen_lr,
                warmup_rounds=10,
                num_epochs_noise_optim=num_epochs_noise_optim,
                num_gen_images=num_gen_images,
                optim_noise_lr=optim_noise_lr
                )

        print(f"Done training global GAN model on task nr {task_id}")
    torch.cuda.empty_cache()

    return curr_global_generator, curr_global_discriminator
