import copy

import torch

from gan_experiments import training_functions


def train_multiband_gan(
        task_id, local_discriminator, local_generator, task_loader, n_epochs, local_dis_lr, local_gen_lr,
        num_gen_images, local_scheduler_rate,
        ):
    training_functions.train_local(
            local_generator=local_generator,
            local_discriminator=local_discriminator,
            n_epochs=n_epochs,
            task_loader=task_loader,
            task_id=task_id,
            local_dis_lr=local_dis_lr,
            local_gen_lr=local_gen_lr,
            num_gen_images=num_gen_images,
            local_scheduler_rate=local_scheduler_rate,
            )
    print("Done training local GAN model")

    curr_global_generator = copy.deepcopy(local_generator)
    curr_global_discriminator = copy.deepcopy(local_discriminator)

    torch.cuda.empty_cache()
    return curr_global_generator, curr_global_discriminator
