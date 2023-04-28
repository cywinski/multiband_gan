import numpy as np
import torch
import torch.autograd as autograd
import wandb
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torch import Tensor
from torch.autograd import Variable


def compute_gradient_penalty(D, real_samples, fake_samples, device, task_ids):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates, task_ids)
    fake = Variable(
        Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False
    ).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def generate_images_grid(generator, device, task_ids, noise=None):
    generations = (
        generator(
            torch.randn(len(task_ids), generator.latent_dim, device=device),
            task_ids.to(device),
        )
        .detach()
        .cpu()
        if noise is None
        else generator(noise, task_ids.to(device)).detach().cpu()
    )
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(len(task_ids) // 4, len(task_ids) // 4),
        axes_pad=0.5,
    )
    for ax, im in zip(grid, generations):
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 0, 1)
        ax.imshow(im.squeeze(), cmap="gray")

    return fig


def prepare_class_samplers(task_id, class_table):
    class_samplers = []
    for task_id in range(task_id):
        local_probs = class_table[task_id] * 1.0 / torch.sum(class_table[task_id])
        class_samplers.append(
            torch.distributions.categorical.Categorical(probs=local_probs)
        )
    return class_samplers


def generate_previous_data(
    n_prev_tasks,
    n_prev_examples,
    curr_global_generator,
    class_table=None,
    biggan_training=False,
):
    curr_global_generator.eval()
    with torch.no_grad():
        if not n_prev_examples:
            return (
                torch.Tensor().to(curr_global_generator.device),
                torch.Tensor().to(curr_global_generator.device),
                torch.Tensor().to(curr_global_generator.device),
            )

        if class_table is None:
            # Generate equally distributed examples from previous tasks
            tasks_dist = [n_prev_examples // n_prev_tasks for _ in range(n_prev_tasks)]
        else:
            # Generate replays based on class distribution
            curr_class_table = class_table[:n_prev_tasks]
            tasks_dist = (
                torch.sum(curr_class_table, dim=1)
                * n_prev_examples
                // torch.sum(curr_class_table)
            )
            tasks_dist[: n_prev_examples - tasks_dist.sum()] += 1

        task_ids = []
        for task_id in range(n_prev_tasks):
            if tasks_dist[task_id]:
                task_ids.append([task_id] * tasks_dist[task_id])

        # Tensor of tasks ids to generate
        task_ids = (
            torch.from_numpy(np.concatenate(task_ids))
            .float()
            .to(curr_global_generator.device)
        )

        if class_table is not None:
            class_samplers = prepare_class_samplers(n_prev_tasks, curr_class_table)

            sampled_classes = []
            for task_id in range(n_prev_tasks):
                if tasks_dist[task_id] > 0:
                    sampled_classes.append(
                        class_samplers[task_id].sample(tasks_dist[task_id].view(-1, 1))
                    )
            sampled_classes = torch.cat(sampled_classes).to(
                curr_global_generator.device
            )
            task_ids = sampled_classes

        random_noise = torch.randn(len(task_ids), curr_global_generator.latent_dim).to(
            curr_global_generator.device
        )

        if biggan_training:
            generations = curr_global_generator(
                random_noise, curr_global_generator.shared(task_ids.long())
            )
        else:
            generations = curr_global_generator(
                random_noise, task_ids
            )

        return generations, random_noise, task_ids


def optimize_noise(
    images,
    generator,
    n_iterations,
    task_id,
    lr,
    log=False,
    labels=None,
    biggan_training=False,
):
    generator.eval()

    images = images.to(generator.device)
    task_ids = (
        (torch.zeros([len(images)]) + task_id).to(generator.device)
        if labels is None
        else labels.to(generator.device)
    )
    if biggan_training:
        task_ids = generator.shared(task_ids.long())

    criterion = torch.nn.MSELoss()

    noise = torch.randn(len(images), generator.latent_dim).to(generator.device)
    noise.requires_grad = True

    optimizer = torch.optim.Adam([noise], lr=lr)
    for i in range(n_iterations):
        optimizer.zero_grad()
        generations = generator(noise, task_ids)
        loss = criterion(generations, images)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(
                f"[Noise optimization] [Epoch {i}/{n_iterations}] [Loss: {loss.item():.3f}]"
            )

        if log:
            wandb.log(
                {
                    f"loss_optimization/task_{task_id}": np.round(loss.item(), 3),
                }
            )

    return noise
