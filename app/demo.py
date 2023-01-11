import torch
from torchvision.utils import save_image
import gradio as gr
import numpy as np
import umap
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch
import sys

sys.path.append("../")

LATENT_SIZE = 100
DEVICE = "cuda"
BASE_LATENT = None
BASE_GENERATIONS = None
REDUCER_UMAP = umap.UMAP()
NUM_NOISE_SAMPLES_TO_PLOT = 300
NOISE_TO_PLOT = torch.randn([NUM_NOISE_SAMPLES_TO_PLOT, LATENT_SIZE], device=DEVICE)
SCENARIOS_MAP = {
    "Class-Incremental 5 tasks": {"num_tasks": 5, "dir": "CI_5"},
    "Class-Incremental 10 tasks": {
        "num_tasks": 10,
        "dir": "CI_10",
    },
    "Class-Incremental 20 tasks": {
        "num_tasks": 20,
        "dir": "CI_20",
    },
    "Dirichlet alpha=1 10 tasks": {
        "num_tasks": 10,
        "dir": "Dirichlet_alpha_1_10",
    },
    "Dirichlet alpha=1 20 tasks": {
        "num_tasks": 20,
        "dir": "Dirichlet_alpha_1_20",
    },
    "Dirichlet alpha=100 10 tasks": {
        "num_tasks": 10,
        "dir": "Dirichlet_alpha_100_10",
    },
}
DISPLAY_IMG_W = 200
DISPLAY_IMG_H = 200


def get_model_dir(dataset, scenario):
    scenario_dict = SCENARIOS_MAP[scenario]
    scenario_dir = scenario_dict["dir"]
    scenario_num_tasks = scenario_dict["num_tasks"]
    return f"../results/{dataset}/{scenario_dir}/model{scenario_num_tasks-1}_curr_global_generator"


def load_generator(dataset, scenario):
    generator = torch.load(get_model_dir(dataset, scenario), map_location=DEVICE)
    generator.eval()
    return generator


def generate(generator, task_id, n_imgs, random_noise=None):
    task_ids = (torch.zeros([n_imgs]) + task_id).to(DEVICE)

    if random_noise is None:
        z = torch.randn(n_imgs, generator.latent_dim).to(DEVICE)
    else:
        random_noise = np.array(random_noise.values, dtype=np.float16).flatten()
        z = torch.from_numpy(random_noise).unsqueeze(0).float().to(DEVICE)

    return generator(z, task_ids)


def predict(seed, dataset, task_id, n_imgs_to_generate, scenario, noise=None):
    torch.manual_seed(seed)
    generator = load_generator(dataset, scenario)
    generations = generate(generator, task_id, n_imgs_to_generate, random_noise=noise)
    save_image(generations, "generations.png", normalize=True)
    return gr.Image.update(value="generations.png")


def update_scenarios(dataset):
    global BASE_LATENT
    BASE_LATENT = None
    dataset_name = dataset.lower()
    if dataset_name in ["mnist", "fashionmnist"]:
        return gr.update(
            choices=["Class-Incremental 5 tasks", "Dirichlet alpha=1 10 tasks"],
            value=None,
        ), gr.update(visible=False, value=None)
    elif dataset_name == "omniglot":
        return gr.update(
            choices=[
                "Class-Incremental 5 tasks",
                "Class-Incremental 20 tasks",
                "Dirichlet alpha=1 20 tasks",
            ]
        ), gr.update(visible=False, value=None)
    elif dataset_name == "doublemnist":
        return gr.update(choices=["Class-Incremental 10 tasks"]), gr.update(
            visible=False, value=None
        )
    elif dataset_name == "cern":
        return gr.update(choices=["Class-Incremental 5 tasks"]), gr.update(
            visible=False, value=None
        )
    elif dataset_name == "celeba":
        return gr.update(
            choices=[
                "Class-Incremental 5 tasks",
                "Dirichlet alpha=1 10 tasks",
                "Dirichlet alpha=100 10 tasks",
            ]
        ), gr.update(visible=False, value=None)
    elif dataset_name == "cifar10":
        return gr.update(choices=["Class-Incremental 5 tasks"]), gr.update(
            visible=False, value=None
        )
    else:
        raise NotImplementedError("Wrong dataset")


def update_task_ids(scenario):
    global BASE_LATENT
    BASE_LATENT = None
    if scenario:
        scenario_dict = SCENARIOS_MAP[scenario]
        return gr.update(
            choices=[t for t in range(scenario_dict["num_tasks"])], visible=True
        )


def plot_latent(extra_noise, dataset, scenario, task_id):
    fig, ax = plt.subplots()
    generator = load_generator(dataset, scenario)
    global BASE_LATENT
    global BASE_GENERATIONS

    if BASE_LATENT is None:
        task_ids = (torch.zeros([NUM_NOISE_SAMPLES_TO_PLOT]) + task_id).to(DEVICE)
        base_generations, base_latent = generator(
            NOISE_TO_PLOT, task_ids, return_emb=True
        )
        base_latent = base_latent.detach().cpu().numpy()
        base_latent = REDUCER_UMAP.fit_transform(base_latent)
        BASE_LATENT = base_latent
        BASE_GENERATIONS = base_generations

    xs = BASE_LATENT[:, 0]
    ys = BASE_LATENT[:, 1]

    ax.scatter(
        x=xs,
        y=ys,
        color="b",
        linewidths=1,
        edgecolors="b",
        label="Random noise",
        alpha=0.5,
    )

    # plot examplary generations in latent space
    for j in range(len(xs)):
        if j % 20 == 0:
            generation = BASE_GENERATIONS[j].cpu().detach().numpy()
            generation = np.swapaxes(generation, 0, 2)
            generation = np.swapaxes(generation, 0, 1)
            imagebox = OffsetImage(
                generation,
                zoom=0.7,
                cmap="gray"
                if dataset.lower() in ["mnist", "fashionmnist", "omniglot", "cern", "doublemnist"]
                else None,
                alpha=0.8,
            )
            ab = AnnotationBbox(imagebox, (xs[j], ys[j]), frameon=False)
            ax.add_artist(ab)

    if extra_noise is not None:
        extra_noise = np.array(extra_noise.values, dtype=np.float16).flatten()
        extra_noise = torch.from_numpy(extra_noise).unsqueeze(0).to(DEVICE)
        task_ids = (torch.zeros([1]) + task_id).to(DEVICE)
        extra_generation, noise_in_latent = generator(
            extra_noise, task_ids, return_emb=True
        )
        noise_in_latent = noise_in_latent.detach().cpu().numpy()
        noise_in_latent = REDUCER_UMAP.transform(noise_in_latent.reshape(1, -1))

        ax.scatter(
            noise_in_latent[0, 0],
            noise_in_latent[0, 1],
            color="r",
            label="Provided noise with the generation",
            linewidths=1,
            marker="X",
        )

        extra_generation = extra_generation[0].cpu().detach().numpy()
        extra_generation = np.swapaxes(extra_generation, 0, 2)
        extra_generation = np.swapaxes(extra_generation, 0, 1)
        imagebox = OffsetImage(
            extra_generation,
            zoom=0.7,
            cmap="gray"
            if dataset.lower() in ["mnist", "fashionmnist", "omniglot", "cern", "doublemnist"]
            else None,
        )
        ab = AnnotationBbox(
            imagebox,
            (noise_in_latent[0, 0] + 0.3, noise_in_latent[0, 1] + 0.3),
            frameon=True,
        )
        ax.add_artist(ab)

    plt.axis("off")
    plt.legend()
    plt.title(f"Translator's latent space for task {task_id}")
    return gr.update(value=fig, visible=True)


with gr.Blocks() as demo:
    gr.Markdown("## MultibandGAN inference")
    with gr.Tab("Inference from random noise"):
        with gr.Row():
            with gr.Column():
                seed = gr.Slider(0, 100, label="Seed", value=42)
                with gr.Row():
                    dataset = gr.Dropdown(
                        choices=[
                            "MNIST",
                            "FashionMNIST",
                            "DoubleMNIST",
                            "Omniglot",
                            "CelebA",
                            "CIFAR10",
                            "CERN",
                        ],
                        label="Dataset",
                    )
                    scenario = gr.Dropdown(label="Scenario", interactive=True)
                task_id = gr.Radio(label="Task ID", interactive=True, visible=False)
                n_imgs_to_generate = gr.Slider(
                    1, 64, label="Number of Images", step=1, value=32, visible=True
                )
                gen_bttn = gr.Button("Generate")

            with gr.Column():
                generated_images = gr.Image(label="Generations", interactive=False)

        # generate images
        gen_bttn.click(
            fn=predict,
            inputs=[seed, dataset, task_id, n_imgs_to_generate, scenario],
            outputs=generated_images,
        )
        # display possible scenarios for dataset
        dataset.change(
            fn=update_scenarios, inputs=[dataset], outputs=[scenario, task_id]
        )
        # display possible task_ids for scenario
        scenario.change(fn=update_task_ids, inputs=scenario, outputs=task_id)

    with gr.Tab("Inference from provided noise"):
        seed = gr.Slider(value=42, visible=False)

        with gr.Row():
            dataset = gr.Dropdown(
                choices=[
                    "MNIST",
                    "FashionMNIST",
                    "DoubleMNIST",
                    "Omniglot",
                    "CelebA",
                    "CIFAR10",
                    "CERN",
                ],
                label="Dataset",
            )
            scenario = gr.Dropdown(label="Scenario", interactive=True)
        task_id = gr.Radio(label="Task ID", interactive=True, visible=False)
        n_imgs_to_generate = gr.Slider(value=1, visible=False)
        noise = gr.DataFrame(
            value=[np.around(np.random.normal(size=10), 3).tolist() for _ in range(10)],
            row_count=10,
            col_count=10,
            headers=None,
            label="Noise",
        )
        gen_bttn = gr.Button("Generate")
        with gr.Row():
            # generated_images = gr.Image(label="Generation", interactive=False).style(
            #     height=DISPLAY_IMG_H, width=DISPLAY_IMG_W
            # )
            latent_plot = gr.Plot(interactive=False, label="Translator's latent space")

        # generate images
        # gen_bttn.click(
        #     fn=predict,
        #     inputs=[seed, dataset, task_id, n_imgs_to_generate, scenario, noise],
        #     outputs=generated_images,
        # )
        # display possible scenarios for dataset
        dataset.change(
            fn=update_scenarios, inputs=[dataset], outputs=[scenario, task_id]
        )
        # display possible task_ids for scenario
        scenario.change(fn=update_task_ids, inputs=scenario, outputs=task_id)
        # plot new noise in generator's latent
        gen_bttn.click(
            fn=plot_latent,
            inputs=[noise, dataset, scenario, task_id],
            outputs=latent_plot,
        )

demo.launch()
