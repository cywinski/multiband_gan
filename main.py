import argparse
import copy
import os
import random
import sys
from glob import glob
from collections import OrderedDict, Counter
import numpy as np
import torch
import torch.utils.data as data

import continual_benchmark.dataloaders as dataloaders
import continual_benchmark.dataloaders.base
from continual_benchmark.dataloaders.datasetGen import data_split
from gan_experiments import models_definition, gan_utils, multiband_training
from gan_experiments.validation import Validator, CERN_Validator
from visualise import *
from utils import count_parameters


def run(args):
    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](
        args.dataroot, args.skip_normalization, args.train_aug
    )

    if args.dataset.lower() == "celeba":
        num_classes = 10
    else:
        num_classes = train_dataset.number_classes

    num_batches = args.num_batches
    train_dataset_splits, val_dataset_splits, task_output_space = data_split(
        dataset=train_dataset,
        dataset_name=args.dataset.lower(),
        num_batches=num_batches,
        num_classes=num_classes,
        random_split=args.random_split,
        random_mini_shuffle=args.random_shuffle,
        limit_data=args.limit_data,
        dirichlet_split_alpha=args.dirichlet,
        reverse=args.reverse,
        limit_classes=args.limit_classes,
    )

    # Calculate constants
    labels_tasks = {}
    for task_name, task in train_dataset_splits.items():
        labels_tasks[int(task_name)] = task.dataset.class_list

    if hasattr(train_dataset.dataset, "classes"):
        tasks_num_classes_dict = {
            task_id: [train_dataset.dataset.classes[i] for i in class_idxs]
            for task_id, class_idxs in labels_tasks.items()
        }
    else:
        tasks_num_classes_dict = None

    n_tasks = len(labels_tasks)

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print("Task order:", task_names)
    if args.rand_split_order:
        random.shuffle(task_names)
        print("Shuffled task order:", task_names)

    if tasks_num_classes_dict is not None:
        print("Classes order: ", end="")
        print([tasks_num_classes_dict[task_name] for task_name in task_names])

    fid_table = OrderedDict()
    precision_table = OrderedDict()
    recall_table = OrderedDict()
    test_fid_table = OrderedDict()
    fid_local_gan = OrderedDict()

    # Prepare GAN
    translator = models_definition.Translator(
        latent_size=args.latent_dim,
        device=device,
        num_embeddings=num_batches if not args.class_cond else num_classes,
        embedding_dim=num_batches if not args.class_cond else num_classes,
    ).to(device)
    local_generator = models_definition.Generator(
        latent_dim=args.latent_dim,
        img_shape=train_dataset[0][0].shape,
        device=device,
        translator=translator,
        num_features=args.g_n_features,
    ).to(device)
    local_discriminator = models_definition.Discriminator(
        img_shape=train_dataset[0][0].shape,
        device=device,
        num_features=args.d_n_features,
        num_embeddings=0 if not args.class_cond else num_classes,
        embedding_dim=0 if not args.class_cond else num_classes,
    ).to(device)

    class_table = (
        torch.zeros(n_tasks, num_classes, dtype=torch.long) if args.class_cond else None
    )

    print(local_generator)
    print(
        f"Generator number of learnable parmeters: {count_parameters(local_generator)}"
    )
    print(local_discriminator)
    print(
        f"Discriminator number of learnable parmeters: {count_parameters(local_discriminator)}"
    )

    local_train_loaders = []
    global_train_loaders = []
    val_loaders = []
    for task_name in range(n_tasks):
        # Manually shuffle train dataset to disable shuffling by global DataLoader.
        # This way we are getting the same order of batches each epoch, thus we are
        # able to optimize noise during global training only in first epoch.
        train_data_shuffle = torch.randperm(
            len(train_dataset_splits[task_name].dataset)
        )
        train_dataset_splits[task_name].dataset = data.Subset(
            dataset=train_dataset_splits[task_name].dataset, indices=train_data_shuffle
        )
        global_train_dataset_loader = data.DataLoader(
            dataset=train_dataset_splits[task_name],
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        local_train_dataset_loader = data.DataLoader(
            dataset=train_dataset_splits[task_name],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        local_train_loaders.append(local_train_dataset_loader)
        global_train_loaders.append(global_train_dataset_loader)
        val_data = (
            val_dataset_splits[task_name]
            if args.score_on_val
            else train_dataset_splits[task_name]
        )
        val_loader = data.DataLoader(
            dataset=val_data,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
        )

        val_loaders.append(val_loader)

    if args.dirichlet is not None:
        labels_tasks_str = "_".join(
            [
                "_".join(str(label) for label in labels_tasks[task])
                for task in labels_tasks
            ]
        )
        labels_tasks_str = labels_tasks_str[: min(20, len(labels_tasks_str))]
    else:
        labels_tasks_str = ""

    if not args.skip_validation:
        stats_file_name = f"seed_{args.seed}_batches_{args.num_batches}_labels_{labels_tasks_str}_val_{args.score_on_val}_random_{args.random_split}_shuffle_{args.random_shuffle}_dirichlet_{args.dirichlet}_limit_{args.limit_data}_reverse_{args.reverse}_class_cond_{args.class_cond}"

        print("Removing previous stats files: ")
        for f in glob(
            os.path.join(
                "results", "orig_stats", f"{args.dataset}_{stats_file_name}*.npy"
            )
        ):
            print(f)
            os.remove(f)

        if args.dataset.lower() != "cern":
            validator = Validator(
                n_classes=num_classes,
                device=device,
                dataset=args.dataset,
                stats_file_name=stats_file_name,
                score_model_device=args.score_model_device,
                dataloaders=val_loaders,
            )
        else:
            validator = CERN_Validator(
                dataloaders=val_loaders, stats_file_name=stats_file_name, device=device
            )

    curr_global_generator = None

    tasks_to_learn = task_names if not args.only_task_0 else {0}
    print(f"Tasks to learn: {tasks_to_learn}")
    for task_id in tasks_to_learn:
        print(
            f"###### Task number {task_id} -> {tasks_num_classes_dict[task_id]} ######"
        ) if tasks_num_classes_dict is not None else print(
            f"###### Task number {task_id} ######"
        )

        task_name = task_names[task_id]
        local_train_dataset_loader = local_train_loaders[task_id]
        global_train_dataset_loader = global_train_loaders[task_id]

        if args.training_procedure == "multiband":
            (
                curr_local_generator,
                curr_global_generator,
                curr_global_discriminator,
            ) = multiband_training.train_multiband_gan(
                task_id=task_id,
                local_discriminator=local_discriminator,
                local_generator=local_generator,
                local_task_loader=local_train_dataset_loader,
                global_task_loader=global_train_dataset_loader,
                n_local_epochs=args.num_local_epochs,
                n_global_epochs=args.num_global_epochs,
                local_dis_lr=args.local_dis_lr,
                local_gen_lr=args.local_gen_lr,
                num_gen_images=args.num_gen_images,
                local_scheduler_rate=args.local_scheduler_rate,
                global_scheduler_rate=args.global_scheduler_rate,
                n_critic_steps=args.n_critic_steps,
                lambda_gp=args.lambda_gp,
                batch_size=args.batch_size,
                limit_previous_examples=args.limit_previous,
                curr_global_generator=curr_global_generator,
                global_gen_lr=args.global_gen_lr,
                num_epochs_noise_optim=args.num_epochs_noise_optim,
                optim_noise_lr=args.optim_noise_lr,
                local_b1=args.local_b1,
                local_b2=args.local_b2,
                warmup_rounds=args.global_warmup,
                class_cond=args.class_cond,
                class_table=class_table,
                num_classes=num_classes,
            )
        else:
            print("Wrong training procedure")
            return None

        curr_global_generator.eval()
        torch.save(
            curr_local_generator,
            os.path.join(
                args.rpath,
                args.dataset,
                args.experiment_name,
                f"model{task_id}_curr_local_generator",
            ),
        )
        torch.save(
            curr_global_generator,
            os.path.join(
                args.rpath,
                args.dataset,
                args.experiment_name,
                f"model{task_id}_curr_global_generator",
            ),
        )
        torch.save(
            curr_global_discriminator,
            os.path.join(
                args.rpath,
                args.dataset,
                args.experiment_name,
                f"model{task_id}_curr_global_discriminator",
            ),
        )
        print("Models saved")

        fid_table[task_name] = OrderedDict()
        precision_table[task_name] = OrderedDict()
        recall_table[task_name] = OrderedDict()
        if args.skip_validation:
            for j in range(task_id + 1):
                fid_table[j][task_name] = -1
        else:
            if (args.training_procedure == "multiband") and (
                not args.gen_load_pretrained_models
            ):
                (
                    fid_result,
                    precision,
                    recall,
                    generated_classes,
                ) = validator.calculate_results(
                    curr_global_generator=curr_local_generator,
                    task_id=task_id,
                    calculate_class_dist=args.dataset.lower() == "mnist",
                    batch_size=args.val_batch_size,
                    class_cond=args.class_cond,
                )

                fid_local_gan[task_id] = fid_result
                print(f"FID local GAN: {fid_result}")
                wandb.log({f"local_gan_FID_task_{task_id}": fid_result})
                if len(generated_classes):
                    wandb.log(
                        {
                            f"local_gan_generated_classes_task_{task_id}": wandb.Histogram(
                                generated_classes, num_bins=num_classes
                            )
                        }
                    )
                    print(f"Generated classes: {Counter(generated_classes)}")

            for j in range(task_id + 1):
                val_name = task_names[j]
                print("validation split name:", val_name)
                (
                    fid_result,
                    precision,
                    recall,
                    generated_classes,
                ) = validator.calculate_results(
                    curr_global_generator=curr_global_generator,
                    task_id=j,
                    calculate_class_dist=args.dataset.lower() == "mnist",
                    batch_size=args.val_batch_size,
                    class_cond=args.class_cond,
                )
                fid_table[j][task_name] = fid_result
                precision_table[j][task_name] = precision
                recall_table[j][task_name] = recall
                print(f"FID task {j}: {fid_result}")

                wandb.log({f"global_gan_FID_task_{j}": fid_result})
                if len(generated_classes):
                    wandb.log(
                        {
                            f"global_gan_generated_classes_task_{j}": wandb.Histogram(
                                generated_classes, num_bins=num_classes
                            )
                        }
                    )
                    print(f"Generated classes: {Counter(generated_classes)}")

                if args.class_cond:
                    n_classes_per_task = num_classes // num_batches
                    classes_to_generate = [
                        c for c in range(j * 2, (j * 2) + n_classes_per_task)
                    ]
                    task_ids = torch.cat(
                        [
                            (
                                torch.zeros(
                                    [args.num_gen_images // len(classes_to_generate)]
                                )
                                + c
                            )
                            for c in classes_to_generate
                        ]
                    ).to(curr_global_generator.device)
                    generations = curr_global_generator(
                        torch.randn(len(task_ids), curr_global_generator.latent_dim).to(
                            curr_global_generator.device
                        ),
                        task_ids,
                    )
                else:
                    generations = curr_global_generator(
                        torch.randn(
                            args.num_gen_images, curr_global_generator.latent_dim
                        ).to(curr_global_generator.device),
                        (torch.zeros([args.num_gen_images]) + j).to(
                            curr_global_generator.device
                        ),
                    )

                wandb.log(
                    {
                        f"final_generations_task_{j}": wandb.Image(generations),
                    }
                )

        local_generator = copy.deepcopy(curr_global_generator)
        if args.new_d_every_task:
            print("Building new discriminator")
            local_discriminator = models_definition.Discriminator(
                img_shape=train_dataset[0][0].shape,
                device=device,
                num_features=args.d_n_features,
                num_embeddings=0 if not args.class_cond else num_classes,
                embedding_dim=0 if not args.class_cond else num_classes,
            ).to(device)

        else:
            print("Using discriminator from previous task")
            local_discriminator = copy.deepcopy(curr_global_discriminator)

    return (
        fid_table,
        task_names,
        test_fid_table,
        precision_table,
        recall_table,
        fid_local_gan,
    )


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_run",
        help="Name of current experiment",
    )
    parser.add_argument(
        "--rpath", type=str, default="results/", help="Directory to save results"
    )
    parser.add_argument(
        "--gpuid",
        nargs="+",
        type=int,
        default=[0],
        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat the experiment N times"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Random seed. If defined all random operations will be reproducible",
    )
    parser.add_argument(
        "--dataset", type=str, default="MNIST", help="Dataset to train on"
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="data/",
        help="The root folder of dataset or downloaded data",
    )
    parser.add_argument(
        "--skip_normalization",
        action="store_true",
        help="Loads dataset without normalization",
    )
    parser.add_argument(
        "--train_aug",
        dest="train_aug",
        default=False,
        action="store_true",
        help="Allow data augmentation during training",
    )
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument(
        "--rand_split",
        dest="rand_split",
        default=False,
        action="store_true",
        help="Randomize the classes in splits",
    )
    parser.add_argument(
        "--random_shuffle",
        dest="random_shuffle",
        default=False,
        action="store_true",
        help="Move part of data to next batch",
    )
    parser.add_argument(
        "--random_split",
        dest="random_split",
        default=False,
        action="store_true",
        help="Randomize data in splits",
    )
    parser.add_argument(
        "--limit_data", type=float, default=None, help="limit_data to given %"
    )
    parser.add_argument(
        "--dirichlet",
        default=None,
        type=float,
        help="Alpha parameter for dirichlet data split",
    )
    parser.add_argument(
        "--reverse",
        dest="reverse",
        default=False,
        action="store_true",
        help="Reverse the ordering of batches",
    )
    parser.add_argument("--limit_classes", type=int, default=-1)
    parser.add_argument(
        "--rand_split_order",
        dest="rand_split_order",
        default=False,
        action="store_true",
        help="Randomize the order of splits",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="Latent dimension of Generator"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--score_on_val",
        action="store_true",
        required=False,
        default=False,
        help="Compute FID on validation dataset instead of training dataset",
    )
    parser.add_argument("--val_batch_size", type=int, default=250)
    parser.add_argument(
        "--gen_load_pretrained_models",
        default=False,
        action="store_true",
        help="Load pretrained generative models",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of threads for dataloader"
    )
    parser.add_argument("--skip_validation", default=False, action="store_true")
    parser.add_argument(
        "--score_model_device",
        default="cpu",
        type=str,
        help="Device to score model on",
        choices=["cpu", "gpu"],
    )
    parser.add_argument(
        "--training_procedure",
        type=str,
        default="multiband",
        help="Training procedure multiband|replay",
    )
    parser.add_argument(
        "--num_local_epochs",
        type=int,
        default=120,
        help="Number of epochs to train local GAN",
    )
    parser.add_argument(
        "--num_global_epochs",
        type=int,
        default=200,
        help="Number of epochs to train global GAN",
    )
    parser.add_argument(
        "--num_epochs_noise_optim",
        type=int,
        default=1000,
        help="Number of epochs to optimize noise in global training",
    )
    parser.add_argument("--local_dis_lr", type=float, default=0.0002)
    parser.add_argument("--local_gen_lr", type=float, default=0.0002)
    parser.add_argument("--global_gen_lr", type=float, default=0.001)
    parser.add_argument("--optim_noise_lr", type=float, default=0.1)
    parser.add_argument(
        "--num_gen_images",
        type=int,
        default=32,
        help="Number of images to generate each epoch",
    )
    parser.add_argument("--local_scheduler_rate", type=float, default=0.99)
    parser.add_argument("--global_scheduler_rate", type=float, default=0.99)
    parser.add_argument(
        "--n_critic_steps",
        type=int,
        default=5,
        help="Train the generator every n_critic steps",
    )
    parser.add_argument("--lambda_gp", type=int, default=10)
    parser.add_argument(
        "--limit_previous",
        default=0.5,
        type=float,
        help="How much of previous data we want to generate each epoch",
    )
    parser.add_argument(
        "--d_n_features",
        type=int,
        default=32,
        help="Number of features in discriminator",
    )
    parser.add_argument(
        "--g_n_features", type=int, default=32, help="Number of features in generator"
    )
    parser.add_argument(
        "--local_b1",
        default=0.0,
        type=float,
        help="Beta1 parameter of local Adam optimizer",
    )
    parser.add_argument(
        "--local_b2",
        default=0.9,
        type=float,
        help="Beta2 parameter of local Adam optimizer",
    )
    parser.add_argument(
        "--task_embedding_dim",
        default=5,
        type=int,
        help="Dimension of task embedding used in torch.nn.Embedding()",
    )
    parser.add_argument(
        "--only_task_0",
        dest="only_task_0",
        default=False,
        action="store_true",
        help="Train only local GAN on first task",
    )
    parser.add_argument(
        "--new_d_every_task",
        dest="new_d_every_task",
        default=False,
        action="store_true",
        help="Train new discriminator every task, if False -> model from previous task will be used",
    )
    parser.add_argument(
        "--global_warmup",
        default=10,
        type=int,
        help="Number of epochs for global warmup - only translator training",
    )
    parser.add_argument("--wandb_project", type=str, default="MultibandGAN")
    parser.add_argument(
        "--class_cond",
        dest="class_cond",
        default=False,
        action="store_true",
        help="Use class conditioning during training",
    )
    args = parser.parse_args(argv)

    return args


if __name__ == "__main__":
    args = get_args(sys.argv[1:])

    wandb.init(
        project=f"{args.wandb_project}_{args.dataset}",
        name=f"{args.experiment_name}",
        config=vars(args),
    )

    torch.cuda.set_device(args.gpuid[0])
    device = torch.device("cuda")

    if args.seed:
        print("Using manual seed = {}".format(args.seed))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print(
            "WARNING: Not using manual seed - your experiments will not be reproducible"
        )

    acc_val, acc_test, precision_table, recall_table = {}, {}, {}, {}
    os.makedirs(
        os.path.join(args.rpath, args.dataset, args.experiment_name), exist_ok=True
    )
    with open(
        os.path.join(args.rpath, args.dataset, args.experiment_name, "args.txt"), "w"
    ) as text_file:
        text_file.write(str(args))
    for r in range(args.repeat):
        (
            acc_val[r],
            _,
            acc_test[r],
            precision_table[r],
            recall_table[r],
            fid_local_gan,
        ) = run(args)
    np.save(
        os.path.join(args.rpath, args.dataset, args.experiment_name, "fid.npy"), acc_val
    )
    np.save(
        os.path.join(args.rpath, args.dataset, args.experiment_name, "precision.npy"),
        precision_table,
    )
    np.save(
        os.path.join(args.rpath, args.dataset, args.experiment_name, "recall.npy"),
        recall_table,
    )
    np.save(
        os.path.join(
            args.rpath, args.dataset, args.experiment_name, "fid_local_gan.npy"
        ),
        fid_local_gan,
    )

    plot_final_results(
        [args.experiment_name],
        type="fid",
        fid_local_gan=fid_local_gan,
        rpath=f"{args.rpath}/{args.dataset}/",
    )
    print(fid_local_gan)
