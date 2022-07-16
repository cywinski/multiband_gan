import argparse
import copy
import os
import random
import sys
from collections import OrderedDict

import torch
import torch.utils.data as data

import continual_benchmark.dataloaders as dataloaders
import continual_benchmark.dataloaders.base
from continual_benchmark.dataloaders.datasetGen import data_split
from gan_experiments import models_definition, gan_utils, multiband_training
from gan_experiments.validation import Validator
from visualise import *


# Exemplary run:
# python .\main.py --experiment_name testGAN_new_repo2 --dataset MNIST --gpuid 0 --seed 42 --skip_normalization --num_batches 5 --latent_dim 128 --batch_size 64 --score_on_val
# --num_local_epochs 100 --num_global_epochs 100 --local_dis_lr 0.002 --local_gen_lr 0.002


def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.skip_normalization,
                                                                         args.train_aug)

    if args.dataset.lower() == "celeba":
        num_classes = 10
    else:
        num_classes = train_dataset.number_classes

    num_batches = args.num_batches
    train_dataset_splits, val_dataset_splits, task_output_space = data_split(dataset=train_dataset,
                                                                             dataset_name=args.dataset.lower(),
                                                                             num_batches=num_batches,
                                                                             num_classes=num_classes,
                                                                             random_split=args.random_split,
                                                                             random_mini_shuffle=args.random_shuffle,
                                                                             limit_data=args.limit_data,
                                                                             dirichlet_split_alpha=args.dirichlet,
                                                                             reverse=args.reverse,
                                                                             limit_classes=args.limit_classes)

    # Calculate constants
    labels_tasks = {}
    for task_name, task in train_dataset_splits.items():
        labels_tasks[int(task_name)] = task.dataset.class_list

    n_tasks = len(labels_tasks)

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)
    if args.rand_split_order:
        random.shuffle(task_names)
        print('Shuffled task order:', task_names)
    fid_table = OrderedDict()
    precision_table = OrderedDict()
    recall_table = OrderedDict()
    test_fid_table = OrderedDict()
    fid_local_gan = OrderedDict()

    # Prepare GAN
    local_generator = models_definition.Generator(latent_dim=args.latent_dim, img_shape=train_dataset[0][0].shape,
                                                  device=device).to(device)
    local_discriminator = models_definition.Discriminator(img_shape=train_dataset[0][0].shape, device=device).to(device)
    local_generator.apply(gan_utils.weights_init_normal)
    local_discriminator.apply(gan_utils.weights_init_normal)
    # translate_noise = True

    print(local_generator)
    print(local_discriminator)

    class_table = torch.zeros(n_tasks, num_classes, dtype=torch.long)
    train_loaders = []
    val_loaders = []
    for task_name in range(n_tasks):
        train_dataset_loader = data.DataLoader(dataset=train_dataset_splits[task_name],
                                               batch_size=args.batch_size, shuffle=True,
                                               drop_last=False)

        train_loaders.append(train_dataset_loader)
        val_data = val_dataset_splits[task_name] if args.score_on_val else train_dataset_splits[task_name]
        val_loader = data.DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                                     num_workers=args.workers)

        val_loaders.append(val_loader)

    if args.dirichlet is not None:
        labels_tasks_str = "_".join(["_".join(str(label) for label in labels_tasks[task]) for task in labels_tasks])
        labels_tasks_str = labels_tasks_str[:min(20, len(labels_tasks_str))]
    else:
        labels_tasks_str = ""

    if not args.skip_validation:
        stats_file_name = f"seed_{args.seed}_batches_{args.num_batches}_labels_{labels_tasks_str}_val_{args.score_on_val}_random_{args.random_split}_shuffle_{args.random_shuffle}_dirichlet_{args.dirichlet}_limit_{args.limit_data}"
        # if args.dataset.lower() != "cern":
        validator = Validator(n_classes=num_classes, device=device, dataset=args.dataset,
                              stats_file_name=stats_file_name,
                              score_model_device=args.score_model_device, dataloaders=val_loaders)
        # else:
        #     validator = CERN_Validator(dataloaders=val_loaders, stats_file_name=stats_file_name, device=device)

    curr_global_generator = None
    curr_global_discriminator = None

    wandb.init(
            project="MultibandGAN",
            name=f"{args.experiment_name}",
            config={
                    'local_dis_lr': args.local_dis_lr,
                    'local_gen_lr': args.local_gen_lr,
                    'global_dis_lr': args.global_dis_lr,
                    'global_gen_lr': args.global_gen_lr,
                    'batch_size': args.batch_size,
                    'latent_dim': args.latent_dim,
                    'dataset': args.dataset,
                    'num_local_epochs': args.num_local_epochs,
                    'num_global_epochs': args.num_global_epochs,
                    'local_scheduler_rate': args.local_scheduler_rate,
                    })

    # for task_id in range(len(task_names)):
    for task_id in {0}:
        print("######### Task number {} #########".format(task_id))

        task_name = task_names[task_id]

        print("Train local GAN model")
        train_dataset_loader = train_loaders[task_id]

        if args.training_procedure == "multiband":
            curr_global_generator, curr_global_discriminator = multiband_training.train_multiband_gan(
                    task_id=task_id,
                    local_discriminator=local_discriminator,
                    local_generator=local_generator,
                    task_loader=train_dataset_loader,
                    n_epochs=args.num_global_epochs + args.num_local_epochs if task_id == 0 else args.num_local_epochs,
                    local_dis_lr=args.local_dis_lr,
                    local_gen_lr=args.local_gen_lr,
                    num_gen_images=args.num_gen_images,
                    local_scheduler_rate=args.local_scheduler_rate,
                    )
        else:
            print("Wrong training procedure")
            return None

        fig = gan_utils.generate_images_grid(curr_global_generator, args.num_gen_images, task_id, device,
                                             experiment_name=args.experiment_name)
        fig.savefig(f"results/{args.experiment_name}/generations_task_{task_id}")
        wandb.log({
                f"generations_{args.experiment_name}_{task_id}": fig
                })
        plt.close(fig)

        torch.save(curr_global_generator, f"results/{args.experiment_name}/model{task_id}_curr_generator")
        torch.save(curr_global_discriminator, f"results/{args.experiment_name}/model{task_id}_curr_discriminator")

        # Plotting results for already learned tasks
        # if not args.gen_load_pretrained_models:
        #     vae_utils.plot_results(args.experiment_name, curr_global_decoder, class_table, task_id,
        #                            translate_noise=translate_noise, same_z=False)
        #     if args.training_procedure == "multiband":
        #         vae_utils.plot_results(args.experiment_name, local_vaegan.decoder, class_table, task_id,
        #                                translate_noise=translate_noise, suffix="_local_vae", same_z=False,
        #                                starting_point=local_vaegan.starting_point)
        #         torch.save(local_vaegan, f"results/{args.experiment_name}/model{task_id}_local_vaegan")
        #
        #     torch.save(curr_global_decoder, f"results/{args.experiment_name}/model{task_id}_curr_decoder")
        #     torch.save(curr_global_discriminator, f"results/{args.experiment_name}/model{task_id}_curr_discriminator")

        fid_table[task_name] = OrderedDict()
        precision_table[task_name] = OrderedDict()
        recall_table[task_name] = OrderedDict()
        if args.skip_validation:
            for j in range(task_id + 1):
                fid_table[j][task_name] = -1
        else:
            if (args.training_procedure == "multiband") and (not args.gen_load_pretrained_models):
                fid_result, precision, recall = validator.calculate_results(curr_global_generator=curr_global_generator,
                                                                            task_id=task_id,
                                                                            )

                fid_local_gan[task_id] = fid_result
                print(f"FID local GAN: {fid_result}")
                wandb.log({f"local_gan_FID_task_{task_id}": fid_result})
            for j in range(task_id + 1):
                val_name = task_names[j]
                print('validation split name:', val_name)
                fid_result, precision, recall = validator.calculate_results(curr_global_generator=curr_global_generator,
                                                                            task_id=j)
                fid_table[j][task_name] = fid_result
                precision_table[j][task_name] = precision
                recall_table[j][task_name] = recall
                print(f"FID task {j}: {fid_result}")
                wandb.log({f"global_gan_FID_task_{task_id}": fid_result})

            local_generator = copy.deepcopy(curr_global_generator)
            local_discriminator = copy.deepcopy(curr_global_discriminator)
    return fid_table, task_names, test_fid_table, precision_table, recall_table, fid_local_gan


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default='default_run', help='Name of current experiment')
    parser.add_argument('--rpath', type=str, default='results/', help='Directory to save results')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--seed', type=int, required=False,
                        help="Random seed. If defined all random operations will be reproducible")
    parser.add_argument('--dataset', type=str, default='MNIST', help="Dataset to train on")
    parser.add_argument('--dataroot', type=str, default='data/', help="The root folder of dataset or downloaded data")
    parser.add_argument('--skip_normalization', action='store_true', help='Loads dataset without normalization')
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--num_batches', type=int, default=5)
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--random_shuffle', dest='random_shuffle', default=False, action='store_true',
                        help="Move part of data to next batch")
    parser.add_argument('--random_split', dest='random_split', default=False, action='store_true',
                        help="Randomize data in splits")
    parser.add_argument('--limit_data', type=float, default=None,
                        help="limit_data to given %")
    parser.add_argument('--dirichlet', default=None, type=float,
                        help="Alpha parameter for dirichlet data split")
    parser.add_argument('--reverse', dest='reverse', default=False, action='store_true',
                        help="Reverse the ordering of batches")
    parser.add_argument('--limit_classes', type=int, default=-1)
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--latent_dim', type=int, default=128, help="Latent dimension of Generator")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--score_on_val', action='store_true', required=False, default=False,
                        help="Compute FID on validation dataset instead of training dataset")
    parser.add_argument('--val_batch_size', type=int, default=250)
    parser.add_argument('--gen_load_pretrained_models', default=False, action='store_true',
                        help="Load pretrained generative models")
    parser.add_argument('--workers', type=int, default=0, help="Number of threads for dataloader")
    parser.add_argument('--skip_validation', default=False, action='store_true')
    parser.add_argument('--score_model_device', default="cpu", type=str, help="Device to score model on",
                        choices=["cpu", "gpu"])
    parser.add_argument('--training_procedure', type=str, default='multiband',
                        help='Training procedure multiband|replay')
    parser.add_argument('--num_local_epochs', type=int, default=100,
                        help="Number of epochs to train local GAN")
    parser.add_argument('--num_global_epochs', type=int, default=100,
                        help="Number of epochs to train global GAN")
    parser.add_argument('--local_dis_lr', type=float, default=0.002)
    parser.add_argument('--local_gen_lr', type=float, default=0.002)
    parser.add_argument('--global_gen_lr', type=float, default=0.002)
    parser.add_argument('--global_dis_lr', type=float, default=0.002)
    parser.add_argument('--num_gen_images', type=int, default=16,
                        help="Number of images to generate each epoch")
    parser.add_argument('--local_scheduler_rate', type=float, default=0.99)
    parser.add_argument('--global_scheduler_rate', type=float, default=0.99)

    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])

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
        print("WARNING: Not using manual seed - your experiments will not be reproducible")

    acc_val, acc_test, precision_table, recall_table = {}, {}, {}, {}
    os.makedirs(os.path.join(args.rpath, args.experiment_name), exist_ok=True)
    with open(os.path.join(args.rpath, args.experiment_name, "args.txt"), "w") as text_file:
        text_file.write(str(args))
    for r in range(args.repeat):
        acc_val[r], _, acc_test[r], precision_table[r], recall_table[r], fid_local_gan = run(args)
    np.save(os.path.join(args.rpath, args.experiment_name, "fid.npy"), acc_val)
    np.save(os.path.join(args.rpath, args.experiment_name, "precision.npy"), precision_table)
    np.save(os.path.join(args.rpath, args.experiment_name, "recall.npy"), recall_table)
    np.save(os.path.join(args.rpath, args.experiment_name, "fid_local_gan.npy"), fid_local_gan)

    plot_final_results([args.experiment_name], type="fid", fid_local_vae=fid_local_gan)
    print(fid_local_gan)
