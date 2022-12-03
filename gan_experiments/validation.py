import os

import numpy as np
import torch

from gan_experiments.fid import calculate_frechet_distance
from gan_experiments.prd import compute_prd_from_embedding, prd_to_max_f_beta_pair
from scipy.stats import wasserstein_distance


class Validator:
    def __init__(
        self,
        n_classes,
        device,
        dataset,
        stats_file_name,
        dataloaders,
        score_model_device=None,
    ):
        self.n_classes = n_classes
        self.device = device
        self.dataset = dataset
        self.score_model_device = score_model_device
        self.dataloaders = dataloaders

        print("Preparing validator")
        if dataset in ["MNIST", "Omniglot"]:  # , "DoubleMNIST"]:
            if dataset in ["Omniglot"]:
                from gan_experiments.evaluation_models.lenet_Omniglot import Model
            # elif dataset == "DoubleMNIST":
            #     from gan_experiments.evaluation_models.lenet_DoubleMNIST import Model
            else:
                from gan_experiments.evaluation_models.lenet import Model
            net = Model()
            model_path = "gan_experiments/evaluation_models/lenet_" + dataset
            net.load_state_dict(torch.load(model_path))
            net.to(device)
            net.eval()
            self.model = net
            self.dims = 128 if dataset in ["Omniglot", "DoubleMNIST"] else 84  # 128
            self.score_model_func = net.part_forward
        elif dataset.lower() in [
            "celeba",
            "doublemnist",
            "fashionmnist",
            "flowers",
            "cern",
            "cifar10",
        ]:
            from gan_experiments.evaluation_models.inception import InceptionV3

            self.dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            self.model = InceptionV3([block_idx])
            if score_model_device:
                self.model = self.model.to(score_model_device)
            self.model.eval()
            self.score_model_func = lambda batch: self.model(
                batch.to(score_model_device)
            )[0]
        self.stats_file_name = f"{stats_file_name}_dims_{self.dims}"

    def calculate_results(
        self,
        curr_global_generator,
        task_id,
        batch_size,
        starting_point=None,
        calculate_class_dist=True,
    ):
        curr_global_generator.eval()

        test_loader = self.dataloaders[task_id]

        with torch.no_grad():
            distribution_orig = []
            distribution_gen = []
            precalculated_statistics = False
            os.makedirs(os.path.join("results", "orig_stats"), exist_ok=True)
            # os.makedirs(f"results/orig_stats/", exist_ok=True)
            stats_file_path = os.path.join(
                "results",
                "orig_stats",
                f"{self.dataset}_{self.stats_file_name}_{task_id}.npy",
            )
            # stats_file_path = f"results/orig_stats/{self.dataset}_{self.stats_file_name}_{task_id}.npy"
            if os.path.exists(stats_file_path):
                print(
                    f"Loading cached original data statistics from: {self.stats_file_name}"
                )
                distribution_orig = np.load(stats_file_path)
                precalculated_statistics = True

            print("Calculating FID:")
            if not precalculated_statistics:
                for idx, batch in enumerate(test_loader):
                    x = batch[0].to(self.device)
                    y = batch[1]
                    distribution_orig.append(
                        self.score_model_func(x).cpu().detach().numpy()
                    )

                distribution_orig = np.array(np.concatenate(distribution_orig)).reshape(
                    -1, self.dims
                )
                np.save(stats_file_path, distribution_orig)

            if calculate_class_dist:
                if self.dataset.lower() != "mnist":
                    raise NotImplementedError  # Missing classifier for this dataset
                generated_classes = []

            examples_to_generate = len(distribution_orig)
            while examples_to_generate:
                n_batch_to_generate = min(batch_size, examples_to_generate)
                z = torch.randn(
                    [n_batch_to_generate, curr_global_generator.latent_dim]
                ).to(self.device)

                task_ids = torch.zeros(n_batch_to_generate) + task_id
                example = curr_global_generator(z, task_ids)

                if self.dataset.lower() in ["fashionmnist", "doublemnist"]:
                    example = example.repeat([1, 3, 1, 1])

                distribution_gen.append(self.score_model_func(example).cpu().detach())

                if calculate_class_dist:
                    generated_classes.append(
                        self.model(example).cpu().detach().argmax(1)
                    )

                examples_to_generate -= n_batch_to_generate
            generated_classes = [
                item.item() for sublist in generated_classes for item in sublist
            ]
            print("Classified generated classes")
            distribution_gen = (
                torch.cat(distribution_gen).numpy().reshape(-1, self.dims)
            )
            num_data_for_prd = min(len(distribution_orig), len(distribution_gen))

            precision, recall = compute_prd_from_embedding(
                eval_data=distribution_gen[
                    np.random.choice(
                        len(distribution_gen), num_data_for_prd, replace=False
                    )
                ],
                ref_data=distribution_orig[
                    np.random.choice(
                        len(distribution_orig), num_data_for_prd, replace=False
                    )
                ],
            )
            precision, recall = prd_to_max_f_beta_pair(precision, recall)
            print(f"Precision:{precision},recall: {recall}")

            return (
                calculate_frechet_distance(distribution_gen, distribution_orig),
                precision,
                recall,
                generated_classes,
            )


# def compute_results_from_examples(self, args, generations, task_id, join_tasks=False):
#     distribution_orig = []
#     precalculated_statistics = False
#     stats_file_path = f"results/orig_stats/compare_files_{args.dataset}_{args.experiment_name}_{task_id}.npy"
#     test_loader = self.dataloaders[task_id]

#     if os.path.exists(stats_file_path) and not join_tasks:
#         print(f"Loading cached original data statistics from: {self.stats_file_name}")
#         distribution_orig = np.load(stats_file_path)
#         precalculated_statistics = True
#     print("Calculating FID:")
#     if not precalculated_statistics:
#         if join_tasks:
#             for task in range(task_id + 1):
#                 test_loader = self.dataloaders[task]
#                 for idx, batch in enumerate(test_loader):
#                     x = batch[0].to(self.device)
#                     if args.dataset.lower() in ["fashionmnist", "doublemnist"]:
#                         x = x.repeat([1, 3, 1, 1])
#                     distribution_orig.append(
#                         self.score_model_func(x).cpu().detach().numpy()
#                     )
#         else:
#             for idx, batch in enumerate(test_loader):
#                 x = batch[0].to(self.device)
#                 if args.dataset.lower() in ["fashionmnist", "doublemnist"]:
#                     x = x.repeat([1, 3, 1, 1])
#                 distribution_orig.append(
#                     self.score_model_func(x).cpu().detach().numpy()
#                 )

#     if args.dataset.lower() in ["mnist", "fashionmnist", "omniglot", "doublemnist"]:
#         generations = generations.reshape(-1, 1, 28, 28)
#     elif args.dataset.lower() in ["celeba", "flowers"]:
#         generations = generations.reshape(-1, 3, 64, 64)
#     generations = torch.from_numpy(generations).to(self.device)
#     if args.dataset.lower() in ["fashionmnist", "doublemnist"]:
#         generations = generations.repeat([1, 3, 1, 1])

#     if not precalculated_statistics:
#         distribution_orig = np.array(np.concatenate(distribution_orig)).reshape(
#             -1, self.dims
#         )
#         if not join_tasks:
#             os.makedirs("/".join(stats_file_path.split("/")[:-1]), exist_ok=True)
#             np.save(stats_file_path, distribution_orig)

#     distribution_gen = []
#     batch_size = args.val_batch_size
#     max_len = min(len(generations), len(distribution_orig))
#     for idx in range(0, max_len, batch_size):
#         start_point = idx
#         end_point = min(max_len, idx + batch_size)
#         distribution_gen.append(
#             self.score_model_func(generations[start_point:end_point])
#             .cpu()
#             .detach()
#             .numpy()
#             .reshape(-1, self.dims)
#         )
#     distribution_gen = np.concatenate(distribution_gen)

#     print(f"Orig:{len(distribution_orig)}, Gen:{len(distribution_gen)}")
#     precision, recall = compute_prd_from_embedding(
#         eval_data=distribution_orig,
#         ref_data=distribution_gen[
#             np.random.choice(len(distribution_gen), len(distribution_orig), True)
#         ],  # TODO go back to FALSE
#     )
#     precision, recall = prd_to_max_f_beta_pair(precision, recall)
#     print(f"Precision:{precision},recall: {recall}")

#     return (
#         calculate_frechet_distance(
#             distribution_gen[
#                 np.random.choice(len(distribution_gen), len(distribution_orig), True)
#             ],  # TODO go back to FALSE
#             distribution_orig,
#         ),
#         precision,
#         recall,
#     )


class CERN_Validator:
    def __init__(self, dataloaders, stats_file_name, device):
        self.dataloaders = dataloaders
        self.stats_file_name = stats_file_name
        self.device = device

    def sum_channels_parallel(self, data):
        coords = np.ogrid[0 : data.shape[1], 0 : data.shape[2]]
        half_x = data.shape[1] // 2
        half_y = data.shape[2] // 2

        checkerboard = (coords[0] + coords[1]) % 2 != 0
        checkerboard.reshape(-1, checkerboard.shape[0], checkerboard.shape[1])

        ch5 = (data * checkerboard).sum(axis=1).sum(axis=1)

        checkerboard = (coords[0] + coords[1]) % 2 == 0
        checkerboard = checkerboard.reshape(
            -1, checkerboard.shape[0], checkerboard.shape[1]
        )

        mask = np.zeros((1, data.shape[1], data.shape[2]))
        mask[:, :half_x, :half_y] = checkerboard[:, :half_x, :half_y]
        ch1 = (data * mask).sum(axis=1).sum(axis=1)

        mask = np.zeros((1, data.shape[1], data.shape[2]))
        mask[:, :half_x, half_y:] = checkerboard[:, :half_x, half_y:]
        ch2 = (data * mask).sum(axis=1).sum(axis=1)

        mask = np.zeros((1, data.shape[1], data.shape[2]))
        mask[:, half_x:, :half_y] = checkerboard[:, half_x:, :half_y]
        ch3 = (data * mask).sum(axis=1).sum(axis=1)

        mask = np.zeros((1, data.shape[1], data.shape[2]))
        mask[:, half_x:, half_y:] = checkerboard[:, half_x:, half_y:]
        ch4 = (data * mask).sum(axis=1).sum(axis=1)

        # assert all(ch1+ch2+ch3+ch4+ch5 == data.sum(axis=1).sum(axis=1))==True

        return np.stack([ch1, ch2, ch3, ch4, ch5])

    def calculate_results(
        self,
        curr_global_generator,
        task_id,
        batch_size,
        starting_point=None,
        calculate_class_dist=False,
    ):
        curr_global_generator.eval()
        test_loader = self.dataloaders[task_id]
        with torch.no_grad():
            distribution_orig = []
            distribution_gen = []

            precalculated_statistics = False
            os.makedirs(os.path.join("results", "orig_stats"), exist_ok=True)
            stats_file_path = os.path.join(
                "results",
                "orig_stats",
                f"CERN_{self.stats_file_name}_{task_id}.npy",
            )
            if os.path.exists(stats_file_path):
                print(
                    f"Loading cached original data statistics from: {self.stats_file_name}"
                )
                distribution_orig = np.load(stats_file_path)
                precalculated_statistics = True

            print("Calculating CERN scores:")
            for idx, batch in enumerate(test_loader):
                x = batch[0].to(self.device)
                y = batch[1]
                z = torch.randn([len(y), curr_global_generator.latent_dim]).to(
                    self.device
                )
                
                y = y.sort()[0]
                
                if starting_point != None:
                    task_ids = torch.zeros(len(y)) + starting_point
                else:
                    task_ids = torch.zeros(len(y)) + task_id
                
                example = curr_global_generator(z, task_ids)
                
                if not precalculated_statistics:
                    distribution_orig.append(
                        self.sum_channels_parallel(x.cpu().detach().numpy().squeeze(1))
                    )
                distribution_gen.append(
                    self.sum_channels_parallel(
                        example.cpu().detach().numpy().squeeze(1)
                    )
                )

            distribution_gen = np.hstack(distribution_gen)
            # distribution_gen = np.array(np.concatenate(distribution_gen)).reshape(-1, self.dims)
            if not precalculated_statistics:
                distribution_orig = np.hstack(distribution_orig)
                np.save(stats_file_path, distribution_orig)

            return (
                wasserstein_distance(
                    distribution_orig.reshape(-1), distribution_gen.reshape(-1)
                ),
                0,
                0,
                []
            )

    # def compute_results_from_examples(
    #     self, args, generations, task_id, join_tasks=False
    # ):
    #     distribution_orig = []
    #     precalculated_statistics = False
    #     stats_file_path = f"results/orig_stats/compare_files_{args.dataset}_{args.experiment_name}_{task_id}.npy"
    #     test_loader = self.dataloaders[task_id]
    #     if os.path.exists(stats_file_path) and not join_tasks:
    #         print(
    #             f"Loading cached original data statistics from: {self.stats_file_name}"
    #         )
    #         distribution_orig = np.load(stats_file_path)
    #         precalculated_statistics = True
    #     print("Calculating results:")
    #     if not precalculated_statistics:
    #         if join_tasks:
    #             for task in range(task_id + 1):
    #                 test_loader = self.dataloaders[task]
    #                 for idx, batch in enumerate(test_loader):
    #                     x = batch[0]
    #                     distribution_orig.append(
    #                         self.sum_channels_parallel(x.numpy().squeeze(1))
    #                     )
    #         else:
    #             for idx, batch in enumerate(test_loader):
    #                 x = batch[0]
    #                 distribution_orig.append(
    #                     self.sum_channels_parallel(x.numpy().squeeze(1))
    #                 )

    #     generations = generations.reshape(-1, 44, 44)
    #     # generations = torch.from_numpy(generations).to(self.device)

    #     if not precalculated_statistics:
    #         distribution_orig = np.hstack(distribution_orig)
    #         if not join_tasks:
    #             os.makedirs("/".join(stats_file_path.split("/")[:-1]), exist_ok=True)
    #             np.save(stats_file_path, distribution_orig)

    #     distribution_gen = []
    #     batch_size = args.val_batch_size
    #     # distribution_gen.append(self.sum_channels_parallel(example.cpu().detach().numpy().squeeze(1)))
    #     distribution_gen = self.sum_channels_parallel(generations)

    #     return (
    #         wasserstein_distance(
    #             distribution_orig.reshape(-1), distribution_gen.reshape(-1)
    #         ),
    #         0,
    #         0,
    #     )
