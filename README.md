# Multiband GAN

Multiband GAN is an adaptation of *multiband training* introduced in [Multiband VAE](https://github.com/KamilDeja/multiband_vae) method to the Generative Adversarial Networks architecture. To stabilize the training, instead of standard GAN architecture, we use WGAN with gradient penalty. We are able to outperform other known methods on the most popular continual learning benchmarks.


## Experiments

### Multiband VAE
CL Benchmark|Num. of tasks|FID $\downarrow$|Recall $\uparrow$| Precision $\uparrow$
:---:|:---:|:---:|:---:|:---: 
Split-MNIST Class Incremental|5|23|92|98
MNIST Dirichlet $\alpha=1$|10|30|92|97
Split-FashionMNIST Class Incremental|5|56|65|72
FashionMNIST Dirichlet $\alpha=1$|10|77|58|69
Split-Omniglot Class Incremental|5|12|98|96
Split-Omniglot Class Incremental|20|24|95|91
Omniglot Dirichlet $\alpha=1$|20|24|96|91
FashionMNIST $\rightarrow$ MNIST|10|49|68|70
MNIST $\rightarrow$ FashionMNIST|10|49|70|70
Split-CelebA Class Incremental|5|95|28.5|23.2
CelebA Dirichlet $\alpha=1$|10|93|33|22
CelebA Dirichlet $\alpha=100$|10|89|36.2|28


### Multiband GAN
CL Benchmark|Num. of tasks|FID $\downarrow$|Recall $\uparrow$| Precision $\uparrow$
:---:|:---:|:---:|:---:|:---: 
Split-MNIST Class Incremental|5|9|99|99
MNIST Dirichlet $\alpha=1$|10|44|96|97
Split-FashionMNIST Class Incremental|5|42|81|80
FashionMNIST Dirichlet $\alpha=1$|10|56|79|77
Split-Omniglot Class Incremental|5|3|97|95
Split-Omniglot Class Incremental|20|6|89|81
Omniglot Dirichlet $\alpha=1$|20|6|93|85
FashionMNIST $\rightarrow$ MNIST|10|40|78|79
MNIST $\rightarrow$ FashionMNIST|10|34|84|83
Split-CelebA Class Incremental|5|57|66|64
CelebA Dirichlet $\alpha=1$|10|60|66|67
CelebA Dirichlet $\alpha=100$|10|49|70|76
Split-CIFAR10 Class Incremental|5|113|65|37

## Reproducing results
1. Split-MNIST Class Incremental 5 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name CI_5 --dataset MNIST --gpuid 0 --seed 42 --num_batches 5 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --global_warmup 5
```

2. MNIST Dirichlet $\alpha=1$ 10 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name Dirichlet_alpha_1_10 --dataset MNIST --gpuid 0 --seed 42 --num_batches 10 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --dirichlet 1 --global_warmup 5
```

3. Split-FashionMNIST Class Incremental 5 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name CI_5 --dataset FashionMNIST --gpuid 0 --seed 42 --num_batches 5 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --global_warmup 5
```

4. FashionMNIST Dirichlet $\alpha=1$ 10 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name Dirichlet_alpha_1_10 --dataset FashionMNIST --gpuid 0 --seed 42 --num_batches 10 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --dirichlet 1 --global_warmup 5
```

5. Split-Omniglot Class Incremental 5 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name CI_5 --dataset Omniglot --gpuid 0 --seed 42 --num_batches 5 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --global_warmup 5
```

6. Split-Omniglot Class Incremental 20 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name CI_20 --dataset Omniglot --gpuid 0 --seed 42 --num_batches 20 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --global_warmup 5
```

7. Omniglot Dirichlet $\alpha=1$ 20 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name Dirichlet_alpha_1_20 --dataset Omniglot --gpuid 0 --seed 42 --num_batches 20 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --global_warmup 5 --dirichlet 1
```

8. FashionMNIST $\rightarrow$ MNIST Class Incremental 10 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name CI_10 --dataset DoubleMNIST --gpuid 0 --seed 42 --num_batches 10 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --global_warmup 5
```

9. MNIST $\rightarrow$ FashionMNIST Class Incremental 10 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name CI_10_reverse --dataset DoubleMNIST --gpuid 0 --seed 42 --num_batches 10 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --global_warmup 5 --reverse
```

10. Split-CelebA Class Incremental 5 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name CI_5 --dataset CelebA --gpuid 0 --seed 42 --num_batches 5 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 50 --g_n_features 50 --global_warmup 5
```

11. CelebA Dirichlet $\alpha=1$ 10 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name Dirichlet_alpha_1_10 --dataset CelebA --gpuid 0 --seed 42 --num_batches 10 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 50 --g_n_features 50 --global_warmup 5 --dirichlet 1
```

12. CelebA Dirichlet $\alpha=100$ 10 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name Dirichlet_alpha_100_10 --dataset CelebA --gpuid 0 --seed 42 --num_batches 10 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 50 --g_n_features 50 --global_warmup 5 --dirichlet 100
```

13. Split-CIFAR10 Class Incremental 5 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name CI_5 --dataset CIFAR10 --gpuid 0 --seed 42 --num_batches 5 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 128 --g_n_features 128 --global_warmup 5
```

14. CERN Class Incremental 5 tasks
```
python3 main.py --wandb_project MultibandGAN --experiment_name Class-Incremental_5_tasks --dataset CERN --gpuid 0 --seed 42 --num_batches 5 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --global_warmup 5
```