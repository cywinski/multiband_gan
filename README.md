# Multiband GAN

## Reproducing results
1. MNIST
```
python3 main.py --wandb_project final_MultibandGAN --experiment_name CI_5_warmup_5 --dataset MNIST --gpuid 2 --seed 42 --num_batches 5 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --global_warmup 5
```

2. FashionMNIST
```
python3 main.py --wandb_project final_MultibandGAN --experiment_name CI_5_warmup_5 --dataset FashionMNIST --gpuid 2 --seed 42 --num_batches 5 --latent_dim 100 --batch_size 64 --score_on_val --num_local_epochs 120 --num_global_epochs 200 --local_dis_lr 0.0002 --local_gen_lr 0.0002 --local_scheduler_rate 0.99 --lambda_gp 10 --num_gen_images 64 --local_b1 0.0 --local_b2 0.9 --num_epochs_noise_optim 1000 --global_gen_lr 0.001 --optim_noise_lr 0.1 --global_scheduler_rate 0.99 --limit_previous 0.5 --d_n_features 32 --g_n_features 32 --global_warmup 5
``` 