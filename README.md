# Multiband GAN

## How to run
### MNIST dataset
```
python3 main.py 
    --experiment_name my_cool_experiment 
    --dataset MNIST 
    --gpuid 0
    --seed 42 
    --num_batches 5 
    --batch_size 64 
    --score_on_val 
    --gan_type wgan  
    --limit_previous 0.5
```