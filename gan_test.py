import torch
import matplotlib.pyplot as plt
from gan_experiments import gan_utils
from visualise import *



generator = torch.load(f"results/MNIST/CI_5/model1_curr_global_generator", map_location="cuda")


# class_table = curr_global_decoder.class_table
batch_size = 32
n_prev_examples = 40
recon_prev, z_prev, task_ids_prev = gan_utils.generate_previous_data(
    4,
    n_prev_examples=n_prev_examples,
    curr_global_generator=generator)


fig = plt.figure()
for i in range(40):
    plt.subplot(4,10,i+1)
    plt.tight_layout()
    plt.imshow(recon_prev[i][0].cpu(), cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(task_ids_prev[i]))
    plt.xticks([])
    plt.yticks([])
    
plt.show()
fig.savefig('./test.png')

