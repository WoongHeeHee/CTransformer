# Hyperparameters #

d_embed = 512
d_model = 512
h = 8
d_ff = 2048
n_layer = 6
batch_size = 64
num_epochs = 100

train_data_path = "data/train.tsv"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_seed = 42