import torch
from torch import nn
import numpy as np
import random_init.layers as layers
import random_init.utils as utils
import random_init.models as models
from tqdm import tqdm
import random


CUDA = True
HEIGHT = 800
WIDTH = 800
OUTPUT_DIR = 'images/'
SCALE = 100.0
SHIFT = 1.0
DIMS = 6
IMAGE_PREFIX = 'kaiming-uniform-init'
# SEEDS = [5000, 11032, 56, 42, 119, 131, 51438, 2193]
HIDDEN = [20, 20, 20, 20, 20, 20, 20, 20, 20]
ACTIVATION = 'relu'
NUM_IMAGES = 5

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cuda:0' if CUDA else 'cpu'

for seed in tqdm(range(NUM_IMAGES)):
    seed = random.randrange(2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    inp = utils.generate_nd_inputs(WIDTH, HEIGHT, SCALE, SHIFT, True, -1.0, 1.0, 3)
    inp_tensor = torch.from_numpy(inp).float()
    inp_tensor = inp_tensor.to(device)
    dnn = models.HomogeneousMLP(DIMS, 3, layers.KaimingUniformFC, HIDDEN, ACTIVATION)
    dnn.to(device)
    output = dnn(inp_tensor)
    output = output.cpu().detach().numpy()
    # Scale the numbers back to [0-255]
    output = output * 255.0

    image_name = f'{IMAGE_PREFIX}_{DIMS}-dim-inp_{SCALE}-scale_{SHIFT}-shift_activation-{ACTIVATION}_{len(HIDDEN)}-hidden_seed-{seed}.jpg'
    utils.save_image(OUTPUT_DIR, image_name, output, WIDTH, HEIGHT)

