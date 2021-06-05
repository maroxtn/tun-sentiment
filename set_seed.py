import torch

import random
import numpy as np

import os
import yaml


def set_seed():
    """Set seed for reproducibility.
    """

    seed = get_seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_seed():
    """"Get seed from model_names.yaml
    """

    config = yaml.load(open("model_names.yaml"), Loader=yaml.FullLoader)
    return int(config["seed"])