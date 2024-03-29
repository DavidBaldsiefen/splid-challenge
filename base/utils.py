import pandas as pd
from tqdm import tqdm
from fastcore.basics import Path
import random
import numpy as np
import tensorflow as tf

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)