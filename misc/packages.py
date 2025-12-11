# →
# conceptual framework
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, degrees

# ml models
import warnings
warnings.filterwarnings("ignore") # suppress warnings

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import geopandas as gpd
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import random
from scipy.stats import norm
from sklearn.metrics import r2_score
from IPython.display import display, Math
from sklearn.inspection import permutation_importance
from collections import defaultdict

plt.style.use("~/geoscience/carbon_estimation/MNRAS.mplstyle")
%matplotlib inline

import json
import pathlib

from tqdm import tqdm

import os
import shutil
from deephyper.evaluator import RunningJob


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing
