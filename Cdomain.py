from Aconfig import Config
from Aconfig import DEVICE
import torch
import numpy as np
import matplotlib.pyplot as plt

config = Config()


x_min = config.x_min
x_max = config.x_max
t_min = config.t_min
t_max = config.x_max

x_min.to(DEVICE), x_max.to(DEVICE), t_min.to(DEVICE), t_max.to(DEVICE)

t_mid = (t_max + t_min)/2
x_mid = (x_max + x_min)/2


