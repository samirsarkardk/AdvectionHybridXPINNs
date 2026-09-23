from Aconfig import Config
from Aconfig import DEVICE
import torch
import numpy as np
import matplotlib.pyplot as plt

config = Config()
delta = config.delta


x_min = config.x_min
x_max = config.x_max
t_min = config.t_min
t_max = config.x_max

x_min.to(DEVICE), x_max.to(DEVICE), t_min.to(DEVICE), t_max.to(DEVICE)

t_mid = (t_max + t_min)/2

t_mid.to(DEVICE)

def subdomain1(delta):

    x1_min = x_min
    x1_max = x_max

    t1_min = t_min
    t1_max = t_mid - delta

    return x1_min, x1_max, t1_min, t1_max


def subdomain2(delta):
    x2_min = x_min
    x2_max = x_max
    t2_min = t_mid + delta
    t2_max = t_max

    return x2_min, x2_max, t2_min, t2_max





def subdomain3(delta, N=2000):

    # Generate candidate points
    x = torch.rand(N * 10, 1)
    t = torch.rand(N * 10, 1)

    # Points in the four corner subdomains
    mask = ~(
        (
            (x <= x_max) &
            (t <= t_mid - delta)
        )
        |
        (
            (x <= x_max ) &
            (t >= t_mid + delta)
        )
       
    )

    x3 = x[mask]
    t3 = t[mask]

    return x3[:N], t3[:N]



# x5, t5 = subdomain5(delta, N=2000)

# print(x5.shape)
# print(t5.shape)




# N = 2000

# x1_min, x1_max, t1_min, t1_max = subdomain1(delta)

# x1 = x1_min + (x1_max - x1_min) * torch.rand(N,1)
# t1 = t1_min + (t1_max - t1_min) * torch.rand(N,1)
# X = torch.cat((x1,t1), dim=1)


# print(x1.shape)
# print(t1.shape)
# print(X.shape)


