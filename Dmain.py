from Cdomain import (subdomain5, subdomain4, subdomain3, subdomain2, subdomain1)
from Aconfig import (DEVICE, Config)
from Bmodel import (PINN1, PINN2, PINN3, PINN4, PINN5)
import torch
import numpy as np
import matplotlib.pyplot as plt

config = Config()
delta = config.delta

model1 = PINN1().to(DEVICE)
model2 = PINN2().to(DEVICE)
model3 = PINN3().to(DEVICE)
model4 = PINN4().to(DEVICE)
model5 = PINN5().to(DEVICE)


N = 2000


x1_min, x1_max, t1_min, t1_max = subdomain1(delta)

x1 = x1_min + (x1_max - x1_min) * torch.rand(N,1)
t1 = t1_min + (t1_max - t1_min) * torch.rand(N,1)
X = torch.cat((x1,t1), dim=1)


print(x1.shape)
print(t1.shape)
print(X.shape)