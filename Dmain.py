from Cdomain import (subdomain5, subdomain4, subdomain3, subdomain2, subdomain1)
from Aconfig import (DEVICE, Config)
from Bmodel import (PINN1, PINN2, PINN3, PINN4, PINN5)
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

config = Config()
delta = config.delta

model1 = PINN1().to(DEVICE)
model2 = PINN2().to(DEVICE)
model3 = PINN3().to(DEVICE)
model4 = PINN4().to(DEVICE)
model5 = PINN5().to(DEVICE)


N = 2000

# -------------------------------------------------------------------------------------------------
# SUBDOMAIN POINTS GENERATIONS
# -------------------------------------------------------------------------------------------------


# Subdomain1 points generation
x1_min, x1_max, t1_min, t1_max = subdomain1(delta)

x1 = x1_min + (x1_max - x1_min) * torch.rand(N,1)
t1 = t1_min + (t1_max - t1_min) * torch.rand(N,1)

# subdomain2 points generation
x2_min, x2_max, t2_min, t2_max = subdomain2(delta)

x2 = x2_min + (x2_max - x2_min) * torch.rand(N,1)
t2 = t2_min + (t2_max - t2_min) * torch.rand(N,1)

# subdomain3 points generation
x3_min, x3_max, t3_min, t3_max = subdomain3(delta)

x3 = x3_min + (x3_max - x3_min) * torch.rand(N,1)
t3 = t3_min + (t3_max - t3_min) * torch.rand(N,1)

# Subdoamin4 points generation
x4_min, x4_max, t4_min, t4_max = subdomain4(delta)

x4 = x4_min + (x4_max - x4_min) * torch.rand(N,1)
t4 = t4_min + (t4_max - t4_min) * torch.rand(N,1)

# subdomain5 points generation
x5, t5 = subdomain5(delta, N)

# -------------------------------------------------------------------------------------------------
# SUBDOMAIN LOSS FUNCTION FOR EACH SUBDOMAINS
# -------------------------------------------------------------------------------------------------


def Subdomain1Loss(x,t):
    x = x1
    t = t1
    x_i = torch.linspace(0,x1_max-delta, steps=100, device= DEVICE).reshape(-1,1)
    t_i = torch.zeros_like(x_i)
    x_i.requires_grad_(True)
    t_i.requires_grad_(True)
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_pred1 = model1(x,t)
    u_pred0 = model1(x_i, t_i)
    u_pred_left_b = model1(torch.zeros_like(t),t)
    InitialLoss = nn.MSELoss()(u_pred0, torch.sin(x_i))
    BoundaryLoss = nn.MSELoss()(u_pred_left_b, torch.zeros_like(u_pred_left_b))
    u_t = torch.autograd.grad(u_pred1, t, grad_outputs= torch.ones_like(u_pred1), create_graph= True)[0]
    u_x = torch.autograd.grad(u_pred1, x, grad_outputs= torch.ones_like(u_pred1), create_graph= True)[0]
    u_xx = torch.autograd.grad(u_x, x , grad_outputs= torch.ones_like(u_x), create_graph= True)[0]
    residual = u_t - u_xx
    pdeloss = nn.MSELoss()(residual, torch.zeros_like(residual))
    loss1 = pdeloss + InitialLoss + BoundaryLoss
    return loss1




optimizer = torch.optim.Adam(model1.parameters(), lr= config.learning_rate)

for epoch in range(config.num_epochs):
    optimizer.zero_grad()

    loss = Subdomain1Loss(x1,t1)

    loss.backward()

    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e}")



