from Cdomain import (subdomain5, subdomain4, subdomain3, subdomain2, subdomain1)
from Aconfig import (DEVICE, Config)
from Bmodel import (PINN1, PINN2, PINN3, PINN4)
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, model1, model2, model3, model4):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def forward(self, x, t):
        return (
            0.25 * self.model1(x, t) +
            0.25 * self.model2(x, t) +
            0.25 * self.model3(x, t) +
            0.25 * self.model4(x, t)
        )



config = Config()
delta = config.delta

model1 = PINN1().to(DEVICE)
model2 = PINN2().to(DEVICE)
model3 = PINN3().to(DEVICE)
model4 = PINN4().to(DEVICE)
model5 = CombinedModel(model1, model2, model3, model4).to(DEVICE)


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
    x_i = torch.linspace(x1_min,x1_max, steps=100, device= DEVICE).reshape(-1,1)
    t_i = torch.zeros_like(x_i)

    x_b = torch.zeros_like(x_i)
    t_b = torch.linspace(t1_min,t1_max, steps=100, device= DEVICE).reshape(-1,1)

    x_b.requires_grad_(True)
    t_b.requires_grad_(True)

    x_i.requires_grad_(True)
    t_i.requires_grad_(True)

    x.requires_grad_(True)
    t.requires_grad_(True)

    u_pred1 = model1(x,t)
    u_pred_i = model1(x_i, t_i)
    u_pred_left_b = model1(x_b,t_b)

    InitialLoss = nn.MSELoss()(u_pred_i, torch.sin(x_i))

    BoundaryLoss = nn.MSELoss()(u_pred_left_b, torch.zeros_like(u_pred_left_b))

    u_t = torch.autograd.grad(u_pred1, t, grad_outputs= torch.ones_like(u_pred1), create_graph= True)[0]
    u_x = torch.autograd.grad(u_pred1, x, grad_outputs= torch.ones_like(u_pred1), create_graph= True)[0]
    u_xx = torch.autograd.grad(u_x, x , grad_outputs= torch.ones_like(u_x), create_graph= True)[0]

    residual = u_t - u_xx

    Pdeloss = nn.MSELoss()(residual, torch.zeros_like(residual))

    loss1 = Pdeloss + InitialLoss + BoundaryLoss

    return loss1


def Subdomain2Loss(x,t):

    x = x2
    t = t2

    x_i = torch.linspace(x2_min , x2_max, steps=100, device= DEVICE).reshape(-1,1)
    t_i = torch.zeros_like(x_i)

    x_b = torch.ones_like(t_i)
    t_b = torch.linspace(t2_min, t2_max , steps=100, device= DEVICE).reshape(-1,1)

    x_b.requires_grad_(True)
    t_b.requires_grad_(True)

    x_i.requires_grad_(True)
    t_i.requires_grad_(True)

    x.requires_grad_(True)
    t.requires_grad_(True)

    u_pred2 = model2(x,t)

    u_pred_i = model2(x_i, t_i)

    u_pred_right_b = model2(x_b,t_b)

    InitialLoss = nn.MSELoss()(u_pred_i, torch.sin(x_i))

    BoundaryLoss = nn.MSELoss()(u_pred_right_b, torch.zeros_like(u_pred_right_b))

    u_t = torch.autograd.grad(u_pred2, t, grad_outputs= torch.ones_like(u_pred2), create_graph= True)[0]
    u_x = torch.autograd.grad(u_pred2, x, grad_outputs= torch.ones_like(u_pred2), create_graph= True)[0]
    u_xx = torch.autograd.grad(u_x, x , grad_outputs= torch.ones_like(u_x), create_graph= True)[0]

    residual = u_t - u_xx

    Pdeloss = nn.MSELoss()(residual, torch.zeros_like(residual))

    loss2 = Pdeloss + InitialLoss + BoundaryLoss
    return loss2


def Subdomain3Loss(x,t):

    x = x3
    t = t3

    t_b = torch.linspace(t3_min, t3_max , steps=100, device= DEVICE).reshape(-1,1)
    x_b = torch.zeros_like(t_b)

    x_b.requires_grad_(True)
    t_b.requires_grad_(True)

    x.requires_grad_(True)
    t.requires_grad_(True)

    u_pred3 = model3(x,t)

    u_pred_left_b = model3(x_b,t_b)

    BoundaryLoss = nn.MSELoss()(u_pred_left_b, torch.zeros_like(u_pred_left_b))

    u_t = torch.autograd.grad(u_pred3, t, grad_outputs= torch.ones_like(u_pred3), create_graph= True)[0]
    u_x = torch.autograd.grad(u_pred3, x, grad_outputs= torch.ones_like(u_pred3), create_graph= True)[0]
    u_xx = torch.autograd.grad(u_x, x , grad_outputs= torch.ones_like(u_x), create_graph= True)[0]

    residual = u_t - u_xx

    Pdeloss = nn.MSELoss()(residual, torch.zeros_like(residual))

    loss3 = Pdeloss + BoundaryLoss
    return loss3


def Subdomain4Loss(x,t):

    x = x4
    t = t4

    t_b = torch.linspace(t4_min, t4_max , steps=100, device= DEVICE).reshape(-1,1)
    x_b = torch.ones_like(t_b)

    x_b.requires_grad_(True)
    t_b.requires_grad_(True)

    x.requires_grad_(True)
    t.requires_grad_(True)

    u_pred4 = model4(x,t)

    u_pred_right_b = model4(x_b,t_b)

    BoundaryLoss = nn.MSELoss()(u_pred_right_b, torch.zeros_like(u_pred_right_b))

    u_t = torch.autograd.grad(u_pred4, t, grad_outputs= torch.ones_like(u_pred4), create_graph= True)[0]
    u_x = torch.autograd.grad(u_pred4, x, grad_outputs= torch.ones_like(u_pred4), create_graph= True)[0]
    u_xx = torch.autograd.grad(u_x, x , grad_outputs= torch.ones_like(u_x), create_graph= True)[0]

    residual = u_t - u_xx

    Pdeloss = nn.MSELoss()(residual, torch.zeros_like(residual))

    loss4 = Pdeloss + BoundaryLoss
    return loss4


def Subdomain5Loss(x,t):

    x = x5
    t = t5

    x = x.reshape(-1, 1).requires_grad_(True)
    t = t.reshape(-1, 1).requires_grad_(True)

    x_i = torch.linspace(x1_max , x2_min, steps=50, device= DEVICE).reshape(-1,1)
    t_i = torch.zeros_like(x_i)

    x_b_left = torch.zeros_like(t_i)
    t_b_left = torch.linspace(t1_max, t2_min , steps=50, device= DEVICE).reshape(-1,1)

    x_b_right = torch.ones_like(x_i)
    t_b_right = torch.linspace(t1_max, t2_min , steps=50, device= DEVICE).reshape(-1,1)

    x_b_left.requires_grad_(True)
    t_b_left.requires_grad_(True)

    x_b_right.requires_grad_(True)
    t_b_right.requires_grad_(True)

    x_i.requires_grad_(True)
    t_i.requires_grad_(True)

    x.requires_grad_(True)
    t.requires_grad_(True)

    u_pred5 = model5(x,t)

    u_pred_i = model5(x_i, t_i)

    u_pred_right_b = model5(x_b_right, t_b_right)
    u_pred_left_b = model5(x_b_left, t_b_left)

    InitialLoss = nn.MSELoss()(u_pred_i, torch.sin(x_i))

    BoundaryLossLeft = nn.MSELoss()(u_pred_left_b, torch.zeros_like(u_pred_left_b))
    BoundaryLossRight = nn.MSELoss()(u_pred_right_b, torch.zeros_like(u_pred_right_b))

    u_t = torch.autograd.grad(u_pred5, t, grad_outputs= torch.ones_like(u_pred5), create_graph= True)[0]
    u_x = torch.autograd.grad(u_pred5, x, grad_outputs= torch.ones_like(u_pred5), create_graph= True)[0]
    u_xx = torch.autograd.grad(u_x, x , grad_outputs= torch.ones_like(u_x), create_graph= True)[0]

    residual = u_t - u_xx

    Pdeloss = nn.MSELoss()(residual, torch.zeros_like(residual))

    loss5 = Pdeloss + InitialLoss + BoundaryLossLeft + BoundaryLossRight

    return loss5

optimizer1 = torch.optim.Adam(model1.parameters(), lr= config.learning_rate)
optimizer2 = torch.optim.Adam(model2.parameters(), lr= config.learning_rate)
optimizer3 = torch.optim.Adam(model3.parameters(), lr= config.learning_rate)
optimizer4 = torch.optim.Adam(model4.parameters(), lr= config.learning_rate)
optimizer5 = torch.optim.Adam(model5.parameters(), lr= config.learning_rate)



for epoch in range(config.num_epochs):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    optimizer4.zero_grad()
    optimizer5.zero_grad()

    loss1 = Subdomain1Loss(x1,t1)
    loss1.backward()
    optimizer1.step()
    loss2 = Subdomain2Loss(x2,t2)
    loss2.backward()
    optimizer2.step()
    loss3 = Subdomain3Loss(x3,t3)
    loss3.backward()
    optimizer3.step()
    loss4 = Subdomain4Loss(x4,t4)
    loss4.backward()
    optimizer4.step()

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    optimizer4.zero_grad()
    loss5 = Subdomain5Loss(x5,t5)
    loss5.backward()
    optimizer5.step()

    

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss1: {loss1.item():.6e} | Loss2: {loss2.item():.6e} | Loss3: {loss3.item():.6e} | Loss4: {loss4.item():.6e} | Loss5: {loss5.item():.6e} ")


torch.save(model1.state_dict(), "model1.pth")
torch.save(model2.state_dict(), "model2.pth")
torch.save(model3.state_dict(), "model3.pth")
torch.save(model4.state_dict(), "model4.pth")
