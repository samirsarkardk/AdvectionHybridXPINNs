from Cdomain import subdomain1, subdomain2, subdomain3, subdomain4, subdomain5
import torch
from Aconfig import Config, DEVICE
import matplotlib.pyplot as plt

config = Config()
delta = config.delta
x_min = config.x_min
x_max = config.x_max
t_min = config.t_min
t_max = config.t_max
x_max.to(DEVICE), t_min.to(DEVICE), x_min.to(DEVICE),t_max.to(DEVICE)


N = 500


# ============================================================
# Generate 500 points in Subdomain 1
# ============================================================

x1_min, x1_max, t1_min, t1_max = subdomain1(delta)

x1 = x1_min + (x1_max - x1_min) * torch.rand(N, 1, device=DEVICE)
t1 = t1_min + (t1_max - t1_min) * torch.rand(N, 1, device=DEVICE)


# ============================================================
# Generate 500 points in Subdomain 2
# ============================================================

x2_min, x2_max, t2_min, t2_max = subdomain2(delta)

x2 = x2_min + (x2_max - x2_min) * torch.rand(N, 1, device=DEVICE)
t2 = t2_min + (t2_max - t2_min) * torch.rand(N, 1, device=DEVICE)


# ============================================================
# Generate 500 points in Subdomain 3
# ============================================================

x3_min, x3_max, t3_min, t3_max = subdomain3(delta)

x3 = x3_min + (x3_max - x3_min) * torch.rand(N, 1, device=DEVICE)
t3 = t3_min + (t3_max - t3_min) * torch.rand(N, 1, device=DEVICE)


# ============================================================
# Generate 500 points in Subdomain 4
# ============================================================

x4_min, x4_max, t4_min, t4_max = subdomain4(delta)

x4 = x4_min + (x4_max - x4_min) * torch.rand(N, 1, device=DEVICE)
t4 = t4_min + (t4_max - t4_min) * torch.rand(N, 1, device=DEVICE)


# ============================================================
# Generate 500 points in Subdomain 5
# ============================================================

x5, t5 = subdomain5(delta, N=N)


# ============================================================
# Convert to NumPy for plotting
# ============================================================

x1_plot = x1.cpu().numpy().flatten()
t1_plot = t1.cpu().numpy().flatten()

x2_plot = x2.cpu().numpy().flatten()
t2_plot = t2.cpu().numpy().flatten()

x3_plot = x3.cpu().numpy().flatten()
t3_plot = t3.cpu().numpy().flatten()

x4_plot = x4.cpu().numpy().flatten()
t4_plot = t4.cpu().numpy().flatten()

x5_plot = x5.cpu().numpy().flatten()
t5_plot = t5.cpu().numpy().flatten()


# ============================================================
# Plot all five subdomains
# ============================================================

plt.figure(figsize=(9, 9))

plt.scatter(x1_plot, t1_plot, s=10, label="Subdomain 1")
plt.scatter(x2_plot, t2_plot, s=10, label="Subdomain 2")
plt.scatter(x3_plot, t3_plot, s=10, label="Subdomain 3")
plt.scatter(x4_plot, t4_plot, s=10, label="Subdomain 4")
plt.scatter(x5_plot, t5_plot, s=10, label="Subdomain 5")

# Draw the entire domain boundary
plt.xlim(x_min, x_max)
plt.ylim(t_min, t_max)

plt.xlabel("$x$")
plt.ylabel("$t$")
plt.title("Decomposed Domain with 500 Points per Subdomain")

plt.legend()
plt.grid(True)
plt.gca().set_aspect("equal", adjustable="box")

plt.show()