import numpy as np
from delaySensitiveGOL import delaySensitiveGOL
import matplotlib.pyplot as plt

# Define the parameters
delays = [1, 2, 3, 4]
taus = [0.0, 0.05, 0.1, 0.2, 0.3]

fig, axes = plt.subplots(len(taus), len(delays), figsize=(12, 10))

for i, tau in enumerate(taus):
    for j, delay in enumerate(delays):
        # Generate or load the final state image here (dummy image used for now)
        gol = delaySensitiveGOL(
            sizeXY=(50, 50),
            timeStamps=2000,
            initConfig=None,
            rule="life",
            seed=42,
            density=0.5,
            alpha=0.5,
            delay=delay,
            tau=tau,
            averageTimeStamp=1500,
            extremes=(0, 1)
        )
        gol.simulate(vis=False, save=False)
        axes[i, j].imshow(gol.imgs[-1], cmap='Greys')
        axes[i, j].axis('off')

# Set column labels for delay values
for ax, delay in zip(axes[0], delays):
    ax.set_title(f"Delay = {delay}")

# Set row labels for tau values
for ax, tau in zip(axes[:, 0], taus):
    ax.set_ylabel(f"Tau = {tau}", rotation=90, size='large', labelpad=20)

plt.tight_layout()
plt.show()
