
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
import numpy as np
from KDEpy import TreeKDE

# Create 2D data of shape (obs, dims)
data = np.random.randn(2**10, 2)

grid_points = 2**6  # Grid points in each dimension
N = 16  # Number of contours

fig = plt.figure(figsize=(12, 4))

for plt_num, norm in enumerate([1, 2, np.inf], 1):

    ax = fig.add_subplot(1, 3, plt_num)
    ax.set_title(f'Norm $p={norm}$')

    # Compute the kernel density estimate
    kde = FFTKDE(kernel='gaussian',bw=10, norm=norm)
    grid, points = kde.fit(data).evaluate(grid_points)

    # The grid is of shape (obs, dims), points are of shape (obs, 1)
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points, grid_points).T

    # Plot the kernel density estimatea
    ax.contour(x, y, z, N, linewidths=0.8, colors='k')
    ax.contourf(x, y, z, N, cmap="RdBu_r")
    #ax.plot(data[:, 0], data[:, 1], 'ok', ms=3)

plt.tight_layout()