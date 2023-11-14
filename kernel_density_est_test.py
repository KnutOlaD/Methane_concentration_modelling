#script that will test out the KDEpy package

#Example from the package itself

from KDEpy import FFTKDE
import matplotlib.pyplot as plt
import numpy as np

customer_ages = [40, 56, 20, 35, 27, 24, 29, 37, 39, 46]

# Distribution of customers
x, y = FFTKDE(kernel="gaussian", bw="silverman").fit(customer_ages).evaluate()
#plt.plot(x, y)

# Distribution of customer income (weight each customer by their income)
customer_income = [152, 64, 24, 140, 88, 64, 103, 148, 150, 132]

# The `bw` parameter can be manually set, e.g. `bw=5`
x, y = FFTKDE(bw="silverman").fit(customer_ages, weights=customer_income).evaluate()
#plt.figure()
#plt.plot(x, y)
#plt.show()

# Create 2D data of shape (obs, dims)
data = np.random.randn(2**4, 2)

grid_points = 2**7  # Grid points in each dimension
N = 16  # Number of contours

fig = plt.figure(figsize=(12, 4))

for plt_num, norm in enumerate([1, 3, 5], 1):

    ax = fig.add_subplot(1, 3, plt_num)
    ax.set_title(f'Norm $p={norm}$')

    # Compute the kernel density estimate
    kde = FFTKDE(kernel='gaussian', norm=norm)
    grid, points = kde.fit(data).evaluate(grid_points)

    # The grid is of shape (obs, dims), points are of shape (obs, 1)
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points, grid_points).T

    # Plot the kernel density estimate
    ax.contour(x, y, z, N, linewidths=0.8, colors='k')
    ax.contourf(x, y, z, N, cmap="RdBu_r")
    ax.plot(data[:, 0], data[:, 1], 'ok', ms=3)

plt.tight_layout()
plt.show()

#Create a kernel function with age-based bandwidth and custom weights



bw = dissipation_energy_timescale * time