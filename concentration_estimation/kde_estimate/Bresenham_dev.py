import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import multivariate_normal
from numba.typed import List
from numba import jit, prange


@jit(nopython=True)
def bresenham(x0, y0, x1, y1):
    """
    Bresenham's Line Algorithm to generate points between (x0, y0) and (x1, y1)
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

@jit(nopython=True)
def identify_shadowed_cells(x0, y0, xi, yj, legal_grid):
    """
    Identify shadowed cells by tracing from edges inward.
    Cells start as potentially shadowed and are marked free 
    if they have line of sight to kernel center.
    """
    grid_size = legal_grid.shape[0]
    # Start with all cells potentially shadowed
    shadowed = np.ones((grid_size, grid_size), dtype=np.bool_)
    
    # Trace from edges
    for edge_x in [0, grid_size-1]:
        for y in range(grid_size):
            los_cells = bresenham(x0,y0,edge_x, y)
            # Mark cells as free until hitting illegal cell
            for cell in los_cells:
                if not legal_grid[cell[0], cell[1]]:
                    break
                shadowed[cell[0], cell[1]] = False
                
    for edge_y in [0, grid_size-1]:
        for x in range(grid_size):
            los_cells = bresenham(x0, y0, x, edge_y)
            for cell in los_cells:
                if not legal_grid[cell[0], cell[1]]:
                    break
                shadowed[cell[0], cell[1]] = False
    
    # Convert to list format
    shadowed_cells = []
    for i in range(len(xi)):
        for j in range(len(yj)):
            if shadowed[xi[i], yj[j]]:
                shadowed_cells.append((xi[i], yj[j]))
                
    return shadowed_cells


# Example usage
x0, y0 = 5, 5
xi = np.arange(11)
yj = np.arange(11)
grid_size = 11
legal_grid = np.ones((11, 11), dtype=bool)
legal_grid[7, 8] = False  # Example of an illegal cell
legal_grid[8,7] = False
legal_grid[7,7] = False

#creatte an island in some other octant
legal_grid[:3,4:6] = False
legal_grid[2,3] = False
legal_grid[2,2] = False

import time
time_start = time.time()
n=0
while n<1000000:
    #shadowed_cells = identify_shadowed_cells(x0, y0, xi, yj, legal_grid)
    shadowed_cells = identify_shadowed_cells(x0, y0, xi, yj, legal_grid)
    n+=1
time_end = time.time()
print(f'Time taken: {time_end - time_start:.6f} s')

shadowed_cells = identify_shadowed_cells(x0, y0, xi, yj, legal_grid)
shadow_x, shadow_y = zip(*shadowed_cells)
legal_grid[shadow_x, shadow_y] = False


# Plot all shadowed cells in yellow, the illegal cell in red, and the origin as a green dot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(0, 11, 1))
ax.grid(False)  # Disable the grid

# Plot a Gaussian centered at the origin
x, y = np.mgrid[0:11, 0:11]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
rv = multivariate_normal([5, 5], [[4, 0], [0, 4]])
density = ax.pcolor(x, y, rv.pdf(pos), label='Kernel density',cmap = 'viridis')

# Plot the shadowed cells
for cell in shadowed_cells:
    ax.add_patch(plt.Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, color='yellow', alpha=1))

# Plot the legal and illegal cells
for m in range(11):
    for n in range(11):
        if legal_grid[m, n]:
            ax.add_patch(plt.Rectangle((m - 0.5, n - 0.5), 1, 1, fill=None, edgecolor='black'))
        else:
            ax.add_patch(plt.Rectangle((m - 0.5, n - 0.5), 1, 1, color='black'))

# Calculate the line from 5,5 to 8,1 and plot it
cells = bresenham(x0, y0, 8, 1)
for cell in cells:
    ax.add_patch(plt.Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, color='grey', alpha=0.7))

# Plot the line
ax.plot([x0, 8], [y0, 1], color='white', marker='o')

# Create legend patches
shadowed_patch = mpatches.Patch(color='yellow', label='Blocked cells')
illegal_patch = mpatches.Patch(color='black', label='Bathymetry/land')
plt.legend(handles=[illegal_patch, shadowed_patch])

# Plot the origin
ax.plot(x0, y0, color='green', marker='o')
ax.text(x0+2.3, y0, 'x_0, y_0', color='white', fontsize=10, ha='right',
        bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0'))

# Plot the endpoint
ax.plot(8, 1, color='white', marker='o')
ax.text(8+0.34, 1, 'x_1, y_1', color='white', fontsize=10, ha='left',
        bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0'))

plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.xlabel('x')
plt.ylabel('y')

plt.show()

# Plot the grid and the shadowed cells for each iteration
not_now = 1
if not_now == 0:
    for i in xi:
        for j in yj:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xticks(np.arange(0, 10, 1))
            ax.set_yticks(np.arange(0, 10, 1))
            ax.grid(True)

            # Plot the legal and illegal cells
            for m in range(10):
                for n in range(10):
                    if legal_grid[m, n]:
                        ax.add_patch(plt.Rectangle((m - 0.5, n - 0.5), 1, 1, fill=None, edgecolor='black'))
                    else:
                        ax.add_patch(plt.Rectangle((m - 0.5, n - 0.5), 1, 1, color='red'))

            # Get the intersecting cells
            cells = bresenham(x0, y0, i, j)

            # Plot the intersecting cells
            for cell in cells:
                ax.add_patch(plt.Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, color='blue', alpha=0.5))

            # Identify shadowed cells
            shadowed_cells = identify_shadowed_cells(x0, y0, [i], [j], legal_grid)

            # Plot the shadowed cells
            for cell in shadowed_cells:
                ax.add_patch(plt.Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, color='yellow', alpha=0.5))

            # Plot the line
            ax.plot([x0, i], [y0, j], color='green', marker='o')

            plt.xlim(-1, 10)
            plt.ylim(-1, 10)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Line from ({x0}, {y0}) to ({i}, {j})')
            plt.show()
