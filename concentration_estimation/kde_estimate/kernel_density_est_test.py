#script that will test out the KDEpy package

#Example from the package itself

from KDEpy import TreeKDE
import matplotlib.pyplot as plt
import numpy as np




# Create 2D data of shape (obs, dims)
data = np.random.randn(2**4, 2)

grid_points = 2**7  # Grid points in each dimension
N = 16  # Number of contours

#fig = plt.figure(figsize=(12, 4))

#add some weights
weights = np.ones(data.shape[0])
'''
for plt_num, norm in enumerate([1, 3, 5], 1):

    ax = fig.add_subplot(1, 3, plt_num)
    ax.set_title(f'Norm $p={norm}$')

    # Compute the kernel density estimate
    kde = TreeKDE(kernel='gaussian',bw = 1, norm=norm)
    grid, points = kde.fit(data,weights).evaluate(grid_points)

    # The grid is of shape (obs, dims), points are of shape (obs, 1)
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points, grid_points).T

    # Plot the kernel density estimate
    ax.contour(x, y, z, N, linewidths=0.8, colors='k')
    ax.contourf(x, y, z, N, cmap="RdBu_r")
    ax.plot(data[:, 0], data[:, 1], 'ok', ms=3)
'''
#plt.tight_layout()
#plt.show()

#Make a time series with plots where the bandwidth is a function of time
# Create 2D data of shape (obs, dims)
data = np.random.randn(2**4, 2)

grid_points = 2**7  # Grid points in each dimension
#Create a grid point tuple which is not square which gives the x and y axis
#different resolution
grid_points = (2**7,2**8)  # Grid points in each dimension

N = 16  # Number of contours

bw = np.arange(0.1, 10, 0.1)

data = 2*np.random.randn(2**4, 2)

#make an example where a new point is added every timestep

#adding datapoints one at a time
#data = np.array([])
#let the weithgs be zero for all inactvie points
bwdata = np.ones(len(bw))*0.1

weights = np.zeros(data.shape[0])

#create a meshgrod from the grid_points tuple to plot on
x,y = np.meshgrid(np.linspace(-4,4,grid_points[0]),np.linspace(-4,4,grid_points[1]))


plt.figure()
for i in range(0, 10):
    
    

    #activate one point
    weights[:i] = 1.

    #create grid
    # Compute the kernel density estimate
    kde = TreeKDE(kernel='epa',bw = bwdata,norm=2)
    grid, points = kde.fit(data,weights).evaluate(
        np.meshgrid(np.linspace(-4,4,grid_points[0]),
                    np.linspace(-4,4,grid_points[1])))

    # The grid is of shape (obs, dims), points are of shape (obs, 1)
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = points.reshape(grid_points[0], grid_points[1]).T

    #increase the bandwidth of all active points
    bwdata[:i] = bwdata[:i] + 0.1

    # Plot the kernel density estimate
    plt.contour(x, y, z, N, linewidths=0.8, colors='k')
    plt.contourf(x, y, z, N, cmap="RdBu_r")
    #put limits on the x an y axis
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.plot(data[:i, 0], data[:i, 1], 'ok', ms=3)
    #set fixed climits
    plt.pause(0.1)
    plt.clf()
    




#Create a kernel function with age-based bandwidth and custom weights

#dissipation_energy_timescale = 1.0
#time = np.arange(0, 100, 0.1)

#bw = dissipation_energy_timescale * time

