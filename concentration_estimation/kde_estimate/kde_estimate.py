#script that will test out the KDEpy package

#Example from the package itself

from KDEpy import TreeKDE
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import netCDF4 as nc
import utm
from scipy.sparse import coo_matrix as coo_matrix
from scipy.sparse import csr_matrix as csr_matrix
import pickle
#set path to load_plot_opendrift
import sys
sys.path.append(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\src\Dispersion_modelling\concentration_estimation')
import load_plot_opendrift as lpo


#path to data
datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_test.nc'
#Load the opendrift data
particles = lpo.load_nc_data(datapath)
#add utm coordinates to the dictionary
particles = lpo.add_utm(particles)
#get grid parameters
bin_x,bin_y,bin_z,bin_time = lpo.get_grid_params(particles) #gives

#make a utm plot of all the particles at the last timestep 
plt.figure()
plt.scatter(particles['UTM_x'][:,-1],particles['UTM_y'][:,-1],s=1)

#find all particles that have z value less than 10
z = particles['z'][:,-1]
z = z[~np.isnan(z)]
z = z[z<10]
#find the indices of the particles that have z value less than 10
ind = np.where(np.abs(particles['z'][:,-1])<10)[0]

particles_test_set = particles['UTM_x'][ind,-1],particles['UTM_y'][ind,-1]
#use only particles that have utmx larger than 0.5e6 and utmy values less than 8.0e6
ind = np.where((particles_test_set[0]>0.5e6) & (particles_test_set[1]<8.0e6))[0]
particles_test_set = particles_test_set[0][ind],particles_test_set[1][ind]
#make a plot of the particles that have z value less than 10
plt.figure()
plt.scatter(particles_test_set[0],particles_test_set[1],s=1)
#create a grid with resolution 1000 m for the particles_test_set dataset

#remove minimum xutm and yutm values such that the grid is centered around the particles
#bin_x_test = bin_x_test - np.min(particles_test_set[0])
#bin_y_test = bin_y_test - np.min(particles_test_set[1])

#do the same for the particles_test_set
particles_test_set = particles_test_set[0].data[:] - np.min(particles_test_set[0].data[:]),particles_test_set[1].data[:] - np.min(particles_test_set[1].data[:])
#divide by 1000 to get the grid in km
particles_test_set = np.array(particles_test_set)
#flip it around
particles_test_set = np.flipud(np.transpose(particles_test_set))
bin_x_test = np.max(particles_test_set[:,0])/1000
bin_y_test = np.max(particles_test_set[:,1])/1000
#for particles_test_set too
particles_test_set = particles_test_set/1000

#grid should start from zero and go to 1.1e6 - 0.5e6 in x direction and 8.0e6 - 7.65e6 in y direction
grid_points = (int(np.ceil(bin_x_test)),int(np.ceil(bin_y_test)))  # Grid points in each dimension

#use only every tenth particle
particles_test_set = particles_test_set[::2,:]

#Plot the points in the grid
plt.figure()
plt.scatter(particles_test_set[:,0],particles_test_set[:,1],s=1)


#set number of contours: N
N = 16  # Number of contours

#create kernel
kde = TreeKDE(kernel='epa',bw = 10, norm=2)
#evaluate the kernel on the datapoints and grid
grid, points = kde.fit(particles_test_set).evaluate(grid_points)
#get x,y, and z values
x,y = np.unique(grid[:,0]),np.unique(grid[:,1])
z = points.reshape(grid_points[0], grid_points[1]).T

plt.figure()
#plot the kde on the grid

# Plot the kernel density estimate
plt.contour(x, y, z, N, linewidths=0.8, colors='k')
plt.contourf(x, y, z, N, cmap="RdBu_r")
#put limits on the x an y axis
plt.xlim([0,bin_x_test])
plt.ylim([0,bin_y_test])
plt.plot(particles_test_set[:,0],particles_test_set[:,1], 'ok', ms=3)

