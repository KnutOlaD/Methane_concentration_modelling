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
import numba


#path to data
datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_test.nc'
#Load the opendrift data
particles = lpo.load_nc_data(datapath)
#add utm coordinates to the dictionary
particles = lpo.add_utm(particles)
#Add a bandwidth parameter to the particles dictionary that has the same
#size as the particles['UTM_x'] and particles['UTM_y'] arrays
particles['bandwidth'] = np.ones(particles['UTM_x'].shape)*0.1 #Initial bandwith is small. 
#the initial weight of all particles
weight_initial = 1  #corresponds to release weight... 
#and a weight parameter
particles['weights'] = np.ones(particles['UTM_x'].shape)*weight_initial
#set horizontal and vertical resolution of the grid
dxy_grid = 500 #m
dz_grid = 10 #m
#get grid parameters
bin_x,bin_y,bin_z,bin_time = lpo.get_grid_params(particles,resolution=[dxy_grid,dz_grid]) #
#Find the lower left corner of the grid in utm coordinates
ll_corner = [np.min(np.min(particles['UTM_x'][:,-1])),np.min(np.min(particles['UTM_y'][:,:]))]
#kernel choice
kernel_c = 'epa'
#set threshold for deactivating particles
deactivate_thr = 10**-6 #this should be in moles/m and correspond to background concentration
#calculate all the horizontal concentration fields using kde for the first timestep.
#this will be len(bin_z) concentration fields of size len(bin_x)*len(bin_y).
#set initial time
t_now = 0

#all particles at the first timestep:
particles_t_now = [particles['UTM_x'][:,0].compressed(),particles['UTM_y'][:,0].compressed()]
#find maximum depth bin in dataset at t_now
max_depth_bin = int(np.ceil(np.max(np.abs(particles['z'][:,t_now])/dz_grid)))
#digitize the particles in the depth bins at t_now and find out where we have particles
zbinlocs_t_now = np.digitize(np.abs(particles['z'][:,t_now]).compressed(),bin_z)

#Get the number of grid points in the horizontal fields (let this be the same for all fields)
grid_points = (int(np.ceil(np.max((bin_x-ll_corner[0])/dxy_grid))),
               int(np.ceil(np.max(bin_y-ll_corner[1])/dxy_grid)))  # Grid points in each dimension

#Create a concentration field SPARSE object that shall contain all the concentration fields for all 
#depth layers and all time steps. Make a dictionary for this
CONCENTRATION = {}
#The structure of the dictionary will be CONCENTRATION[(LAYER,TIMESTEP)] = SPARSE MATRIX

#only loop through the depth bins that have particles in them, i.e. use only integers found in
# zbinlocs_t_now
zbinlocs_t_now_unique = np.unique(zbinlocs_t_now)


from KDEpy import FFTKDE

'''
#make it faster with numba
@jit
def get_kde_estimates(particles_t_now_z_here,
                      zbinlocs_t_now,
                      grid_points,
                      kernel_c):
    
    Function that will calculate the concentration fields for all depth bins at time t_now
    

    for z_here in np.unique(zbinlocs_t_now):
        #find the indices of the particles that are in the jth depth bin
        ind = np.where(zbinlocs_t_now == z_here)[0]
        #get the particles that are in the jth depth bin
        particles_t_now_z_here = np.flipud(np.transpose(np.array([particles['UTM_x'][ind,t_now].compressed(),
                                        particles['UTM_y'][ind,t_now].compressed()])))
        #Get particles locations fit with grid coordinates
        particles_t_now_z_here = (particles_t_now_z_here-ll_corner)/dxy_grid
        #create kernel
        kde = TreeKDE(kernel=kernel_c,bw = 1, norm=2)
        #evaluate the kernel on the datapoints and grid
        grid, points = kde.fit(particles_t_now_z_here).evaluate(grid_points)
        #get x,y, and z values
        x,y = np.unique(grid[:,0]),np.unique(grid[:,1])
        z = points.reshape(grid_points[0], grid_points[1]).T
        #deactivate all particles located in bins with values lower than the deactivation threshold
        #do this by manipulating the particles dictionary
        ind_deactivate = np.where(z<deactivate_thr)[0]
        #set the weights of the particles to zero for all eternity if the particles are in a bin with concentration lower than the deactivation threshold
        particles['weights'][ind_deactivate,t_now] = 0
        #set all values in z that is lower than the deactivation threshold to zero (this is just to make sparse matrix work)
        z[z<deactivate_thr] = 0
        #make a sparse matrix from the z values and store in CONCENTRATION
        CONCENTRATION[(z_here,t_now)] = csr_matrix(z)
        #plot the concentration field
        print(z_here)

#call the function
get_kde_estimates(particles,zbinlocs_t_now,grid_points,kernel_c)



#plot the concentration field
#Number of contours
N = 64
#Get the utm coordinates using ll_corner and dxy_grid
x = x*dxy_grid+np.min(ll_corner[0])
y = y*dxy_grid+np.min(ll_corner[1])
#plot the concentration field
plt.figure()
plt.contourf(x, y, z, N, cmap="RdBu_r")
plt.colorbar()
plt.show()
'''

#Transform the x and y coordinates to utm coordinates by using the dxy    
#do the same for the particles_test_set
particles_test_set = particles_test_set[0].data[:] - np.min(particles_test_set[0].data[:]),particles_test_set[1].data[:] - np.min(particles_test_set[1].data[:])
#divide by 1000 to get the grid in km
particles_test_set = np.array(particles_test_set)
#flip it around
particles_test_set = np.flipud(np.transpose(particles_test_set))

np.sum((m*dz_grid < np.abs(particles['z'][:,t_now])) & (np.abs(particles['z'][:,t_now]) < n*dz_grid)) > 0
#set number of contours: N
N = 64  # Number of contours

#create kernel
kde = FFTKDE(kernel='epa',bw = 10, norm=2)
#evaluate the kernel on the datapoints and grid
grid, points = kde.fit(particles_test_set).evaluate(grid_points)
#get x,y, and z values
x,y = np.unique(grid[:,0]),np.unique(grid[:,1])
z = points.reshape(grid_points[0], grid_points[1]).T


#find all particles that have z value less than 10
z = particles['z'][:,-1]
z = z[~np.isnan(z)]
z = z[z<100]
#find the indices of the particles that have z value less than 10
ind = np.where(np.abs(particles['z'][:,-1])<400)[0]

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
bin_x_test = np.max(particles_test_set[:,0])/dxy_grid
bin_y_test = np.max(particles_test_set[:,1])/dxy_grid
#for particles_test_set too
particles_test_set = particles_test_set/1000

#grid should start from zero and go to 1.1e6 - 0.5e6 in x direction and 8.0e6 - 7.65e6 in y direction
grid_points = (int(np.ceil(bin_x_test)),int(np.ceil(bin_y_test)))  # Grid points in each dimension

#Plot the points in the grid
plt.figure()
plt.scatter(particles_test_set[:,0],particles_test_set[:,1],s=1)




#set number of contours: N
N = 64  # Number of contours

#create kernel
kde = TreeKDE(kernel='epa',bw = 10, norm=2)
#evaluate the kernel on the datapoints and grid
grid, points = kde.fit(particles_test_set).evaluate(grid_points)
#get x,y, and z values
x,y = np.unique(grid[:,0]),np.unique(grid[:,1])
z = points.reshape(grid_points[0], grid_points[1]).T

#Transform the x and y coordinates to utm coordinates by using the dxy_grid resolution parameter
x = x*dxy_grid+np.min(ll_corner[0])
y = y*dxy_grid+np.min(ll_corner[1])

plt.figure()
#plot the kde on the grid

# Plot the kernel density estimate
plt.contour(x, y, z, N, linewidths=0.1, colors='k')
plt.contourf(x, y, z, N, cmap="RdBu_r")
#plt.contourf(x, y, z, N, cmap="Spectral")
#put limits on the x an y axis
plt.xlim([np.min(x),np.max(x)])
plt.ylim([np.min(y),np.max(y)])

#plt.plot(particles_test_set[:,0],particles_test_set[:,1], 'ok', ms=0.1)

#Plot the kernel density estimate using pcolor and pcolormesh
#plt.figure()
#plt.imshow(z)
#plt.colorbar()
#set interpolation to flat

'''



import random 
import math 
import time

#@jit(nopython=True)
def some_function(n):
    z = 0
    for i in range(n):
        x=random.random()
        y = random.random()
        z += math.sqrt(x**2+y**2)
    return z

start = time.time()
some_function(10000000)
end = time.time()
print(end-start)

'''