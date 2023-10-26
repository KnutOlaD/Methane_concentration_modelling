'''
Load and plot opendrift data

Author: Knut Ola Dølven

'''

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import netCDF4 as nc

#List of variables in the script:
#datapath: path to the netcdf file containing the opendrift data
#ODdata: netcdf file containing the opendrift data
#particles: dictionary containing information about the drift particles

#datapath
datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_test.nc'

#load data
ODdata = nc.Dataset(datapath)

#check the variables in the file
print(ODdata.variables.keys())

#Create a dictionary with the variables (just more used to working with dictionaries)
particles = {'lon':ODdata.variables['lon'][:],
                    'lat':ODdata.variables['lat'][:],
                    'z':ODdata.variables['z'][:],
                    'time':ODdata.variables['time'][:],
                    'status':ODdata.variables['status'][:],
                    'trajectory':ODdata.variables['trajectory'][:]}

#remove ODdata from memory
#del ODdata

#Create a matrix which contains the weighing of all the particles, i.e. the amount of methane associated
#with each particle. This has the same shape as the lon/lat matrices
weighing = np.zeros(particles['lon'].shape)
#Find the first value of each particle which is not masked
first_not_masked = np.argmax(~particles['lon'].mask, axis=1)
#Loop over all particles and assign the first_not_masked and all subsequent indices 
#a weighinging equal to the absolute value of the depth.

for i in range(particles['lon'].shape[0]):
    #Loop over all time steps
    for j in range(first_not_masked[i],particles['lon'].shape[1]):
        #Assign the weighing
        weighing[i,j:] = np.abs(particles['z'][i,j])*np.ones(particles['z'][i,j:].shape)


### GREATE A GRID TO BIN THE PARTICLES IN ###
#Vectors with UTM coordinates from the lon/lat matrices using the utm package
UTM_x,UTM_y = utm.from_latlon(
    particles['lat'],
    particles['lon'])
#Assign utm values into the dictionary
particles['UTM_x'] = UTM_x
particles['UTM_y'] = UTM_y

#Create a grid with 800 m resolution using the UTM coordinates
grid_resolution = 800
#Find the minimum and maximum UTM coordinates
UTM_x_min = np.min(particles['UTM_x'])
UTM_x_max = np.max(particles['UTM_x'])
UTM_y_min = np.min(particles['UTM_y'])
UTM_y_max = np.max(particles['UTM_y'])

#Create a grid with 800 m resolution
grid_x = np.arange(UTM_x_min,UTM_x_max,grid_resolution)
grid_y = np.arange(UTM_y_min,UTM_y_max,grid_resolution)

#Create a meshgrid
grid_x,grid_y = np.meshgrid(grid_x,grid_y)

#Create a matrix with the grid coordinates
grid_coordinates = np.array([grid_x.flatten(),grid_y.flatten()]).T

#Loop over each grid cell and find the particles which are inside the grid cell
#and the weighing of each particle. Sum the weighing of all particles inside the grid cell
#and assign this value to the grid cell
grid_weighing = np.zeros(grid_coordinates.shape[0])
for i in range(grid_coordinates.shape[0]):
    #Find the particles inside the grid cell
    particles_inside = (particles['UTM_x'] > 
                        grid_coordinates[i,0]) & (particles['UTM_x'] < 
                        grid_coordinates[i,0]+grid_resolution) & (particles['UTM_y'] > 
                        grid_coordinates[i,1]) & (particles['UTM_y']
                        < grid_coordinates[i,1]+grid_resolution)
    #Sum the weighing of all particles inside the grid cell
    grid_weighing[i] = np.sum(weighing[particles_inside])

#Plot the number of weights in each grid cell in a contour plot
plt.figure()
plt.contourf(grid_x,grid_y,grid_weighing.reshape(grid_x.shape))
plt.colorbar()
plt.title('Number of weights in each grid cell')
plt.xlabel('UTM x')
plt.ylabel('UTM y')
plt.show()








#Plot the weighing of all the particles
plt.figure()
plt.scatter(particles['lon'][:,0],particles['lat'][:,0],c=weighing[:,0])
plt.colorbar()
plt.title('Weighing of particles')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


#Plot all particles at the last time step to show the weighing
plt.figure()
plt.scatter(particles['lon'][:,-1],particles['lat'][:,-1],c=weighing[:,-1])
plt.colorbar()
plt.title('Weighing of particles at the first time step')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


#Plot the trajectories of a single particle
plt.figure()
plt.plot(particles['lon'][0,:],particles['lat'][0,:])
plt.title('Trajectory of a single particle')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

#Plot an image of the status matrix
plt.figure()
plt.imshow(particles['status'])
plt.colorbar()
plt.title('Status matrix')
plt.xlabel('Time step')
plt.ylabel('Particle index')
plt.show()
