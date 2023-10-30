'''
Load and plot opendrift data

Author: Knut Ola Dølven

'''
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import netCDF4 as nc
import utm
from scipy.sparse import coo_matrix as coo_matrix

#List of variables in the script:
#datapath: path to the netcdf file containing the opendrift data
#ODdata: netcdf file containing the opendrift data
#particles: dictionary containing information about the drift particles


def load_nc_data(filename):
    '''
    Load netcdf file and return a dictionary with the variables
    '''
    #load data
    ODdata = nc.Dataset(filename)

    #check the variables in the file
    print(ODdata.variables.keys())

    #Create a dictionary with the variables (just more used to working with dictionaries)
    particles = {'lon':ODdata.variables['lon'][:],
                        'lat':ODdata.variables['lat'][:],
                        'z':ODdata.variables['z'][:],
                        'time':ODdata.variables['time'][:],
                        'status':ODdata.variables['status'][:],
                        'trajectory':ODdata.variables['trajectory'][:]}

    return particles



#load data:
datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_test.nc'
particles = load_nc_data(datapath)

#get utm coordinates:
UTM_x = np.ma.zeros(particles['lon'].shape)
UTM_x.mask = particles['lon'].mask
UTM_y = np.ma.zeros(particles['lon'].shape)
UTM_y.mask = particles['lon'].mask
#Loop over all time steps
for i in range(particles['lon'].shape[1]):
    #Find the UTM coordinates
    UTM_x[:,i],UTM_y[:,i],zone_number,zone_letter = utm.from_latlon(
    particles['lat'][:,i],
    particles['lon'][:,i])

#Create a 4 dimensional grid, covering the whole area of the drift and all timesteps
#Define grid spatial resolution
grid_resolution = 1000 #in meters
#Define temporal resolution
grid_temporal_resolution = particles['time'][1] - particles['time'][0] #the time resolution from OD.
#Define vertical resolution
grid_vertical_resolution = 10 #in meters

### CREATE HORIZONTAL GRID ###
UTM_x_min = np.min(UTM_x)
UTM_x_max = np.max(UTM_x)
UTM_y_min = np.min(UTM_y)
UTM_y_max = np.max(UTM_y)

#Create a list of lists of sparse matrices which contains the 2d horizontal field at each depth 
#at each time step. The list is structured as [Depth][Time step]

#Define the bin edges using the grid resolution and min/max values
bin_x = np.arange(UTM_x_min,UTM_x_max,grid_resolution)
bin_y = np.arange(UTM_y_min,UTM_y_max,grid_resolution)

#Create a meshgrid with the coordinates of each grid cell
grid_x,grid_y = np.meshgrid(bin_x,bin_y)

### CREATE VERTICAL GRID ###
bin_z = np.arange(0,np.max(np.max(np.abs(particles['z']))+grid_vertical_resolution
                             ),grid_vertical_resolution)

### CREATE TEMPORAL GRID ###
bin_t = np.arange(0,np.max(particles['time'])+grid_temporal_resolution,
                  grid_temporal_resolution)

### CREATE GRID OBJECT ###
#This is a list of lists containing all the horizontal fields for each depth layer and 
#at each timestep. 

GRID = list()

for i in range(bin_z):
    for j in range(bin_t):
        GRID[i][j] = coo_matrix(grid_x.shape)