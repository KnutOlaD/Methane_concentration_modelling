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
from scipy.sparse import csr_matrix as csr_matrix
import pickle

#List of variables in the script:
#datapath: path to the netcdf file containing the opendrift data
#ODdata: netcdf file containing the opendrift data
#particles: dictionary containing information about the drift particles
#GRID: list of lists containing the horizontal fields at each depth and time step as sparse matrices


#create a list of lists containing the horizontal fields at each depth and time step as sparse matrices
GRID = []

################################
########## FUNCTIONS ###########
################################

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

###########################################################################################

def create_grid(particles,
                resolution=[1000,10],
                savefile_path=False,
                grid_params=[],
                fill_data = False):
    '''
    Creates a list of lists containing sparse matrices for each horizontal field at each 
    depth level and time step. 
    Saves the grid object as a pickle file for later use as option.
    Creates a csr matrix with GRID[timestep][depth_level]

    Input:
    particles: dictionary containing the variables from the netcdf file. Must contain the UTM coordinates
    resolution: list containing the horizontal, vertical and temporal resolution of the grid
        default is [1000,10] which means 1000 meters in the horizontal, 10 meters in the vertical.
    savefile_path: path to where the grid object should be saved as a pickle file
    grid_params: list containing the grid parameters. If this is not empty, the grid object will not be created
    but parameters from the grid_params list will be used. The list should contain the following:
    [bin_x,bin_y,bin_z,bin_time].
    fill_data: boolean. If True, the grid object will be filled with the horizontal fields from
    the particles dictionary

    Output:
    GRID: list of lists containing the sparse matrices for each horizontal field at each
        depth level and time step
    bin_x: bin edges for the horizontal grid
    bin_y: bin edges for the horizontal grid
    bin_z: bin edges for the vertical grid
    bin_time: bin edges for the temporal grid
            
    '''
    #Create a 4 dimensional grid, covering the whole area of the drift and all timesteps
    #Define grid spatial resolution

    if not grid_params:
        grid_resolution = resolution[0] #in meters
        #Define temporal resolution
        grid_temporal_resolution = particles['time'][1] - particles['time'][0] #the time resolution from OD.
        #Define vertical resolution
        grid_vertical_resolution = resolution[1] #in meters

        ### CREATE HORIZONTAL GRID ###
        UTM_x_min = np.min(particles['UTM_x'])
        UTM_x_max = np.max(particles['UTM_x'])
        UTM_y_min = np.min(particles['UTM_y'])
        UTM_y_max = np.max(particles['UTM_y'])

        #Define the bin edges using the grid resolution and min/max values
        bin_x = np.arange(UTM_x_min-grid_resolution,
                          UTM_x_max+grid_resolution,
                          grid_resolution)
        bin_y = np.arange(UTM_y_min-grid_resolution,
                          UTM_y_max+grid_resolution,
                          grid_resolution)

        #Create a horizontal matrix with sizes bin_x and bin_y containing only zeroes
        #This is the matrix that will be filled with the horizontal fields
        H_0 = np.zeros((bin_x.shape[0],bin_y.shape[0]))

        ### CREATE VERTICAL GRID ###
        #Define the bin edges using the grid resolution and min/max values
        bin_z = np.arange(0,
                          np.max(np.abs(particles['z']))+grid_vertical_resolution,
                          grid_vertical_resolution)

        ### CREATE TEMPORAL GRID ###
        #Define the bin edges using the grid resolution and min/max values
        bin_time = np.arange(np.min(particles['time']),
                             np.max(particles['time'])+grid_temporal_resolution,
                             grid_temporal_resolution)
    
    #Loop over all time steps and depth levels and create a sparse matrix for each
    GRID = []
    #GRID.append([])
    for i in range(0,bin_time.shape[0]):
        #Create a sparse matrix for each depth level and timestep
        print(i)
        GRID.append([])
        #Bin all the particles at time j into different depth levels
        z_indices = np.digitize(particles['z'][:,i],bin_z) 
        for j in range(0,bin_z.shape[0]):
            if fill_data == True:
                #Bin the particles in the first time step and depth level to the grid
                #Binned x coordinates:
                x = particles['UTM_x'][z_indices == i,j]
                #Binned y coordinates:
                y = particles['UTM_y'][z_indices == i,j]
                #Find the index locations of the x and y coordinates
                x_indices = np.digitize(x,bin_x)
                y_indices = np.digitize(y,bin_y)
                #Create a horizontal field with the same size as the grid and fill with the 
                #horizontal coordinates, adding up duplicates in the x_indices/y_indices list
                #Find duplicates in the x_indices/y_indices list
                #x_indices_unique = np.unique(x_indices)
                
            else:
                #Create a sparse matrix for each time step
                GRID[i].append([])
                #Create a sparse matrix for each depth level
                GRID[i][j] = csr_matrix(H_0)

    
    if savefile_path == True:
        #save the grid object as a pickle file for later use
        #filename
        f = savefile_path
        with open('grid_object.pickle', 'wb') as f:
            pickle.dump(GRID, f)
    
    return GRID,bin_x,bin_y,bin_z,bin_time

###########################################################################################

def add_utm(particles):
    '''
    Adds utm coordinates to the particles dictionary
    
    Input:
    particles: dictionary containing the variables from the netcdf file. Must contain the lat/lon coordinates

    Output:
    particles: dictionary containing the variables from the netcdf file. Now also contains the UTM coordinates
    
    '''
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
    
    particles['UTM_x'] = UTM_x
    particles['UTM_y'] = UTM_y

    return particles

###########################################################################################

def get_grid_params(particles,resolution=[1000,10]):
    '''
    Gets the grid parameters, i.e. the bin edges for the horizontal, vertical and temporal grids
    from the particles dictionary

    Input:
    particles: dictionary containing the variables from the netcdf file. Must contain the UTM coordinates
    resolution: list containing the horizontal, vertical and temporal resolution of the grid
        default is [1000,10] which means 1000 meters in the horizontal, 10 meters in the vertical.

    Output:
    bin_x: bin edges for the horizontal grid
    bin_y: bin edges for the horizontal grid
    bin_z: bin edges for the vertical grid
    bin_time: bin edges for the temporal grid 
    '''
    #Get min/max values for the horizontal grid
    UTM_x_min = np.min(particles['UTM_x'])
    UTM_x_max = np.max(particles['UTM_x'])
    UTM_y_min = np.min(particles['UTM_y'])
    UTM_y_max = np.max(particles['UTM_y'])

    grid_resolution = resolution[0] #in meters
    #Define temporal resolution
    grid_temporal_resolution = particles['time'][1] - particles['time'][0] #the time resolution from OD.
    #Define vertical resolution
    grid_vertical_resolution = resolution[1] #in meters
    
    #Define the bin edges using the grid resolution and min/max values
    bin_x = np.arange(UTM_x_min-grid_resolution,
                        UTM_x_max+grid_resolution,
                        grid_resolution)
    bin_y = np.arange(UTM_y_min-grid_resolution,
                          UTM_y_max+grid_resolution,
                          grid_resolution)
    #Define the bin edges using the grid resolution and min/max values
    bin_z = np.arange(0,
                          np.max(np.abs(particles['z']))+grid_vertical_resolution,
                          grid_vertical_resolution)
    #Define the bin edges using the grid resolution and min/max values
    bin_time = np.arange(np.min(particles['time']),
                         np.max(particles['time'])+grid_temporal_resolution,
                         grid_temporal_resolution)

    return bin_x,bin_y,bin_z,bin_time


#Just load the grid object to make it faster
with open('grid_object.pickle', 'rb') as f:
    GRID = pickle.load(f)

#Add utm coordinates to the particles dictionary
particles = add_utm(particles)

#Create a zero grid
GRID,bin_x,bin_y,bin_z,bin_time = create_grid(particles,
                                              savefile_path=True,
                                              resolution=[5000,50])

### Try to fill the first sparse matrix with the horizontal field at the first time step and depth level
#Get locations from the utm coordinates in the particles dictionary 
#Get the grid parameters

#Get the grid parameters
#bin_x,bin_y,bin_z,bin_time = get_grid_params(particles)

#Get the last timestep
bin_time_number = np.digitize(particles['time'][0],bin_time)
bin_time_number = len(particles['time'])-1

#Get the utm coordinates of the particles in the first time step
x = particles['UTM_x'][:,bin_time_number]
y = particles['UTM_y'][:,bin_time_number]
z = np.abs(particles['z'][:,bin_time_number])

#Get the bin numbers for the particles
bin_x_number = np.digitize(x.compressed(),bin_x)
bin_y_number = np.digitize(y.compressed(),bin_y)
bin_z_number = np.digitize(z.compressed(),bin_z)

#Simplified vertical profile of length the same as the vertical grid
vertical_profile = np.ones(bin_z.shape[0])
#Should be an exponential with around 100 at the bottom and 10 at the surface
vertical_profile = np.round(np.exp(np.arange(0,np.max(np.abs(particles['z'][:,0])),1)/44))


#Fill the GRID with the horizontal field at all timesteps
for j in range(0,len(particles['time']-1)): 
    print(j)
    #Get the utm coordinates of the particles at time step j but only non-masked values
    bin_x_number = np.digitize(particles['UTM_x'][:,j].compressed(),bin_x)
    bin_y_number = np.digitize(particles['UTM_y'][:,j].compressed(),bin_y)
    bin_z_number = np.digitize(np.abs(particles['z'][:,j]).compressed(),bin_z)
    for i in range(0,len(bin_z_number)):
        GRID[j][bin_z_number[i]][bin_x_number[i],bin_y_number[i]] += 1

#Plot the horizontal field at the first time step and depth level 1
plt.figure()
plt.imshow(np.flipud(GRID[len(particles['time'])-1][1].todense().T))
plt.colorbar()
#set a smaller color range
plt.clim(0,10)
plt.show()

#Plot the horizontal field at the first time step and depth level
#Create a gif of the horizontal fields at the first depth level
import imageio
images = []
for i in range(0,len(GRID)):
    #make sure the color range is the same for all images
    #flip the image left to right
    images.append(np.flipud(GRID[i][1].todense().T))
    #make sure the color range is the same for all images
imageio.mimsave(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\results\Concentration_plots_gifs\horizontal_field.gif', images)

#Create a weighing matrix for the horizontal field 


    










        















