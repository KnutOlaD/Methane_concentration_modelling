'''
Load and plot opendrift data

Author: Knut Ola Dølven

I think this one runs the big simulation.. 

TO DO

-SET ALL WEIGHTS TO 1
-SET ALL BANDWIDTHS TO 1
- RUN TEST SIMULATION

'''
#Working on plotting function for everything that's correctly projected

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import netCDF4 as nc
import utm
from scipy.sparse import coo_matrix as coo_matrix
from scipy.sparse import csr_matrix as csr_matrix
import pickle
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numba
from numpy.ma import masked_invalid
import imageio
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import seaborn as sns
import time
from numba import jit, prange
import numpy.ma as ma
from pyproj import Proj, Transformer
import  xarray as xr

############################
###DEFINE SOM COOL COLORS###
############################

color_1 = '#7e1e9c'
color_2 = '#014d4e'

###############   
###SET RULES###
###############

plotting = False
#Set plotting style
plt.style.use('dark_background') 
#fit wind data
fit_wind_data = True
#fit gas transfer velocity
fit_gt_vel = True
#set path for wind model pickle file
wind_model_path = 'C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\interpolated_wind_sst_fields_test.pickle'
#plot wind data
plot_wind_field = False
#plot gas transfer velocity
plot_gt_vel = False
#Use all depth layers
use_all_depth_layers = False
### CONSTANTS ###
#max kernel bandwidth
max_ker_bw = 7000 #Used to be 8000
#atmospheric background concentration
#atmospheric_conc = ((44.64*2)/1000000) #mol/m3 #1911.8 ± 0.6 ppb #44.64 #From Helge
#atmospheric_conc = (3.3e-09)*1000 #mol/m3 #ASSUMING SATURATION CONCENTRATION EVERYWHERE. 
atmospheric_conc = 0 #We assume equilibrium with the atmosphere
#oceanic background concentration
#background_ocean_conc = (3.3e-09)*1000 #mol/m3
background_ocean_conc = 0 #We assume equilibrium with the atmosphere
#Oswald solubility coeffocient
oswald_solu_coeff = 0.28 #(for methane)
#Set projection
projection = ccrs.LambertConformal(central_longitude=0.0, central_latitude=70.0, standard_parallels=(70.0, 70.0))
#grid size
dxy_grid = 800. #m
dz_grid = 40. #m
#grid cell volume
V_grid = dxy_grid*dxy_grid*dz_grid
#age constant
age_constant = 10 #m per hour, see figure.
#Initial bandwidth
initial_bandwidth = 50 #m
#set colormap
#colromap = 'magma'
colormap = sns.color_palette("rocket", as_cmap=True)
#K value for the microbial oxidation (MOx) (see under mox section for more values)
R_ox = 3.6*10**-7 #s^-1
#total seabed release that enters the water column as dissoved gas
sum_sb_release = 0.02695169330621381 #mol/s 
sum_sb_release_hr = sum_sb_release*3600 #mol/hr
#number of seed particles
num_seed = 500
#weights full simulation
weights_full_sim = sum_sb_release_hr/num_seed #mol/hr
total_seabed_release = num_seed*weights_full_sim*(30*24) #Old value: 20833 #Whats the unit here?
#only for top layer trigger
kde_all = True
#Weight full sim - think this is wrong
#weights_full_sim = 0.16236 #mol/hr #Since we're now releasing only 500 particles per hour (and not 2000)
#I think this might be mmol? 81.18
#Set manual border for grid
manual_border = True
#what am I doing now?
run_test = False
run_full = True
#KDE dimensionality
kde_dim = 2
#Silvermans coefficients
silverman_coeff = (4/(kde_dim+2))**(1/(kde_dim+4))
silverman_exponent = 1/(kde_dim+4)
#Set bandwidth estimator preference
h_adaptive = 'Local_Silverman' #alternatives here are 'Local_Silverman', 'Time_dep' and 'No_KDE'
#Get new bathymetry or not?
get_new_bathymetry = False
#How should data be loaded/created
load_from_nc_file = True
load_from_hdf5 = False #Fix this later if needed
create_new_datafile = False

#create a list of lists containing the horizontal fields at each depth and time step as sparse matrices
GRID = []

start_time_whole_script = time.time()

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
    particles = {'lon':ODdata.variables['lon'][:].copy(),
                        'lat':ODdata.variables['lat'][:].copy(),
                        'z':ODdata.variables['z'][:].copy(),
                        'time':ODdata.variables['time'][:].copy(),
                        'status':ODdata.variables['status'][:].copy(),
                        'trajectory':ODdata.variables['trajectory'][:].copy()} #this is 

    #example of usage
    #datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_test.nc'
    #particles = load_nc_data(datapath)
    return particles

###########################################################################################

#@numba.jit(nopython=False)
def create_grid(time,
                UTM_x_minmax,
                UTM_y_minmax,
                maxdepth,
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
    maxdepth: maximum depth of the drift

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
        grid_temporal_resolution = time[1] - time[0] #the time resolution from OD.
        #Define vertical resolution
        grid_vertical_resolution = resolution[1] #in meters

        ### CREATE HORIZONTAL GRID ###
        UTM_x_min = UTM_x_minmax[0]
        UTM_x_max = UTM_x_minmax[1]
        UTM_y_min = UTM_y_minmax[0]
        UTM_y_max = UTM_y_minmax[1]

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
                          maxdepth+grid_vertical_resolution,
                          grid_vertical_resolution)

        ### CREATE TEMPORAL GRID ###
        #Define the bin edges using the grid resolution and min/max values
        bin_time = np.arange(np.min(time),
                             np.max(time)+grid_temporal_resolution,
                             grid_temporal_resolution)
    
    #Loop over all time steps and depth levels and create a sparse matrix for each
    GRID = []
    #GRID.append([])
    for i in range(0,bin_time.shape[0]):
        #Create a sparse matrix for each depth level and timestep
        print(i)
        GRID.append([])
        #Bin all the particles at time j into different depth levels
        for j in range(0,bin_z.shape[0]):
            #Create a sparse matrix for each time step
            GRID[i].append([])
            #Create a sparse matrix for each depth level
            GRID[i][j] = csr_matrix(H_0)
            #if fill_data == True:
                #Bin the particles in the first time step and depth level to the grid
                #Binned x coordinates:
                #x = UTM_x[z_indices == i,j]
                #Binned y coordinates:
                #y = UTM_y[z_indices == i,j]
                #Find the index locations of the x and y coordinates
                #x_indices = np.digitize(x,bin_x)
                #y_indices = np.digitize(y,bin_y)
                #Create a horizontal field with the same size as the grid and fill with the 
                #horizontal coordinates, adding up duplicates in the x_indices/y_indices list
                #Find duplicates in the x_indices/y_indices list
                #x_indices_unique = np.unique(x_indices)
            #else:
    
    if savefile_path == True:
        #save the grid object as a pickle file for later use
        #filename
        f = savefile_path
        with open('grid_object.pickle', 'wb') as f:
            pickle.dump(GRID, f)
    
    return GRID,bin_x,bin_y,bin_z,bin_time

###########################################################################################

def add_utm(particles,utm_zone = 33,utm_letter = 'W'):
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
        #set all values outside of the UTM domain to nan
        #...
        #Find the UTM coordinates
        valid_indices = ~particles['lat'][:,i].mask & ~particles['lon'][:,i].mask
        UTM_x[valid_indices,i], UTM_y[valid_indices,i], _, _ = utm.from_latlon(particles['lat'][valid_indices,i], 
        particles['lon'][valid_indices,i], force_zone_number=utm_zone, force_zone_letter=utm_letter)
    
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

#############################################################################################################

def calc_schmidt_number(T=5,gas='methane'):
    '''
    Calculates the Schmidt number for methane, co2, oxygen or nitrogen in 
    seawater at a given temperature. Temperature default value is 5 degrees celcius.

    Input:
    T: temperature in degrees celcius
    gas: string containing the name of the gas. Default is methane. Options are 
    'methane', 'carbon dioxide', 'oxygen' and 'nitrogen'

    Output:
    Sc: Schmidt number

    '''
    
    #Coefficient matrix for methane, oxygen, nitrogen, and carbon dioxide
    #Coefficients from Table 1 in Wanninkhof (2014)

    coeffmat = np.array([[2101.2,-131.54,4.4931,-0.08676,0.00070663],
                        [2116.8,-136.25,4.7353,-0.092307,0.0007555],
                        [1920.4,-135.6,5.2122,-0.10939,0.00093713],
                        [2304.8,-162.75,6.2557,-0.13129,0.0011255]])

    if gas == 'methane':
        coeff = coeffmat[0]
    elif gas == 'carbon dioxide':
        coeff = coeffmat[1]
    elif gas == 'oxygen':
        coeff = coeffmat[2]
    elif gas == 'nitrogen':
        coeff = coeffmat[3]
    else:
        print('Error: Gas name not recognized')
    
    Sc = coeff[0] + coeff[1]*T + coeff[2]*T**2 + coeff[3]*T**3 + coeff[4]*T**4
    
    #everything is good.

    return Sc

#############################################################################################################

def calc_gt_vel(u10=5,temperature=20,gas='methane'):
    ''' 
    Calculates the gas transfer velocity using the Wanninkhof (2014) formulation.

    Input:
    C_o: concentration in the surface layer
    C_a: concentration in the atmosphere
    Sc: Schmidt number #
    u10: wind speed at 10 meters height
    temperature: temperature in degrees celcius in air(?)
    gas: string containing the name of the gas. Default is methane. Options are
    'methane', 'carbon dioxide', 'oxygen' and 'nitrogen'

    Output:
    k: gas transfer velocity #cm/hr #It says in the paper at least ... 

    '''

    #Calculate the Schmidt number
    Sc = calc_schmidt_number(T=temperature,gas=gas)

    #make this such that we can calculate the gas transfer velocity for the whole grid and just
    #grab the data... 
    #Calculate the gas transfer velocity constant
    k = (0.251 * u10**2 * (Sc/660)**(-0.5)) #cm/hr

    #Calculate the atmospheric flux
    #F = k * (C_o - C_a) #mol/m2/day??? WTF? Per day???

    return k

def calc_mox_consumption(C_o,R_ox):
    '''
    Calculates the consumption of oxygen due to methane oxidation

    Input:
    C_o: concentration of methane (mol/m3)
    R_ox: rate of oxygen consumption due to methane oxidation (1/s)

    Output:
    O2_consumption: consumption of oxygen due to methane oxidation
    '''
    CH4_consumption = R_ox * C_o

    return CH4_consumption

@numba.jit(parallel=True,nopython=True)
def histogram_estimator_numba(x_pos,y_pos,grid_x,grid_y,bandwidths = None,weights=None):
    '''
    Input:
    x_pos (np.array): x-coordinates of the particles
    y_pos (np.array): y-coordinates of the particles
    grid_x (np.array): grid cell boundaries in the x-direction
    grid_y (np.array): grid cell boundaries in the y-direction

    Output:
    particle_count: np.array of shape (grid_size, grid_size)
    total_weight: np.array of shape (grid_size, grid_size)
    average_bandwidth: np.array of shape (grid_size, grid_size)
    '''

    #get size of grid in x and y direction
    grid_size_x = len(grid_x)
    grid_size_y = len(grid_y)

    # Initialize the histograms
    particle_count = np.zeros((grid_size_x, grid_size_y), dtype=np.int32)
    total_weight = np.zeros((grid_size_x,grid_size_y), dtype=np.float64)
    sum_bandwidth = np.zeros((grid_size_x,grid_size_y), dtype=np.float64)
    
    #Normalize the particle positions to the grid
    x_pos = (x_pos - grid_x[0])/(grid_x[1]-grid_x[0])
    y_pos = (y_pos - grid_y[0])/(grid_y[1]-grid_y[0])
    
    # Create a 2D histogram of particle positions
    for i in numba.prange(len(x_pos)):
        if np.isnan(x_pos[i]) or np.isnan(y_pos[i]):
            continue
        x = int(x_pos[i])
        y = int(y_pos[i])
        if x >= grid_size_x or y >= grid_size_y or x < 0 or y < 0: #check if the particle is outside the grid
            continue
        total_weight[y, x] += weights[i] #This is just the mass in each cell
        particle_count[y, x] += 1
        sum_bandwidth[y, x] += bandwidths[i]*weights[i] #weighted sum of bandwidths
    
    #print(np.shape(particle_count))

    return particle_count, total_weight, sum_bandwidth 

def histogram_estimator(x_pos, y_pos, grid_x, grid_y, bandwidths=None, weights=None):
    '''
    Input:
    x_pos (np.array): x-coordinates of the particles
    y_pos (np.array): y-coordinates of the particles
    grid_x (np.array): grid cell boundaries in the x-direction
    grid_y (np.array): grid cell boundaries in the y-direction
    bandwidths (np.array): bandwidths of the particles
    weights (np.array): weights of the particles

    Output:
    particle_count: np.array of shape (grid_size, grid_size)
    total_weight: np.array of shape (grid_size, grid_size)
    average_bandwidth: np.array of shape (grid_size, grid_size)
    '''

    # Get size of grid in x and y direction
    grid_size_x = len(grid_x)
    grid_size_y = len(grid_y)

    # Initialize the histograms
    particle_count = np.zeros((grid_size_x, grid_size_y), dtype=np.int32)
    total_weight = np.zeros((grid_size_x, grid_size_y), dtype=np.float64)
    cell_bandwidth = np.zeros((grid_size_x, grid_size_y), dtype=np.float64)
    
    # Normalize the particle positions to the grid
    grid_x0 = grid_x[0]
    grid_y0 = grid_y[0]
    grid_x1 = grid_x[1]
    grid_y1 = grid_y[1]
    x_pos = (x_pos - grid_x0) / (grid_x1 - grid_x0)
    y_pos = (y_pos - grid_y0) / (grid_y1 - grid_y0)
    
    # Filter out NaN values
    valid_mask = ~np.isnan(x_pos) & ~np.isnan(y_pos)
    x_pos = x_pos[valid_mask]
    y_pos = y_pos[valid_mask]
    weights = weights[valid_mask]
    bandwidths = bandwidths[valid_mask]
    
    # Convert positions to integer grid indices
    x_indices = x_pos.astype(np.int32)
    y_indices = y_pos.astype(np.int32)
    
    # Boundary check
    valid_mask = (x_indices >= 0) & (x_indices < grid_size_x) & (y_indices >= 0) & (y_indices < grid_size_y)
    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    weights = weights[valid_mask]
    bandwidths = bandwidths[valid_mask]
    
    # Accumulate weights and counts
    np.add.at(total_weight, (x_indices, y_indices), weights) #This is just the mass in each cell
    np.add.at(particle_count, (x_indices, y_indices), 1)
    np.add.at(cell_bandwidth, (x_indices, y_indices), bandwidths * weights)

    cell_bandwidth = np.divide(cell_bandwidth, total_weight, out=np.zeros_like(cell_bandwidth), where=total_weight!=0)

    return total_weight, particle_count, cell_bandwidth

def generate_gaussian_kernels(num_kernels, ratio, stretch=1):
    """
    Generates Gaussian kernels and their bandwidths. The function generates a kernel with support
    equal to the bandwidth multiplied by the ratio and the ratio sets the "resolution" of the 
    gaussian bandwidth family, i.e. ratio = 1/3 means that one kernel will be created for 0.33, 0.66, 1.0 etc.
    The kernels are stretched in the x-direction by the stretch factor.


    Parameters:
    num_kernels (int): The number of kernels to generate.
    ratio (float): The ratio between the kernel bandwidth and integration support.
    stretch (float): The stretch factor of the kernels. Defined as the ratio between the bandwidth in the x and y directions.

    Returns:
    gaussian_kernels (list): List of Gaussian kernels.
    bandwidths_h (np.array): Array of bandwidths associated with each kernel.
    kernel_origin (list): List of kernel origins.
    """

    gaussian_kernels = [np.array([[1]])]
    bandwidths_h = np.zeros(num_kernels)
    #kernel_origin = [np.array([0, 0])]

    for i in range(1, num_kernels):
        a = np.arange(-i, i + 1, 1).reshape(-1, 1)
        b = np.arange(-i, i + 1, 1).reshape(1, -1)
        h = (i * ratio) #+ ratio * len(a) #multiply with 2 here, since it goes in all directions (i.e. the 11 kernel is 22 wide etc.). 
        #impose stretch and calculate the kernel
        h_a = h*stretch
        h_b = h
        kernel_matrix = ((1 / (2 * np.pi * h_a * h_b)) * np.exp(-0.5 * ((a / h_a) ** 2 + (b / h_b) ** 2)))
        #append the kernel matrix and normalize (to make sure the sum of the kernel is 1)
        gaussian_kernels.append(kernel_matrix / np.sum(kernel_matrix))
        bandwidths_h[i] = h
        #kernel_origin.append(np.array([0, 0]))

    return gaussian_kernels, bandwidths_h#, kernel_origin

##############################
### ONGOING OPTIMIZASATION ###
##############################

@jit(nopython=True, parallel=True)
def _process_kernels(non_zero_indices, kde_pilot, cell_bandwidths, kernel_bandwidths, 
                    gaussian_kernels, illegal_cells, gridsize_x, gridsize_y):
    
    """Numba-optimized kernel processing"""
    
    n_u = np.zeros((gridsize_x, gridsize_y))
    
    for idx in prange(len(non_zero_indices)):
        i, j = non_zero_indices[idx]
        
        # Get kernel index and kernel
        kernel_index = np.argmin(np.abs(kernel_bandwidths - cell_bandwidths[i, j]))
        kernel = gaussian_kernels[kernel_index].copy()  # Need copy for numba
        kernel_size = len(kernel) // 2
        
        # Window boundaries
        i_min = max(i - kernel_size, 0)
        i_max = min(i + kernel_size + 1, gridsize_x)
        j_min = max(j - kernel_size, 0)
        j_max = min(j + kernel_size + 1, gridsize_y)
        
        # Handle illegal cells
        illegal_window = illegal_cells[i_min:i_max, j_min:j_max]
        if np.any(illegal_window):
            illegal_sum = 0.0
            for ii in range(i_max - i_min):
                for jj in range(j_max - j_min):
                    if illegal_window[ii, jj]:
                        illegal_sum += kernel[ii, jj]
                        kernel[ii, jj] = 0
            weighted_kernel = kernel * (kde_pilot[i,j] + illegal_sum)
        else:
            weighted_kernel = kernel * kde_pilot[i, j]
            
        # Add contribution
        n_u[i_min:i_max, j_min:j_max] += weighted_kernel[
            max(0, kernel_size - i):kernel_size + min(gridsize_x - i, kernel_size + 1),
            max(0, kernel_size - j):kernel_size + min(gridsize_y - j, kernel_size + 1)
        ]
    
    return n_u

def grid_proj_kde(grid_x, grid_y, kde_pilot, gaussian_kernels, 
                  kernel_bandwidths, cell_bandwidths, illegal_cells=None):
    """Optimized version of grid_proj_kde"""
    
    # Initialize illegal cells if None
    if illegal_cells is None:
        illegal_cells = np.zeros((len(grid_x), len(grid_y)), dtype=np.bool_)
    
    # Get grid sizes
    gridsize_x, gridsize_y = len(grid_x), len(grid_y)
    
    # Convert gaussian_kernels to homogeneous float64 arrays
    gaussian_kernels = [np.ascontiguousarray(kernel, dtype=np.float64) for kernel in gaussian_kernels]
    
    # Pre-compute non-zero indices
    non_zero_indices = np.array(np.where(kde_pilot > 0)).T
    
    # Convert inputs to contiguous arrays with consistent dtypes
    kde_pilot = np.ascontiguousarray(kde_pilot, dtype=np.float64)
    cell_bandwidths = np.ascontiguousarray(cell_bandwidths, dtype=np.float64)
    kernel_bandwidths = np.ascontiguousarray(kernel_bandwidths, dtype=np.float64)
    illegal_cells = np.ascontiguousarray(illegal_cells, dtype=np.bool_)
    
    # Process kernels using numba-optimized function
    n_u = _process_kernels(non_zero_indices, kde_pilot, cell_bandwidths,
                          kernel_bandwidths, gaussian_kernels, illegal_cells,
                          gridsize_x, gridsize_y)
    
    return n_u

##############################
### ONGOING OPTIMIZASATION ###
##############################

#Function to calculate the grid projected kernel density estimator
def grid_proj_kde_deprec(grid_x, 
                  grid_y, 
                  kde_pilot, 
                  gaussian_kernels, 
                  kernel_bandwidths, 
                  cell_bandwidths,
                  illegal_cells = None):
    """
    Projects a kernel density estimate (KDE) onto a grid using Gaussian kernels.

    Parameters:
    grid_x (np.array): Array of grid cell boundaries in the x-direction.
    grid_y (np.array): Array of grid cell boundaries in the y-direction.
    kde_pilot (np.array): The pilot KDE values on the grid.
    gaussian_kernels (list): List of Gaussian kernel matrices.
    kernel_bandwidths (np.array): Array of bandwidths associated with each Gaussian kernel.
    cell_bandwidths (np.array): Array of bandwidths of the particles.
    illegal_cells = array of size grid_x,grid_y with True/False values for illegal cells

    Returns:
    np.array: The resulting KDE projected onto the grid.

    Notes:
    - This function only works with a simple histogram estimator as the pilot KDE.
    - The function assumes that the Gaussian kernels are symmetric around their center.
    - The grid size is determined by the lengths of grid_x and grid_y.
    - The function iterates over non-zero values in the pilot KDE and applies the corresponding Gaussian kernel.
    - The appropriate Gaussian kernel is selected based on the bandwidth of each particle.
    - The resulting KDE is accumulated in the output grid n_u.
    - Uses the reflection method to handle boundary conditions.
    """
    # ONLY WORKS WITH SIMPLE HISTOGRAM ESTIMATOR ESTIMATE AS PILOT KDE!!!

    if illegal_cells is None:
        illegal_cells = np.zeros((len(grid_x), len(grid_y)), dtype=bool)

    # Get the grid size
    gridsize_x = len(grid_x)
    gridsize_y = len(grid_y)

    n_u = np.zeros((gridsize_x, gridsize_y))

    # Get the indices of non-zero kde_pilot values
    non_zero_indices = np.argwhere(kde_pilot > 0)
   
    # Find the closest kernel indices for each particle bandwidth
    # kernel_indices = np.argmin(np.abs(kernel_bandwidths[:, np.newaxis] - cell_bandwidths[tuple(non_zero_indices.T)]), axis=0)
    
    for idx in non_zero_indices:
        i, j = idx
        # Get the appropriate kernel for the current particle bandwidth
        # find the right kernel index
        kernel_index = np.argmin(np.abs(kernel_bandwidths - cell_bandwidths[i, j])) #Can be vectorized
        # kernel_index = kernel_indices[i * grid_size + j]
        kernel = gaussian_kernels[kernel_index]
        kernel_size = len(kernel) // 2  # Because it's symmetric around the center.

        # Define the window boundaries
        i_min = max(i - kernel_size, 0)
        i_max = min(i + kernel_size + 1, gridsize_x)
        j_min = max(j - kernel_size, 0)
        j_max = min(j + kernel_size + 1, gridsize_y)

        #Check if there are illegal cells in the kernel area and run reflect_kernel_contribution if there are
        #if np.any(illegal_cells[i_min:i_max, j_min:j_max]):

        #Handle illegal cells
        if np.any(np.argwhere(illegal_cells[i_min:i_max, j_min:j_max])):
            illegal_indices = np.argwhere(illegal_cells[i_min:i_max, j_min:j_max])
            #Sum contribution for all illegal cells in the kernel
            illegal_kernel_sum = np.sum(kernel[illegal_indices[:,0],illegal_indices[:,1]])
            #set them to zero
            kernel[illegal_indices[:,0],illegal_indices[:,1]] = 0
            #calculat the weighted kernel sum
            weighted_kernel = kernel*(kde_pilot[i,j]+illegal_kernel_sum)
        else:
            weighted_kernel = kernel * kde_pilot[i, j]

        # Add the contribution to the result matrix
        n_u[i_min:i_max, j_min:j_max] += weighted_kernel[
            max(0, kernel_size - i):kernel_size + min(gridsize_x - i, kernel_size + 1),
            max(0, kernel_size - j):kernel_size + min(gridsize_y - j, kernel_size + 1)
        ]

    return n_u

def kernel_matrix_2d_NOFLAT(x,y,x_grid,y_grid,bw,weights,ker_size_frac=4,bw_cutoff=2):
    ''' 
    Creates a kernel matrices for a 2d gaussian kernel with bandwidth bw and a cutoff at 
    2*bw for all datapoints and sums them onto grid x_grid,ygrid. The kernel matrices are 
    created by binning the kernel values (the 2d gaussian) are created with a grid with
    adaptive resolution such that the kernel resolution fits within the x_grid/y_grid grid resolution. 
    Normalizes with the sum of the kernel values (l2norm). Assumes uniform x_grid/y_grid resolution.
    Input: 
    x: x-coordinates of the datapoints
    y: y-coordinates of the datapoints
    x_grid: x-coordinates of the grid
    y_grid: y-coordinates of the grid
    bw: bandwidth of the kernel (vector of length n with the bandwidth for each datapoint)
    weights: weights for each datapoint
    ker_size_frac: the fraction of the grid size of the underlying grid that the kernel grid should be
    bw_cutoff: the cutoff for the kernel in standard deviations

    Output:
    GRID_active: a 3d matrix with the kernel values for each datapoint
    '''

    #desired fractional difference between kernel grid size and grid size
    ker_size_frac = 4 #1/3 of the grid size of underlying grid
    bw_cutoff = 2 #cutoff for the kernel in standard deviations

    #calculate the grid resolution
    dxy_grid = x_grid[1]-x_grid[0]

    #create a grid for z values
    GRID_active = np.zeros((len(x_grid),len(y_grid)))

    for i in range(len(x)):
        #calculate the kernel for each datapoint
        #kernel_matrix[i,:] = gaussian_kernel_2d(grid_points[0]-x[i],grid_points[1]-y[i],bw=bw)
        #create a matrix for the kernel that makes sure the kernel resolution fits
        #within the grid resolution (adaptive kernel size). ker_size is the number of points in each direction
        #in the kernel. Can also add in weight of kernel here to save time, but let's do that later if needed.
        ker_size = int(np.ceil((bw_cutoff*2*bw[i]*ker_size_frac)/dxy_grid))
        a = np.linspace(-bw_cutoff*bw[i],bw_cutoff*bw[i],ker_size)
        b = np.linspace(-bw_cutoff*bw[i],bw_cutoff*bw[i],ker_size)
        #create the 2d coordinate matrix
        a = a.reshape(-1,1)
        b = b.reshape(1,-1)
        #kernel_matrix[i,:] = #gaussian_kernel_2d_sym(a,b,bw=1, norm='l2norm')
        kernel_matrix = ((1/(2*np.pi*bw[i]**2))*np.exp(-0.5*((a/bw[i])**2+(b/bw[i])**2)))/np.sum(((1/(2*np.pi*bw[i]**2))*np.exp(-0.5*((a/bw[i])**2+(b/bw[i])**2))))
        #add the kernel_matrix values by binning them into the grid using digitize
        #get the indices of the grid points that are closest to the datapoints
        lx = a+x[i]
        ly = b+y[i]
        #get the indices of the grid points that are closest to the datapoints
        ix = np.digitize(lx,x_grid)
        iy = np.digitize(ly,y_grid)

        #if any values in ix or iy is outside the grid, remove the kernel entirely and skip to next iteration
        if np.any(ix >= len(x_grid)) or np.any(iy >= len(y_grid)) or np.any(ix < 0) or np.any(iy < 0):
            continue

        #add the kernel values to the grid
        GRID_active[ix,iy] += kernel_matrix*weights[i]

    return GRID_active


from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, ScalarFormatter, LogLocator
import matplotlib.patches as patches

def plot_2d_data_map_loop(data,
                          lon,
                          lat,
                          projection,
                          levels,
                          timepassed,
                          colormap,
                          title,
                          unit,
                          savefile_path=False,
                          show=False,
                          adj_lon = [0,0],
                          adj_lat = [0,0],
                          bar_position = [0.195,0.12,0.54558,0.03],
                          dpi=150,
                          log_scale=False,
                          figuresize=[14,10],
                          starttimestring='May 20, 2021',
                          endtimestring='May 20, 2021',
                          maxnumticks = 10,
                          plot_progress_bar = True,
                          plot_model_domain = False,
                          contoursettings = [2,'0.8',0.1]):
    '''
    Plots 2d data on a map with time progression bar.

    Input:
    data: 2d data
    lon: longitude
    lat: latitude
    timepassed: i and length of time vector
    projection: projection
    levels: levels for the contour plot
    colormap: colormap
    title: title of the plot
    unit: unit of the data
    savefile_path: path to where the plot should be saved
    show: boolean. If True, the plot will be shown
    adj_lon: adjustment of the longitude extent
    adj_lat: adjustment of the latitude extent
    bar_position: position of the time progression bar
    dpi: dpi of the plot
    log_scale: boolean. If True, the colorbar will be logarithmic, uses min/max of levels input
    figuresize: size of the figure
    plot_model_domain: [min_lon,max_lon,min_lat,max_lat,linewidth,color] for the model domain
    contoursettings = [contourfrac,contourcolor,contourlinewidth] for the contour lines

    Output:
    fig: figure object
    '''

    
    # Cache transformed coordinates
    if np.shape(lon) != np.shape(data):
        lon, lat = np.meshgrid(lon, lat)
    
    # Create figure only once
    fig = plt.figure(figsize=kwargs.get('figuresize', [14, 10]))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
    ax = fig.add_subplot(gs[0], projection=projection)
    
    # Pre-transform coordinates
    transformed_coords = ccrs.PlateCarree().transform_points(
        projection, lon, lat
    )
    
    # Optimize contour plotting
    if kwargs.get('log_scale', False):
        norm = LogNorm(vmin=np.min(levels), vmax=np.max(levels))
        contourf = ax.contourf(lon, lat, data, 
                             levels=levels,
                             cmap=colormap,
                             norm=norm,
                             transform=ccrs.PlateCarree(),
                             zorder=0,
                             extend='both')
    else:
        contourf = ax.contourf(lon, lat, data,
                             levels=levels,
                             cmap=colormap,
                             transform=ccrs.PlateCarree(),
                             zorder=0,
                             extend='max')
    
    # Reduce gridline density
    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                     draw_labels=False,
                     linewidth=0.5,
                     color='white',
                     alpha=0.5,
                     linestyle='--',
                     xlocs=np.linspace(lon.min(), lon.max(), 5),
                     ylocs=np.linspace(lat.min(), lat.max(), 5))
    
    # Add essential features only
    ax.add_feature(cfeature.LAND, facecolor='0.2', zorder=2)
    ax.add_feature(cfeature.COASTLINE, zorder=3, color='0.5', linewidth=0.5)
    
    # Optimize colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label(unit, fontsize=16)
    
    if kwargs.get('plot_progress_bar', True):
        ax2 = plt.subplot(gs[1])
        pos = ax.get_position()
        bar_position = kwargs.get('bar_position', [0.195,0.12,0.54558,0.03])
        ax2.set_position([pos.x0, bar_position[1], pos.width, bar_position[3]])
        ax2.fill_between([0, timepassed[0]], [0, 0], [1, 1], color='grey')
        ax2.set_yticks([])
        ax2.set_xticks([0, timepassed[1]])
        ax2.set_xticklabels([
            kwargs.get('starttimestring', 'May 20, 2021'),
            kwargs.get('endtimestring', 'May 20, 2021')
        ], fontsize=16)
    
    # Save/show plot
    if 'savefile_path' in kwargs:
        plt.savefig(kwargs['savefile_path'],
                   dpi=kwargs.get('dpi', 150),
                   bbox_inches='tight')
    if kwargs.get('show', False):
        plt.show()
    
    return fig

def fit_wind_sst_data(bin_x,bin_y,bin_time,run_test=False):
    '''
    loads and fits wind and sea surface temperature data onto model grid given by bin_x, bin_y and bin_time.
    '''
    print('Fitting wind data')
    #load the wind data
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\ERAV_all_2018.pickle', 'rb') as f:
        lons, lats, times, sst, u10, v10, ws = pickle.load(f)
    #create a time vector on unix timestamps
    #create datetime vector
    datetime_vector = pd.to_datetime(times)
    # Convert datetime array to Unix timestamps
    wind_time_unix = (datetime_vector.astype(np.int64) // 10**9).values
    #create a time vector that starts on May 20 if using the test data
    ocean_time_unix = bin_time
    #reverse the lats coordinates and associated matrices (sst, u10, v10, ws)
    lats = lats[::-1]
    #the matrices should be reversed only in the first dimension
    sst = sst[:,::-1,:]
    u10 = u10[:,::-1,:]
    v10 = v10[:,::-1,:]
    ws = ws[:,::-1,:]
    ### FIT THE WIND DATASET ONTO THE MODEL GRID ###
    #create a lon/lat meshgrid
    lon_mesh,lat_mesh = np.meshgrid(lons,lats)
    # Create empty arrays for UTM coordinates
    UTM_x_wind = np.empty_like(lon_mesh)
    UTM_y_wind = np.empty_like(lat_mesh)
    utm_zone = 33 #force zone number to be 33
    # Convert each point in the grid to UTM
    for i in range(lon_mesh.shape[0]):
        for j in range(lon_mesh.shape[1]):
            UTM_x_wind[i, j], UTM_y_wind[i, j], _, _ = utm.from_latlon(lat_mesh[i, j], lon_mesh[i, j],force_zone_number=utm_zone, force_zone_letter='W')
    #create meshgrids for bin_x and bin_y
    bin_x_mesh,bin_y_mesh = np.meshgrid(bin_x,bin_y)
    #Loop over and 
    #1. Create objects to store wind fields and sst fields for each model timestep
    #2. Find the closest neighbour in time in the wind field
    #3. Interpolate the wind field onto the model grid
    #4. Interpolate the sst field onto the model grid
    #create objects to store wind fields and sst fields for each model timestep
    ws_interp = np.zeros((len(bin_time),len(bin_y),len(bin_x)))
    sst_interp = np.zeros((len(bin_time),len(bin_y),len(bin_x)))
    for i in range(0,len(bin_time)):
        print(i)
        #find the closest neighbour in time in the wind field
        time_diff = np.abs(ocean_time_unix[i] - wind_time_unix)
        time_index = np.argmin(time_diff)
        #interpolate ws and sst onto the model grid
        ws_interp[i,:,:] = griddata((UTM_x_wind.flatten(), UTM_y_wind.flatten()), ws[time_index,:,:].flatten(), (bin_x_mesh, bin_y_mesh), method='cubic')
        sst_interp[i,:,:] = griddata((UTM_x_wind.flatten(), UTM_y_wind.flatten()), sst[time_index,:,:].flatten(), (bin_x_mesh, bin_y_mesh), method='nearest')
    #store the interpolated wind fields and sst fields in a pickle file
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\model_grid\\interpolated_wind_sst_fields_test.pickle', 'wb') as f:
        pickle.dump([ws_interp,sst_interp,bin_x_mesh,bin_y_mesh,ocean_time_unix], f)
    return None


@jit(nopython=True)
def histogram_variance_numba(binned_data, bins): #here, suggest to multiply with (M-1)/M to get unbiased estimate
    if np.sum(binned_data) == 0:
        return 0.0
    hist, bin_edges = np.histogram(binned_data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean = np.sum(hist * bin_centers) / np.sum(hist) #Weighted mean position
    variance = np.sum(hist * (bin_centers - mean) ** 2) / np.sum(hist)
    return variance

@jit(nopython=True)
def histogram_std(binned_data, effective_samples=None, bin_size=1):
    '''Calculate the simple variance of the binned data'''
    if np.sum(binned_data) == 0:
        return 0
        
    if effective_samples is None:
        effective_samples = np.sum(binned_data)
    
    grid_size = len(binned_data)
    X = np.arange(0, grid_size * bin_size, bin_size)
    Y = np.arange(0, grid_size * bin_size, bin_size)
    
    sum_data = np.sum(binned_data)
    mu_x = np.sum(binned_data * X) / sum_data
    mu_y = np.sum(binned_data * Y) / sum_data
    
    # Vectorized variance calculation
    variance = (np.sum(binned_data * ((X - mu_x)**2 + (Y - mu_y)**2)) / 
               (sum_data - 1)) - bin_size * bin_size / 12
    
    return np.sqrt(variance)


def histogram_variance(binned_data, bin_size=1):
    '''
    Calculate the simple variance of the binned data using ...
    '''
    #check that there's data in the binned data
    if np.sum(binned_data) == 0:
        return 0
    #get the central value of all bins
    grid_size = len(binned_data)
    #Central point of all grid cells
    X = np.arange(0,grid_size*bin_size,bin_size) #I think this doesnt work.. 
    Y = np.arange(0,grid_size*bin_size,bin_size)
    #Calculate the average position in the binned data
    mu_x = np.sum(binned_data*X)/np.sum(binned_data)
    mu_y = np.sum(binned_data*Y)/np.sum(binned_data)
    #Calculate the variance
    var_y = np.sum(binned_data*(X-mu_x)**2)/np.sum(binned_data)
    var_x = np.sum(binned_data*(Y-mu_y)**2)/np.sum(binned_data)
    #Calculate the covariance
    cov_xy = np.sum(binned_data*(X-mu_x)*(Y-mu_y))/np.sum(binned_data)
    #Calculate the total variance
    variance_data = var_x+var_y+2*cov_xy*0
    #https://towardsdatascience.com/on-the-statistical-analysis-of-rounded-or-binned-data-e24147a12fa0
    #Sheppards correction
    variance_data = variance_data - 1/12*(3*bin_size**2)
    return variance_data

def window_sum(data):
    # Filter out zero values
    non_zero_data = data[data != 0]
    return np.sum(non_zero_data)

@jit(nopython=True)
def calculate_autocorrelation_numba(data):
    num_rows, num_cols = data.shape
    max_lag = min(num_rows, num_cols) - 1
    
    autocorr_rows = np.zeros(max_lag)
    autocorr_cols = np.zeros(max_lag)
    
    row_denominators = np.array([1 / (num_cols - k) for k in range(1, max_lag + 1)])
    col_denominators = np.array([1 / (num_rows - k) for k in range(1, max_lag + 1)])
    
    for k in range(1, max_lag + 1):
        autocorr_rows[k - 1] = np.mean([row_denominators[k - 1] * np.sum(data[row, :num_cols - k] * data[row, k:]) for row in range(num_rows)])
    
    for k in range(1, max_lag + 1):
        autocorr_cols[k - 1] = np.mean([col_denominators[k - 1] * np.sum(data[:num_rows - k, col] * data[k:, col]) for col in range(num_cols)])
    
    return autocorr_rows, autocorr_cols

@jit(nopython=True)
def calculate_autocorrelation(data, bin_size=1):
    '''Calculate autocorrelation for rows and columns'''
    num_rows, num_cols = data.shape
    max_lag = min(num_rows, num_cols) - 1
    
    autocorr_rows = np.zeros(max_lag)
    autocorr_cols = np.zeros(max_lag)
    
    # Precompute denominators
    row_denominators = 1.0 / np.arange(num_cols - 1, num_cols - max_lag - 1, -1)
    col_denominators = 1.0 / np.arange(num_rows - 1, num_rows - max_lag - 1, -1)
    
    # Vectorized autocorrelation calculation
    for k in range(1, max_lag + 1):
        row_sum = 0.0
        col_sum = 0.0
        
        for i in range(num_rows):
            row_sum += np.sum(data[i, :num_cols-k] * data[i, k:])
        for j in range(num_cols):
            col_sum += np.sum(data[:num_rows-k, j] * data[k:, j])
            
        autocorr_rows[k-1] = row_sum * row_denominators[k-1] / num_rows
        autocorr_cols[k-1] = col_sum * col_denominators[k-1] / num_cols
    
    return autocorr_rows, autocorr_cols

def get_integral_length_scale(histogram_prebinned, window_size):
    '''
    Processes the histogram_prebinned data to calculate the integral length scale
    for all non-zero elements using a specified window size.
    
    Input:
    histogram_prebinned: 2D matrix with histogram data
    window_size: Size of the window to use for the calculations
    
    Output:
    integral_length_scale_matrix: 2D matrix with the integral length scale values
    '''
    rows, cols = histogram_prebinned.shape
    integral_length_scale_matrix = np.zeros_like(histogram_prebinned, dtype=float)

    #find all non-zero indices in histogram_prebinned
    non_zero_indices = np.argwhere(histogram_prebinned > 0)

    #pad histogram_prebinned to avoid edge effects
    histogram_prebinned_padded = np.pad(histogram_prebinned, window_size // 2, mode='reflect')

    for idx in non_zero_indices:
        i, j = idx
        window = histogram_prebinned_padded[i:i + window_size, j:j + window_size]
        if np.any(window != 0):
            autocorr_rows, autocorr_cols = calculate_autocorrelation(window)
            autocorr = (autocorr_rows + autocorr_cols) / 2
            integral_length_scale = np.sum(autocorr) / autocorr[0]
            integral_length_scale_matrix[i, j] = integral_length_scale

    return integral_length_scale_matrix

#Reflect kernel density at predefined boundaries
def reflect_with_shadow(x, y, xi, yj, legal_grid):
    """
    Helper function to reflect (xi, yj) back to a legal position
    across the barrier while respecting the shadow.
    """
    x_reflect, y_reflect = xi, yj

    # Reflect along x-axis if needed
    while not legal_grid[x_reflect, yj] and x_reflect != x:
        x_reflect += np.sign(x - xi)  # Step towards the particle

    # Reflect along y-axis if needed
    while not legal_grid[xi, y_reflect] and y_reflect != y:
        y_reflect += np.sign(y - yj)  # Step towards the particle
    
    # Check final reflection position legality
    if legal_grid[x_reflect, y_reflect]:
        return x_reflect, y_reflect
    else:
        return None, None  # No valid reflection found

#Make a bresenham line
def bresenham(x0, y0, x1, y1): 
    """
    Bresenham's Line Algorithm to generate points between (x0, y0) and (x1, y1)

    Intput:
    x0: x-coordinate of the starting point
    y0: y-coordinate of the starting point
    x1: x-coordinate of the ending point
    y1: y-coordinate of the ending point

    Output:
    points: List of points between (x0, y0) and (x1, y1)
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1 # Step direction for x
    sy = 1 if y0 < y1 else -1 # Step direction for y
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

#Identify shadowed cells
def identify_shadowed_cells(x0, y0, xi, yj, legal_grid):
    """
    Identify shadowed cells in the legal grid.

    Input: 
    x0: x-coordinate of the kernel origin grid cell (for grid projected)
    y0: y-coordinate of the kernel origin grid cell (for grid projected)
    xi: x-coordinates of the kernel
    yj: y-coordinates of the kernel
    legal_grid: 2D boolean array with legal cells (true means legal)
    """
    shadowed_cells = []
    for i in xi:
        for j in yj:
            cells = bresenham(x0, y0, i, j)
            for cell in cells:
                if not legal_grid[cell[0], cell[1]]:
                    shadowed_cells.append((i,j))
    return shadowed_cells

def process_bathymetry(bathymetry_path, bin_x, bin_y, transformer, output_path):
    """
    Process bathymetry data and return bathymetry values with corresponding lon/lat grids.

    Parameters:
    -----------
    bathymetry_path : str
        Path to the bathymetry data file
    bin_x : np.array
        Array of x-coordinates (UTM) for target grid
    bin_y : np.array
        Array of y-coordinates (UTM) for target grid
    transformer : Transformer
        Transformer object for coordinate transformation
    output_path : str
        Path to save the processed data

    Returns:
    --------
    tuple : (interpolated_bathymetry, latitude_grid, longitude_grid)
        interpolated_bathymetry : np.array
            Interpolated bathymetry data
        latitude_grid : np.array
            Grid of latitude values
        longitude_grid : np.array
            Grid of longitude values
    """
    # Input validation
    if not os.path.exists(bathymetry_path):
        raise FileNotFoundError(f"Bathymetry file not found: {bathymetry_path}")
    
    try:
        # Load bathymetry data
        bathy_data = xr.open_dataset(bathymetry_path)
        x_coords = bathy_data['x'].values
        y_coords = bathy_data['y'].values
        bathymetry = bathy_data['z'].values

        # Transform source coordinates to lat/lon
        lon_coords_mesh, lat_coords_mesh = transformer.transform(
            *np.meshgrid(x_coords, y_coords)
        )
        
        # TODO: Verify if this +45 adjustment is needed for your region
        lon_coords_mesh += 45

        # Create target grid in UTM coordinates
        bin_x_mesh, bin_y_mesh = np.meshgrid(bin_x, bin_y)
        
        # Convert target UTM grid to lat/lon
        lat_grid, lon_grid = utm.to_latlon(
            bin_x_mesh, 
            bin_y_mesh, 
            zone_number=33,  # Verify UTM zone for your region
            zone_letter='W'
        )

        # Prepare coordinates for interpolation
        points = np.column_stack((lon_coords_mesh.ravel(), lat_coords_mesh.ravel()))
        values = bathymetry.ravel()
        target_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

        # Interpolate bathymetry data
        interpolated_bathymetry = griddata(
            points, 
            values, 
            target_points, 
            method='nearest'
        )
        interpolated_bathymetry = interpolated_bathymetry.reshape(lat_grid.shape)
        
        # Set positive depths to zero (assuming depths are negative)
        interpolated_bathymetry[interpolated_bathymetry > 0] = 0

        # Save the processed data
        with open(output_path, 'wb') as f:
            pickle.dump({
                'bathymetry': interpolated_bathymetry,
                'latitude': lat_grid,
                'longitude': lon_grid
            }, f)

        return interpolated_bathymetry, lat_grid, lon_grid
    
    except Exception as e:
        raise RuntimeError(f"Error processing bathymetry data: {str(e)}")

@jit(nopython=True)
def _process_window_statistics(data_subset, subset_counts, pad_size, window_size, 
                             stats_threshold, silverman_coeff, silverman_exponent, dxy_grid):
    """Compute statistics for a single window"""
    if np.sum(subset_counts) < stats_threshold:
        std = window_size/2
        n_eff = np.sum(data_subset)/window_size
        integral_length_scale = window_size
    else:
        # Assuming histogram_std is available to Numba
        std = histogram_std(data_subset, None, 1)
        # Assuming calculate_autocorrelation is available to Numba
        autocorr_rows, autocorr_cols = calculate_autocorrelation(data_subset)
        autocorr = (autocorr_rows + autocorr_cols) / 2
        if autocorr.any():
            non_zero_idx = np.where(autocorr != 0)[0][0]
            integral_length_scale = np.sum(autocorr) / autocorr[non_zero_idx]
        else:
            integral_length_scale = 0.000001
        n_eff = np.sum(data_subset) / integral_length_scale
    
    h = np.sqrt((silverman_coeff * n_eff**(-silverman_exponent)) * std) * dxy_grid
    return std, n_eff, integral_length_scale, h

@jit(nopython=True, parallel=True)
def compute_adaptive_bandwidths(preGRID_active_padded, preGRID_active_counts_padded,
                              window_size, pad_size, stats_threshold,
                              silverman_coeff, silverman_exponent, dxy_grid):
    """
    Compute adaptive bandwidths for all windows with integrated statistics processing
    
    Parameters:
    -----------
    preGRID_active_padded : np.ndarray
        Padded grid of active particles
    preGRID_active_counts_padded : np.ndarray
        Padded grid of particle counts
    window_size : int
        Size of the processing window
    pad_size : int
        Size of padding around the grid
    stats_threshold : float
        Threshold for statistical calculations
    silverman_coeff : float
        Coefficient for Silverman's rule
    silverman_exponent : float
        Exponent for Silverman's rule
    dxy_grid : float
        Grid spacing
    """
    # Convert to float64 and ensure contiguous
    preGRID_active_padded = np.ascontiguousarray(preGRID_active_padded.astype(np.float64))
    preGRID_active_counts_padded = np.ascontiguousarray(preGRID_active_counts_padded.astype(np.float64))
    
    shape = preGRID_active_padded.shape
    std_estimate = np.zeros((shape[0]-2*pad_size, shape[1]-2*pad_size), dtype=np.float64)
    N_eff = np.zeros_like(std_estimate)
    h_matrix_adaptive = np.zeros_like(std_estimate)
    integral_length_scale_matrix = np.zeros_like(std_estimate)
    
    # Main processing loop with parallel support
    for row in prange(pad_size, shape[0]-pad_size):
        for col in range(pad_size, shape[1]-pad_size):
            if preGRID_active_counts_padded[row, col] > 0:
                # Extract data subset
                data_subset = preGRID_active_padded[
                    row-pad_size:row+pad_size+1,
                    col-pad_size:col+pad_size+1
                ]
                subset_counts = preGRID_active_counts_padded[
                    row-pad_size:row+pad_size+1,
                    col-pad_size:col+pad_size+1
                ]
                
                # Skip if center cell is empty
                if data_subset[pad_size,pad_size] == 0:
                    continue
                
                # Normalize data
                total_sum = np.sum(data_subset)
                if total_sum > 0:
                    data_subset = (data_subset/total_sum)*subset_counts
                
                row_idx = row - pad_size
                col_idx = col - pad_size
                
                # Process statistics
                total_counts = np.sum(subset_counts)
                if total_counts < stats_threshold:
                    # Simple estimators for low particle counts
                    std = window_size/2
                    n_eff = np.sum(data_subset)/window_size
                    integral_length_scale = window_size
                else:
                    # Full statistical analysis
                    # Calculate standard deviation
                    std = histogram_std(data_subset, None, 1)
                    
                    # Calculate autocorrelation
                    autocorr_rows, autocorr_cols = calculate_autocorrelation(data_subset)
                    autocorr = (autocorr_rows + autocorr_cols) / 2
                    
                    # Calculate integral length scale
                    if autocorr.any():
                        non_zero_idx = np.where(autocorr != 0)[0][0]
                        integral_length_scale = np.sum(autocorr) / autocorr[non_zero_idx]
                    else:
                        integral_length_scale = 0.000001
                    
                    n_eff = np.sum(data_subset) / integral_length_scale
                
                # Calculate bandwidth using Silverman's rule
                h = np.sqrt((silverman_coeff * n_eff**(-silverman_exponent)) * std) * dxy_grid
                
                # Store results
                std_estimate[row_idx, col_idx] = std
                N_eff[row_idx, col_idx] = n_eff
                integral_length_scale_matrix[row_idx, col_idx] = integral_length_scale
                h_matrix_adaptive[row_idx, col_idx] = h
    
    return std_estimate, N_eff, integral_length_scale_matrix, h_matrix_adaptive

@jit(nopython=True, parallel=True)
def compute_adaptive_bandwidths_old(preGRID_active_padded, preGRID_active_counts_padded,
                              window_size, pad_size, stats_threshold,
                              silverman_coeff, silverman_exponent, dxy_grid):
    """Compute adaptive bandwidths for all windows"""
    shape = preGRID_active_padded.shape
    std_estimate = np.zeros((shape[0]-2*pad_size, shape[1]-2*pad_size))
    N_eff = np.zeros_like(std_estimate)
    h_matrix_adaptive = np.zeros_like(std_estimate)
    integral_length_scale_matrix = np.zeros_like(std_estimate)
    
    for row in prange(pad_size, shape[0]-pad_size):
        for col in range(pad_size, shape[1]-pad_size):
            if preGRID_active_counts_padded[row, col] > 0:
                data_subset = preGRID_active_padded[row-pad_size:row+pad_size+1,
                                                  col-pad_size:col+pad_size+1]
                subset_counts = preGRID_active_counts_padded[row-pad_size:row+pad_size+1,
                                                           col-pad_size:col+pad_size+1]
                
                if data_subset[pad_size,pad_size] == 0:
                    continue
                
                data_subset = (data_subset/np.sum(data_subset))*subset_counts
                
                row_idx = row - pad_size
                col_idx = col - pad_size
                
                std, n_eff, ils, h = _process_window_statistics(
                    data_subset, subset_counts, pad_size, window_size,
                    stats_threshold, silverman_coeff, silverman_exponent, dxy_grid
                )
                
                std_estimate[row_idx, col_idx] = std
                N_eff[row_idx, col_idx] = n_eff
                integral_length_scale_matrix[row_idx, col_idx] = ils
                h_matrix_adaptive[row_idx, col_idx] = h
    
    return std_estimate, N_eff, integral_length_scale_matrix, h_matrix_adaptive


def load_variable(args):
    """Helper function to load a single variable"""
    var_name, variable = args
    return var_name, variable[:]

def load_netcdf_optimized(datapath):
    """Optimized loading of NetCDF data"""
    # Open dataset with memory mapping
    ODdata = nc.Dataset(datapath, 'r', mmap=True)
    print('Loading all data into memory from nc file...')
    
    # Initialize dictionary
    particles_all_data = {}
    
    # Load variables with progress bar
    for var_name in tqdm(ODdata.variables, desc="Loading variables"):
        particles_all_data[var_name] = ODdata.variables[var_name][:].copy()
    
    # Add UTM coordinates
    particles_all_data = add_utm(particles_all_data)
    print('Data loaded from nc file.')
    
    return particles_all_data


#################################
########## INITIATION ###########
#################################

#if __name__ == '__main__':

#Just load the grid object to make it faster
#with open('grid_object.pickle', 'rb') as f:
#   GRID = pickle.load(f)

if run_test == True:
    datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_test.nc'#test dataset
    ODdata = nc.Dataset(datapath, 'r', mmap=True)
    particles = load_nc_data(datapath)
    particles = add_utm(particles)
    #adjust the time vector to start on May 20 2018
    minlon = 12.5
    maxlon = 21
    minlat = 68.5
    maxlat = 72
    maxdepth = 15*20

if run_full == True:
    datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_norkyst_unlimited_vdiff.nc'#real dataset
    #datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_norkyst.nc'#real dataset
    ODdata = nc.Dataset(datapath, 'r', mmap=True)
    #number of particles
    n_particles = first_timestep_lon = ODdata.variables['lon'][:, 0].copy()
    #Create an empty particles dictionary with size n_particlesx2 (one for current and one for previous timestep)
    particles = {'lon':np.ma.zeros((len(n_particles),2)),
                'lat':np.ma.zeros((len(n_particles),2)),
                'z':np.ma.zeros((len(n_particles),2)),
                'time':np.ma.zeros(2),}
    #first timestep:
    particles['lon'][:,0] = ODdata.variables['lon'][:, 0].copy()
    particles['lat'][:,0] = ODdata.variables['lat'][:, 0].copy() 
    particles['z'][:,0] = ODdata.variables['z'][:, 0].copy()
    particles['timefull'] = ODdata.variables['time'][:].copy()
    particles['time'][0] = ODdata.variables['time'][0].copy()
    #second timestep:
    particles['lon'][:,1] = ODdata.variables['lon'][:, 1].copy()
    particles['lat'][:,1] = ODdata.variables['lat'][:, 1].copy()
    particles['z'][:,1] = ODdata.variables['z'][:, 1].copy()
    #and current time step
    particles['time'][1] = ODdata.variables['time'][1].copy()
    #add utm
    particles = add_utm(particles)
        
    #unmasked_first_timestep_lon = first_timestep_lon[~first_timestep_lon.mask]
    #set limits for grid manually since we dont know how this will evolv
    #loop over all timesteps to check the limits of the grid or define boundaries manually
    if manual_border == True:
        minlon = 12.5
        maxlon = 21
        minlat = 68.5
        maxlat = 72
        maxdepth = 15*20
    else:
        for i in range(0,ODdata.variables['lon'].shape[1]):
            #get max and min unmasked lat/lon values
            print(i)
            if i == 0:
                minlon = np.min(ODdata.variables['lon'][:,i].compressed())
                maxlon = np.max(ODdata.variables['lon'][:,i].compressed())
                minlat = np.min(ODdata.variables['lat'][:,i].compressed())
                maxlat = np.max(ODdata.variables['lat'][:,i].compressed())
                maxdepth = np.max(np.abs(ODdata.variables['z'][:,i].compressed()))
            else:
                minlon = np.min([np.min(ODdata.variables['lon'][:,i].compressed()),minlon])
                maxlon = np.max([np.max(ODdata.variables['lon'][:,i].compressed()),maxlon])
                minlat = np.min([np.min(ODdata.variables['lat'][:,i].compressed()),minlat])
                maxlat = np.max([np.max(ODdata.variables['lat'][:,i].compressed()),maxlat])
                maxdepth = np.max([np.max(np.abs(ODdata.variables['z'][:,i].compressed())),maxdepth])
    
#get the min/max values for the UTM coordinates using the utm package and the minlon/maxlon/minlat/maxlat values
minUTMxminUTMy = utm.from_latlon(minlat,minlon)
minUTMxmaxUTMy = utm.from_latlon(minlat,maxlon)
maxUTMxminUTMy = utm.from_latlon(maxlat,minlon)
maxUTMxmaxUTMy = utm.from_latlon(maxlat,maxlon)
# Example: Forcing all coordinates to UTM zone 33N
zone_number = 33

minUTMxminUTMy = utm.from_latlon(minlat, minlon, force_zone_number=zone_number, force_zone_letter='W')
minUTMxmaxUTMy = utm.from_latlon(minlat, maxlon, force_zone_number=zone_number, force_zone_letter='W')
maxUTMxminUTMy = utm.from_latlon(maxlat, minlon, force_zone_number=zone_number, force_zone_letter='W')
maxUTMxmaxUTMy = utm.from_latlon(maxlat, maxlon, force_zone_number=zone_number, force_zone_letter='W')

#Create a test data set of binned values in a 10x10 grid containing 50% zeros
histogram_prebinned = np.random.choice([0, 1], size=(10, 10), p=[0.5, 0.5])
#replace all 1s with some random values
histogram_prebinned_uneven_weights = histogram_prebinned.copy()
histogram_prebinned_uneven_weights[histogram_prebinned_uneven_weights == 1] = np.random.randint(1, 10, size=np.sum(histogram_prebinned_uneven_weights == 1))

N = np.shape(histogram_prebinned)[0]*np.shape(histogram_prebinned)[1]
#Calculate the effective number of samples
effective_samples = (np.sum(histogram_prebinned)**2)/np.sum(histogram_prebinned**2)
effective_samples_uneven = (np.sum(histogram_prebinned_uneven_weights)**2)/np.sum(histogram_prebinned_uneven_weights**2)

print('Effective samples:',effective_samples)
print('Effective samples uneven:',effective_samples_uneven)

plt.imshow(histogram_prebinned, cmap='viridis')
plt.colorbar()

###### SET UP GRIDS FOR THE MODEL ######
print('Creating the output grid...')
#MODEELING OUTPUT GRID
if run_test == True:
    GRID,bin_x,bin_y,bin_z,bin_time = create_grid(np.ma.filled(np.array(particles['time']),np.nan),
                                                [np.max([100000-dxy_grid-1,minUTMxminUTMy[0]]),np.min([1000000-dxy_grid-1,maxUTMxmaxUTMy[0]])],
                                                [np.max([dxy_grid+1,minUTMxminUTMy[1]]),np.min([10000000-dxy_grid-1,maxUTMxmaxUTMy[1]])],
                                                maxdepth+25,
                                                savefile_path=False,
                                                resolution=np.array([dxy_grid,dz_grid]))
elif run_full == True:
    GRID,bin_x,bin_y,bin_z,bin_time = create_grid(np.ma.filled(np.array(particles['timefull']),np.nan),
                                                [np.max([100000-dxy_grid-1,minUTMxminUTMy[0]]),np.min([1000000-dxy_grid-1,maxUTMxmaxUTMy[0]])],
                                                [np.max([dxy_grid+1,minUTMxminUTMy[1]]),np.min([10000000-dxy_grid-1,maxUTMxmaxUTMy[1]])],
                                                maxdepth+25,
                                                savefile_path=False,
                                                resolution=np.array([dxy_grid,dz_grid]))
#CREATE ONE ACTIVE HORIZONTAL MODEL FIELD
GRID_active = np.zeros((len(bin_x),len(bin_y)))
#ATMOSPHERIC FLUX GRID
GRID_atm_flux = np.zeros((len(bin_time),len(bin_x),len(bin_y)))
#MOX CONSUMPTION GRID
GRID_mox = np.zeros((len(bin_time),len(bin_x),len(bin_y)))
#Total atmoshperic flux vector (as function of time, in mol/hr)
total_atm_flux = np.zeros(len(bin_time))
#Total MoX consumption vector (as function of time, in mol/hr)
total_mox = np.zeros(len(bin_time))
#particle weightloss history
particles_atm_loss = np.zeros((len(bin_time)))
#particle mox loss history
particles_mox_loss = np.zeros((len(bin_time)))
#Mass that leaves the model domain
particles_mass_out = np.zeros((len(bin_time)))
#Mass lost to killing of particles
particles_mass_died = np.zeros((len(bin_time)))
#local integral length scale
integral_length_scale_windows = GRID
#local std
standard_deviations_windows = GRID

#Create coordinates for plotting
bin_x_mesh,bin_y_mesh = np.meshgrid(bin_x,bin_y)
#And lon lat coordinates
lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh.T,bin_y_mesh.T,zone_number=33,zone_letter='W')

#Create datetime vector from bin_time
if run_test == True: 
    timedatetime = pd.to_datetime(bin_time,unit='s')-pd.to_datetime('2020-01-01')+pd.to_datetime('2018-05-20')
else:
    timedatetime = pd.to_datetime(bin_time,unit='s')

###### GENERATE GAUSSIAN KERNELS ######
print('Generating gaussian kernels...')
#generate gaussian kernels
gaussian_kernels, gaussian_bandwidths_h = generate_gaussian_kernels(20, 1/3, stretch=1)
#Get the bandwidth in real distances (this is easy since the grid is uniform)
gaussian_bandwidths_h = gaussian_bandwidths_h*(bin_x[1]-bin_x[0])
print('done.')
############################
###### FIRST TIMESTEP ######
############################

#Get the last timestep (to get the size of the thing)
bin_time_number = np.digitize(particles['time'][0],bin_time)
bin_time_number = len(particles['time'])-1

#Get the utm coordinates of the particles in the first time step
x = particles['UTM_x'][:,0]
y = particles['UTM_y'][:,0]
z = np.abs(particles['z'][:,0])

#Get the bin numbers for the particles
bin_x_number = np.digitize((x.compressed()),bin_x)
bin_y_number = np.digitize((x.compressed()),bin_y)
bin_z_number = np.digitize((x.compressed()),bin_z)

##################################
### CALCULATE GRID CELL VOLUME ###
##################################

grid_resolution = [dxy_grid,dxy_grid,dz_grid] #in meters
V_grid = grid_resolution[0]*grid_resolution[1]*grid_resolution[2]

###############################################
### LOAD AND/OR FIT WIND AND SST FIELD DATA ###
###############################################

if fit_wind_data == True:
    fit_wind_sst_data(bin_x,bin_y,bin_time) #This functino saves the wind and sst fields in a pickle file
    #LOAD THE PICKLE FILE (no matter what)
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\model_grid\\interpolated_wind_sst_fields_test.pickle', 'rb') as f:
    ws_interp,sst_interp,bin_x_mesh,bin_y_mesh,ocean_time_unix = pickle.load(f)

#########################################################
######### CALCULATE GAS TRANSFER VELOCITY FIELDS ########
#########################################################

#interpolate nans in the wind field and sst field
ws_interp = np.ma.filled(ws_interp,np.nan)
sst_interp = np.ma.filled(sst_interp,np.nan)

GRID_gt_vel = np.zeros((len(bin_time),len(bin_y),len(bin_x)))

#Calculate the gas transfer velocity
if fit_gt_vel == True:
    print('Fitting gas transfer velocity')
    for gtvel in range(0,len(bin_time)):
        print(gtvel)
        GRID_gt_vel[gtvel,:,:] = calc_gt_vel(u10=ws_interp[gtvel,:,:],temperature=sst_interp[gtvel,:,:],gas='methane')

        tmp = GRID_gt_vel[gtvel,:,:]

        valid_mask = ~np.isnan(tmp)
        invalid_mask = np.isnan(tmp)
        valid_coords = np.array(np.nonzero(valid_mask)).T
        invalid_coords = np.array(np.nonzero(invalid_mask)).T
        valid_values = tmp[valid_mask]
        tmp[invalid_mask] = griddata(valid_coords, valid_values, invalid_coords, method='nearest')

        GRID_gt_vel[gtvel,:,:] = tmp

    #save the GRID_gt_vel
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\model_grid\\gt_vel\\GRID_gt_vel.pickle', 'wb') as f:
        pickle.dump(GRID_gt_vel, f)
else:#load the GRID_gt_vel
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\model_grid\\gt_vel\\GRID_gt_vel.pickle', 'rb') as f:
        GRID_gt_vel = pickle.load(f)

################################################################
### ADD DICTIONARY ENTRIES FOR PARTICLE WEIGHT AND BANDWIDTH ###
################################################################

### Weight ###
particles['weight'] = np.ma.zeros(particles['z'].shape)
#add mask
particles['weight'].mask = particles['lon'].mask

### Bandwidth ###
#Define inital bandwidth
initial_bandwidth = initial_bandwidth #meters
#Define the bandwidth aging constant
age_constant = age_constant #meters spread every hour
#Define the bandwidth matrix
particles['bw'] = np.ma.zeros(particles['lon'].shape)
#Add the initial bandwidth to the particles at all timesteps
particles['bw'][:,0] = initial_bandwidth
#Add mask
particles['bw'].mask = particles['lon'].mask

#-----------------------------------------#
#CREATE A MATRIX FOR REMOVING THE DIAGONAL#
#-----------------------------------------#

#For test run only. This is to remove the diagonal and the artifact area
if run_test == True: 
    diag_rm_mat = np.ones(np.shape(GRID[0][0][:,:]))   
    #remove diagonal and artifact area..... 
    diag_thr = 0.4
    #GRID_atm_flux_mm_m2 = np.flip(GRID_atm_flux_mm_m2,axis=1)
    for i in range(0,len(diag_rm_mat)):
        for j in range(0,len(diag_rm_mat[i])):
            if i < int(j*diag_thr):
                diag_rm_mat[i][j] = 0

###################################################
###### ADD DICTIONARY ENTRY FOR PARTICLE AGE ######
###################################################

initial_age = 0
particles['age'] = np.ma.zeros(particles['lon'].shape)
#add mask
particles['age'].mask = particles['lon'].mask
#and for total particles
total_parts = np.zeros(len(bin_time))

####################################################################################
###### ADD VECTOR TO STORE INTEGRAL LENGTH SCALE AND OTHER STUFF OF THE FIELD ######
####################################################################################

integral_length_scale_full = np.zeros([len(bin_time),len(bin_z)])
h_values_full = np.zeros([len(bin_time),len(bin_z)])
h_values_std_full = np.zeros([len(bin_time),len(bin_z)])
#std_full = np.zeros(len(bin_time))
h_list = list()
neff_list = list()
std_list = list()

######################################################################
###### FIND ILLEGAL CELLS IN THE GRID USING THE BATHYMETRY DATA ######
######################################################################
print('Getting bathymetry and illegal cells')
# Define projections and transformer
polar_stereo_proj = Proj(proj="stere", lat_ts=75, lat_0=90, lon_0=-45, datum="WGS84")
wgs84 = Proj(proj="latlong", datum="WGS84")
transformer = Transformer.from_proj(polar_stereo_proj, wgs84)

# Get the bathymetry data
if get_new_bathymetry == True:
    bathymetry_path = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\bathymetry\IBCAO_v4_400m.nc'
    output_path = 'C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\bathymetry\\interpolated_bathymetry.pickle'
    bathymetry_data = process_bathymetry(bathymetry_path, bin_x, bin_y, transformer, output_path)
    interpolated_bathymetry = bathymetry_data[0]
    lat_mesh_map = bathymetry_data[1]
    lon_mesh_map = bathymetry_data[2]
else:
    # Load the interpolated bathymetry data from pickle file
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\bathymetry\\interpolated_bathymetry.pickle', 'rb') as f:
        bathymetry_data = pickle.load(f)
    interpolated_bathymetry = bathymetry_data['bathymetry']
    lat_mesh_map = bathymetry_data['latitude']
    lon_mesh_map = bathymetry_data['longitude']

del bathymetry_data

# Create matrices for illegal cells using the bathymetry data and delta z
illegal_cells = np.zeros([len(bin_x), len(bin_y), len(bin_z)])
bin_z_bath_test = bin_z.copy()
bin_z_bath_test[0] = 1  # Ensure the surface boundary is respected

# Loop through all grid cells and check if they are illegal
for i in range(len(bin_x)):
    for j in range(len(bin_y)):
        for k in range(len(bin_z)):
            if bin_z_bath_test[k] > np.abs(interpolated_bathymetry[j, i]):
                illegal_cells[i, j, k] = 1
# Plotting
if plotting == True:
    # Plot the illegal cell matrices in a 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    for i in range(3):
        for j in range(3):
            axs[i, j].pcolor(lon_mesh_grid, lat_mesh_grid, illegal_cells[:, :, i + j].T, cmap='rocket')
            axs[i, j].set_title(f'Illegal cells at z={bin_z[i + j]}')
    plt.show()
    # Make a contour plot where the depth levels in bin_z are used as contours
    plt.figure(figsize=(10, 10))
    contourf_plot = plt.contourf(lon_mesh_grid, lat_mesh_grid, np.abs(interpolated_bathymetry), cmap='rocket', levels=bin_z, extend='max')
    plt.contour(lon_mesh_grid, lat_mesh_grid, np.abs(interpolated_bathymetry), levels=bin_z, colors='black', linewidths=0.5)
    plt.contour(lon_mesh_grid, lat_mesh_grid, np.abs(interpolated_bathymetry), levels=[0], colors='white', linewidths=1)
    plt.colorbar(contourf_plot, label='Depth (m)', extend='max')
    plt.show()

print('done.')

#################################################################################
########### CREATE MAP FOR PLOTTING OF COASTLINE WHERE BATHYMETRY > 0 ###########
#################################################################################

# Create a map for plotting the coastline where bathymetry > 0
coastline_map = np.zeros_like(interpolated_bathymetry)
coastline_map[interpolated_bathymetry >= 0] = 1

#############################################################################################
################### END INITIAL CONDITIONS ### END INITIAL CONDITIONS #######################
#############################################################################################

#---------------------------------------------------#
#####################################################
#####  MODEL THE CONCENTRATION AT EACH TIMESTEP #####
### AKA THIS IS WHERE THE ACTUAL MODELING HAPPENS ###
#####################################################
#---------------------------------------------------#
print('Starting to loop through all timesteps...')

time_steps_full = len(ODdata.variables['time'])

#age_vector = np.zeros(len(particles['z']), dtype=bool) #This vector is True if the particle has an age.. 

kde_time_vector = np.zeros(time_steps_full-1)
h_estimate_vector = np.zeros(time_steps_full-1)
elapsed_time_timestep = np.zeros(time_steps_full-1)

run_all = True

#test_layers
GRID_top = np.zeros((len(bin_time),len(bin_x),len(bin_y)))
GRID_hs = np.zeros((len(bin_time),len(bin_x),len(bin_y)))
GRID_stds = np.zeros((len(bin_time),len(bin_x),len(bin_y)))
GRID_neff = np.zeros((len(bin_time),len(bin_x),len(bin_y)))

#Particles that left the domain vector
particles_that_left = np.array([])

if run_all == True:

    #Start looping through.
    for kkk in range(1,time_steps_full-1): 
        start_time_full = time.time()
        start_time = time.time()        
        print(f"Time step {kkk}")

        #------------------------------------------------------#
        # LOADING PARTICLES INTO MEMORY (NOT MEMORY OPTIMIZED) #
        #------------------------------------------------------#

        if kkk==1:         
            ### Load all data into memory ###
            if load_from_nc_file == True:
                #import multiprocessing as mp
                #from tqdm import tqdm
                #particles_all_data = load_netcdf_optimized(datapath)
                ODdata = nc.Dataset(datapath, 'r', mmap=True)
                print('Loading all data into memory from nc file...')
                particles_all_data = {}
                for var_name in ODdata.variables:
                    particles_all_data[var_name] = ODdata.variables[var_name][:]
                particles_all_data = add_utm(particles_all_data)
                print('Data loaded from nc file.')

            elif load_from_hdf5 == True:
                import h5py
                target_location = 'C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\OpenDrift\\particles_all_data.h5'
                with h5py.File(target_location, 'r') as h5file:
                    particles_all_data = {var_name: h5file[var_name][:] for var_name in h5file.keys()}
                print(f'Data loaded from {target_location}')
                #add UTM
                particles_all_data = add_utm(particles_all_data)
            
            else:
                print('No data loaded. I guess data is already in!?')
            
        #Create a h5py file of the datafile for faster loading in the future
        if create_new_datafile == True:
            import h5py
            ODdata = nc.Dataset(datapath, 'r', mmap=True) #load nc file
            target_location = 'C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\OpenDrift\\particles_full_vdiff.h5'
            # Open the HDF5 file in write mode
            with h5py.File(target_location, 'w') as h5file:
                print('Loading all data into mamory from h5 file...')
                for var_name in ODdata.variables:
                    print(f'Writing {var_name} to HDF5 file...')
                    # Write each variable to the HDF5 file incrementally
                    h5file.create_dataset(var_name, data=ODdata.variables[var_name][:])

                print('Data saved to particles_all_data.h5')
                # Load the data from the HDF5 file

        #########################################################
        ############### DATA LOADED INTO MEMORY #################
        #########################################################
        
        #-----------------------------#
        # START WITH THE CALCULATIONS #
        #-----------------------------#

        # Replace the 0th index with the 1st index
        particles['lon'][:,0] = particles['lon'][:,1]
        particles['lat'][:,0] = particles['lat'][:,1]
        particles['z'][:,0] = particles['z'][:,1]
        particles['time'][0] = particles['time'][1]
        particles['bw'][:,0] = particles['bw'][:,1]

        # Assign the data from preloaded data
        particles['lon'][:, 1] = particles_all_data['lon'][:, kkk].copy()
        particles['lat'][:, 1] = particles_all_data['lat'][:, kkk].copy()
        particles['z'][:, 1] = particles_all_data['z'][:, kkk].copy()
        particles['time'][1] = particles_all_data['time'][kkk].copy()
        particles['UTM_x'][:,1] = particles_all_data['UTM_x'][:,kkk].copy()
        particles['UTM_y'][:,1] = particles_all_data['UTM_y'][:,kkk].copy()
        particles['UTM_x'][:,0] = particles_all_data['UTM_x'][:,kkk-1].copy()
        particles['UTM_y'][:,0] = particles_all_data['UTM_y'][:,kkk-1].copy()
        particles = add_utm(particles)

        # Update masks
        particles['weight'][:, 1].mask = particles['z'][:, 1].mask
        particles['bw'][:, 1].mask = particles['z'][:, 1].mask

        # Add weights if it's the first timestep
        if kkk == 1:
            particles['weight'][:,1][np.where(particles['weight'][:,1].mask == False)] = weights_full_sim
            particles['weight'][:,0] = particles['weight'][:,1]
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"Data loading: {elapsed_time:.6f} seconds")

        #--------------------------------------------------------#
        # Count mass that left due to particles dying of old age #
        #--------------------------------------------------------#

        # Get deactivated indices safely
        deactivated_indices = np.where((particles['z'][:,1].mask == True) & 
                                    (particles['z'][:,0].mask == False))[0]

        #make sure deactivated_indices is int
        deactivated_indices = deactivated_indices.astype(np.int64)

        # Only process if we have deactivated particles
        if deactivated_indices.size > 0:
            particles_mass_died[kkk] = np.sum(particles['weight'][deactivated_indices,0])  # Use previous timestep weights
        else:
            particles_mass_died[kkk] = 0

        #------------------------------------------#
        # DEACTIVATE PARTICLES OUTSIDE OF THE GRID #
        #------------------------------------------#
        #Boundaries.
        max_x = np.max(bin_x)
        min_x = np.min(bin_x)
        max_y = np.max(bin_y)
        min_y = np.min(bin_y)
        #create a vector for the indices that are outside the grid
        outside_grid = np.where((particles['UTM_x'][:,1] < min_x) | 
                            (particles['UTM_x'][:,1] > max_x) | 
                            (particles['UTM_y'][:,1] < min_y) | 
                            (particles['UTM_y'][:,1] > max_y))[0]

        #Mask all particles that are outside the grid
        particles['z'][outside_grid,1].mask = True
        particles_mass_out[kkk] = np.sum(particles['weight'][outside_grid,1])
        particles['weight'][outside_grid,1].mask = True
        particles['bw'][outside_grid,1].mask = True
        particles['UTM_x'][outside_grid,1].mask = True
        particles['UTM_y'][outside_grid,1].mask = True
        particles['lon'][outside_grid,1].mask = True
        particles['lat'][outside_grid,1].mask = True

        # Set up vector if it doesnt exist. 
        if 'particles_that_left' not in locals():
            particles_that_left = np.array([], dtype=int)

        # Then mask the particles
        if particles_that_left.size > 0:
            for field in ['z', 'weight', 'bw', 'UTM_x', 'UTM_y', 'lon', 'lat']:
                particles[field][particles_that_left, 1].mask = True

        # Add new outside_grid particles to particles_that_left
        if outside_grid.size > 0:
            # Ensure outside_grid is integer type
            outside_grid = outside_grid.astype(np.int64)
            # Concatenate and maintain integer type
            particles_that_left = np.unique(np.concatenate((particles_that_left, outside_grid))).astype(np.int64)
        #--------------------------------------#
        #MODIFY PARTICLE WEIGHTS AND BANDWIDTHS#
        #--------------------------------------#

        #Unmask particles that were masked in the previous timestep
        #particles['z'][:,j].mask = particles['z'][:,j-1].mask
        # Get the indices where particles['age'][:,j] is not masked
        #do some binning on those
        bin_z_number = np.digitize(
        np.abs(particles['z'][:,1][np.where(
            particles['z'][:,1].mask == False)]),bin_z)

        # Get the indices where particles['age'][:,j] is not masked and equal to 0
        activated_indices = np.where((particles['z'][:,1].mask == False) & (particles['z'][:,0].mask == True))[0]
        already_active = np.where((particles['z'][:,1].mask == False) & (particles['z'][:,0].mask == False))[0]

        #activated_indices = unmasked_indices[0][
        #    particles['age'][:,1][unmasked_indices] == 0]
        #already active indices
        #already_active = unmasked_indices[0][
        #    particles['age'][:,1][unmasked_indices] != 0]
        
        #print the number of true 
        ### ADD INITIAL WEIGHT IF THE PARTICLE HAS JUST BEEN ACTIVATED ###
        if run_test == True:
            if activated_indices.any(): #If there are new particles added at this timestep
                # Use these indices to modify the original particles['weight'] array
                particles['weight'][activated_indices,1] = (np.round(
                    np.exp((np.abs(particles['z'][:,1][activated_indices]
                                    )+10)/44)))*0.037586 #moles per particle
                #Make the mask false for all subsquent timesteps
                particles['weight'][activated_indices,1].mask = False
                #do this for all the maske
                #same for bandwidth
                particles['bw'][activated_indices,1] = initial_bandwidth
        elif run_full == True:
            particles['weight'][activated_indices,1].mask = False
            particles['weight'][activated_indices,1] = weights_full_sim
            particles['bw'][activated_indices,1].mask = False
            particles['bw'][activated_indices,1] = initial_bandwidth

        ### MODIFY ALREADY ACTIVE ###
        #add the weight of the particle to the current timestep 
        if already_active.any():
            ##### MOX LOSS #####
            particles['weight'][already_active,1] = particles[
                'weight'][already_active,0]-(particles['weight'][
                    already_active,0]*R_ox*3600) #mol/hr
            particles_mox_loss[kkk] = np.nansum(particles['weight'][already_active,0]*R_ox*3600)

            ##### ATM LOSS #####
            #This calculates the loss to the atmosphere for the methane released in the previous timestep... (mostly uses j-1 idx)
            #Find all particles located in the surface layer and create an index vector (to avoid double indexing numpy problem)
            already_active_surface = already_active[np.where(np.abs(particles['z'][already_active,0])<bin_z[1])[0]] #all particles with surface z
            #find the gt_vel for the surface_layer_idxs
            gt_idys = np.digitize(np.abs(particles['UTM_y'][already_active_surface,0]),bin_y)
            gt_idxs = np.digitize(np.abs(particles['UTM_x'][already_active_surface,0]),bin_x)
            #make sure all gt_idys and gt_idxs are within the grid
            gt_idys[gt_idys >= len(bin_y)] = len(bin_y)-1
            gt_idxs[gt_idxs >= len(bin_x)] = len(bin_x)-1
            #Distribute the atmospheric loss on these particles depending on their weight
            #replace any nans in GRID_gt_cel[j-1][gt_idys,gt_idxs] with the nearest non nan value in the grid
            #GRID_gt_vel[j-1][gt_idys,gt_idxs] = np.nan_to_num(GRID_gt_vel[j-1][gt_idys,gt_idxs])
            #Each particle have contributed with a certain amount gt_vel*weight in the PREVIOUS timestep. This is loss to atmosphere.
            particleweighing = (particles['weight'][already_active_surface,0]*GRID_gt_vel[kkk-1][gt_idys,gt_idxs])/np.nansum(
                particles['weight'][already_active_surface,0]*GRID_gt_vel[kkk-1][gt_idys,gt_idxs])
            #particles['weight'][already_active,j][surface_layer_idx] = particles['weight'][already_active,j][surface_layer_idx] - (gt_vel_loss*particles['weight'][already_active,j-1][surface_layer_idx]*total_atm_flux[j-1])/np.nansum((gt_vel_loss*particles['weight'][already_active,j-1][surface_layer_idx])) #mol/hr
            particles['weight'][already_active_surface,1] = particles['weight'][already_active_surface,0] - (
                particleweighing*total_atm_flux[kkk-1]) #mol/hr
            particles_atm_loss[kkk] = np.nansum((particleweighing*total_atm_flux[kkk-1])/np.nansum(particleweighing))
            #weigh this with the gt_vel
            #USING THE TOTAL ATM FLUX HERE.. 
            #remove particles with weight less than 0
            #if particles['weight'][already_active,j][particles['weight'][already_active,j]<0].any():
            #    break
            particles['weight'][already_active,1][particles['weight'][already_active,1]<0] = 0
            #add the bandwidth of the particle to the current timestep
            particles['bw'][already_active,1] = particles['bw'][already_active,1] + age_constant
            #limit the bandwidth to a maximum value
            particles['bw'][already_active,1][particles['bw'][already_active,1]>max_ker_bw] = max_ker_bw

        #finished with modifying weights, replace weights[0] with weights[1] for next step
        particles['weight'][:,0] = particles['weight'][:,1]


        #--------------------------------------------------#
        #FIGURE OUT WHERE PARTICLES ARE LOCATED IN THE GRID#
        #--------------------------------------------------#
        
        #.... And create a sorted matrix for all the active particles according to
        #which depth layer they are currently located in. 

        #Get sort indices
        sort_indices = np.argsort(bin_z_number)
        #sort
        bin_z_number = bin_z_number[sort_indices]
        #get indices where bin_z_number changes
        change_indices = np.where(np.diff(bin_z_number) != 0)[0]
        #Trigger if you want to loop through all depth layers
        #if use_all_depth_layers == True: #is this depracated?? I think so. 
        #    change_indices = np.array([0,len(bin_z_number)])
        
        #Define the [location_x,location_y,location_z,weight,bw] for the particle. This is the active particle matrix
        parts_active = [particles['UTM_x'][:,1].compressed()[sort_indices],
                        particles['UTM_y'][:,1].compressed()[sort_indices],
                        particles['z'][:,1].compressed()[sort_indices],
                        bin_z_number,
                        particles['weight'][:,1].compressed()[sort_indices],
                        particles['bw'][:,1].compressed()[sort_indices]]
        
        #keep track of number of particles
        total_parts[kkk] = len(parts_active[0])
        
        #-----------------------------------#
        #INITIATE FOR LOOP OVER DEPTH LAYERS#
        #-----------------------------------#

        #add one right hand side limit to change_indices
        change_indices = np.append(change_indices,len(bin_z_number))
        #add a zero at the beginning (to include depth layer 0)
        change_indices = np.insert(change_indices, 0, 0)

        ###########################################
        ###########################################
        ###########################################

        for i in range(0,len(change_indices)-1): #This essentially loops over all particles (does it???)
            
            #-----------------------------------------------------------#
            #DEFINE ACTIVE GRID AND ACTIVE PARTICLES IN THIS DEPTH LAYER#
            #-----------------------------------------------------------#

            #Define GRID_active by creating a zero matrix of same size as the grid
            GRID_active = np.zeros((len(bin_x),len(bin_y)))

            #Define active particle matrix in depth layer i
            parts_active_z = [parts_active[0][change_indices[i]:change_indices[i+1]+1],
                            parts_active[1][change_indices[i]:change_indices[i+1]+1],
                            parts_active[2][change_indices[i]:change_indices[i+1]+1],
                            parts_active[3][change_indices[i]:change_indices[i+1]+1],
                            parts_active[4][change_indices[i]:change_indices[i+1]+1],
                            parts_active[5][change_indices[i]:change_indices[i+1]+1]]

            #-----------------------------------------------------#
            #CALCULATE THE CONCENTRATION FIELD IN THE ACTIVE LAYER#
            #-----------------------------------------------------#


            #Set any particle that has left the model domain to have zero weight and location
            #at the model boundary

            #Remove particles that are outside the model domain (they are not allowed to return later if the current will bring them back)
            # Assuming parts_active_z is a list of arrays with 'x' and 'y' coordinates

            # Create masks for x and y coordinates
            mask_x = (parts_active_z[0] >= bin_x[0]) & (parts_active_z[0] <= bin_x[-1])
            mask_y = (parts_active_z[1] >= bin_y[0]) & (parts_active_z[1] <= bin_y[-1])
            #find all false values in the mask
            mask_x_false = np.where(mask_x == False)
            mask_y_false = np.where(mask_y == False)

            # Combine masks to filter out values outside the grid
            mask = mask_x & mask_y

            # Apply the mask to filter the arrays
            for jj in range(len(parts_active_z)):
                parts_active_z[jj] = parts_active_z[jj][mask]

            #-------------------------------------#
            #CALCULATE THE KERNEL DENSITY ESTIMATE#
            #-------------------------------------#

            if kde_all == True or i==0: #dont bother with the kde if there are no particles in the depth layer
                #print('Doing kde for depth layer',i)
                ### THIS IS THE OLD WAY OF DOING IT ###            
                #ols_GRID_active = kernel_matrix_2d_NOFLAT(parts_active_z[0],
                                            #parts_active_z[1],
                                            #bin_x,
                                            #bin_y,
                                            #parts_active_z[5],
                                            #parts_active_z[4])
                ########################################
                ###################
                ### preGRIDding ###
                ###################
            
                
                #Stop and go out of the if if there are no particles in the depth layer
                if len(parts_active_z[0]) == 0:
                    continue

                #Time the kde step
                start_time = time.time()
                #pre-kernel density estimate using the histogram estimator
                preGRID_active,preGRID_active_counts,preGRID_active_bw = histogram_estimator(parts_active_z[0],
                                                    parts_active_z[1],
                                                    bin_x,
                                                    bin_y,
                                                    parts_active_z[5],
                                                    parts_active_z[4])
                
                ###########################
                ### Using no KDE at all ###
                ###########################

                if h_adaptive == 'No_KDE':
                    GRID_active = preGRID_active
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                
                
                #######################################
                ### Using time dependent bandwidths ###
                #######################################

                if h_adaptive == 'Time_dep':
                    GRID_active = grid_proj_kde(bin_x,
                                                bin_y,
                                                preGRID_active,
                                                gaussian_kernels,
                                                gaussian_bandwidths_h,
                                                preGRID_active_bw,
                                                illegal_cells=illegal_cells[:,:,i],)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"KDE took {elapsed_time:.6f} seconds")
                    #store the time it took to calculate the kde 
                    kde_time_vector[kkk] = elapsed_time
   
                ##############################################
                ### Using local Silverman AKA Adaptive KDE ###
                ##############################################
                
                if h_adaptive == 'Local_Silverman' and preGRID_active.any():
                    start_time = time.time()
                    
                    # Compute integral length scale
                    autocorr_rows, autocorr_cols = calculate_autocorrelation(preGRID_active_counts)
                    autocorr = (autocorr_rows + autocorr_cols) / 2
                    
                    if autocorr.any() > 0:
                        integral_length_scale_full[kkk, i] = (np.sum(autocorr) / autocorr[np.argwhere(autocorr != 0)[0]])
                        window_size = int(integral_length_scale_full[kkk, i])
                    else:
                        integral_length_scale_full[kkk,i] = 0
                        window_size = 7
                    
                    window_size = np.clip(window_size, 7, 15)
                    if window_size % 2 == 0:
                        window_size += 1
                    
                    pad_size = window_size//2

                    
                    # Pad arrays
                    preGRID_active_padded = np.pad(preGRID_active, pad_size, mode='reflect')
                    preGRID_active_counts_padded = np.pad(preGRID_active_counts, pad_size, mode='reflect')
                    
                    # Compute statistics and bandwidths
                    std_estimate, N_eff, integral_length_scale_matrix, h_matrix_adaptive = compute_adaptive_bandwidths(
                        preGRID_active_padded, preGRID_active_counts_padded,
                        window_size, pad_size, (window_size**2)/2,
                        silverman_coeff, silverman_exponent, dxy_grid
                    )

                    # Get summary statistics if the matrix is not empty
                    if h_matrix_adaptive.any():
                        h_values_full[kkk,i] = np.mean(h_matrix_adaptive[h_matrix_adaptive>0])
                        h_values_std_full[kkk,i] = np.std(h_matrix_adaptive[h_matrix_adaptive>0])
                    #plt.hist(h_matrix_adaptive[h_matrix_adaptive>0].flatten(),bins=50)
                    #plt.show()
                    #plt.close()
                    #square it to get h (silverman in 2 dimensions give the square root)
                    #h_matrix_adaptive = h_matrix_adaptive**2

                    end_time = time.time()
                    time_to_estimate_h = end_time-start_time
                    h_estimate_vector[kkk] = time_to_estimate_h

                    #Do the KDE using the grid_proj_kde function
                    GRID_active = grid_proj_kde(bin_x,
                                                bin_y,
                                                preGRID_active,
                                                gaussian_kernels,
                                                gaussian_bandwidths_h,
                                                h_matrix_adaptive*0.5, #because it's symmetric
                                                illegal_cells = illegal_cells[:,:,i])

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    kde_time_vector[kkk] = elapsed_time
                    
                    #make a plot if kkk is modulus 50
                    if kkk % 50 == 0:
                        plt.figure()
                        plt.subplot(1,2,1)
                        plt.imshow(h_matrix_adaptive)
                        plt.subplot(1,2,2)
                        #and a histogram of hs
                        plt.hist(h_matrix_adaptive[h_matrix_adaptive>0])
                        plt.show()

                        print(f"Integral_length_scale = {integral_length_scale_full[kkk,:]*dxy_grid} meter")
                        print(f"Depth layer {i}, time step {kkk}")
                        print(f"KDE {elapsed_time:.6f} seconds")

                        #Estimate time left
                        #calculate avareage time step of the previous 10 timesteps
                        print(f"Estimated time left is at least {((np.mean(kde_time_vector[kkk-10:kkk])*time_steps_full)/3600):.2f} hours")


                if run_test == True:
                    GRID_active = diag_rm_mat*(GRID_active/V_grid) #Dividing by V_grid to get concentration in mol/m^3
                elif run_full == True:
                    GRID_active = GRID_active/V_grid

                print("\nMemory state:")
                print(f"Is GRID[{kkk}][{i}] None? {GRID[kkk][i] is None}")
                print(f"GRID[{kkk}][{i}] reference count: {sys.getrefcount(GRID[kkk][i]) if GRID[kkk][i] is not None else 'N/A'}")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                print("\nDetailed reference tracking:")
                
                # Create sparse matrix
                sparse_matrix = csr_matrix(GRID_active)
                print(f"1. New sparse matrix id: {id(sparse_matrix)}")
                print(f"   Reference count: {sys.getrefcount(sparse_matrix)}")
                
                # Assign to GRID
                GRID[kkk][i] = sparse_matrix
                print(f"\n2. GRID[{kkk}][{i}] id: {id(GRID[kkk][i])}")
                print(f"   Reference count: {sys.getrefcount(GRID[kkk][i])}")
                
                # Delete original reference
                del sparse_matrix
                print(f"\n3. Final reference count: {sys.getrefcount(GRID[kkk][i])}")

                GRID_top[kkk,:,:] = GRID_active
                integral_length_scale_windows[kkk][i] = csr_matrix(integral_length_scale_matrix)
                standard_deviations_windows[kkk][i] = csr_matrix(std_estimate)
                GRID_mox[kkk,:,:] = GRID_active*(R_ox*3600*V_grid) #need this to adjust the weights for next loop. MOx in each layer
                total_mox[kkk] = np.nansum(GRID_mox[kkk,:,:])

                #-------------------------------#
                #CALCULATE ATMOSPHERIC FLUX/LOSS#
                #-------------------------------#
                #*dxy_grid**2
                if i == 0:
                    #GRID_atm_flux[kkk,:,:] = np.multiply(GRID_gt_vel[kkk,:,:].T,
                    #    (((GRID_active+background_ocean_conc)-atmospheric_conc))
                    #    )*dxy_grid*0.01 
                    # If we assume equilibrium concentration this simplifies to
                    GRID_atm_flux[kkk,:,:] = np.multiply(GRID_gt_vel[kkk,:,:].T,GRID_active)*dxy_grid*dxy_grid*0.01
                    #    )
                    # 
                    # #This is in mol/hr for each gridcell. The gt_vel is cm/hr, multiply with 0.01 to get m/hr
                    #and GRID_active is in concentration
                    #GRID_active = (GRID_active*V_grid - GRID_atm_flux[kkk,:,:])/V_grid #We do this through weighing of particles, but need to account for loss on this ts as well
                    total_atm_flux[kkk] = np.nansum(GRID_atm_flux[kkk,:,:])#....but not in the atmospheric flux.. 

                    #fill the test layers
                    GRID_top[kkk,:,:] = GRID_active
                    GRID_hs[kkk,:,:] = h_matrix_adaptive
                    GRID_stds[kkk,:,:] = std_estimate
                    GRID_neff[kkk,:,:] = N_eff

        end_time_full = time.time()
        elapsed_time_full = end_time_full - start_time_full
        elapsed_time_timestep[kkk] = elapsed_time_full
        print(f"Total time= {elapsed_time_full:.5f} seconds.")

        #Plot and maintain a plot of elapsed time at every 25th timestep
        if kkk % 50 == 0:
            fig, ax1 = plt.subplots()

            # Plot elapsed time on the left y-axis
            ax1.plot(elapsed_time_timestep[:kkk], linewidth=2, color=color_1)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Elapsed time [s]', color=color_1)
            ax1.tick_params(axis='y', labelcolor=color_1)
            # Create a second y-axis to plot the number of particles
            ax2 = ax1.twinx()
            ax2.plot(total_parts[:kkk], linewidth=2, color=color_2)
            ax2.set_ylabel('Number of particles', color=color_2)
            ax2.tick_params(axis='y', labelcolor=color_2)
            ax1.set_xlim([0, len(elapsed_time_timestep)])
            ax1.set_ylim([np.min(elapsed_time_timestep[10:kkk]),np.max(elapsed_time_timestep[10:])])

            plt.show()

        
end_time_whole_script = time.time()

total_computation_time = end_time_whole_script-start_time_whole_script

print(f"Full calculation time was: {total_computation_time}")

if plotting == True:
    #create a gif from the images
    imageio.mimsave(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\results\concentration\create_gif\concentration.gif', images_conc, duration=0.5)

#----------------------#
#PICKLE FILES FOR LATER#
#----------------------#

#Save the GRID, GRID_atm_flux and GRID_mox, ETC to pickle files
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID.pickle', 'wb') as f:
    pickle.dump(GRID, f)
    #create a sparse matrix first
#load the GRID file
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_mox.pickle', 'rb') as f:
    GRID_mox = pickle.load(f)

#GRID_atm_sparse = csr_matrix(GRID_atm_flux)    
GRID_atm_sparse = GRID_atm_flux
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_atm_flux.pickle', 'wb') as f:
    pickle.dump(GRID_atm_sparse, f)
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_atm_flux.pickle', 'rb') as f:
    GRID_atm = pickle.load(f)
#GRID_mox_sparse = csr_matrix(GRID_mox)
GRID_mox_sparse = GRID_mox
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_mox.pickle', 'wb') as f:
    pickle.dump(GRID_mox_sparse, f)
#and wind, sst, and gt_vel fields
if fit_wind_data == True:
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\ws_interp.pickle', 'wb') as f:
        pickle.dump(ws_interp, f)
    with open('C:\\Users\\kdo000\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\sst_interp.pickle', 'wb') as f:
        pickle.dump(sst_interp, f)
    with open('C:\\Users\\kdo000\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_gt_vel.pickle', 'wb') as f:
        pickle.dump(GRID_gt_vel, f)
#save a short textfile with the settings
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\settings.txt', 'w') as f:
    f.write('Settings for the test run\n')
    f.write('--------------------------------\n')
    f.write('Number of particles: '+str(...)+'\n')
    f.write('Number of timesteps: '+str(744)+'\n')
    f.write('Grid horizontal resolution: '+str(dxy_grid)+'\n')
    f.write('Grid vertical resolution: '+str(dz_grid)+'\n')
    f.write('Grid cell volume: '+str(V_grid)+'\n')
    f.write('Initial bandwidth: '+str(initial_bandwidth)+'\n')
    f.write('Age constant: '+str(age_constant)+'\n')
    f.write('Max kernel bandwidth: '+str(max_ker_bw)+'\n')
    f.write('Run test: '+str(run_test)+'\n')
    f.write('Run full: '+str(run_full)+'\n')
    f.write('Use all depth layers: '+str(use_all_depth_layers)+'\n')
    f.write('Atmospheric background concentration: '+str(atmospheric_conc)+'\n')
    f.write('Background ocean concentration: '+str(background_ocean_conc)+'\n')
    f.write('--------------------------------\n')

######################################################################################################
#----------------------------------------------------------------------------------------------------#
###PLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTING###
#----------------------------------------------------------------------------------------------------#
######################################################################################################

#Remove the first 

#Get grid on lon/lat and limits for the figures. 
lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh,bin_y_mesh,zone_number=33,zone_letter='W')

min_lon = np.min(lon_mesh)
max_lon = np.max(lon_mesh)
min_lat = np.min(lat_mesh)
max_lat = np.max(lat_mesh)

#-------------------------------#
#PLOT THE ATMOSPHERIC FLUX FIELD#
#-------------------------------#

#load the GRID_atm_flux
#with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_atm_flux.pickle', 'rb') as f:
#    GRID_atm_flux = pickle.load(f)

#################################################################################
############ PLOTTING TIMESERIES OF DIFFUSIVE ATMOSPHERIC FLUX FIELD ############
#################################################################################

#OR ANY OTHER FIELD, REALLY, JUST CHANGE THE GRID VARIABLE...

#Calculate atmospheric flux field per square meter per hour
GRID_atm_flux_m2 = (GRID_atm_flux/(dxy_grid**2))#
GRID_gt_vel = GRID_gt_vel #Here, just in cm/hr for convention. 


GRID_generic = GRID_atm_flux_m2
images_atm_rel = []
time_steps = len(bin_time)
levels_atm = np.linspace(np.nanmin(np.nanmin(GRID_generic)),np.nanmax(np.nanmax(GRID_generic)),100)
#levels_atm = levels_atm[1:-1]
levels_atm = levels_atm[:-50]*0.25

#datetimevector for the progress bar
times = pd.to_datetime(bin_time,unit='s')#-pd.to_datetime('2020-01-01')+pd.to_datetime('2018-05-20')

twentiethofmay = 720
time_steps = 1495

do = False
if do == True:
    for i in range(twentiethofmay,time_steps,1):
        fig = plot_2d_data_map_loop(data=GRID_generic[800, :, :].T,
                                    lon=lon_mesh,
                                     lat=lat_mesh,
                                    projection=projection,
                                    levels=levels_atm,
                                    timepassed=[i-twentiethofmay, time_steps-twentiethofmay],
                                    colormap=colormap,
                                    title='Atmospheric flux [mol m$^{-2}$ hr$^{-1}$]' + str(times[i])[5:-3],
                                    unit='mol m$^{-2}$ hr$^{-1}$',
                                    savefile_path='C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run_25m\\make_gif\\atm_flux' + str(i) + '.png',
                                    adj_lon = [1,-1],
                                    adj_lat = [0,-0.5],
                                    show=False,
                                    dpi=90,
                                    figuresize = [12,10],
                                    log_scale = True,
                                    starttimestring = '20 May 2018',
                                    endtimestring = '21 June 2018',
                                    maxnumticks = 10,
                                    plot_progress_bar = True,
                                    #plot_model_domain = [min_lon,max_lon,min_lat,max_lat,0.5,[0.4,0.4,0.4]],
                                    contoursettings = [2,'0.8',0.1])
        images_atm_rel.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run_25m\\make_gif\\atm_flux' + str(i) + '.png'))
        plt.close(fig)  # Close the figure to avoid displaying it

    #create gif
    imageio.mimsave('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run_25m\\make_gif\\atm_flux.gif', images_atm_rel, duration=0.5)

#Do the same proceedure for the gt_vel field



#############################################################
############ PLOTTING TIMESERIES OF GT_VEL FIELD ############
#############################################################

#load the GRID_gt_vel
#with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_gt_vel.pickle', 'rb') as f:
#    GRID_gt_vel = pickle.load(f)

plot_atm=False
if plot_atm == True:
    GRID_generic = GRID_gt_vel
    images_atm_rel = []
    #datetimevector for the progress bar
    times = pd.to_datetime(bin_time,unit='s')#-pd.to_datetime('2020-01-01')+pd.to_datetime('2018-05-20')

    images_gt_vel = []
    time_steps = len(bin_time)
    levels_gt = np.linspace(np.nanmin(np.nanmin(GRID_gt_vel)),np.nanmax(np.nanmax(GRID_gt_vel)),20)

    for i in range(twentiethofmay,time_steps,2):
        fig = plot_2d_data_map_loop(data=GRID_gt_vel[i, :, :],
                                    lon=lon_vec,
                                    lat=lat_vec,
                                    projection=projection,
                                    levels=levels_gt,
                                    timepassed=[i-twentiethofmay, time_steps-twentiethofmay],
                                    colormap=colormap,
                                    title='Gas transfer velocity [cm hr$^{-1}$]' + str(times[i]),
                                    unit='cm hr$^{-1}$',
                                    savefile_path=r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\results\atmosphere\gt_vel_gif\gt_vel' + str(i) + '.png',
                                    adj_lon = [1,-1],
                                    adj_lat = [0,-0.7],
                                    show=False,
                                    dpi=90,
                                    figuresize = [12,10],
                                    log_scale = False,
                                    starttimestring = '20 May 2018',
                                    endtimestring = '20 June 2018')
        images_gt_vel.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\gt_vel_gif\\gt_vel' + str(i) + '.png'))
        plt.close(fig)  # Close the figure to avoid displaying it

    #create gif
    imageio.mimsave('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\gt_vel_gif\\gt_vel.gif', images_gt_vel, duration=0.5)

###########################################################################
############ PLOTTING 2D FIELD OF ACCUMULATED ATMOSPHERIC FLUX ############
###########################################################################

GRID_atm_flux_sum = np.nansum(GRID_atm_flux[twentiethofmay:time_steps,:,:],axis=0) ##Calculate the sum of all timesteps in GRID_atm_flux in moles
total_sum = np.nansum(np.nansum(GRID_atm_flux_sum))#total sum
percent_of_release = np.round((total_sum/total_seabed_release)*100,4) #why multiply with 100??? Because it's percantage dumb-ass
GRID_atm_flux_sum = GRID_atm_flux_sum/(dxy_grid**2)#/1000000 #convert to mol. THIS IS ALREADY IN MOLAR. But divide to get per square meter
levels = np.linspace(np.nanmin(np.nanmin(GRID_atm_flux_sum)),np.nanmax(np.nanmax(GRID_atm_flux_sum)),100)
levels = levels[:]
#flip the lon_vector right/left


plot_2d_data_map_loop(data=GRID_atm_flux_sum.T,
                    lon=lon_mesh,
                    lat=lat_mesh,
                    projection=projection,
                    levels=levels,
                    timepassed=[1, time_steps],
                    colormap=colormap,
                    title='Total released methane = '+str(np.round(total_sum,2))+' mol, $\sim'+str(percent_of_release)+'\%$',
                    unit='mol m$^{-2}$',
                    savefile_path='C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run_25m\\atm_flux_sum.png',
                    show=True,
                    adj_lon = [0,0],
                    adj_lat = [0,0],
                    dpi=60,
                    figuresize = [12,10],
                    log_scale = True,
                    plot_progress_bar = False,
                    maxnumticks = 9,
                    plot_model_domain = False,#[min_lon,max_lon,min_lat,max_lat,0.5,[0.4,0.4,0.4]],
                    contoursettings = [4,'0.8',0.1])

#######################################################################
############ PLOTTING TIMESERIES OF TOTAL ATMOSPHERIC FLUX ############
#######################################################################

#get corresponding time vector
times_totatm =  pd.to_datetime(bin_time,unit='s')


# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Total atmospheric flux
ax1.plot(times_totatm[twentiethofmay:time_steps], total_atm_flux[twentiethofmay:time_steps], color='red')
ax1.set_title('Atmospheric Flux')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('mol hr{^-1}')
ax1.grid(True, alpha=0.3)

# Plot 2: Total MOx
ax2.plot(times_totatm[twentiethofmay:time_steps], total_mox[twentiethofmay:time_steps], color='blue')
ax2.set_title('Microbial Oxidation')
ax2.set_xlabel('Timestep')
ax2.set_ylabel('mol hr{^-1}')
ax2.grid(True, alpha=0.3)

# Plot 3: Total particles
ax3.plot(times_totatm[twentiethofmay:time_steps], particles_mass_out[twentiethofmay:time_steps], color='green')
ax3.set_title('mol hr{^-1}')
ax3.set_xlabel('Timestep')
ax3.set_ylabel('Number of Particles')
ax3.grid(True, alpha=0.3)

# Remove the fourth subplot
ax4.plot(times_totatm[twentiethofmay:time_steps], particles_mass_died[twentiethofmay:time_steps], color='green')
ax4.set_title('mol hr{^-1}')
ax4.set_xlabel('Timestep')
ax4.set_ylabel('Number of Particles')
ax4.grid(True, alpha=0.3)

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(times_totatm,total_atm_flux,label='Total atmospheric flux',color='blue')
ax.set_ylabel('Total atmospheric flux [mol/hr]',fontdict={'fontsize':16})
ax.set_title('Total atmospheric flux',fontdict={'fontsize':16})
#set fontsize for the xticks
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',fontsize=14)
#add yy axis showing the number of active particles
ax2 = ax.twinx()
ax2.plot(times_totatm,total_mox,color='0.4',label='Number of active particles')
ax2.set_ylabel('Number of active particles',fontdict={'fontsize':16})
ax2.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',fontsize=14)
#save figure
plt.savefig('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run\\total_atm_flux.png')

#find the dominant periods in the total_atm_flux dataset
#use the periodogram
from scipy.signal import periodogram
f, Pxx = periodogram(total_atm_flux,1/3600,window='hann',nfft=1024)
plt.plot(f,Pxx)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power spectral density [mol$^2$ hr]')
plt.xlim([0,0.00009])
#plot a vertical line at the semidiurnal and diurnal frequency
f_semi = 1/(12.42*3600)
f_semi_idx = np.where(np.abs(f-f_semi) == np.min(np.abs(f-f_semi)))[0]
Pxx_semi = Pxx[f_semi_idx]
plt.plot(f_semi,Pxx_semi,'ro')
f_diurnal = 1/(24*3600)
f_diurnal_idx = np.where(np.abs(f-f_diurnal) == np.min(np.abs(f-f_diurnal)))[0]
Pxx_diurnal = Pxx[f_diurnal_idx]
plt.plot(f_diurnal,Pxx_diurnal,'bo')
#plot grid
plt.grid()

#find tidal frequency in the periodogram
f_tidal = 1/(12.42*3600)
f_tidal_idx = np.where(np.abs(f-f_tidal) == np.min(np.abs(f-f_tidal)))[0]
Pxx_tidal = Pxx[f_tidal_idx]
plt.plot(f_tidal,Pxx_tidal,'ro')

#Check what's going on in the top layer, i.e. we need to loop through 
#GRID and sum all the top layers

time_steps = len(bin_time)

plot_all = False
if plot_all == True:

    images_field_test = []
    for n in range(0, len(GRID), 10):
        print(n)
        #GRID_top_sum = np.sum((GRID[n][:].toarray()))
        GRID_top_sum = GRID_mox[n,:,:]
        levels = np.linspace(0, np.max(maxmed)+20**-5, 20)
        levels = levels[:-10]
        #plot an imshow in the figure
        #plt.imshow(GRID[n][0].toarray())
        #skip timestep if GRID_top_sum only has zeros.. 
        #if np.sum(GRID_top_sum) == 0:
        #    continue
        fig = plot_2d_data_map_loop(data = GRID_top_sum,
                                lon = lon_mesh[0,:],
                                lat = lat_mesh[:,0],
                                projection = projection,
                                levels = levels,
                                timepassed = [n,time_steps],
                                colormap = colormap,
                                title = 'concentration [mol]'+str(times[n]),
                                unit = 'mol',
                                savefile_path = 'C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run_full\\make_gif\\mox_field_test'+str(n)+'.png',
                                show = False,
                                adj_lon = [0,0],
                                adj_lat = [0,-2.5],
                                bar_position = [0.315,0.12,0.49558,0.03],
                                dpi = 90,
                                log_scale = False)
        images_field_test.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run_full\\make_gif\\mox_field_test'+str(n)+'.png'))
        plt.close(fig)
    
#Create a field with all the layers summed up
GRID_sum = np.zeros(GRID[0][0].toarray().shape)
for n in range(0,744):
    print(n)
    GRID_sum = GRID_sum + np.sum(GRID[n][:].toarray(),axis=0)

#first loop through to find max median
maxmed = []
maxmed = np.max(np.max(GRID_mox))
for n in range(0,744):
    print(n)
    maxmed = np.append(maxmed,np.max(GRID_mox, axis=0))

plt.plot(GRID_top_sum)

#fill in the gasp in total_mox and total_atm_flux (just remove all zeros)

total_mox_new = total_mox[total_mox != 0]
total_atm_flux_new = total_atm_flux[total_atm_flux != 0]
#make a time vector where the same zero indices are removed
times_new = times[total_mox != 0]

#set the ylimits
ylims = [0,np.max([np.max(total_mox_new),np.max(total_atm_flux_new)])]

#make a yy plot of both of them on the same figure
fig, ax1 = plt.subplots()
ax1.plot(times_new[170:-170],total_mox_new[170:-170],color='blue',label='Total microbial oxidation')
ax1.set_ylabel('Total microbial oxidation [mol/hr]', color='blue')
ax1.set_xlabel('Time')
ax1.set_title('Total microbial oxidation and total atmospheric flux')
#set ylims
ax1.set_ylim(ylims)
ax2 = ax1.twinx()
ax2.plot(times_new[170:-170],total_atm_flux_new[170:-170],color='red',label='Total atmospheric flux')
ax2.set_ylabel('Total atmospheric flux [mol/hr]', color='red')
#set ylims
ax2.set_ylim(ylims)
#set xticks such that only 4 ticks are shown
#ax1.set_xticks(np.linspace(0,len(times_new[170:-170]),5))
fig.tight_layout()
plt.show()

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(times_new[170:-170],total_atm_flux_new[170:-170],label='Total atmospheric flux',color='blue')
ax.set_ylabel('Total atmospheric flux [mol/hr]',fontdict={'fontsize':16})
ax.set_title('Total MOx and atmospheric flux, approx. 1 to 10 relationship',fontdict={'fontsize':16})
ax.set_ylim([0,np.max(total_mox_new)])
#set fontsize for the xticks
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',fontsize=14)
#add yy axis showing the number of active particles
ax2 = ax.twinx()
ax2.plot(times_new[170:-170],total_mox_new[170:-170],color='0.4',label='Number of active particles')
ax2.set_ylabel('Total MOx [mol/hr]',fontdict={'fontsize':16})
ax2.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',fontsize=14)
ax2.set_ylim([0,np.max(total_mox_new)])
#save figure
plt.savefig('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run\\total_atm_flux_sameaxis.png')





####################################################################################################
#DEPRACATED#DEPRACATED#DEPRACATED#DEPRACATED#DEPRACATED#DEPRACATED#DEPRACATED#DEPRACATED#DEPRACATED#
####################################################################################################
'''


    #calculate wind speed
    wind_speed = np.sqrt(u10**2 + v10**2)
    wind_speed = np.transpose(wind_speed,(2,1,0))
    sst = np.transpose(sst,(2,1,0)) #Just to get it to match with the UTM coordinate matrix and have time as the last dimension.
    del u10,v10

    #Calculate utm coordinates for the wind field
    #loop over to be sure, so much weird going on now. 
    #UTM_wind = np.zeros((len(lats),len(lons)))
    #for i in range(0,len(lats)):
    #    for j in range(0,len(lons)):
    #        UTM_wind[i,j] = utm.from_latlon(lats[i],lons[j])[0]

    latmesh,lonmesh = np.meshgrid(lats,lons)

    UTM_wind = utm.from_latlon(latmesh,lonmesh)
    
    UTM_wind_x_mesh = UTM_wind[0]
    UTM_wind_y_mesh = UTM_wind[1]
       

    from scipy.interpolate import griddata

    wind_coo_mat = np.array([UTM_x_wind_mesh.flatten(),UTM_y_wind_mesh.flatten()]).T
    
    #Flatten wind and sst data
    wind_speed_flat = wind_speed[:,:,0].flatten()
    sst_flat = sst[:,:,0].flatten()

    #Create 2d array of fine grid
    #create mesh
    bin_x_mesh,bin_y_mesh = np.meshgrid(bin_x.T,bin_y.T)

    #plot the outlines of the two domains in the same figure
    plt.plot(bin_x_mesh,bin_y_mesh,'r.')
    plt.plot(UTM_wind_x_mesh,UTM_wind_y_mesh,'b.')

    GRID_coo_mat = np.array([bin_x_mesh.flatten(),bin_y_mesh.flatten()]).T

    #Interpolate the wind field to the grid
    wind_speed_interp = griddata(wind_coo_mat,wind_speed_flat,GRID_coo_mat,method='linear')

    #reshape the interpolated wind field
    wind_speed_interp = wind_speed_interp.reshape(bin_x_mesh.shape)

    #Make a plot of the original and interpolated wind speed field with utm corrdinates in the same figure 
    #using two subplots side by side. Use contourf
    
    levels = np.arange(0,21,1)
    colormap = plt.cm.get_cmap('magma',20)
    fig,ax = plt.subplots(2,2,figsize=(10,10))
    ###WIND SPEED###
    #original data
    ax[0].contourf(UTM_x_wind_mesh,UTM_y_wind_mesh,wind_speed[:,:,0].T)
    ax[0].contour(UTM_x_wind_mesh,UTM_y_wind_mesh,wind_speed[:,:,0].T,levels=levels,colors='black')
    ax[0].set_title('Original wind speed field at time 0')
    
    #interpolated data
    ax[1].contourf(bin_x_mesh,bin_y_mesh,wind_speed_interp)
    ax[1].contour(bin_x_mesh,bin_y_mesh,wind_speed_interp,levels=levels,colors='black')
    ax[1].set_title('Interpolated wind speed at time 0')



    #ax[0].contourf(UTM_x_wind_mesh,UTM_y_wind_mesh,wind_speed[:,:,0].T)
    #ax[0].set_title('Original wind speed field at time 0')
    #ax[1].contourf(bin_x_mesh,bin_y_mesh,wind_speed_interp)
    #ax[1].set_title('Interpolated wind speed field')
    #plt.show()

    ws = np.sqrt(u10**2 + v10**2)   
    U_constant = 5 #m/s
    T_constant = 10 #degrees celcius

        if plot_wind_field == True:
        levels_w = np.arange(-1, 24, 2)
        levels_sst = np.arange(np.round(np.nanmin(sst_interp))-2, np.round(np.nanmax(sst_interp))+1, 1)
        #do the same plot but just on lon lat coordinates
        #convert bin_x_mesh and bin_y_mesh to lon/lat
        lon_mesh,lat_mesh = utm.to_latlon(bin_x_mesh,bin_y_mesh,zone_number=33,zone_letter='V')
        colormap = 'magma'

        import imageio
        import matplotlib.gridspec as gridspec

        images_wind = []
        images_sst = []
        time_steps = len(bin_time)

        #datetimevector
        times = pd.to_datetime(bin_time,unit='s')-pd.to_datetime('2020-01-01')+pd.to_datetime('2018-05-20')

        for i in range(time_steps):
            #WIND FIELD PLOT
            fig = plt.figure(figsize=(7, 7))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])  # Create a GridSpec object

            lons_zoomed = lon_mesh
            lats_zoomed = lat_mesh
            ws_zoomed = ws_interp[i,:,:]

            ax1 = plt.subplot(gs[0])  # Create the first subplot for the contour plot
            contourf = ax1.contourf(lons_zoomed, lats_zoomed, ws_zoomed, levels=levels_w,cmap=colormap)
            cbar = plt.colorbar(contourf, ax=ax1)
            cbar.set_label('[m/s]')
            cbar.set_ticks(levels_w[1:-1])
            ax1.set_title('Wind speed, '+str(times[i])[:10])
            contour = ax1.contour(lons_zoomed, lats_zoomed, ws_zoomed, levels = levels_w, colors = 'w', linewidths = 0.2)
            ax1.clabel(contour, inline=True, fontsize=8)
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')

            ax2 = plt.subplot(gs[1])  # Create the second subplot for the progress bar
            ax2.set_position([0.12,0.12,0.6246,0.03])
            ax2.set_xlim(0, time_steps)  # Set the limits to match the number of time steps
            #ax2.plot([i, i], [0, 1], color='w')  # Plot a vertical line at the current time step
            ax2.fill_between([0, i], [0, 0], [1, 1], color='grey')
            ax2.set_yticks([])  # Hide the y-axis ticks
            ax2.set_xticks([0,time_steps])  # Set the x-axis ticks at the start and end
            ax2.set_xticklabels(['May 20, 2018', 'June 20, 2018'])  # Set the x-axis tick labels to the start and end time

            plt.savefig('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\gt_vel\\create_gif\\gt_vel'+str(i)+'.png')
            images_wind.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\gt_vel\\create_gif\\gt_vel'+str(i)+'.png'))
            plt.close()

            #SST PLOT
            fig = plt.figure(figsize=(7, 7))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])

            lons_zoomed = lon_mesh
            lats_zoomed = lat_mesh
            ws_zoomed = sst_interp[i,:,:]

            ax1 = plt.subplot(gs[0])  # Create the first subplot for the contour plot
            contourf = ax1.contourf(lons_zoomed, lats_zoomed, ws_zoomed, levels=levels_sst,cmap = colormap)
            cbar = plt.colorbar(contourf, ax=ax1)
            cbar.set_label('[m/s]')
            cbar.set_ticks(levels_sst[1:-1])
            ax1.set_title('Wind speed, '+str(times[i])[:10])
            contour = ax1.contour(lons_zoomed, lats_zoomed, ws_zoomed, levels = levels_sst, colors = 'w', linewidths = 0.2)
            ax1.clabel(contour, inline=True, fontsize=8)
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')

            ax2 = plt.subplot(gs[1])  # Create the second subplot for the progress bar
            ax2.set_position([0.12,0.12,0.6246,0.03])
            ax2.set_xlim(0, time_steps)  # Set the limits to match the number of time steps
            #ax2.plot([i, i], [0, 1], color='w')  # Plot a vertical line at the current time step
            ax2.fill_between([0, i], [0, 0], [1, 1], color='grey')
            ax2.set_yticks([])  # Hide the y-axis ticks
            ax2.set_xticks([0,time_steps])  # Set the x-axis ticks at the start and end
            ax2.set_xticklabels(['May 20, 2018', 'June 20, 2018'])  # Set the x-axis tick labels to the start and end time

            plt.savefig('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\sst\\create_gif\\sst_field'+str(i)+'.png')
            images_sst.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\sst\\create_gif\\sst_field'+str(i)+'.png'))
            plt.close()

        #create a gif
        imageio.mimsave('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\wind\\create_gif\\wind_field.gif', images_wind, duration=0.5)
        #and for sst
        imageio.mimsave('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\sst\\create_gif\\sst_field.gif', images_sst, duration=0.5)


'''

############################
#######PLOTTING WIND########
############################

if plot_wind_field == True:

    #load the wind data
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\model_grid\\interpolated_wind_sst_fields_test.pickle', 'rb') as f:
        ws_interp,sst_interp,bin_x_mesh,bin_y_mesh,ocean_time_unix = pickle.load(f)

    levels_w = np.arange(-1, 24, 2)
    levels_sst = np.arange(np.round(np.nanmin(sst_interp))-1, np.round(np.nanmax(sst_interp))+1, 1)-273.15
    levels_gt_vel = np.arange(0, 0.5, 0.05)
    #do the same plot but just on lon lat coordinates
    #convert bin_x_mesh and bin_y_mesh to lon/la
    lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh,bin_y_mesh,zone_number=33,zone_letter='W')
    #datetimevector
    if run_test == True:
        times = pd.to_datetime(bin_time,unit='s')-pd.to_datetime('2020-01-01')+pd.to_datetime('2018-05-20')
    else:
        times = pd.to_datetime(bin_time,unit='s')

    images_wind = []
    images_sst = []
    time_steps = len(bin_time)

    for i in range(time_steps):
        fig = plot_2d_data_map_loop(ws_interp[i,:,:],lon_mesh,
                        lat_mesh,projection,levels_w,[i,time_steps],
                        colormap,'Wind speed, '+str(times[i])[:10],
                        'm s$^{-1}$',
                        savefile_path='C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\wind\\create_gif\\wind'+str(i)+'.png',
                        dpi=90)
        #append to gif list
        images_wind.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\wind\\create_gif\\wind'+str(i)+'.png'))
        plt.close(fig)

        #SST PLOT
        fig = plot_2d_data_map_loop(sst_interp[i,:,:]-273.15,lon_mesh,
                        lat_mesh,projection,levels_sst,[i,time_steps],
                        colormap,'Sea surface temperature, '+str(times[i])[:10],
                        '°C',savefile_path='C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\sst\\create_gif\\sst'+str(i)+'.png',
                        dpi=90)
        #append to gif list
        images_sst.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\sst\\create_gif\\sst'+str(i)+'.png'))

        plt.close(fig)
    #create a gif
    #imageio.mimsave('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\wind\\create_gif\\wind_field.gif', images_wind, duration=0.5)
    #and for sst
    imageio.mimsave('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\sst\\create_gif\\sst_field.gif', images_sst, duration=0.5)

if plot_gt_vel == True:
    levels_gt = np.arange(np.round(np.nanmin(GRID_gt_vel)), np.round(np.nanmax(GRID_gt_vel))+0.2, 10)
    #do the same plot but just on lon lat coordinates
    lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh,bin_y_mesh,zone_number=33,zone_letter='W')
    #craete gif image list
    images_gt_vel = []
    for i in range(0,len(bin_time)):
        fig = plot_2d_data_map_loop(GRID_gt_vel[i,:,:],
                                    lon_mesh,
                                    lat_mesh,
                                    projection,
                                    levels_gt,
                                    [i,len(bin_time)],
                                    colormap,
                                    'Gas transfer velocity'+str(times[i])[:10],
                                    'm d$^{-1}$',
                                    savefile_path='C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\gt_vel\\create_gif\\gt_vel'+str(i)+'.png',
                                    show=False,
                                    dpi=90)
        images_gt_vel.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\gt_vel\\create_gif\\gt_vel'+str(i)+'.png'))
        #save figure
        plt.savefig('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\gt_vel\\gt_vel'+str(i)+'.png')
        plt.close(fig)

    #create a gif
    imageio.mimsave('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\gt_vel\\gt_vel.gif', images_gt_vel, duration=0.5)



#MAP PLOTTING.

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # Add this import


def plot_coastline_with_coordinates(coastline_map, lon_grid, lat_grid):
    """
    Plot coastline using proper coordinate grids
    """
    # Create figure with Lambert Conformal projection
    fig = plt.figure(figsize=(12, 8))
    proj = ccrs.LambertConformal(
        central_longitude=0.0,
        central_latitude=70.0,
        standard_parallels=(70.0, 70.0)
    )
    
    # Setup map
    ax = plt.axes(projection=proj)
    
    # Plot coastline data
    plt.pcolormesh(
        lon_grid, 
        lat_grid, 
        coastline_map,
        transform=ccrs.PlateCarree(),
        cmap='binary'
    )
    
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )
    
    # Use the correct formatter classes
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    # Customize gridlines
    gl.top_labels = False
    gl.right_labels = False
    
    # Set extent based on your grids
    ax.set_extent([
        lon_grid.min(), 
        lon_grid.max(), 
        lat_grid.min(), 
        lat_grid.max()
    ], crs=ccrs.PlateCarree())
    
    plt.title('Coastline Map')
    plt.show()

def plot_utm_with_latlon_grid(coastline_map, bin_x, bin_y, utm_zone=33):
    """Plot coastline in UTM coordinates with lat/lon grid overlay"""
    
    # Create transformers
    utm_to_latlon = Transformer.from_crs(
        f"EPSG:326{utm_zone}", "EPSG:4326", always_xy=True
    )
    latlon_to_utm = Transformer.from_crs(
        "EPSG:4326", f"EPSG:326{utm_zone}", always_xy=True
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot coastline in UTM
    plt.pcolormesh(bin_x, bin_y, coastline_map, cmap='binary')
    
    # Generate lat/lon grid lines
    lat_lines = np.arange(68.5, 72.5, 0.5)
    lon_lines = np.arange(12.5, 21.5, 0.5)
    
    # Plot longitude lines
    for lon in lon_lines:
        lats = np.linspace(lat_lines.min(), lat_lines.max(), 100)
        x, y = latlon_to_utm.transform(
            np.full_like(lats, lon), lats
        )
        plt.plot(x, y, '--', color='gray', alpha=0.5, linewidth=0.5)
        # Add labels
        if y[0] > bin_y.min() and y[0] < bin_y.max():
            plt.text(x[0], y[0], f'{lon}°E', fontsize=8)
    
    # Plot latitude lines
    for lat in lat_lines:
        lons = np.linspace(lon_lines.min(), lon_lines.max(), 100)
        x, y = latlon_to_utm.transform(lons, np.full_like(lons, lat))
        plt.plot(x, y, '--', color='gray', alpha=0.5, linewidth=0.5)
        # Add labels
        if x[0] > bin_x.min() and x[0] < bin_x.max():
            plt.text(x[0], y[0], f'{lat}°N', fontsize=8)
    
    plt.xlabel('UTM Easting (m)')
    plt.ylabel('UTM Northing (m)')
    plt.title('Coastline Map (UTM) with Lat/Lon Grid')
    
    # Set limits to match bin_x and bin_y
    plt.xlim(bin_x.min(), bin_x.max())
    plt.ylim(bin_y.min(), bin_y.max())
    
    plt.show()

plot_coastline_with_coordinates(coastline_map, lon_grid, lat_grid)
###
