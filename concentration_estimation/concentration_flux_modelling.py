'''
The main script for the dissolved gas concentration model framework.

For the script to work, it needs to import from akd_estimator.py located here: https://github.com/KnutOlaD/akd_estimator

Main Components:
1. Grid creation and particle initialization
2. Concentration field estimation using KDE
3. Process calculations (oxidation, atmospheric flux)
4. Multi-layer visualization and animation

Script Structure:
1. Import libraries and modules
2. Define constants and parameters
3. Functions
4. Load data and preprocess
5. Create grid and initialize particles
6. Main simulation loop
7. Visualization

Script main results:

1. Creates the GRID object, a list of lists containing the sparse matrices for each horizontal field at each depth level and time step. 
2. Saves the grid object as a pickle file for later use.
3. Creates the GRID_atm_flux object, a grid storing all the 2d atmospheric fluxes at each timestep
4. (...)

Author: Knut Ola Dølven
Date: 2024
License: MIT

'''

#########################
### IMPORT LIBRARIES ####
#########################

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import utm
from scipy.sparse import csr_matrix as csr_matrix
import pickle
from scipy.interpolate import griddata
import pandas as pd
import cartopy.crs as ccrs
import seaborn as sns
import time
from pyproj import Proj, Transformer
import xarray as xr
from scipy.spatial import cKDTree
import sys
import os

### import the self-made modules ###
#Set up paths
source_root = r"\src" #must be source roots where the akd_estimator is located.
project_root = r"\src\Methane_dispersion_modelling\concentration_estimation"
# Set project root directory explicitly
# Set module paths relative to project root
akd_path = source_root+ '\\akd_estimator'
#add the paths
sys.path.append(akd_path)
sys.path.append(project_root)
import akd_estimator as akd
import geographic_plotter as gp

###############

###############   
###SET RULES###
###############

### PLOT SETTINGS ###
color_1 = '#7e1e9c'
color_2 = '#014d4e'
colormap = 'magma'
plotting = False #plotting?
plot_gt_vel = False #plot gas transfer velocity
plot_wind_data = False #plot wind data
estimate_verttrans = False #estimate vertical transport?
plt.style.use('dark_background') #plotting style

### TRIGGERS ###
save_data_to_file = False #save output?
fit_wind_data = False #fit wind data
fit_gt_vel = False #Do a new fit of gas transfer velocity
wind_model_path = 'C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\interpolated_wind_sst_fields_test.pickle'  #path to wind model
kde_all = False #only do KDE for top layer trigger
manual_border = True #Set manual border for grid
manual_border_corners = [12.5,21,68.5,72,50*60] #Manual limitations for the grid [minlon,maxlon,minlat,maxlat,maxdepth]
get_new_bathymetry = False #Get new bathymetry or not?
redistribute_lost_mass = True #redistribute mass of deactivated particles?
h_adaptive = 'No_KDE' ##Set bandwidth estimator preference, alternatives are 'Local_Silverman', 'Time_dep' and 'No_KDE'
load_from_nc_file = True #Load data from netcdf file

### CONSTANTS ###
max_ker_bw = 10000 #max kernel bandwidth in meters
max_adaptation_window = 10000 #max adaptation window size in meters
gaussian_kernel_resolution = 1/3 # set resolution for the gaussian kernels (how many steps in bandwdith as a factor of the cell size)
atmospheric_conc = 0 #Atmospheric background concentration. We assume equilibrium with the atmosphere
background_ocean_conc = 0 #ocean background concentration, we    assume equilibrium with the atmosphere
oswald_solu_coeff = 0.28 ##Oswald solubility coeffocient (for methane)
projection = ccrs.LambertConformal(central_longitude=0.0, central_latitude=70.0, standard_parallels=(70.0, 70.0)) #Set projection
dxy_grid = 800. #horizontal grid size in meters
dz_grid = 25. #vertical grid size in meters
dt_grid = 3600. #temporal grid size in seconds
V_grid = dxy_grid*dxy_grid*dz_grid #grid volume in m^3
R_ox = 2.15*10**-7 #s^-1 average is R_ox = 3.6*10**-7 #s^-1
sum_sb_release = 0.02695169330621381 #mol/hr 
sum_sb_release_hr = sum_sb_release*dt_grid #mol released each time step
num_seed = 500
mass_full_sim = sum_sb_release_hr/num_seed #mol/particle #mass full simulation
twentiethofmay = 720 #Test period starts on May 20th
time_steps = 1495 #Test period ends on June 20th
redistribution_limit = 5000 #meters
lifespan = 24*7*4 # particle lifespan in hours
#For time dependent bandwidth
initial_bandwidth = 0.0000000000001
age_constant = 0.0000000000001

### PATHS DO DATA ###
# Sigma-level dataset with vertical diffusion in the whole wc and fallback = 10^-15
datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_norkyst_unlimited_vdiff_30s_fb_-15.nc'

# Timing
start_time_whole_script = time.time()

################################
########## FUNCTIONS ###########
################################

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
    
   
    if savefile_path == True:
        #save the grid object as a pickle file for later use
        #filename
        f = savefile_path
        with open('grid_object.pickle', 'wb') as f:
            pickle.dump(GRID, f)
    
    return bin_x,bin_y,bin_z,bin_time

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

#############################################################################################################

def fit_wind_sst_data(bin_x,bin_y,bin_time):
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
        
        # TODO: Verify if this +45 adjustment is needed
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

def find_nearest_grid_cell(lon_cwc, lat_cwc, depth_cwc, lon_mesh, lat_mesh, bin_z):
    """Find nearest grid cell for a given coordinate"""
    
    # Handle depth first using digitize
    depth_idx = np.digitize(depth_cwc, bin_z) - 1
    
    # Reshape mesh coordinates into (n_points, 2) array
    points = np.column_stack((lon_mesh.flatten(), lat_mesh.flatten()))
    
    # Build KDTree
    tree = cKDTree(points)
    
    # Find nearest neighbor
    distance, index = tree.query([lon_cwc, lat_cwc])
    
    # Convert flat index back to 2D indices
    lat_idx = index // lon_mesh.shape[1]
    lon_idx = index % lon_mesh.shape[1]
    
    # Optional: Print distance to nearest cell
    print(f"Distance to nearest cell: {distance:.2f} degrees")
    
    return lon_idx, lat_idx, depth_idx


# =============================================================================
# INITIATION
# =============================================================================

#if __name__ == '__main__':

# ---------------------------------------------------------------------
# LOAD AND USE FIRST TIMESTEP TO DEFINE THE GRID AND PARTICLE VARIABLES
# ---------------------------------------------------------------------

ODdata = nc.Dataset(datapath, 'r', mmap=True) #load the netcdf file
n_particles = first_timestep_lon = ODdata.variables['lon'][:, 0].copy() #get number of particles
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
particles['time'][1] = ODdata.variables['time'][1].copy()
#add utm dictionary entries (particles['UTM_x'] and particles['UTM_y'])
particles = add_utm(particles)

if manual_border == True:
    minlon = manual_border_corners[0]
    maxlon = manual_border_corners[1]
    minlat = manual_border_corners[2]
    maxlat = manual_border_corners[3]
    maxdepth = manual_border_corners[4]
else: #get the min/max values for the grid by looping over all timesteps
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

#get the corners in UTM coordinates using the utm package and the minlon/maxlon/minlat/maxlat values and force all coordinates to UTM zone
zone_number = 33
minUTMxminUTMy = utm.from_latlon(minlat, minlon, force_zone_number=zone_number, force_zone_letter='W')
minUTMxmaxUTMy = utm.from_latlon(minlat, maxlon, force_zone_number=zone_number, force_zone_letter='W')
maxUTMxminUTMy = utm.from_latlon(maxlat, minlon, force_zone_number=zone_number, force_zone_letter='W')
maxUTMxmaxUTMy = utm.from_latlon(maxlat, maxlon, force_zone_number=zone_number, force_zone_letter='W')

# ---------------------------------------------------------------------
# CREATE GRID
# ---------------------------------------------------------------------

print('Creating the output grid...')
#Create modeling output grid
bin_x,bin_y,bin_z,bin_time = create_grid(np.ma.filled(np.array(particles['timefull']),np.nan),
                                            [np.max([100000-dxy_grid-1,minUTMxminUTMy[0]]),np.min([1000000-dxy_grid-1,maxUTMxmaxUTMy[0]])],
                                            [np.max([dxy_grid+1,minUTMxminUTMy[1]]),np.min([10000000-dxy_grid-1,maxUTMxmaxUTMy[1]])],
                                            maxdepth+25,
                                            savefile_path=False,
                                            resolution=np.array([dxy_grid,dz_grid]))

# ---------------------------------------------------------------------
# ADD DICTIONARY ENTRIES FOR PARTICLE WEIGHT, BANDWIDTH, AGE, AND VERTICAL TRANSPORT 
# ---------------------------------------------------------------------

particles['weight'] = np.ma.zeros(particles['z'].shape) #particle mass
particles['weight'].mask = particles['lon'].mask #add mask
particles['bw'] = np.ma.zeros(particles['lon'].shape) #bandwidth
particles['bw'].mask = particles['lon'].mask #add mask
particles['z_transport'] = np.ma.zeros(particles['z'].shape)
particles['z_transport'].mask = particles['lon'].mask #add mask
initial_age = 0
particles['age'] = np.ma.zeros(particles['lon'].shape) #particle age
particles['age'].mask = particles['lon'].mask #add mask
#and for total particles
total_parts = np.zeros(len(bin_time))

# ---------------------------------------------------------------------
# PRECOMPUTE GAUSSIAN KERNELS
# ---------------------------------------------------------------------

print('Generating gaussian kernels...')
#Calculate how many kernels we need to span from 0 to the maximum bandwidth
num_gaussian_kernels = int(np.ceil((max_ker_bw/dxy_grid)/gaussian_kernel_resolution))
#generate gaussian kernels
gaussian_kernels, gaussian_bandwidths_h = akd.generate_gaussian_kernels(num_gaussian_kernels, gaussian_kernel_resolution, stretch=1)
#Get the bandwidth in real distances (this is easy since the grid is uniform)
gaussian_bandwidths_h = gaussian_bandwidths_h*(bin_x[1]-bin_x[0])
print('done.')

# ---------------------------------------------------------------------
# BIN THE FIRST TIMESTEP 
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# CREATE GRID OF PERMISSIBLE GRID CELLS USING BATHYMETRY DATA
# ---------------------------------------------------------------------

print('Getting bathymetry and impermissible cells')
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

# Create matrices for impermissible cells using the bathymetry data and delta z
impermissible_cells = np.zeros([len(bin_x), len(bin_y), len(bin_z)])
bin_z_bath_test = bin_z.copy()
bin_z_bath_test[0] = 1  # Ensure the surface boundary is respected

# Loop through all grid cells and check if they are impermissible
for i in range(len(bin_x)):
    for j in range(len(bin_y)):
        for k in range(len(bin_z)):
            if bin_z_bath_test[k] > np.abs(interpolated_bathymetry[j, i]):
                impermissible_cells[i, j, k] = 0.1

print('done.')

# ---------------------------------------------------------------------
# DEFINE MATRICES, VECTORS, ETC FOR THE MODELING LOOP
# ---------------------------------------------------------------------

### RELATING TO THE GRID ###
GRID_active = np.zeros((len(bin_x),len(bin_y))) # active grid (temporary grid where we do kde, atmospheric flux, etc)
GRID_atm_flux = np.zeros((len(bin_time),len(bin_x),len(bin_y))) # atmospheric flux grid (grid storing all the 2d atmospheric fluxes)
GRID_mox = np.zeros((len(bin_time),len(bin_x),len(bin_y))) # Grid that stores depth-integrated mox consumption for the complete 4d field
GRID_gt_vel = np.zeros((len(bin_time),len(bin_y),len(bin_x))) #Gas transfer velocity grid for each timestep
bin_x_mesh,bin_y_mesh = np.meshgrid(bin_x,bin_y) #Create meshgrid for the bin_x and bin_y
lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh.T,bin_y_mesh.T,zone_number=33,zone_letter='W') #Create lat/lon meshgrid (mostly for plotting)
GRID_top = np.zeros((len(bin_time),len(bin_x),len(bin_y))) #top layer grid
GRID_hs = np.zeros((len(bin_time),len(bin_x),len(bin_y))) #integrated h grid for each timestep
GRID_stds = np.zeros((len(bin_time),len(bin_x),len(bin_y))) #integrated std grid for each timestep
GRID_neff = np.zeros((len(bin_time),len(bin_x),len(bin_y))) #integrated neff grid for each timestep
# Initialize the main 4-dimensional GRID as nested list with proper dimensions
time_steps_full = len(ODdata.variables['time'])-50 #number of timesteps, define this now to get correct length of vectors
GRID = [[None for _ in range(len(bin_z))] for _ in range(time_steps_full)]
# Initialize GRID for storing vertical transort in each grid cell
GRID_vtrans = [[None for _ in range(len(bin_z))] for _ in range(time_steps_full)]
integral_length_scale_windows = GRID #integral length scale grid
standard_deviations_windows = GRID #standard deviation grid
neff_windows = GRID #neff grid


### VECTORS WITH RESULTS ###
time_steps_full = len(ODdata.variables['time'])-50 #number of timesteps, define this now to get correct length of vectors
total_atm_flux = np.zeros(len(bin_time)) #Total space integrated atmoshperic flux time-series (as function of time, in mol/hr)
particles_atm_loss = np.zeros((len(bin_time))) #particle weightloss history (accumulated particle mass loss to the atmosphere at each timestep)
particles_mox_loss = np.zeros((len(bin_time))) #particle mox loss history (accumulated particle mox loss at each timestep)
particles_mass_out = np.zeros((len(bin_time))) #Mass that leaves the model domain (accumulated particle mass out at each timestep)
particles_mass_back = np.zeros((len(bin_time))) #Mass that re-enters the model domain (accumulated particle mass out at each timestep)
particles_mass_died = np.zeros((len(bin_time))) #Mass lost to particle deactivation (when particles were not able to re-distribute mass)
particles_that_left = np.array([]) #particles that left the domain
particles_that_comes_back = np.array([]) #particles that re-enter the domain
timedatetime = pd.to_datetime(bin_time,unit='s') #Create datetime vector from bin_time
integral_length_scale_full = np.zeros([len(bin_time),len(bin_z)]) #integral length scales for the whole field
h_values_full = np.zeros([len(bin_time),len(bin_z)]) #h values for the whole field
h_values_std_full = np.zeros([len(bin_time),len(bin_z)]) #standard deviations for the whole field
h_estimate_vector = np.zeros(time_steps_full-1) #vector to store the h estimate
h_list = list() #list to store h values
neff_list = list() #list to store neff values
std_list = list() #list to store std values
particle_mass_redistributed = np.zeros(len(bin_time)) #the amount of redistributed mass at each timestep. 
outside_grid = [] #list to store particles that are outside the grid
particle_lifespan_matrix = np.zeros((lifespan,int(np.max(bin_z)+dz_grid),4)) #matrix to store properties across the lifespan of particles (mass, age, etc)  
depth_bins_lifespan = np.arange(0,np.max(bin_z)+dz_grid/2,dz_grid/2) #depth bins for the lifespan matrix
lost_particles_due_to_nandepth = 0 #This is to check how much is lost to things erraneously going into bathymetry. Tests show its less then 10 particles 

### RELATING TO PERFORMANCE EVALUATION ###
kde_time_vector = np.zeros(time_steps_full-1) #vector to store the time it takes to do the KDE
elapsed_time_timestep = np.zeros(time_steps_full-1) #vector to store the elapsed time for each timestep
num_timesteps = time_steps_full
num_layers = len(bin_z)

### SOME EXTRA THINGS TO AVOID "NOT DEFINED" ERRORS ###
h_matrix_adaptive = np.zeros(np.shape(GRID_active))
std_estimate = np.zeros(np.shape(GRID_active))
N_eff = np.zeros(np.shape(GRID_active))


# ---------------------------------------------------------------------
# PROJECT AND/OR LOAD WIND AND SST FIELD DATA (IF SELECTED) AND GET GAS TRANSFER VELOCITY FIELDS
# ---------------------------------------------------------------------

if fit_wind_data == True:
    fit_wind_sst_data(bin_x,bin_y,bin_time) #This functino saves the wind and sst fields in a pickle file
    #LOAD THE PICKLE FILE (no matter what)
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\model_grid\\interpolated_wind_sst_fields_test.pickle', 'rb') as f:
    ws_interp,sst_interp,bin_x_mesh,bin_y_mesh,ocean_time_unix = pickle.load(f)

#interpolate nans in the wind field and sst field
ws_interp = np.ma.filled(ws_interp,np.nan) 
sst_interp = np.ma.filled(sst_interp,np.nan)

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


# =============================================================================
# END INITIAL CONDITIONS 
# =============================================================================

# =============================================================================
#####################################################
#####  MODEL THE CONCENTRATION AT EACH TIMESTEP #####
#####################################################
# =============================================================================

print('Starting to loop through all timesteps...')

#Start looping through.
for kkk in range(1,time_steps_full-1): 
    start_time_full = time.time() #for performance evaluation
    start_time = time.time() #this too...
    print(f"Time step {kkk}")

    # ------------------------------------------------------
    # LOADING PARTICLES INTO MEMORY (NOT MEMORY OPTIMIZED, REQUIRES 64 GB RAM) 
    # ------------------------------------------------------

    # Add option here to use memory vs cpu optimized version?
    if kkk==1:         
        ### Load all data into memory if first timestep ###
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
           
    #----------------------------------------------------#
    # DEFINE THE ACTIVE PARTICLE DATASETS AT T=1 AND T=0 #
    #----------------------------------------------------#

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
    particles['age'][:,1].mask = particles['z'][:, 1].mask
    particles['z_transport'][:,1].mask = particles['z'][:,1].mask

    # Add weights and reset the aging parameter if it's the first timestep
    if kkk == 1:
        particles['weight'][:,1][np.where(particles['weight'][:,1].mask == False)] = mass_full_sim
        particles['weight'][:,0] = particles['weight'][:,1]
        particles['age'][:,1][np.where(particles['weight'][:,1].mask == False)] = 0
        particles['age'][:,0] = particles['age'][:,1]

    #set all nan values in particles['z'] to the deepest layer - this is to solve a problem with a few particles being stuck in the seafloor when using sigma-layer parametrization.
    lost_particles_due_to_nandepth += len(particles['z'][np.isnan(particles['z'])])
    particles['z'][np.isnan(particles['z'])] = -(np.max(bin_z)-1)
    print({'Particles in seafloor: '+str(lost_particles_due_to_nandepth)})
    
    # Find active particles for later processing.. 
    active_particles = np.where(particles['z'][:,1].mask == False)[0]

    # -----------------------------
    # TAKE CARE OF DYING PARTICLES 
    # -----------------------------

    #This adds quite a bit of computation time...
    if redistribute_lost_mass == True:

        # Get deactivated indices safely
        deactivated_indices = np.where((particles['z'][:,1].mask == True) & 
                                    (particles['z'][:,0].mask == False))[0]
        deactivated_indices = deactivated_indices.astype(np.int64)
        #remove all outside_grid -indices from the deactivated list to get the indices of particles that died (and not just left the grid)
        particles_that_died = deactivated_indices[~np.isin(deactivated_indices, outside_grid)]

        ### Redistribute weights of particles that died to nearby particles ###
        if deactivated_indices.size > 0:
            particles_mass_died[kkk] = np.sum(particles['weight'][particles_that_died,0])
            # Create KDTree for active particles
            active_positions = np.column_stack((
                particles['UTM_x'][active_particles,1],
                particles['UTM_y'][active_particles,1]
            ))
            tree = cKDTree(active_positions)
            
            # Get dead particle positions
            dead_positions = np.column_stack((
                particles['UTM_x'][particles_that_died,0],
                particles['UTM_y'][particles_that_died,0]
            ))
            dead_weights = particles['weight'][particles_that_died,0]
            
            # Query KDTree for all dead particles at once
            distances, indices = tree.query(dead_positions, 
                                        k=10,  # Get 10 nearest neighbors
                                        distance_upper_bound=redistribution_limit)
            
            # Process each dead particle's redistribution
            valid_mask = distances < redistribution_limit
            weights = np.zeros_like(distances)
            weights[valid_mask] = 1 / (distances[valid_mask] + 1e-10)
            
            # Normalize weights row-wise
            row_sums = weights.sum(axis=1, keepdims=True)
            weights = np.divide(weights, row_sums, where=row_sums > 0)
            
            # Update particle weights
            for i, dead_weight in enumerate(dead_weights):
                valid_neighbors = indices[i][valid_mask[i]]
                neighbor_weights = weights[i][valid_mask[i]]
                particles['weight'][active_particles[valid_neighbors], 1] += dead_weight * neighbor_weights
                particle_mass_redistributed[kkk] += np.sum(dead_weight * neighbor_weights)

            print(f"Redistributed {np.sum(dead_weights):.2f} moles of methane from {len(particles_that_died)} particles")

        else:
            particles_mass_died[kkk] = 0 #store info..
    else:
        particles_mass_died[kkk] = 0
        particle_mass_redistributed[kkk]=0

    # ------------------------------------------
    # DEACTIVATE PARTICLES OUTSIDE OF THE GRID 
    # ------------------------------------------
    #Boundaries.
    max_x = np.max(bin_x)
    min_x = np.min(bin_x)
    max_y = np.max(bin_y)
    min_y = np.min(bin_y)

    outside_grid_prev_TS = outside_grid

    # Then find which active particles are outside grid
    outside_grid = np.where(
        (particles['UTM_x'][active_particles,1] < min_x) | 
        (particles['UTM_x'][active_particles,1] > max_x) | 
        (particles['UTM_y'][active_particles,1] < min_y) | 
        (particles['UTM_y'][active_particles,1] > max_y)
    )[0]

    outside_grid_old = np.where(
        (particles['UTM_x'][active_particles,0] < min_x) | 
        (particles['UTM_x'][active_particles,0] > max_x) | 
        (particles['UTM_y'][active_particles,0] < min_y) | 
        (particles['UTM_y'][active_particles,0] > max_y)
    )[0]

    inside_grid = ~outside_grid

    #Find all particles what were outside the grid in the previous timestep and is back in the grid
    outside_coming_in = ~outside_grid
    outside_coming_in = outside_coming_in[np.isin(outside_coming_in,outside_grid_old)]

    #make sure outside_grid only have particles that left the domain in the current timestep

    # Mask those particles
    particles['z_transport'][outside_grid,1].mask = True
    particles['z'][outside_grid,1].mask = True
    particles['weight'][outside_grid,1].mask = True
    particles['bw'][outside_grid,1].mask = True
    particles['UTM_x'][outside_grid,1].mask = True
    particles['UTM_y'][outside_grid,1].mask = True
    particles['lon'][outside_grid,1].mask = True
    particles['lat'][outside_grid,1].mask = True

    ### FIX NOTATION HERE ###
    outside_leaving = outside_grid[~np.isin(outside_grid,outside_grid_old)]

    # Add new outside_grid particles to particles_that_left
    if outside_leaving.size > 0:
        # Ensure outside_leaving is integer type
        outside_leaving = outside_leaving.astype(np.int64)
        # Concatenate and maintain integer type
        particles_that_left = np.unique(np.concatenate((particles_that_left, outside_leaving))).astype(np.int64)

    if outside_coming_in.size > 0:
        # Ensure outside_coming_in is integer type
        outside_coming_in = outside_coming_in.astype(np.int64)
        # Concatenate and maintain integer type
        particles_that_comes_back = np.unique(np.concatenate((particles_that_comes_back, outside_coming_in))).astype(np.int64)
    

    # Set up vector if it doesnt exist. 
    if 'particles_that_left' not in locals():
        particles_that_left = np.array([], dtype=int)

    # Then mask the particles
    if particles_that_left.size > 0:
        for field in ['z', 'weight', 'bw', 'UTM_x', 'UTM_y', 'lon', 'lat']:
            #Count the mass
            particles_mass_out[kkk] = np.sum(particles['weight'][outside_leaving,1])   
            particles_mass_back[kkk] = np.sum(particles['weight'][outside_coming_in,1])   
            particles[field][particles_that_left, 1].mask = True
    
    print(f"{particles_mass_out[kkk]} moles lost due to {len(outside_leaving)} particles leaving.")
    #print(f"{particles_mass_back[kkk]} moles gained due to {len(outside_coming_in)} particles re-entering.")

    # --------------------------------------
    # MODIFY PARTICLE WEIGHTS AND BANDWIDTHS
    # --------------------------------------

    #Unmask particles that were masked in the previous timestep and do some binning
    bin_z_number = np.digitize(
    np.abs(particles['z'][:,1][np.where(
        particles['z'][:,1].mask == False)]),bin_z)

    # Get the indices where particles['age'][:,j] is not masked and equal to 0
    activated_indices = np.where((particles['z'][:,1].mask == False) & (particles['z'][:,0].mask == True))[0]
    already_active = np.where((particles['z'][:,1].mask == False) & (particles['z'][:,0].mask == False))[0]

    ### ADD INITIAL WEIGHT IF THE PARTICLE HAS JUST BEEN ACTIVATED ###
    particles['weight'][activated_indices,1].mask = False
    particles['weight'][activated_indices,1] = mass_full_sim
    particles['bw'][activated_indices,1].mask = False
    particles['bw'][activated_indices,1] = initial_bandwidth
    particles['age'][activated_indices,1].mask = False
    particles['age'][activated_indices,1] = 0

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
        
        ##### ADD AGE #####
        particles['age'][already_active,1] = particles['age'][already_active,0] + 1
        particles['age'].mask[already_active,1] = False

        ##### CALCULATE VERTICAL DISPLACEMENT #####
        particles['z_transport'][already_active,1] = particles['z'][already_active,1]-particles['z'][already_active,0]

        ##### ADRESS PROBLEMATIC VALUES #####
        particles['weight'][already_active,1][particles['weight'][already_active,1]<0] = 0
        #add the bandwidth of the particle to the current timestep
        if np.isnan(age_constant) == False:
            particles['bw'][already_active,1] = particles['bw'][already_active,1] + age_constant
        #limit the bandwidth to a maximum value
        particles['bw'][already_active,1][particles['bw'][already_active,1]>max_ker_bw] = max_ker_bw

    #finished with modifying weights, replace weights[0] with weights[1] for next step
    particles['weight'][:,0] = particles['weight'][:,1]
    particles['age'][:,0] = particles['age'][:,1]

    # -------------------------------------------------------------------------------------- #
    # STORE STATS OF PARTICLES TO GET INFO ABOUT HOW METHANE IS DISTRIBUTED 
    # ---------------------------------------------------------------------------------------#

    # First get only unmasked active particles
    truly_active = np.where((particles['z'][:,1].mask == False) & 
                        (particles['weight'][:,1].mask == False))[0]

    ages = particles['age'][truly_active, 1].astype(int)
    ages[ages >= lifespan] = lifespan - 1
    depths = np.abs(particles['z'][truly_active, 1].astype(int))
    weights = particles['weight'][truly_active, 1]
    #Create a particle_lifespan_matrix entry for atmospheric loss using the particleweighing matrix
    atmospheric_loss = np.zeros(len(depths))
    add_atm_loss_at_idx = np.where(np.isin(truly_active, already_active_surface))[0]
    np.add.at(atmospheric_loss, add_atm_loss_at_idx, particleweighing*total_atm_flux[kkk-1])

    # Get ages and depths for only unmasked active particles
    np.add.at(particle_lifespan_matrix[:, :, 0], (ages, depths), weights)
    np.add.at(particle_lifespan_matrix[:, :, 1], (ages, depths), weights * R_ox * 3600)
    np.add.at(particle_lifespan_matrix[:, :, 2], (ages, depths), atmospheric_loss) 
    #the particle_lifespan_matrix[:,:,3] should include the number of active particles at a certain age and depth
    #np.add.ad(particle_lifespan_matrix[:,:,3],(ages,depths),)

    ## Handle atmospheric flux - only for unmasked surface particles
    #surface_mask = np.isin(truly_active, already_active_surface)
    #if surface_mask.any():
    #    surface_indices = np.searchsorted(already_active_surface, truly_active[surface_mask])
    #    np.add.at(particle_lifespan_matrix[:, :, 2], 
    #            (ages[surface_mask], depths[surface_mask]),
    #            particleweighing[surface_indices] * total_atm_flux[kkk-1])        

    # --------------------------------------------------
    # FIGURE OUT WHERE PARTICLES ARE LOCATED IN THE GRID
    # --------------------------------------------------
    
    #.... And create a sorted matrix for all the active particles according to
    #which depth layer they are currently located in. 

    #Get sort indices
    sort_indices = np.argsort(bin_z_number)
    #sort
    bin_z_number = bin_z_number[sort_indices]
    #get indices where bin_z_number changes
    change_indices = np.where(np.diff(bin_z_number) != 0)[0]
    
    #Define the [location_x,location_y,location_z,weight,bw] for the particle. This is the active particle matrix
    parts_active = [particles['UTM_x'][:,1].compressed()[sort_indices],
                    particles['UTM_y'][:,1].compressed()[sort_indices],
                    particles['z'][:,1].compressed()[sort_indices],
                    bin_z_number,
                    particles['weight'][:,1].compressed()[sort_indices],
                    particles['bw'][:,1].compressed()[sort_indices],
                    particles['z_transport'][:,1].compressed()[sort_indices]]
    
    #keep track of number of particles
    total_parts[kkk] = len(parts_active[0])

    # -----------------------------------
    # INITIATE FOR LOOP OVER DEPTH LAYERS
    # -----------------------------------

    #add one right hand side limit to change_indices
    change_indices = np.append(change_indices,len(bin_z_number))
    #add a zero at the beginning (to include depth layer 0)
    change_indices = np.insert(change_indices, 0, 0)

    ###########################################
    ###########################################
    ###########################################

    for i in range(0,len(change_indices)-1): #This essentially loops over all particles (does it???)
        
        # -----------------------------------------------------------
        # DEFINE ACTIVE GRID AND ACTIVE PARTICLES IN THIS DEPTH LAYER
        # -----------------------------------------------------------

        #Define GRID_active by creating a zero matrix of same size as the grid
        GRID_active = np.zeros((len(bin_x),len(bin_y)))

        #Define active particle matrix in depth layer i
        parts_active_z = [parts_active[0][change_indices[i]:change_indices[i+1]+1],
                        parts_active[1][change_indices[i]:change_indices[i+1]+1],
                        parts_active[2][change_indices[i]:change_indices[i+1]+1],
                        parts_active[3][change_indices[i]:change_indices[i+1]+1],
                        parts_active[4][change_indices[i]:change_indices[i+1]+1],
                        parts_active[5][change_indices[i]:change_indices[i+1]+1],
                        parts_active[6][change_indices[i]:change_indices[i+1]+1]]

        # -----------------------------------------------------
        # CALCULATE THE CONCENTRATION FIELD IN THE ACTIVE LAYER
        # -----------------------------------------------------

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

        # -------------------------------------
        # CALCULATE THE KERNEL DENSITY ESTIMATE
        # -------------------------------------

        if (kde_all and i < 12) or i == 0 or kkk>= 720:  # Perform KDE for first 10 layers or layer 0
            #print('Doing kde for depth layer',i)

            # ------------------------------
            # preGRIDding 
            # ------------------------------
        
            #Stop and go out of the if if there are no particles in the depth layer
            if len(parts_active_z[0]) == 0:
                continue

            #Time the kde step
            start_time = time.time()
            #pre-kernel density estimate using the histogram estimator
            preGRID_active,preGRID_active_counts,preGRID_active_bw = akd.histogram_estimator(parts_active_z[0],
                                                parts_active_z[1],
                                                bin_x,
                                                bin_y,
                                                parts_active_z[5],
                                                parts_active_z[4])

            preGRID_vert,preGRID_vert_counts,preGRID_vert_bw = akd.histogram_estimator(parts_active_z[0],
                                                parts_active_z[1],
                                                bin_x,
                                                bin_y,
                                                parts_active_z[5],
                                                parts_active_z[6])
            
            # ------------------------------
            # Using no KDE at all 
            # ------------------------------


            if h_adaptive == 'No_KDE':
                GRID_active = preGRID_active
                end_time = time.time()
                elapsed_time = end_time - start_time
            
            
            # ------------------------------
            # Using time dependent bandwidths 
            # ------------------------------

            if h_adaptive == 'Time_dep':
                GRID_active = akd.grid_proj_kde(bin_x,
                                            bin_y,
                                            preGRID_active,
                                            gaussian_kernels,
                                            gaussian_bandwidths_h,
                                            preGRID_active_bw,
                                            illegal_cells=impermissible_cells[:,:,i],)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"KDE took {elapsed_time:.6f} seconds")
                #store the time it took to calculate the kde 
                kde_time_vector[kkk] = elapsed_time

            # -------------------------------------------------
            # Using local Silverman AKA Adaptive KDE 
            # -------------------------------------------------
            
            if h_adaptive == 'Local_Silverman' and preGRID_active.any():
                start_time = time.time()
                
                # Compute integral length scale
                autocorr_rows, autocorr_cols = akd.calculate_autocorrelation(preGRID_active_counts)
                autocorr = (autocorr_rows + autocorr_cols) / 2
                
                if autocorr.any() > 0:
                    integral_length_scale_full[kkk, i] = (np.sum(autocorr) / autocorr[np.argwhere(autocorr != 0)[0]])
                    window_size = max(max_adaptation_window,int(integral_length_scale_full[kkk, i]))
                else:
                    integral_length_scale_full[kkk,i] = 0
                    window_size = 7
                
                window_size = np.clip(window_size, 7, int(max_adaptation_window/dxy_grid))
                if window_size % 2 == 0:
                    window_size += 1
                
                pad_size = window_size//2
                
                # Pad arrays
                preGRID_active_padded = np.pad(preGRID_active, pad_size, mode='reflect')
                preGRID_active_counts_padded = np.pad(preGRID_active_counts, pad_size, mode='reflect')
                
                # Compute statistics and bandwidths
                std_estimate, N_eff, integral_length_scale_matrix, h_matrix_adaptive = akd.compute_adaptive_bandwidths(
                    preGRID_active_padded, 
                    preGRID_active_counts_padded,
                    window_size, 
                    (window_size**2)/4,  #Every fourth cell must contain data. 
                    grid_cell_size=dxy_grid
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
                GRID_active = akd.grid_proj_kde(bin_x,
                                            bin_y,
                                            preGRID_active,
                                            gaussian_kernels,
                                            gaussian_bandwidths_h,
                                            h_matrix_adaptive, #because it's symmetric
                                            illegal_cells = impermissible_cells[:,:,i])
                if estimate_verttrans == True: 
                    GRID_active_verttrans = akd.grid_proj_kde(bin_x,
                                            bin_y,
                                            preGRID_vert,
                                            gaussian_kernels,
                                            gaussian_bandwidths_h,
                                            h_matrix_adaptive, #because it's symmetric
                                            illegal_cells = impermissible_cells[:,:,i])
                

                end_time = time.time()
                elapsed_time = end_time - start_time
                kde_time_vector[kkk] = elapsed_time
                
                #make a plot if kkk is modulus 250
                if kkk % 250 == 0:
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

            #Get concentration
            GRID_active = GRID_active/V_grid

            #-------------------------------------------------
            # ASSIGN VALUES TO SPARSE GRIDS
            #-------------------------------------------------

            # Make explicit copies to avoid values affecting each other
            grid_copy = GRID_active.copy()
            #vtrans_copy = GRID_active.copy()

            # Create sparse matrices from copies
            sparse_grid = csr_matrix(grid_copy)
            #sparse_vtrans = csr_matrix(vtrans_copy)
            
            GRID[kkk][i] = sparse_grid
            #GRID_vtrans[kkk][i] = sparse_vtrans

            # Cleanup
            #del grid_copy, vtrans_copy
            del sparse_grid#, sparse_vtrans

            #Other stuff that could be added... 
            #GRID_top[kkk,:,:] = GRID_copy
            #GRID_mox[kkk,:,:] = GRID_active*(R_ox*3600*V_grid)
            #integral_length_scale_windows[kkk][i] = csr_matrix(integral_length_scale_matrix)
            #standard_deviations_windows[kkk][i] = csr_matrix(std_estimate)

            # -------------------------------
            # CALCULATE ATMOSPHERIC FLUX/LOSS
            # -------------------------------

            # dxy_grid**2
            if i == 0:
                #GRID_atm_flux[kkk,:,:] = np.multiply(GRID_gt_vel[kkk,:,:].T,
                #    (((GRID_active+background_ocean_conc)-atmospheric_conc))
                #    )*dxy_grid*0.01 
                # If we assume equilibrium concentration this simplifies to
                GRID_atm_flux[kkk,:,:] = np.multiply(GRID_gt_vel[kkk,:,:].T,GRID_active)*dxy_grid*dxy_grid*0.01
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

# ----------------------
# PICKLE FILES FOR LATER
# ----------------------

#Save the GRID, GRID_atm_flux and GRID_mox, ETC to pickle files
#with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\GRID.pickle', 'wb') as f:
#    pickle.dump(GRID, f)
    #create a sparse matrix first
#load the GRID file
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\data_02diff\\GRID.pickle', 'rb') as f:
    GRID = pickle.load(f)
'''
if save_data_to_file == True:
    #GRID_atm_sparse = csr_matrix(GRID_atm_flux)    
    GRID_atm_sparse = GRID_atm_flux
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_atm_flux.pickle', 'wb') as f:
        pickle.dump(GRID_atm_sparse, f)
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_atm_flux.pickle', 'rb') as f:
        GRID_atm_flux = pickle.load(f)
    #GRID_mox_sparse = csr_matrix(GRID_mox)
    #GRID_mox_sparse = GRID_mox
    #with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_mox.pickle', 'wb') as f:
    #    pickle.dump(GRID_mox_sparse, f)
    #and wind, sst, and gt_vel fields
    if fit_wind_data == True:
        with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\ws_interp.pickle', 'wb') as f:
            pickle.dump(ws_interp, f)
        with open('C:\\Users\\kdo000\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\sst_interp.pickle', 'wb') as f:
            pickle.dump(sst_interp, f)
        with open('C:\\Users\\kdo000\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_gt_vel.pickle', 'wb') as f:
            pickle.dump(GRID_gt_vel, f)
    #and vectors for total values
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\total_atm_flux.pickle', 'wb') as f:
        pickle.dump(total_atm_flux, f)
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\particles_mass_died.pickle', 'wb') as f:
        pickle.dump(particles_mass_died, f)
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\particles_mass_out.pickle', 'wb') as f:
        pickle.dump(particles_mass_out, f)
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\particles_mox_loss.pickle', 'wb') as f:
        pickle.dump(particles_mox_loss, f)
    #and number of particles
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\total_parts.pickle', 'wb') as f:
        pickle.dump(total_parts, f)
    #and the bandwidths
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\h_values_full.pickle', 'wb') as f:
        pickle.dump(h_values_full, f)
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\h_values_std_full.pickle', 'wb') as f:
        pickle.dump(h_values_std_full, f)
    #and the integral length scales
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\integral_length_scale_full.pickle', 'wb') as f:
        pickle.dump(integral_length_scale_full, f)
    #and the time it took to estimate the bandwidths
    #Save the GRID_vtrans as well
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_vtrans.pickle', 'wb') as f:
        pickle.dump(GRID_vtrans, f)


    #save a short textfile with the settings etc..
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\settings.txt', 'w') as f:
        f.write('Model run summary\n')
        f.write('--------------------------------\n')
        f.write('GRID SETTINGS\n')
        f.write('Grid horizontal resolution: '+str(dxy_grid)+' m\n')
        f.write('Grid vertical resolution: '+str(dz_grid)+' m\n')
        f.write('Grid cell volume: '+str(V_grid)+' m³\n')
        f.write('\nKERNEL SETTINGS\n')
        f.write('Bandwidth estimator: '+str(h_adaptive)+'\n')
        f.write('Gaussian kernel set resolution (num kernels/grid cell length): '+str(gaussian_kernel_resolution)+'\n')
        f.write('Max adaptation window: '+str(max_adaptation_window)+'\n')
        f.write('Initial bandwidth: '+str(initial_bandwidth)+' m\n')
        f.write('Age constant: '+str(age_constant)+'\n')
        f.write('Max kernel bandwidth: '+str(max_ker_bw)+' m\n')
        f.write('\nBUDGET AND PROCESS COEFFICIENTS\n')
        f.write('Atmospheric background: '+str(atmospheric_conc)+' mol/m³\n')
        f.write('Ocean background: '+str(background_ocean_conc)+' mol/m³\n')
        f.write('Oswald solubility coefficient: '+str(oswald_solu_coeff)+'\n')
        f.write('Oxidation rate (R_ox): '+str(R_ox)+' s⁻¹\n')
        f.write('Total seabed release: '+str(num_seed*mass_full_sim*(30*24))+' mol\n')
        f.write('\nPARTICLE SETTINGS\n')
        f.write('Number of seed particles: '+str(num_seed)+'\n')
        f.write('Initial particle weight: '+str(mass_full_sim)+' mol/hr\n')
        f.write('Redistribute lost mass: '+str(redistribute_lost_mass)+'\n')
        f.write('Redistribution limit: '+str(redistribution_limit)+' m\n')
        f.write('\nTIME SETTINGS AND COMPUTATION TIME\n')
        f.write('Start timestep: '+str(twentiethofmay)+'\n')
        f.write('End timestep: '+str(time_steps)+'\n')
        f.write('Total computation time: '+str(total_computation_time)+' s\n')
        f.write('\nOXIDATION SETTINGS\n')
        f.write('\nDATA PATHS\n')
        f.write('Data path: '+str(datapath)+'\n')
        f.write('Wind model path: '+str(wind_model_path)+'\n')
        f.write('--------------------------------\n')
'''
######################################################################################################
#----------------------------------------------------------------------------------------------------#
###PLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTING###
#----------------------------------------------------------------------------------------------------#
######################################################################################################

# Define plotting style 
dark_mode = True
if dark_mode == True:
    plt.style.use('dark_background')
    colormap = 'rocket'
else:
    plt.style.use('default')
    colormap = 'rocket_r'
#Get the lan/lot mesh for the grid 
lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh,bin_y_mesh,zone_number=33,zone_letter='W')
#And limits
min_lon = np.min(lon_mesh)
max_lon = np.max(lon_mesh)
min_lat = np.min(lat_mesh)
max_lat = np.max(lat_mesh)
#Date and timestep vectors
times_totatm =  pd.to_datetime(bin_time,unit='s')
twentiethofmay = 720
time_steps = 1495

#Some triggers
plot_atm_flux = False
plot_concentration = False
create_video_layers = False
create_gif_layers = False
create_video_atm_flux = False
create_gif_atm_flux = False
plot_loss = False
plot_cross_section = False
plot_cross_section_location = False
plot_lifetime_plots = False

# ------------------------------------------ #
# PLOT SEVERAL DEPTH LAYERS OF CONCENTRATION #
# ------------------------------------------ #

if plot_concentration == True:

    # ----------------------------
    # DEFINE THE BATHYMETRY LINES
    bathymetry_lines = gp.get_bathymetry_lines(impermissible_cells, lon_mesh, lat_mesh)

    # ----------------------
    # MAKE ONE PLOT FOR AVERAGE CONCENTRATION FOR EACH LAYER 

    #define dummy timestep
    timestep = 1000

    #Calculate the average for each layer
    GRID_average = np.zeros([10,389,501])
    for n in range(1485-720):
        for k in range(10):
            GRID_average[k,:,:] = GRID_average[k,:,:]+GRID[n+720][k].toarray()
    GRID_average = GRID_average/len(range(1485-720)) #This is the time average for each layer
    layers = [GRID_average[i] for i in range(9)] #This is the layers to plot
    vmax = max(np.nanmax(layer) for layer in layers)*0.75 #Set upper limit
    vmin = 0.01*vmax #set lower limit
    levels = np.linspace(vmin, vmax, 100) #Define specific levels for the plot
    titles = [f'{bin_z[i]:.0f}-{bin_z[i+1]:.0f}m {times_totatm[timestep-twentiethofmay]:%d.%b %H:%M}' 
            for i in range(9)] #Define titles for each plot (if plotted many times)
    depthstrings = [f'{bin_z[i]:.0f}-{bin_z[i+1]:.0f}m' for i in range(9)] #Depth strings for the subplots
    timestrings = ['Time averaged concentration, May 20 - June 20'] #Time strings for the main title
    #Define seep site location
    poi={
            'lon': 14.279600,
            'lat': 68.918600,
            'color': 'yellow',
            'size': 24,
            'label': 'Seep site',
            'edgecolor': 'black'
        }
    timestep= 1000 #Set timestep just to avoid errors since the function demands a timestep
    # PLOT!
    fig = gp.plot_multiple_2d_data_on_map(
        data_list=layers,
        lon=lon_mesh,
        lat=lat_mesh,
        projection=projection,
        levels=levels,
        timepassed=[timestep-twentiethofmay,time_steps-twentiethofmay],
        colormap=colormap,
        titles=titles,
        bathymetry_lines = bathymetry_lines,
        unit='mol m$^{-3}$',
        adj_lon = [2.7,-2.5],
        depthstring = depthstrings,
        timestring = timestrings,
        figsize = (18,18),
        nrows = 3,
        ncols = 3,
        adj_lat = [0.15,-1.2],
        log_scale=True,
        plot_progress_bar = False,
        poi=poi
    )
    plt.show()
    fig

    # -----------------------------
    # CREATE ANIMATION FOR THE WHOLE TIMESERIES

    if create_gif_layers == True or create_video_layers == True:

        foldername = 'C:\\Users\\'
        images_layers = [] #empty list to store images
        timestep_vector = np.arange(twentiethofmay,time_steps,1) #timestepvector
        times = times_totatm[timestep_vector] #timevector

        for timestep in timestep_vector:
            print(timestep)
            timestrings = [f'{times[timestep]:%d.%b %H:%M}']
            GRID_tmp = GRID[timestep]
            layers = [GRID_tmp[i].toarray() for i in range(9)]
            fig = gp.plot_multiple_2d_data_on_map(
                data_list=layers,
                lon=lon_mesh,
                lat=lat_mesh,
                projection=projection,
                levels=levels,
                timepassed=[timestep-twentiethofmay,time_steps-twentiethofmay],
                colormap=colormap,
                titles=titles,
                bathymetry_lines = bathymetry_lines,
                unit='mol m$^{-3}$',
                adj_lon = [2.7,-2.5],
                depthstring = depthstrings,
                timestring = timestrings,
                figsize = (18,18),
                nrows = 3,
                ncols = 3,
                adj_lat = [0.15,-1.2],
                log_scale=True,
                plot_progress_bar = False,
                poi=poi
            )
            savefile_path=foldername+'layer'+str(timestep)+'.png'
            fig.savefig(savefile_path,dpi=90,transparent=False,bbox_inches='tight')
            plt.close(fig)  # Close the figure to avoid displaying it bv801

        #
        if create_video_layers == True:
            # Get and sort filenames numerically
            filenames = sorted([f for f in os.listdir(foldername) if f.endswith('.png')], 
                            key=lambda x: int(x.replace('layer', '').replace('.png', '')))
            import cv2
            # Load all images first
            images_layers = []
            for filename in filenames:
                filepath = os.path.join(foldername, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    images_layers.append(img)

            if len(images_layers) > 0:
                # Get size from first image
                size = (images_layers[0].shape[1], images_layers[0].shape[0])  # width, height
                
                # Create VideoWriter object
                output_path = foldername+'layers.mp4'
                fps = 8  # frames per second
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

                # Write frames
                for image in images_layers:
                    out.write(image)  # No need to convert BGR since cv2.imread already returns BGR

                # Release the video writer
                out.release()
            else:
                print("No images found in the specified directory")

        #create gif
        if create_gif_layers == True:
            import imageio
            for filename in filenames:
                images_layers.append(imageio.imread(foldername+filename))
                #create gif

            imageio.mimsave(foldername+'layers.gif', images_layers, duration=0.5)


# ------------------------------- #
# PLOT THE ATMOSPHERIC FLUX FIELD #
# ------------------------------- #

if plot_atm_flux == True:

    #load the GRID_atm_flux
    #with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_atm_flux.pickle', 'rb') as f:
    #    GRID_atm_flux = pickle.load(f)
    foldername = 'C:\\Users\\'

    # -----------------------
    # PLOT ACCUMULATED ATMOSPHERIC FLUX FIELD
        
    GRID_atm_flux_sum = np.nansum(GRID_atm_flux[twentiethofmay:time_steps,:,:],axis=0) ##Calculate the sum of all timesteps in GRID_atm_flux in moles
    total_sum = np.nansum(np.nansum(GRID_atm_flux_sum))#total sum
    percent_of_release = np.round((total_sum/(num_seed*mass_full_sim*(30*24)))*100,4) #why multiply with 100??? Because it's percantage dumb-ass
    GRID_atm_flux_sum = GRID_atm_flux_sum/(dxy_grid**2)#/1000000 #convert to mol.Divide to get per square meter
    levels = np.linspace(np.nanmin(np.nanmin(GRID_atm_flux_sum)),np.nanmax(np.nanmax(GRID_atm_flux_sum)),100)

    gp.plot_2d_data_on_map(data=GRID_atm_flux_sum.T,
                        lon=lon_mesh,
                        lat=lat_mesh,
                        projection=projection,
                        levels=levels,
                        timepassed=[1, time_steps],
                        colormap=colormap,
                        title='Total released methane = '+str(np.round(total_sum,2))+' mol, $\sim'+str(percent_of_release)+'\%$',
                        unit='mol m$^{-2}$',
                        savefile_path=foldername + 'atm_flux_sum.png',
                        show=True,
                        adj_lon = [0.,-1.8],
                        adj_lat = [0.15,0.],
                        dpi=60,
                        figuresize = [12,10],
                        log_scale = True,
                        plot_progress_bar = False,
                        maxnumticks = 9,
                        plot_model_domain = True,#[min_lon,max_lon,min_lat,max_lat,0.5,[0.4,0.4,0.4]],
                        contoursettings = [4,'0.8',0.1],
                        poi=poi)

    # -----------------------
    # PLOTTING TIMESERIES OF DIFFUSIVE ATMOSPHERIC FLUX FIELD

    if create_video_atm_flux == True or create_gif_atm_flux == True:
        
        images_layers = [] #empty list to store images
        timestep_vector = np.arange(twentiethofmay,time_steps,1) #timestepvector

        #Calculate atmospheric flux field per square meter per hour
        GRID_atm_flux_m2 = (GRID_atm_flux/(dxy_grid**2))#

        GRID_generic = GRID_atm_flux_m2#Define the generic grid
        images_atm_rel = [] #Define list to store images
        levels_atm = np.linspace(np.nanmin(np.nanmin(GRID_generic)),np.nanmax(np.nanmax(GRID_generic)),100)
        levels_atm = levels_atm[:-50]*0.29 #Levels for contour plot

        for i in range(twentiethofmay,time_steps,1):
            fig = gp.plot_2d_data_on_map(data=GRID_generic[i, :, :].T,
                                        lon=lon_mesh,
                                        lat=lat_mesh,
                                        projection=projection,
                                        levels=levels_atm,
                                        timepassed=[i-twentiethofmay, time_steps-twentiethofmay],
                                        colormap=colormap,
                                        title='Atmospheric flux [mol m$^{-2}$ hr$^{-1}$]' + str(times[i])[5:-3],
                                        unit='mol m$^{-2}$ hr$^{-1}$',
                                        adj_lon = [0.,-1.8],
                                        adj_lat = [0.15,0.],
                                        show=False,
                                        dpi=90,
                                        figuresize = [12,10],
                                        log_scale = True,
                                        starttimestring = '20 May 2018',
                                        endtimestring = '21 June 2018',
                                        maxnumticks = 10,
                                        plot_progress_bar = True,
                                        plot_model_domain = True,
                                        #plot_model_domain = [min_lon,max_lon,min_lat,max_lat,0.5,[0.4,0.4,0.4]],
                                        contoursettings = [2,'0.8',0.1],
                                        poi=poi
                                        )
            savefile_path=foldername+'atm_flux'+str(timestep)+'.png'
            fig.savefig(savefile_path,dpi=90,transparent=False,bbox_inches='tight')
            plt.close(fig)  # Close the figure to avoid displaying it bv801

        #Create gif if trigger is True
        if create_video_atm_flux == True:
            # Get and sort filenames numerically
            filenames = sorted([f for f in os.listdir(foldername) if f.endswith('.png')], 
                            key=lambda x: int(x.replace('atm_flux', '').replace('.png', '')))
            import cv2
            # Load all images first
            images_layers = []
            for filename in filenames:
                filepath = os.path.join(foldername, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    images_layers.append(img)

            if len(images_layers) > 0:
                # Get size from first image
                size = (images_layers[0].shape[1], images_layers[0].shape[0])  # width, height
                
                # Create VideoWriter object
                output_path = foldername+'atm_flux.mp4'
                fps = 8  # frames per second
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

                # Write frames
                for image in images_layers:
                    out.write(image)  # No need to convert BGR since cv2.imread already returns BGR

                # Release the video writer
                out.release()
            else:
                print("No images found in the specified directory")

        #create gif
        if create_gif_atm_flux == True:
            import imageio
            for filename in filenames:
                images_layers.append(imageio.imread(foldername+filename))
            imageio.mimsave(foldername+'atm_flux.gif', images_layers, duration=0.5)


    # ------------------- #
    # PLOT LOSS VARIABLES #
    # ------------------- #

    if plot_loss == True:

        fig = gp.plot_loss_analysis(times_totatm, total_atm_flux, ws_interp,
                        particles_mox_loss, particles_mass_out,
                        particles_mass_died, particle_mass_redistributed,
                        twentiethofmay, time_steps)

        plt.savefig('methane_loss.png', dpi=300, bbox_inches='tight')
        plt.show()