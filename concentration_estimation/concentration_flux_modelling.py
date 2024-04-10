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
from numba import jit, prange
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from numpy.ma import masked_invalid
import imageio
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm


###############   
###SET RULES###
###############

plotting = False
#Set plotting style
plt.style.use('dark_background') 
#fit wind data
fit_wind_data = False
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
max_ker_bw = 25000
#atmospheric background concentration
#atmospheric_conc = ((44.64*2)/1000000) #mol/m3 #1911.8 ± 0.6 ppb #44.64 #From Helge
atmospheric_conc = (3.3e-09)*1000 #mol/m3 #ASSUMING SATURATION CONCENTRATION EVERYWHERE. 
#oceanic background concentration
background_ocean_conc = (3.3e-09)*1000 #mol/m3
#Oswald solubility coeffocient
oswald_solu_coeff = 0.28 #(for methane)
#Set projection
projection = ccrs.LambertConformal(central_longitude=0.0, central_latitude=70.0, standard_parallels=(70.0, 70.0))
#grid size
dxy_grid = 5000. #m
dz_grid = 25. #m
#grid cell volume
V_grid = dxy_grid*dxy_grid*dz_grid
#age constant
age_constant = 125 #m per hour, see figure.
#Initial bandwidth
initial_bandwidth = 250 #m
#set colormap
colromap = 'magma'

#List of variables in the script:
#datapath: path to the netcdf file containing the opendrift data|
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
                        'trajectory':ODdata.variables['trajectory'][:]} #this is 

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
        z_indices = np.digitize(particles['z'][:,i],bin_z) 
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
        #set all values outside of the UTM domain to nan
        #...
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

def calc_gt_vel(u10=5,temperature=20,grid_area=1,gas='methane'):
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
    k: gas transfer velocity #m/day

    '''

    #Calculate the Schmidt number
    Sc = calc_schmidt_number(T=temperature,gas=gas)

    #make this such that we can calculate the gas transfer velocity for the whole grid and just
    #grab the data... 
    #Calculate the gas transfer velocity constant
    k = (0.251 * u10**2 * (Sc/660)**(-0.5))*grid_area #m/day

    #Calculate the atmospheric flux
    #F = k * (C_o - C_a) #mol/m2/day

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
        #in the kernel. 
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
         #check that no indices are outside the grid (and use only the ones that are inside the grid and not below zero
        #THIS WAS TOO MUCH HASSLE, JUST REMOVE ANY KERNELS THAT TOUCHES THE BOUNDARY
        # Flatten ix and iy
        #ix_flat = ix.flatten() 
        #iy_flat = iy.flatten()
        # Create the mask
        #mask = (ix_flat >= 0) & (ix_flat < len(x_grid)) & (iy_flat >= 0) & (iy_flat < len(y_grid))
        #get any indices that are false in the mask
        #mask_false = np.where(mask == False)
        # Apply the mask
        #ix_flat = ix_flat[mask]
        #iy_flat = iy_flat[mask]
        # Reshape ix and iy back to their original shapes
        #ix = ix_flat.reshape(len(ix_flat),1)
        #iy = iy_flat.reshape(1,len(iy_flat))

        #if any values in ix or iy is outside the grid, remove the kernel entirely and skip to next iteration
        if np.any(ix >= len(x_grid)) or np.any(iy >= len(y_grid)) or np.any(ix < 0) or np.any(iy < 0):
            continue

        #add the kernel values to the grid
        GRID_active[ix,iy] += kernel_matrix*weights[i]
        
    #reshape GRID_active to the grid

    return GRID_active

#def est_aging_constant

#Create a function for aging constant estimation

#################################
########## INITIATION ###########
#################################

if __name__ == '__main__':

    #Just load the grid object to make it faster
    #with open('grid_object.pickle', 'rb') as f:
    #   GRID = pickle.load(f)

    run_test = True
    if run_test == True:
        datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_test.nc'#test dataset
        particles = load_nc_data(datapath)
        particles = add_utm(particles)
        #adjust the time vector to start on May 20 2018

    run_full = False
    if run_full == True:
        datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_norkyst.nc'#real dataset
        ODdata = nc.Dataset(datapath)
        #get all variables that are not masked at the first timestep
        first_timestep_lon = ODdata.variables['lon'][:, 0]
        #unmasked_first_timestep_lon = first_timestep_lon[~first_timestep_lon.mask]
        unmasked_indices = np.where(~first_timestep_lon.mask)
        unmasked_indices = np.array(unmasked_indices)[0] #fix shape/type problem

        timestep_num = 744
        #loop over and store the lon/lat positions of the particles indicated by
        #the unmasked indices array
        ### CALCULATE AGING FUNCTION ###
        calc_age = False
        if calc_age == True:
            lon = np.zeros((len(unmasked_indices),timestep_num))
            lat = np.zeros((len(unmasked_indices),timestep_num))
            for i in range(0,timestep_num):
                lon[:,i] = ODdata.variables['lon'][unmasked_indices,i]
                lat[:,i] = ODdata.variables['lat'][unmasked_indices,i]
                print(i)

            #find and set to nan all values that are outside the UTM domain,
            #that is above 71 degrees north and below 65 degrees north 
            #and west of 20 degrees east and east of 0 degrees east
            #remove everything above 71 degrees north and 20 degrees east
            lon[lat > 71] = np.nan
            lat[lat > 71] = np.nan

           #Calculate the UTM coordinates
            lon_masked = np.ma.masked_invalid(lon)
            lat_masked = np.ma.masked_invalid(lat)
            UTM = utm.from_latlon(lon_masked,lat_masked)
            utm_x = UTM[0]
            utm_y = UTM[1]

            #create a masked utm_x/utm_y array
            utm_x_masked = np.ma.masked_invalid(utm_x)
            utm_y_masked = np.ma.masked_invalid(utm_y)

            for i in range(0,len(utm_x_masked)):
                diff_x[i,:] = utm_x_masked[i,j] - utm_x_masked[:,j]
                diff_y[i,:] = utm_y_masked[i,j] - utm_y_masked[:,j]
            
                #calculate the distance between all particles
            distance = np.sqrt(diff_x**2 + diff_y**2)

            #find all indices in distance where the difference in horizontal distance is less than 
            #100 meters

        #get only the particles that are active (they are nonmasked)
        #particles = {'lon':ODdata.variables['lon'][unmasked_indices],
        #                'lat':ODdata.variables['lat'][unmasked_indices],
        #                'z':ODdata.variables['z'][unmasked_indices],
        #                'time':ODdata.variables['time'][unmasked_indices],
        #                'status':ODdata.variables['status'][unmasked_indices]}
        

run_everything = True# True
if run_everything == True:
    
    ###### SET UP GRIDS FOR THE MODEL ######

    #MODEELING OUTPUT GRID
    GRID,bin_x,bin_y,bin_z,bin_time = create_grid(np.ma.filled(np.array(particles['time']),np.nan),
                                                [np.max([100000-dxy_grid-1,np.min(particles['UTM_x'].compressed())]),np.min([np.max(particles['UTM_x'].compressed()),1000000-dxy_grid-1])],
                                                [np.max([dxy_grid+1,np.min(particles['UTM_y'].compressed())]),np.min([np.max(particles['UTM_y'].compressed()),10000000-dxy_grid-1])],
                                                np.max(np.abs(particles['z'])),
                                                savefile_path=False,
                                                resolution=np.array([dxy_grid,dz_grid]))
    #CREATE ONE ACTIVE HORIZONTAL MODEL FIELD
    GRID_active = np.zeros((len(bin_x),len(bin_y)))
    #ATMOSPHERIC FLUX GRID
    GRID_atm_flux = np.zeros((len(bin_time),len(bin_x),len(bin_y)))

    #Create coordinates for plotting
    bin_x_mesh,bin_y_mesh = np.meshgrid(bin_x,bin_y)
    #And lon lat coordinates
    lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh.T,bin_y_mesh.T,zone_number=33,zone_letter='V')

    ############################
    ###### FIRST TIMESTEP ######
    ############################

    #Get the last timestep (to get the size of the thing)
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

    #########################################
    ### ADD WEIGHTS TO THE FIRST TIMESTEP ###
    #########################################

    vertical_profile = np.ones(bin_z.shape[0])
    #Should be an exponential with around 100 at the bottom and 10 at the surface
    vertical_profile = np.round(np.exp(np.arange(0,np.max(np.abs(particles['z'][:,0])+10),1)/44)) #This is for test run
    #Create a matrix with the same size as particles['z'] and fill with the vertical profile depending
    #on the depth level where the particle was in its first active timestep
    #It's the same initial weight of all particles. 

    weights_full_sim = 0.0408 #mol/hr

    ##################################
    ### CALCULATE GRID CELL VOLUME ###
    ##################################

    grid_resolution = [dxy_grid,dxy_grid,dz_grid] #in meters
    V_grid = grid_resolution[0]*grid_resolution[1]*grid_resolution[2]

    ################################################
    ### WE DONT NEED THIS PART WITH KDE ESTIMATE ###
    ################################################
    #Have a sparse matrix which keeps track of the number of particles in each grid cell. 
    #GRID_part = GRID
    #Establish a matrix for atmospheric flux which is the same size as GRID only with one depth layer
    #GRID_atm_flux = np.array(GRID)[:,0] #this can be just an np array. 
    ################################################
    ################################################
    ################################################

    ############################
    ### LOAD WIND FIELD DATA ###
    ############################

    if fit_wind_data == True:

        with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\ERAV_all_2018.pickle', 'rb') as f:
            lons, lats, times, sst, u10, v10, ws = pickle.load(f)

        #reverse the lats coordinates and associated matrices (sst, u10, v10, ws)
        lats = lats[::-1]
        #the matrices should be reversed only in the first dimension
        sst = sst[:,::-1,:]
        u10 = u10[:,::-1,:]
        v10 = v10[:,::-1,:]
        ws = ws[:,::-1,:]

        #create a time vector on unix timestamps
        #create datetime vector
        datetime_vector = pd.to_datetime(times)

        # Convert datetime array to Unix timestamps
        wind_time_unix = (datetime_vector.astype(np.int64) // 10**9).values

        #create a time vector that starts on May 20 if using the test data
        if run_test == True:
            ocean_time_unix = bin_time - pd.to_datetime('2020-01-01').timestamp() + pd.to_datetime('2018-05-20').timestamp()

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
                UTM_x_wind[i, j], UTM_y_wind[i, j], _, _ = utm.from_latlon(lat_mesh[i, j], lon_mesh[i, j],force_zone_number=utm_zone)

        #interpolate onto the GRID grid where the grid is determined by bin_x and bin_y
        #and the wind field is given by UTM_x_wind and UTM_y_wind
        #interpolate onto the grid

        #create meshgrids for bin_x and bin_y
        bin_x_mesh,bin_y_mesh = np.meshgrid(bin_x,bin_y)

        #Loop over and 
        #1. Create objects to store wind fields and sst fields for each model timestep
        #2. Find the closest neighbour in time in the wind field
        #3. Interpolate the wind field onto the model grid
        #4. Interpolate the sst field onto the model grid
        #5. Store in a pickle file

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
    else:
        #LOAD THE PICKLE FILE
        with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\model_grid\\interpolated_wind_sst_fields_test.pickle', 'rb') as f:
            ws_interp,sst_interp,bin_x_mesh,bin_y_mesh,ocean_time_unix = pickle.load(f)

    #---------------------------------------------#
    ### PLOT THE WIND AND SST DATA AND MAKE GIF ### 
    #---------------------------------------------#

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
    particles['bw'] = np.ma.zeros(particles['z'].shape)
    #Add the initial bandwidth to the particles at all timesteps
    particles['bw'][:,0] = initial_bandwidth
    #Add mask
    particles['bw'].mask = particles['lon'].mask

    ###################################################
    ###### ADD DICTIONARY ENTRY FOR PARTICLE AGE ######
    ###################################################

    initial_age = 0
    particles['age'] = np.ma.zeros(particles['z'].shape)
    #add mask
    particles['age'].mask = particles['z'].mask
    
    #########################################################
    ######### CALCULATE GAS TRANSFER VELOCITY FIELDS ########
    #########################################################

    #Calculate the gas transfer velocity
    calculate_new = False
    if calculate_new == True:
        GRID_gt_vel = calc_gt_vel(u10=ws_interp,
                                    temperature=sst_interp,
                                    dxy_grid=dxy_grid**2,
                                    gas='methane')
    
        #save the GRID_gt_vel
        with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\model_grid\\gt_vel\\GRID_gt_vel.pickle', 'wb') as f:
            pickle.dump(GRID_gt_vel, f)
    else:#load the GRID_gt_vel
        with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\model_grid\\gt_vel\\GRID_gt_vel.pickle', 'rb') as f:
            GRID_gt_vel = pickle.load(f)
        
    if plot_gt_vel == True:
        levels_gt = np.arange(np.round(np.nanmin(GRID_gt_vel))-0.2, np.round(np.nanmax(GRID_gt_vel))+0.2, 0.2)
        #do the same plot but just on lon lat coordinates
        #convert bin_x_mesh and bin_y_mesh to lon/lat
        lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh,bin_y_mesh,zone_number=33,zone_letter='V')
        colormap = 'magma'

        import imageio
        import matplotlib.gridspec as gridspec

        images_gt_vel = []
        time_steps = len(bin_time)

        #datetimevector
        times = pd.to_datetime(bin_time,unit='s')-pd.to_datetime('2020-01-01')+pd.to_datetime('2018-05-20')

        for i in range(time_steps):
            #Gas transfer velocity plot
            fig = plt.figure(figsize=(7, 7))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])  # Create a GridSpec object

            lons_zoomed = lon_mesh
            lats_zoomed = lat_mesh
            ws_zoomed = GRID_gt_vel[i,:,:]

            ax1 = plt.subplot(gs[0])  # Create the first subplot for the contour plot
            contourf = ax1.contourf(lons_zoomed, lats_zoomed, ws_zoomed, levels=levels_gt,cmap=colormap)
            cbar = plt.colorbar(contourf, ax=ax1)
            cbar.set_label('[$(m^2d)^{-1}$]')
            cbar.set_ticks(levels_gt[1:-1])
            ax1.set_title('Gas transfer velocity, '+str(times[i])[:10])
            contour = ax1.contour(lons_zoomed, lats_zoomed, ws_zoomed, levels = levels_gt, colors = 'w', linewidths = 0.2)
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
            images_gt_vel.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\gt_vel\\create_gif\\gt_vel'+str(i)+'.png'))
            plt.close()        

        #create gif
        imageio.mimsave('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\gt_vel\\create_gif\\gt_vel.gif', images_gt_vel, duration=0.5)

    #############################################################################################
    ################### END INITIAL CONDITIONS ### END INITIAL CONDITIONS #######################
    #############################################################################################

    #---------------------------------------------------#
    #####################################################
    #####  MODEL THE CONCENTRATION AT EACH TIMESTEP #####
    ### AKA THIS IS WHERE THE ACTUAL MODELING HAPPENS ###
    #####################################################
    #---------------------------------------------------#

    for j in range(1,len(particles['time'])-1): 
        
        print(j)

        #--------------------------------------#
        #MODIFY PARTICLE WEIGHTS AND BANDWIDTHS#
        #--------------------------------------#

        #Do some binning
        bin_z_number = np.digitize(np.abs(particles['z'][:,j]).compressed(),bin_z)

        # Get the indices where particles['age'][:,j] is not masked
        unmasked_indices = np.where(particles['z'][:,j].mask == False)
        # Get the indices where particles['age'][:,j] is not masked and equal to 0
        activated_indices = unmasked_indices[0][particles['age'][:,j][unmasked_indices] == 0]
        #already active indices
        already_active = unmasked_indices[0][particles['age'][:,j][unmasked_indices] != 0]

        ### ADD INITIAL WEIGHT IF THE PARTICLE HAS JUST BEEN ACTIVATED ###
        if run_test == True:
            if activated_indices.any(): #If there are new particles added at this timestep
                # Use these indices to modify the original particles['weight'] array
                particles['weight'][activated_indices,j] = np.round(np.exp((np.abs(particles['z'][:,j][activated_indices])+10)/44)) #moles per particle
                #same for bandwidth
                particles['bw'][activated_indices,j] = initial_bandwidth
        elif run_full == True:
            particles['weight'][particles['age'][:,j].mask==0] = weights_full_sim
            particles['bw'][particles['age'][:,j].mask==0] = initial_bandwidth

        ### MODIFY ALREADY ACTIVE ###
        #add the weight of the particle to the current timestep 
        if already_active.any():
            particles['weight'][already_active,j] = particles['weight'][already_active,j-1]#no change for now
            #add the bandwidth of the particle to the current timestep
            particles['bw'][already_active,j] = particles['bw'][already_active,j] + age_constant
            #limit the bandwidth to a maximum value
            particles['bw'][already_active,j][particles['bw'][already_active,j]>max_ker_bw] = max_ker_bw
    
        ### ADD AGE TO THE NEXT TIMESTEP ###
        particles['age'][unmasked_indices,j+1] = particles['age'][unmasked_indices,j] + 1

        #--------------------------------------------------#
        #FIGURE OUT WHERE PARTICLES ARE LOCATED IN THE GRID#
        #--------------------------------------------------#

        #Get sort indices
        sort_indices = np.argsort(bin_z_number)
        #sort
        bin_z_number = bin_z_number[sort_indices]
        #get indices where bin_z_number changes
        change_indices = np.where(np.diff(bin_z_number) != 0)[0]
        #Trigger if you want to loop through all depth layers
        if use_all_depth_layers == True:
            change_indices = np.array([0,len(bin_z_number)])
        
        #Define the [location_x,location_y,location_z,weight,bw] for the particle. This is the active particle matrix
        parts_active = [particles['UTM_x'][:,j].compressed()[sort_indices],
                        particles['UTM_y'][:,j].compressed()[sort_indices],
                        particles['z'][:,j].compressed()[sort_indices],
                        bin_z_number[sort_indices],
                        particles['weight'][:,j].compressed()[sort_indices],
                        particles['bw'][:,j].compressed()[sort_indices]]

        #-----------------------------------#
        #INITIATE FOR LOOP OVER DEPTH LAYERS#
        #-----------------------------------#

        #add one right hans side limit to change_indices
        change_indices = np.append(change_indices,len(bin_z_number))
        for i in range(0,len(change_indices)-1): #This essentially loops over all particles
            
            #-----------------------------------------------------------#
            #DEFINE ACTIVE GRID AND ACTIVE PARTICLES IN THIS DEPTH LAYER#
            #-----------------------------------------------------------#

            #Define GRID_active by decompressing GRID[j][i][:,:] 
            GRID_active = GRID[j][i][:,:].toarray()

            #Define active particle matrix in depth layer i
            parts_active_z = [parts_active[0][change_indices[i]:change_indices[i+1]],
                            parts_active[1][change_indices[i]:change_indices[i+1]],
                            parts_active[2][change_indices[i]:change_indices[i+1]],
                            parts_active[3][change_indices[i]:change_indices[i+1]],
                            parts_active[4][change_indices[i]:change_indices[i+1]],
                            parts_active[5][change_indices[i]:change_indices[i+1]]]

            #-----------------------------------------------------#
            #CALCULATE THE CONCENTRATION FIELD IN THE ACTIVE LAYER#
            #-----------------------------------------------------#

            GRID_active = kernel_matrix_2d_NOFLAT(parts_active_z[0],
                                        parts_active_z[1],
                                        bin_x,
                                        bin_y,
                                        parts_active_z[5],
                                        parts_active_z[4])

            GRID_active = GRID_active/(V_grid*1000) #Dividing by V_grid to get concentration in mol/kg
            #GRID_active = GRID_active.T #This just works

            #----------------------------#
            #PLOT THE CONCENTRATION FIELD#
            #----------------------------#

            bin_x_mesh,bin_y_mesh = np.meshgrid(bin_x,bin_y)
            #and for GRID_active
            images_conc = []
            levels_atm = np.linspace(0, 5*10**-9, 8)
            time_steps = len(bin_time)

            ### PLOTTING ###
            if plotting == True:
                
                fig = plt.figure(figsize=(14, 10))
                gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])  # Create a GridSpec object

                #obtain lon_mesh from bin_x_mesh and bin_y_mesh
                lon_mesh,lat_mesh = utm.to_latlon(bin_x_mesh,bin_y_mesh,zone_number=33,zone_letter='V')


                lons_zoomed = lon_mesh
                lats_zoomed = lat_mesh
                #make input a vertically averaged concentration
                ws_zoomed = GRID_active.T#convert to mmol/hr
                ws_zoomed = ws_zoomed*10**9 #convert to nmol/hr
                #calculate sum
                #ws_zoomed = np.sum(ws_zoomed,axis=0)                

                ax = fig.add_subplot(gs[0],projection=projection)

                # Add a filled contour plot with a lower zorder

                contourf = ax.contourf(lons_zoomed, lats_zoomed, ws_zoomed, levels=levels_atm,cmap=colormap,transform=ccrs.PlateCarree(), zorder=0)
                #cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                #plot a contour not on map projection¨
                #plot a transparent grey shade indicating the model domain

                # Add the colorbar to the new axes
                cbar = plt.colorbar(contourf, ax=ax)
                cbar.set_label('nmol/kg',fontsize = 16)
                cbar.set_ticks(levels_atm[1:-1])
                cbar.ax.tick_params(labelsize=14)
                #cbar = plt.colorbar(contourf, ax=ax)
                #cbar.set_label('Released methane [mol]')
                #cbar.set_ticks(levels_atm[1:-1])
                ax.set_title('Concentration in surface layer [nmol/kg]',fontsize=16)
                contour = ax.contour(lons_zoomed, lats_zoomed, ws_zoomed, levels = levels_atm, colors = 'w', linewidths = 0.2,transform=ccrs.PlateCarree(), zorder=1)
                # Add the land feature with a higher zorder
                ax.add_feature(cfeature.LAND, facecolor='0.2', zorder=2)
                # Add the coastline with a higher zorder
                ax.add_feature(cfeature.COASTLINE, zorder=3, color = 'white' ,linewidth = 0.5)
                # Set the geographical extent of the plot
                ax.set_extent([min_lon, max_lon-5, min_lat+0.5, max_lat-1.5])
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False ,linewidth=0.5, color='white', alpha=0.5, linestyle='--')
                gl.top_labels = True
                gl.left_labels = True
                gl.xlabel_style = {'size':14}
                gl.ylabel_style = {'size':14}

                ax2 = plt.subplot(gs[1])  # Create the second subplot for the progress bar
                fig.subplots_adjust(hspace=0.2)
                ax2.set_position([0.195,0.12,0.54558,0.03])
                ax2.set_xlim(0, time_steps)  # Set the limits to match the number of time steps
                #ax2.plot([i, i], [0, 1], color='w')  # Plot a vertical line at the current time step
                ax2.fill_between([0, i], [0, 0], [1, 1], color='grey')
                ax2.set_yticks([])  # Hide the y-axis ticks
                ax2.set_xticks([0,time_steps])  # Set the x-axis ticks at the start and end
                ax2.set_xticklabels(['May 20, 2018', 'June 20, 2018'],fontsize=16)  # Set the x-axis tick labels to the start and end time
                plt.show()
                #save the figure
                fig.savefig(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\results\concentration\horizontal_field_surface'+str(j)+'.png')

                #add the figure to the list of images
                images_conc.append(imageio.imread(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\results\concentration\horizontal_field_surface'+str(j)+'.png'))

                plt.close(fig)

            #-------------------------------#
            #CALCULATE ATMOSPHERIC FLUX/LOSS#
            #-------------------------------#

            #GRID_gt_vel = calc_gt_vel(u10=U_constant,temperature=T_constant,gas='methane')
            #CALCULATE ATMOSPHERIC FLUX (AFTER PREVIOUS TIMESTEP - OCCURS BETWEEN TIMESTEPS)
            
            if j != 0 and i == 0:
                GRID_atm_flux[j,:,:] = np.multiply(GRID_gt_vel[j,:,:].T,
                    (((GRID_active+background_ocean_conc)-atmospheric_conc))
                    )
                GRID_active = GRID_active - GRID_atm_flux[j,:,:]

            #-----------------------------------------#
            #CALCULATE LOSS DUE TO MICROBIAL OXIDATION#
            #-----------------------------------------#

            #Half life rates with sources per day:
            # 0.0014 Steinle et all., 2016 in North sea, 10 degrees
            # 0.05 Mau et al., 2017 West of Svalbard
            # <0.085 Gr~undker et al., 2021
            
            #0.02/(3600*24)
            #use just e-7 per second, that's kind of in the middle of the pack
            #R_ox = (10**-7)*3600 #half life per hour
        
            #loop over all depths and calculate the Mox consumption
            #for k in range(0,len(bin_z_number)):
            #    GRID[j][bin_z_number[k]][bin_x_number[i],bin_y_number[i]] = (
            #        GRID[j][bin_z_number[k]][bin_x_number[i],bin_y_number[i]] - 
            #        (GRID[j][bin_z_number[k]][bin_x_number[i],bin_y_number[i]]*R_ox)
            #    )


            #-----------------------------------#
            #ASSUME COMPLETE MIXING IN GRID CELL#
            #-----------------------------------#

            ### TRY WITHOUT THIS PART FIRST (BUT SHOULD BE ADDED IN LATER) ###
            #Set the weight to the average of the weights of the particles in the previous grid cell. 
            #if j != 0 and GRID_part[j-1][bin_z_number[i]][bin_x_number[i],bin_y_number[i]] > 1:
            #    particles['weight'][i,j] = (V_grid * 
            #                                GRID[j][bin_z_number[i]][bin_x_number[i],bin_y_number[i]])/(
            #                                    GRID_part[j-1] [bin_z_number[i]][bin_x_number[i],bin_y_number[i]]
            #                                )
            ###################################
            
            #-----------------------------------------#
            #CALCULATE CONCENTRATION IN THE GRID CELLS#
            #-----------------------------------------#

            #GRID[j][bin_z_number[i]][bin_x_number[i],bin_y_number[i]] += particles['weight'][i,j]/V_grid
            #Add the number of particles to the particles matrix (which keeps track of the amount of particles per grid cell). 
            #GRID_part[j][bin_z_number[i]][bin_x_number[i],bin_y_number[i]] += 1
        
        #Calculate the totaø atmospheric flux. 
        #GRID_atm_flux[j][1][:,:] = (GRID[j][1][:,:]-atmospheric_conc)*0.4

        #bin_z_number_old = bin_z_number

    #create a gif
    #imageio.mimsave(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\results\concentration\horizontal_field_gif.gif', images_conc, duration=0.5)


######################################################################################################
#----------------------------------------------------------------------------------------------------#
#####PLOTTINGSAVINGPLOTTINGSAVINGPLOTTINGSAVINGPLOTTINGSAVINGPLOTTINGSAVINGPLOTTINGSAVINGPLOTTING#####
#----------------------------------------------------------------------------------------------------#
######################################################################################################
    
#

    #Save GRID_atm_flux
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\\GRID_atm_flux.pickle', 'wb') as f:
        pickle.dump(GRID_atm_flux, f)

    #save a short textfile with the settings
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\settings.txt', 'w') as f:
        f.write('Settings for the test run\n')
        f.write('--------------------------------\n')
        f.write('Number of particles: '+str(6200)+'\n')
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

    #-------------------------------#
    #PLOT THE ATMOSPHERIC FLUX FIELD#
    #-------------------------------#

    #load the GRID_atm_flux
    with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_atm_flux.pickle', 'rb') as f:
        GRID_atm_flux = pickle.load(f)

    images_atm_rel = []
    time_steps = len(bin_time)
    levels_atm = np.linspace(np.nanmin(np.nanmin(GRID_atm_flux))*10**9,np.nanmax(np.nanmax(GRID_atm_flux))*10**9,8)

    #datetimevector for the progress bar
    times = pd.to_datetime(bin_time,unit='s')-pd.to_datetime('2020-01-01')+pd.to_datetime('2018-05-20')

    min_lon = np.min(lon_mesh)
    max_lon = np.max(lon_mesh)
    min_lat = np.min(lat_mesh)
    max_lat = np.max(lat_mesh)

    for i in range(time_steps):
        #Atmospheric release plot

        lons_zoomed = lon_mesh
        lats_zoomed = lat_mesh
        ws_zoomed = GRID_atm_flux[i,:,:]#convert to mmol/hr

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])  # Create a GridSpec object
        ax = fig.add_subplot(gs[0],projection=projection)

        # Add a filled contour plot with a lower zorder
        contourf = ax.contourf(lons_zoomed, lats_zoomed, ws_zoomed, levels=levels_atm,cmap=colormap,transform=ccrs.PlateCarree(), zorder=0)
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Released methane [mol/hr]',fontsize = 16)
        cbar.set_ticks(levels_atm[1:-1])
        cbar.ax.tick_params(labelsize=14)
        ax.set_title('Released methane [mmol/hr]',fontsize=16)
        contour = ax.contour(lons_zoomed, lats_zoomed, ws_zoomed, levels = levels_atm, colors = 'w', linewidths = 0.2,transform=ccrs.PlateCarree(), zorder=1)
        # Add the land feature with a higher zorder
        ax.add_feature(cfeature.LAND, facecolor='0.2', zorder=2)
        # Add the coastline with a higher zorder
        ax.add_feature(cfeature.COASTLINE, zorder=3, color = '0.6' ,linewidth = 0.5)
        # Set the geographical extent of the plot
        ax.set_extent([min_lon, max_lon-5, min_lat+0.5, max_lat-1.5])
        #Plot a red dot at the location of tromsø
        ax.plot(18.9553,69.6496,marker='o',color='red',markersize=5,transform=ccrs.PlateCarree())
        #with a text label
        ax.text(19.0553,69.58006,'Tromsø',transform=ccrs.PlateCarree(),color='white',fontsize=14)

        #Try to fix labels
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False ,linewidth=0.5, color='white', alpha=0.5, linestyle='--')
        gl.top_labels = True
        gl.left_labels = True
        gl.xlabel_style = {'size':14}
        gl.ylabel_style = {'size':14}

        ax2 = plt.subplot(gs[1])  # Create the second subplot for the progress bar
        fig.subplots_adjust(hspace=0.2)
        ax2.set_position([0.195,0.12,0.54558,0.03])
        ax2.set_xlim(0, time_steps)  # Set the limits to match the number of time steps
        #ax2.plot([i, i], [0, 1], color='w')  # Plot a vertical line at the current time step
        ax2.fill_between([0, i], [0, 0], [1, 1], color='grey')
        ax2.set_yticks([])  # Hide the y-axis ticks
        ax2.set_xticks([0,time_steps])  # Set the x-axis ticks at the start and end
        ax2.set_xticklabels(['May 20, 2018', 'June 20, 2018'],fontsize=16)  # Set the x-axis tick labels to the start and end time


        plt.savefig('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run\\atm_flux'+str(i)+'.png')
        images_atm_rel.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run\\atm_flux'+str(i)+'.png'))
        plt.close()

    #create gif
    imageio.mimsave('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run\\atm_flux.gif', images_atm_rel, duration=0.5)


    #set all values located above the line given by longitude=[13,15,17] and latitude = [71,72,73] to zero





    #Calculate the sum of all timesteps in GRID_atm_flux
    GRID_atm_flux_sum = np.nansum(GRID_atm_flux,axis=0)
    #total sum
    total_sum = np.nansum(np.nansum(GRID_atm_flux_sum*dxy_grid**2))
    GRID_atm_flux_sum = GRID_atm_flux_sum
    GRID_atm_flux_sum = GRID_atm_flux_sum*3600 #mol/m^2


    levels = np.linspace(np.nanmin(np.nanmin(GRID_atm_flux_sum)),np.nanmax(np.nanmax(GRID_atm_flux_sum)),8)
    lons_zoomed = lon_mesh
    lats_zoomed = lat_mesh
    ws_zoomed = GRID_atm_flux_sum

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(1, 1)  # Create a GridSpec object
    ax = fig.add_subplot(gs[0],projection=projection)

    # Add a filled contour plot with a lower zorder
    contourf = ax.contourf(lons_zoomed, lats_zoomed, ws_zoomed, levels=levels,cmap=colormap,transform=ccrs.PlateCarree(), zorder=0)
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Methane [mol/m$^2$]',fontsize = 16)
    cbar.set_ticks(levels[1:-1])
    cbar.ax.tick_params(labelsize=14)
    ax.set_title('Released methane [mol/m$^2$], total = '+str(np.round(total_sum,2))+' mol',fontsize=16)
    contour = ax.contour(lons_zoomed, lats_zoomed, ws_zoomed, levels = levels, colors = '0.9', linewidths = 0.2,transform=ccrs.PlateCarree(), zorder=1)
    # Add the land feature with a higher zorder
    ax.add_feature(cfeature.LAND, facecolor='0.2', zorder=2)
    # Add the coastline with a higher zorder
    ax.add_feature(cfeature.COASTLINE, zorder=3, color = '0.5' ,linewidth = 0.5)
    # Set the geographical extent of the plot
    ax.set_extent([min_lon, max_lon-5, min_lat+0.5, max_lat-1.5])
    #Plot a red dot at the location of tromsø
    ax.plot(18.9553,69.6496,marker='o',color='red',markersize=5,transform=ccrs.PlateCarree())
    #with a text label
    ax.text(19.0553,69.58006,'Tromsø',transform=ccrs.PlateCarree(),color='white',fontsize=12)
    #add text in lower right corner with the total sum
    #ax.text(18.5,68.5,'Total release= '+str(np.round(total_sum,2))+' mol',transform=ccrs.PlateCarree(),color='white',fontsize=14)

    #Try to fix labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True ,linewidth=0.5, color='white', alpha=0.5, linestyle='--')
    gl.top_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'size':14}
    gl.ylabel_style = {'size':14}

    plt.show()

    #save figure
    plt.savefig('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run\\atm_flux_sum.png')



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

'''