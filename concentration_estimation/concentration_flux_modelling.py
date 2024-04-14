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
import seaborn as sns


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
dxy_grid = 1600. #m
dz_grid = 20. #m
#grid cell volume
V_grid = dxy_grid*dxy_grid*dz_grid
#age constant
age_constant = 150 #m per hour, see figure.
#Initial bandwidth
initial_bandwidth = 200 #m
#set colormap
#colromap = 'magma'
colormap = sns.color_palette("rocket", as_cmap=True)
#K value for the microbial oxidation (MOx) (see under mox section for more values)
R_ox = 10**-7 #s^-1
#total seabed release
total_seabed_release = 20833
#only for top layer trigger
kde_all = False
#Weight full sim
weights_full_sim = 0.0408 #mol/hr

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
                          dpi=150):
    '''
    Plots 2d data on a map with time progression bar

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

    Output:
    fig: figure object
    '''
    #Check if coordinate input is mesh or vector
    if np.shape(lon) != np.shape(data):
        lon, lat = np.meshgrid(lon, lat)
    
    #get map extent
    min_lon = np.min(lon)
    max_lon = np.max(lon)
    min_lat = np.min(lat)
    max_lat = np.max(lat)
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])  # Create a GridSpec object  # Create a GridSpec object
    ax = fig.add_subplot(gs[0],projection=projection)

    # Add a filled contour plot with a lower zorder
    contourf = ax.contourf(lon, lat, data, levels=levels,cmap=colormap,transform=ccrs.PlateCarree(), zorder=0)
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label(unit, fontsize=16)
    cbar.set_ticks(levels[1:-1])
    cbar.ax.tick_params(labelsize=14)
    ax.set_title(title,fontsize=16)
    contour = ax.contour(lon, lat, data, levels = levels, colors = '0.9', linewidths = 0.2,transform=ccrs.PlateCarree(), zorder=1)
    # Add the land feature with a higher zorder
    ax.add_feature(cfeature.LAND, facecolor='0.2', zorder=2)
    # Add the coastline with a higher zorder
    ax.add_feature(cfeature.COASTLINE, zorder=3, color = '0.5' ,linewidth = 0.5)
    # Set the geographical extent of the plot
    ax.set_extent([min_lon, max_lon-5, min_lat+0.5, max_lat-1.5])
    #Plot a red dot at the location of tromsø
    ax.plot(18.9553,69.6496,marker='o',color='white',markersize=5,transform=ccrs.PlateCarree())
    #with a text label
    ax.text(19.0553,69.58006,'Tromsø',transform=ccrs.PlateCarree(),color='white',fontsize=12)
    #add location marker for seepage
    ax.plot(14.29,68.9179949,marker='o',color='white',markersize=5,transform=ccrs.PlateCarree())
    #add text in lower right corner with the total sum
    ax.text(14.39,68.8479949,'Methane seeps',transform=ccrs.PlateCarree(),color='white',fontsize=12)

    #add text in lower right corner with the total sum
    #Try to fix labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False ,linewidth=0.5, color='white', alpha=0.5, linestyle='--')
    gl.top_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'size':14}
    gl.ylabel_style = {'size':14}
    #savefigure if savefile_path is given

    ax2 = plt.subplot(gs[1])  # Create the second subplot for the progress bar
    fig.subplots_adjust(hspace=0.2)
    ax2.set_position([0.195,0.12,0.54558,0.03])
    ax2.set_xlim(0, time_steps)  # Set the limits to match the number of time steps
    #ax2.plot([i, i], [0, 1], color='w')  # Plot a vertical line at the current time step
    ax2.fill_between([0, timepassed[0]], [0, 0], [1, 1], color='grey')
    ax2.set_yticks([])  # Hide the y-axis ticks
    ax2.set_xticks([0,timepassed[1]])  # Set the x-axis ticks at the start and end
    ax2.set_xticklabels(['May 20, 2018', 'June 20, 2018'],fontsize=16)  # Set the x-axis tick labels to the start and end time


    if savefile_path != False:
        plt.savefig(savefile_path,dpi=dpi)
    #show figure if show is True
    if show == True:
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
    if run_test == True:
        ocean_time_unix = bin_time - pd.to_datetime('2020-01-01').timestamp() + pd.to_datetime('2018-05-20').timestamp()
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
            UTM_x_wind[i, j], UTM_y_wind[i, j], _, _ = utm.from_latlon(lat_mesh[i, j], lon_mesh[i, j],force_zone_number=utm_zone)
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

#################################
########## INITIATION ###########
#################################

#if __name__ == '__main__':

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
    #number of particles
    n_particles = first_timestep_lon = ODdata.variables['lon'][:, 0]
    #Create an empty particles dictionary with size n_particlesx2 (one for current and one for previous timestep)
    particles = {'lon':np.ma.zeros((len(n_particles),2)),
                'lat':np.ma.zeros((len(n_particles),2)),
                'z':np.ma.zeros((len(n_particles),2)),}
    #first timestep:
    particles['lon'][:,0] = ODdata.variables['lon'][:, 0]
    particles['lat'][:,0] = ODdata.variables['lat'][:, 0]
    particles['z'][:,0] = ODdata.variables['z'][:, 0]
    particles['time'] = ODdata.variables['time'][:]
    #second timestep:
    particles['lon'][:,1] = ODdata.variables['lon'][:, 1]
    particles['lat'][:,1] = ODdata.variables['lat'][:, 1]
    particles['z'][:,1] = ODdata.variables['z'][:, 1]
    #add utm
    particles = add_utm(particles)
    #unmasked_first_timestep_lon = first_timestep_lon[~first_timestep_lon.mask]
    #set limits for grid manually since we dont know how this will evolv
    #loop over all timesteps to check the limits of the grid

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
    



    
    
    





        #find all indices in distance where the difference in horizontal distance is less than 
        #100 meters

    #get only the particles that are active (they are nonmasked)
    #particles = {'lon':ODdata.variables['lon'][unmasked_indices],
    #                'lat':ODdata.variables['lat'][unmasked_indices],
    #                'z':ODdata.variables['z'][unmasked_indices],
    #                'time':ODdata.variables['time'][unmasked_indices],
    #                'status':ODdata.variables['status'][unmasked_indices]}

    #force zone number to be 33
    minUTMxminUTMy = utm.from_latlon(minlat,minlon,force_zone_number=33)
    minUTMxmaxUTMy = utm.from_latlon(minlat,maxlon,force_zone_number=33)
    maxUTMxminUTMy = utm.from_latlon(maxlat,minlon,force_zone_number=33)
    maxUTMxmaxUTMy = utm.from_latlon(maxlat,maxlon,force_zone_number=33)

###### SET UP GRIDS FOR THE MODEL ######
print('Creating the output grid...')
#MODEELING OUTPUT GRID
if run_test == True:
    GRID,bin_x,bin_y,bin_z,bin_time = create_grid(np.ma.filled(np.array(particles['time']),np.nan),
                                                [np.max([100000-dxy_grid-1,np.min(particles['UTM_x'].compressed())]),np.min([np.max(particles['UTM_x'].compressed()),1000000-dxy_grid-1])],
                                                [np.max([dxy_grid+1,np.min(particles['UTM_y'].compressed())]),np.min([np.max(particles['UTM_y'].compressed()),10000000-dxy_grid-1])],
                                                np.max(np.abs(particles['z'])),
                                                savefile_path=False,
                                                resolution=np.array([dxy_grid,dz_grid]))
elif run_full == True:
    GRID,bin_x,bin_y,bin_z,bin_time = create_grid(np.ma.filled(np.array(particles['time']),np.nan),
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

#Create coordinates for plotting
bin_x_mesh,bin_y_mesh = np.meshgrid(bin_x,bin_y)
#And lon lat coordinates
lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh.T,bin_y_mesh.T,zone_number=33,zone_letter='V')

#Create datetime vector from bin_time
if run_test == True: 
    timedatetime = pd.to_datetime(bin_time,unit='s')-pd.to_datetime('2020-01-01')+pd.to_datetime('2018-05-20')
else:
    timedatetime = pd.to_datetime(bin_time,unit='s')

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

##################################
### CALCULATE GRID CELL VOLUME ###
##################################

grid_resolution = [dxy_grid,dxy_grid,dz_grid] #in meters
V_grid = grid_resolution[0]*grid_resolution[1]*grid_resolution[2]

###############################################
### LOAD AND/OR FIT WIND AND SST FIELD DATA ###
###############################################

if fit_wind_data == True:
    fit_wind_sst_data(bin_x,bin_y,bin_time,run_test=True)
    #LOAD THE PICKLE FILE (no matter what)
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\model_grid\\interpolated_wind_sst_fields_test.pickle', 'rb') as f:
    ws_interp,sst_interp,bin_x_mesh,bin_y_mesh,ocean_time_unix = pickle.load(f)

#########################################################
######### CALCULATE GAS TRANSFER VELOCITY FIELDS ########
#########################################################

#interpolate nans in the wind field and sst field
ws_interp = np.ma.filled(ws_interp,np.nan)
sst_interp = np.ma.filled(sst_interp,np.nan)

#Calculate the gas transfer velocity
if fit_gt_vel == True:
    print('Fitting gas transfer velocity')
    GRID_gt_vel = calc_gt_vel(u10=ws_interp,
                                temperature=sst_interp-273.15,
                                gas='methane')
    #replace any nans in the grid with nearest neighbour values in the grid using the nearest neighbour interpolation
    # Get the indices of the valid (non-NaN) and invalid (NaN) points
    valid_mask = ~np.isnan(GRID_gt_vel)
    invalid_mask = np.isnan(GRID_gt_vel)
    # Get the coordinates of the valid points
    valid_coords = np.array(np.nonzero(valid_mask)).T
    # Get the coordinates of the invalid points
    invalid_coords = np.array(np.nonzero(invalid_mask)).T
    # Get the values of the valid points
    valid_values = GRID_gt_vel[valid_mask]
    # Use griddata to interpolate the NaN values
    GRID_gt_vel[invalid_mask] = griddata(valid_coords, valid_values, invalid_coords, method='nearest')

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
#and for total particles
total_parts = np.zeros(len(bin_time))

#-----------------------------------------#
#CREATE A MATRIX FOR REMOVING THE DIAGONAL#
#-----------------------------------------#

#For test run only
if run_test == True: 
    diag_rm_mat = np.ones(np.shape(GRID[0][0][:,:]))   
    #remove diagonal and artifact area..... 
    diag_thr = 0.4
    #GRID_atm_flux_mm_m2 = np.flip(GRID_atm_flux_mm_m2,axis=1)
    for i in range(0,len(diag_rm_mat)):
        for j in range(0,len(diag_rm_mat[i])):
            if i < int(j*diag_thr):
                diag_rm_mat[i][j] = 0

        
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
for j in range(1,len(particles['time'])-1): 

    print(j)

    #----------------------------------------------------#
    #IF RUNNING THE WHOLE SIMULATION LOAD TIMESTEP J DATA#
    #----------------------------------------------------#
    
    if run_full == True:
        #LAZY LOAD FROM THE NC FILE
        #Remove previous previous timestep data from the dict
        particles['lon'][:,j-2] = []
        particles['lat'][:,j-2] = []
        particles['z'][:,j-2] = []
        particles['time'][:,j-2] = []
        particles['status'][:,j-2] = []
        #Load the new timestep data
        particles['lon'][:,j] = ODdata.variables['lon'][unmasked_indices,j]
        particles['lat'][:,j] = ODdata.variables['lat'][unmasked_indices,j]
        particles['z'][:,j] = ODdata.variables['z'][unmasked_indices,j]
        #particles['time'][:,j] = ODdata.variables['time'][unmasked_indices,j]
        #particles['status'][:,j] = ODdata.variables['status'][unmasked_indices,j]
        #Get the utm coordinates of the particles in the first time step
        





    #--------------------------------------#
    #MODIFY PARTICLE WEIGHTS AND BANDWIDTHS#
    #--------------------------------------#

    #do some binning
    #bin_z_number = np.digitize(np.abs(particles['z'][:,j]).compressed(),bin_z)
    #np.digitize(np.abs(particles['z'][:,j]),bin_z)

    #Unmask particles that were masked in the previous timestep
    #particles['z'][:,j].mask = particles['z'][:,j-1].mask
    # Get the indices where particles['age'][:,j] is not masked
    unmasked_indices = np.where(
        particles['z'][:,j].mask == False)
    #set the mask for weights to the same as mask for z
    if j == 234:
        print('stop')
    #do some binning on those
    bin_z_number = np.digitize(
        np.abs(particles['z'][:,j][unmasked_indices]),bin_z)
    # Get the indices where particles['age'][:,j] is not masked and equal to 0
    activated_indices = unmasked_indices[0][
        particles['age'][:,j][unmasked_indices] == 0]
    #already active indices
    already_active = unmasked_indices[0][
        particles['age'][:,j][unmasked_indices] != 0]
    

    ### ADD INITIAL WEIGHT IF THE PARTICLE HAS JUST BEEN ACTIVATED ###
    if run_test == True:
        if activated_indices.any(): #If there are new particles added at this timestep
            # Use these indices to modify the original particles['weight'] array
            particles['weight'][activated_indices,j] = (np.round(
                np.exp((np.abs(particles['z'][:,j][activated_indices]
                                )+10)/44)))*0.037586 #moles per particle
            #Make the mask false for all subsquent timesteps
            particles['weight'][activated_indices,j].mask = False
            #do this for all the maske
            #same for bandwidth
            particles['bw'][activated_indices,j] = initial_bandwidth
    elif run_full == True:
        particles['weight'][particles['age'][:,j].mask==0
                            ] = weights_full_sim
        particles['bw'][particles['age'][:,j].mask==0
                        ] = initial_bandwidth

    ### MODIFY ALREADY ACTIVE ###
    #add the weight of the particle to the current timestep 
    ### MODIFY ALREADY ACTIVE ###
    #add the weight of the particle to the current timestep 
    if already_active.any():
        particles['weight'][already_active,j] = particles[
            'weight'][already_active,j-1]-(particles['weight'][
                already_active,j-1]*R_ox*3600) #mol/hr
        particles_mox_loss[j] = np.nansum(particles['weight'][already_active,j-1]*R_ox*3600)
        #Find all particles located in the surface layer and create an index vector (to avoid double indexing numpy problem)
        already_active_surface = already_active[np.where(np.abs(particles['z'][
            already_active,j-1])<bin_z[1])] #Those who where there on the PREVIOUS time step.
       #find the gt_vel for the surface_layer_idxs
        gt_idys = np.digitize(np.abs(particles['UTM_y'][already_active_surface,j-1]),bin_y)
        gt_idxs = np.digitize(np.abs(particles['UTM_x'][already_active_surface,j-1]),bin_x)
        #make sure all gt_idys and gt_idxs are within the grid
        gt_idys[gt_idys >= len(bin_y)] = len(bin_y)-1
        gt_idxs[gt_idxs >= len(bin_x)] = len(bin_x)-1
        #Distribute the atmospheric loss on these particles depending on their weight
        #replace any nans in GRID_gt_cel[j-1][gt_idys,gt_idxs] with the nearest non nan value in the grid
        #GRID_gt_vel[j-1][gt_idys,gt_idxs] = np.nan_to_num(GRID_gt_vel[j-1][gt_idys,gt_idxs])
        #Each particle have contributed with a certain amount gt_vel*weight
        particleweighing = (particles['weight'][already_active_surface,j-1]*GRID_gt_vel[j-1][gt_idys,gt_idxs])/np.nansum(
            particles['weight'][already_active_surface,j-1]*GRID_gt_vel[j-1][gt_idys,gt_idxs])
        #particles['weight'][already_active,j][surface_layer_idx] = particles['weight'][already_active,j][surface_layer_idx] - (gt_vel_loss*particles['weight'][already_active,j-1][surface_layer_idx]*total_atm_flux[j-1])/np.nansum((gt_vel_loss*particles['weight'][already_active,j-1][surface_layer_idx])) #mol/hr
        particles['weight'][already_active_surface,j] = particles['weight'][already_active_surface,j] - (
            particleweighing*total_atm_flux[j-1])/np.nansum(particleweighing) #mol/hr
        particles_atm_loss[j] = np.nansum((particleweighing*total_atm_flux[j-1])/np.nansum(particleweighing))
        #weigh this with the gt_vel
        #USING THE TOTAL ATM FLUX HERE.. 
        #remove particles with weight less than 0
        #if particles['weight'][already_active,j][particles['weight'][already_active,j]<0].any():
        #    break
        particles['weight'][already_active,j][particles['weight'][already_active,j]<0] = 0
        particles['weight'][already_active,j][particles['weight'][already_active,j]<0] = 0
        #add the bandwidth of the particle to the current timestep
        particles['bw'][already_active,j] = particles['bw'][already_active,j] + age_constant
        #limit the bandwidth to a maximum value
        particles['bw'][already_active,j][particles['bw'][already_active,j]>max_ker_bw] = max_ker_bw

    ### ADD AGE TO THE NEXT TIMESTEP ###
    particles['age'][unmasked_indices,j+1] = particles['age'][unmasked_indices,j] + 1

    #--------------------------------------------------#
    #FIGURE OUT WHERE PARTICLES ARE LOCATED IN THE GRID#
    #--------------------------------------------------#
    
    #.... And create a sorted matrix for all the active particles according to
    #which depth layer they are currently located in. 
    '''
    #Get sort indices
    sort_indices = np.argsort(bin_z_number)
    #sort
    #bin_z_number = bin_z_number[sort_indices]
    #get indices where bin_z_number changes
    change_indices = np.where(np.diff(bin_z_number[sort_indices]) != 0)[0]
    #Trigger if you want to loop through all depth layers
    if use_all_depth_layers == True:
        change_indices = np.array([0,len(bin_z_number[sort_indices])])
    
    #Define the [location_x,location_y,location_z,weight,bw] for the particle. This is the active particle matrix
    parts_active = [particles['UTM_x'][:,j].compressed()[sort_indices],
                    particles['UTM_y'][:,j].compressed()[sort_indices],
                    particles['z'][:,j].compressed()[sort_indices],
                    bin_z_number[sort_indices],
                    particles['weight'][:,j].compressed()[sort_indices],
                    particles['bw'][:,j].compressed()[sort_indices]]

    #add one right hands side limit to change_indices

    #-----------------------------------#
    #INITIATE FOR LOOP OVER DEPTH LAYERS#
    #-----------------------------------#

    change_indices = np.append(change_indices,len(bin_z_number)-1)
    #add a zero at the begining
    change_indices = np.insert(change_indices,0,0)
    '''
    ########################################
    ############# degug ####################
    ########################################

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
                    bin_z_number,
                    particles['weight'][:,j].compressed()[sort_indices],
                    particles['bw'][:,j].compressed()[sort_indices]]
    
    #keep track of number of particles
    total_parts[j] = len(parts_active[0])
    
    #-----------------------------------#
    #INITIATE FOR LOOP OVER DEPTH LAYERS#
    #-----------------------------------#

    #add one right hands side limit to change_indices
    change_indices = np.append(change_indices,len(bin_z_number))

    ###########################################
    #############################################
    ###########################################

    for i in range(0,len(change_indices)-1): #This essentially loops over all particles
        
        #-----------------------------------------------------------#
        #DEFINE ACTIVE GRID AND ACTIVE PARTICLES IN THIS DEPTH LAYER#
        #-----------------------------------------------------------#

        #Define GRID_active by decompressing GRID[j][i][:,:] 
        GRID_active = GRID[j][i][:,:].toarray()

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

        #NEED TO ADD SOMETHING HERE THAT TAKES INTO ACCOUNT LOSS TO ATMOSPHERE AT 
        #THE PREVIOUS TIMESTEP

        #Set any particle that has left the model domain to have zero weight and location
        #at the model boundary
        parts_active_z[4][parts_active_z[0] < np.min(bin_x)] = 0
        parts_active_z[4][parts_active_z[0] > np.max(bin_x)] = 0
        parts_active_z[4][parts_active_z[1] < np.min(bin_y)] = 0
        parts_active_z[4][parts_active_z[1] > np.max(bin_y)] = 0
        parts_active_z[0][parts_active_z[0] < np.min(bin_x)] = np.min(bin_x)+1
        parts_active_z[0][parts_active_z[0] > np.max(bin_x)] = np.max(bin_x)-1
        parts_active_z[1][parts_active_z[1] < np.min(bin_y)] = np.min(bin_y)+1
        parts_active_z[1][parts_active_z[1] > np.max(bin_y)] = np.max(bin_y)-1

        if kde_all == True or i==0:
            GRID_active = kernel_matrix_2d_NOFLAT(parts_active_z[0],
                                        parts_active_z[1],
                                        bin_x,
                                        bin_y,
                                        parts_active_z[5],
                                        parts_active_z[4])

            GRID_active = diag_rm_mat*(GRID_active/(V_grid)) #Dividing by V_grid to get concentration in mol/m^3

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
                #image list for gif
                images_conc = []          
                plot_2d_data_map_loop(data = GRID_active.T*10**12,#converting to nmol/kg
                                    lon = lon_mesh,
                                    lat = lat_mesh,
                                    projection = ccrs.PlateCarree(),
                                    levels = levels_atm,
                                    timepassed = [j,time_steps],
                                    colormap = colormap,
                                    title = 'Concentration in surface layer [nmol/kg]',
                                    unit = 'nmol/kg',
                                    savefile_path = 'C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\concentration\\create_gif\\concentration'+str(j)+'.png',
                                    show = False,
                                    dpi = 90)
                #add the figure to the list of images
                images_conc.append(imageio.imread(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\results\concentration\create_gif\concentration'+str(j)+'.png'))
                plt.close(fig)

            #-------------------------------#
            #CALCULATE ATMOSPHERIC FLUX/LOSS#
            #-------------------------------#
            
            if j != 0 and i == 0:
                GRID_atm_flux[j,:,:] = np.multiply(GRID_gt_vel[j,:,:].T,
                    (((GRID_active+background_ocean_conc)-atmospheric_conc))
                    )*0.01*dxy_grid**2 #This is in mol/hr for each gridcell. The gt_vel is PER DAY
                GRID_active = (GRID_active*V_grid - GRID_atm_flux[j,:,:])/V_grid #We do this through weighing of particles, but need to account for loss on this ts as well
                total_atm_flux[j] = np.nansum(GRID_atm_flux[j,:,:])#....but not in the atmospheric flux.. 

            #-----------------------------------------#
            #CALCULATE LOSS DUE TO MICROBIAL OXIDATION#
            #-----------------------------------------#

            #Half life rates with sources per day:
            # 0.0014 Steinle et all., 2016 in North sea, 10 degrees
            # 0.05 Mau et al., 2017 West of Svalbard
            # <0.085 Gr~undker et al., 2021
            
            #0.02/(3600*24)
            #use just e-7 per second, that's kind of in the middle of the pack
            #R_ox = (10**-7) #half life per second
            #R_ox = (10**-7)*3600 #half life per hour

            GRID_mox[j,:,:] = GRID_active*(R_ox*3600*V_grid ) #need this to adjust the weights for next loop
            #need this to adjust the weights for next loop
            GRID_active = GRID_active - GRID_mox[j,:,:] #I adjust the weights instead. 
            total_mox[j] = np.nansum(GRID_mox[j,:,:])
            #update the weights of the particles due to microbial oxidation is done globally at each timestep since
            #the loss is not dependent on the depth layer (this is more efficient)

if plotting == True:
    #create a gif from the images
    imageio.mimsave(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\results\concentration\create_gif\concentration.gif', images_conc, duration=0.5)

#-----------------------------------#
#SAVE STUFF TO PICKLE FILES FOR LATER#
#-----------------------------------#

#Save the GRID, GRID_atm_flux and GRID_mox, ETC to pickle files
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID.pickle', 'wb') as f:
    pickle.dump(GRID, f)
    #create a sparse matrix first
#GRID_atm_sparse = csr_matrix(GRID_atm_flux)    
GRID_atm_sparse = GRID_atm_flux
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\diss_atm_flux\\test_run\\GRID_atm_flux.pickle', 'wb') as f:
    pickle.dump(GRID_atm_sparse, f)
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

######################################################################################################
#----------------------------------------------------------------------------------------------------#
###PLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTINGPLOTTING###
#----------------------------------------------------------------------------------------------------#
######################################################################################################

#Get grid on lon/lat and limits for the figures. 
lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh,bin_y_mesh,zone_number=33,zone_letter='V')

lons_zoomed = lon_mesh
lats_zoomed = lat_mesh

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

GRID_atm_flux = GRID_atm_flux

#Calculate atmospheric flux field per square meter per hour
GRID_atm_flux_mm_m2 = (GRID_atm_flux/(dxy_grid**2))*10**6 #convert to mmol/m^2/hr

images_atm_rel = []
time_steps = len(bin_time)
levels_atm = np.linspace(np.nanmin(np.nanmin(GRID_atm_flux_mm_m2)),np.nanmax(np.nanmax(GRID_atm_flux_mm_m2)),8)

#datetimevector for the progress bar
times = pd.to_datetime(bin_time,unit='s')-pd.to_datetime('2020-01-01')+pd.to_datetime('2018-05-20')

do = False
if do == True:
    for i in range(time_steps):  
        fig = plot_2d_data_map_loop(data = GRID_atm_flux_mm_m2[i,:,:].T*10**3*24*365,
                                    lon = lon_mesh,
                                    lat = lat_mesh,
                                    projection = projection,
                                    levels = levels_atm,
                                    timepassed = [i,time_steps],
                                    colormap = colormap,
                                    title = 'Atmospheric flux [nmol m$^{-2}$ yr$^{-1}$]'+str(times[i]),
                                    unit = 'nmol m$^{-2}$ yr$^{-1}$',
                                    savefile_path = 'C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run\\make_gif\\atm_flux'+str(i)+'.png',
                                    show = False,
                                    dpi = 90)
        images_atm_rel.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run\\make_gif\\atm_flux'+str(i)+'.png'))
        fig.close()

    #create gif
    imageio.mimsave('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run\\make_gif\\atm_flux.gif', images_atm_rel, duration=0.5)

#set all values located above the line given by longitude=[13,15,17] and latitude = [71,72,73] to zero

#Calculate the sum of all timesteps in GRID_atm_flux
GRID_atm_flux_sum = np.nansum(GRID_atm_flux,axis=0) #in moles
#set all negative values in GRID_atm_flux_sum to zero
#GRID_atm_flux_sum[GRID_atm_flux_sum<0] = 0
#total sum
total_sum = np.nansum(np.nansum(GRID_atm_flux_sum))#*dxy_grid

levels = np.linspace(np.nanmin(np.nanmin(GRID_atm_flux_sum)),np.nanmax(np.nanmax(GRID_atm_flux_sum)),8)
#levels = np.linspace(np.nanmin(np.nanmin(GRID_atm_flux_sum)),levels[4],8)
lons_zoomed = lon_mesh
lats_zoomed = lat_mesh
ws_zoomed = GRID_atm_flux_sum

percent_of_release = np.round(total_sum/total_seabed_release*100,3)

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(1, 1)  # Create a GridSpec object
ax = fig.add_subplot(gs[0],projection=projection)

#min_lon = 12
#max_lon = 24.5
#min_lat = 68.5
#max_lat = 71.5

# Add a filled contour plot with a lower zorder
contourf = ax.contourf(lons_zoomed, lats_zoomed, ws_zoomed.T, levels=levels,cmap=colormap,transform=ccrs.PlateCarree(), zorder=0,extend='max')
cbar = plt.colorbar(contourf, ax=ax)
cbar.set_label(r'Methane [mmol m$^{-2}$]', fontsize=16)
cbar.set_ticks(np.round(levels[1:-1],2))
cbar.ax.tick_params(labelsize=14)
ax.set_title(r'Released methane [mmol m$^{-2}$], total = '+str(np.round(total_sum,2))+' mol, $\sim'+str(percent_of_release)+'\%$',fontsize=16)
contour = ax.contour(lons_zoomed, lats_zoomed, ws_zoomed.T, levels = levels, colors = '0.9', linewidths = 0.2,transform=ccrs.PlateCarree(), zorder=1)
# Add the land feature with a higher zorder
ax.add_feature(cfeature.LAND, facecolor='0.2', zorder=2)
# Add the coastline with a higher zorder
ax.add_feature(cfeature.COASTLINE, zorder=3, color = '0.5' ,linewidth = 0.5)
# Set the geographical extent of the plot
ax.set_extent([min_lon, max_lon-5, min_lat+0.5, max_lat-1.5])
#Plot a red dot at the location of tromsø
ax.plot(18.9553,69.6496,marker='o',color='white',markersize=5,transform=ccrs.PlateCarree())
#with a text label
ax.text(19.0553,69.58006,'Tromsø',transform=ccrs.PlateCarree(),color='white',fontsize=12)
#add text in lower right corner with the total sum
#ax.text(18.5,68.5,'Total release= '+str(np.round(total_sum,2))+' mol',transform=ccrs.PlateCarree(),color='white',fontsize=14)
#add dot where the release is (at 14.29,68.9179949)
ax.plot(14.29,68.9179949,marker='o',color='white',markersize=5,transform=ccrs.PlateCarree())
#add text in lower right corner with the total sum
ax.text(14.39,68.8479949,'Seabed release',transform=ccrs.PlateCarree(),color='white',fontsize=12)

#Try to fix labels
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False ,linewidth=0.5, color='white', alpha=0.5, linestyle='--')
gl.top_labels = True
gl.left_labels = True
gl.xlabel_style = {'size':14}
gl.ylabel_style = {'size':14}

plt.show()

#save figure
#plt.savefig('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run\\atm_flux_sum.png')

#plt.show()

#plot total_atm_flux in a nice figure with nice labels etc
#create datetime vector
times = pd.to_datetime(bin_time,unit='s')-pd.to_datetime('2020-01-01')+pd.to_datetime('2018-05-20')
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(times,total_atm_flux,label='Total atmospheric flux',color='blue')
ax.set_ylabel('Total atmospheric flux [mol/hr]',fontdict={'fontsize':16})
ax.set_title('Total atmospheric flux',fontdict={'fontsize':16})
#set fontsize for the xticks
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',fontsize=14)
#add yy axis showing the number of active particles
ax2 = ax.twinx()
ax2.plot(times,total_parts,color='0.4',label='Number of active particles')
ax2.set_ylabel('Number of active particles',fontdict={'fontsize':16})
ax2.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',fontsize=14)
#save figure
plt.savefig('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\diss_atmospheric_flux\\test_run\\total_atm_flux.png')


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
    lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh,bin_y_mesh,zone_number=33,zone_letter='V')
    #datetimevector
    times = pd.to_datetime(bin_time,unit='s')-pd.to_datetime('2020-01-01')+pd.to_datetime('2018-05-20')

    images_wind = []
    images_sst = []
    time_steps = len(bin_time)

    for i in range(time_steps):
        #fig = plot_2d_data_map_loop(ws_interp[i,:,:],lon_mesh,
        #                lat_mesh,projection,levels_w,[i,time_steps],
        #                colormap,'Wind speed, '+str(times[i])[:10],
        #                'm s$^{-1}$',
        #                savefile_path='C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\wind\\create_gif\\wind'+str(i)+'.png',
        #                dpi=90)
        #append to gif list
        #images_wind.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\model_grid\\wind\\create_gif\\wind'+str(i)+'.png'))
        #plt.close(fig)

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
    lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh,bin_y_mesh,zone_number=33,zone_letter='V')
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
