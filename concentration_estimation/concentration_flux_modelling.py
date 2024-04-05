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
#add folder with kde estimator
from kernel_density_estimator import kernel_matrix_2d


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
            #    x = UTM_x[z_indices == i,j]
                #Binned y coordinates:
            #    y = UTM_y[z_indices == i,j]
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
    Sc: Schmidt number
    u10: wind speed at 10 meters height
    temperature: temperature in degrees celcius in air(?)
    gas: string containing the name of the gas. Default is methane. Options are
    'methane', 'carbon dioxide', 'oxygen' and 'nitrogen'

    Output:
    k: gas transfer velocity

    '''

    #Calculate the Schmidt number
    Sc = calc_schmidt_number(T=temperature,gas=gas)

    #make this such that we can calculate the gas transfer velocity for the whole grid and just
    #grab the data... 
    #Calculate the gas transfer velocity
    k = 0.251 * u10**2 * (Sc/660)**(-0.5) #m/day

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

if __name__ == '__main__':

    #Just load the grid object to make it faster
    #with open('grid_object.pickle', 'rb') as f:
    #   GRID = pickle.load(f)

    datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_test.nc'#test dataset
    #datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_norkyst.nc'#real dataset
    particles = load_nc_data(datapath)

    #Add utm coordinates to the particles dictionary
    particles = add_utm(particles)

    #Set horizontal grid resolution
    dxy_grid = 5000 #m
    #Set vertical grid resolution
    dz_grid = 50 #m

run_everything = True
if run_everything == True:
    #Create a zero grid. This grid has all timesteps and all spatial locations, but is a sparse
    #grid, so hopefully it will not take up too much memory.
    GRID,bin_x,bin_y,bin_z,bin_time = create_grid(np.ma.filled(np.array(particles['time']),np.nan),
                                                [np.min(np.ma.filled(np.array(particles['UTM_x']),np.nan)),np.max(np.ma.filled(np.array(particles['UTM_x']),np.nan))],
                                                [np.min(np.ma.filled(np.array(particles['UTM_y']),np.nan)),np.max(np.ma.filled(np.array(particles['UTM_y']),np.nan))],
                                                np.max(np.abs(particles['z'])),
                                                savefile_path=False,
                                                resolution=np.array([5000,50]))

    #bin_x AND bin_y GIVES THE BIN EDGES IN METERS
    
    ### Try to fill the first sparse matrix with the horizontal field at the first time step and depth level
    #Get locations from the utm coordinates in the particles dictionary 
    #Get the grid parameters

    #Get the grid parameters
    #bin_x,bin_y,bin_z,bin_time = get_grid_params(particles)

    #Get the last timestep
    bin_time_number = np.digitize(particles['time'][0],bin_time)
    bin_time_number = len(particles['time'])-1

    ############################
    ###### FIRST TIMESTEP ######
    ############################

    #Get the utm coordinates of the particles in the first time step
    x = particles['UTM_x'][:,bin_time_number]
    y = particles['UTM_y'][:,bin_time_number]
    z = np.abs(particles['z'][:,bin_time_number])

    #Get the bin numbers for the particles
    bin_x_number = np.digitize(x.compressed(),bin_x)
    bin_y_number = np.digitize(y.compressed(),bin_y)
    bin_z_number = np.digitize(z.compressed(),bin_z)

    ######################################################################################
    ### CREATE A VERTICAL PROFILE AT MEASUREMENT LOCATIONS FITTING THE GRID RESOLUTION ###
    ######################################################################################

    vertical_profile = np.ones(bin_z.shape[0])
    #Should be an exponential with around 100 at the bottom and 10 at the surface
    vertical_profile = np.round(np.exp(np.arange(0,np.max(np.abs(particles['z'][:,0])+10),1)/44))
    #Create a matrix with the same size as particles['z'] and fill with the vertical profile depending
    #on the depth level where the particle was in its first active timestep

    #Create new dictionary entry with same size and mask as particles['z']:
    particles['weight'] = np.ma.zeros(particles['z'].shape)
    #add mask
    particles['weight'].mask = particles['z'].mask

    ##################################
    ### CALCULATE GRID CELL VOLUME ###
    ##################################

    grid_resolution = [5000,5000,50]
    V_grid = grid_resolution[0]*grid_resolution[1]*grid_resolution[2]

    ################################################
    ### WE DONT NEED THIS PART WITH KDE ESTIMATE ###
    ################################################
    #Have a sparse matrix which keeps track of the number of particles in each grid cell. 
    GRID_part = GRID
    #Establish a matrix for atmospheric flux which is the same size as GRID only with one depth layer
    GRID_atm_flux = np.array(GRID)[:,0] #this can be just an np array. 
    ################################################
    ################################################
    ################################################

    ############################
    ### DEFINE CONSTANTS ETC ###
    ############################

    #Atmospheric background concentration
    atmospheric_conc = ((44.64*2)/1000000) #mol/m3
    background_ocean_conc = 3e-09 #mol/m3

    #Oswald spøiboøotu coeffocient
    oswald_solu_coeff = 0.28 #(for methane)

    #Set wind speed and temperature to constant value for now
    U_constant = 5 #m/s
    T_constant = 10 #degrees celcius

    ############################################################
    ######### CALCULATE THE GAS TRANSFER VELOCITY FIELD ########
    ############################################################

    #Calculate the gas transfer velocity
    gas_transfer_vel = calc_gt_vel(u10=U_constant,temperature=T_constant,gas='methane')

    #############################################################
    ######### MODEL THE CONCENTRATION AT EACH TIMESTEP ##########
    #############################################################

    #WATCH OUT: PARTICLES['Z'] IS NEGATIVE DOWNWARDS
    #Fill the GRID with the horizontal field at all timesteps

    for j in range(1,len(particles['time']-1)): 
        #Calculate gas transfer velocity:
        #gas_transf_vel = 0.31 * (u10**2 + 0.5 * (Sc/660)**(-2/3)) #m/day
        
        print(j)
        
        #---------------------#
        #PUT PARTICLES IN BINS#
        #---------------------#

        bin_x_number = np.digitize(particles['UTM_x'][:,j].compressed(),bin_x)
        bin_y_number = np.digitize(particles['UTM_y'][:,j].compressed(),bin_y)
        bin_z_number = np.digitize(np.abs(particles['z'][:,j]).compressed(),bin_z)
        part_weights = particles['weight'][:,j].compressed() #the weight of the active particles
        #calculate the bandwidth for the particles
        bw = np.ones(len(bin_x_number))*20000
        #give the particles some weight
        parts_active = particles['weight'][:,j].compressed()
        parts_active += vertical_profile[bin_z_number]
        
        #--------------------------------------------------#
        #CALCULATE THE KDE ESTIMATE OF THE HORISONTAL FIELD#
        #--------------------------------------------------#
        #...using the above and the x_grid/y_grid coordinates given by bin_x and bin_y
        #and the kernel_matrix_2d function.

        z_kernelized = kernel_matrix_2d(particles['UTM_x'][:,j].compressed(),
                                        particles['UTM_y'][:,j].compressed(),
                                        bin_x,bin_y,bw,parts_active)

        #plot the results on the bin_x/bin_y grid to see if it looks good
        #create meshgrid
        bin_x_mesh,bin_y_mesh = np.meshgrid(bin_x,bin_y)
        #and for z_kernelized
        zz = z_kernelized
        N = 8 #number of contours
        fig, ax = plt.subplots()
        CS = ax.contourf(bin_x_mesh,bin_y_mesh,z_kernelized.T,N)
        cbar = fig.colorbar(CS)
        cbar.set_label('Concentration [mol/m3]')
        plt.show()


        #LOOP OVER ALL PARTICLES AND FILL THE GRID

        for i in range(0,len(bin_z_number)): #This essentially loops over all particles
            
            #-------------------------#
            #GIVE NEW PARTICLES WEIGHT#
            #-------------------------#

            #This is output from M2PG1       
            if particles['z'].mask[i,j] == False and particles['z'].mask[i,j-1] == True or j == 0:
                #use the round of the depth
                particles['weight'][i,j] = vertical_profile[
                    int(np.abs(particles['z'][i,j]))]
                
            #-------------------------------#
            #CALCULATE ATMOSPHERIC FLUX/LOSS#
            #-------------------------------#
                
            #CALCULATE ATMOSPHERIC FLUX (AFTER PREVIOUS TIMESTEP - OCCURS BETWEEN TIMESTEPS)
            GRID_atm_flux[j][bin_x_number[i],bin_y_number[i]] = gas_transfer_vel*(
                (((GRID[j][1][bin_x_number[i],bin_y_number[i]]+3e-09)-atmospheric_conc)
                ))
            #CALCULATE LOSS
            GRID[j][1][bin_x_number[i],bin_y_number[i]] = (
                GRID[j][1][bin_x_number[i],bin_y_number[i]] - 
                GRID_atm_flux[j][bin_x_number[i],bin_y_number[i]])

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


                
            GRID[j][bin_z_number[i]][bin_x_number[i],bin_y_number[i]] += particles['weight'][i,j]/V_grid
            #Add the number of particles to the particles matrix (which keeps track of the amount of particles per grid cell). 
            GRID_part[j][bin_z_number[i]][bin_x_number[i],bin_y_number[i]] += 1
        
        #Calculate the totaø atmospheric flux. 
        #GRID_atm_flux[j][1][:,:] = (GRID[j][1][:,:]-atmospheric_conc)*0.4

        bin_z_number_old = bin_z_number

    #numba. test. 

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

    from PIL import Image

    # Create a list to store the images
    images = []

    for i in range(0,len(GRID)):
        fig, ax = plt.subplots()
        im = ax.imshow(np.flipud(GRID[i][1].todense().T))
        im.set_cmap('viridis')
        im.set_clim(0,50)
        fig.colorbar(im, ax=ax)

        # Draw the figure first
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Normalize the data to the range [0, 255]
        data = ((data - data.min()) * (1/(data.max() - data.min()) * 255)).astype('uint8')

        images.append(Image.fromarray(data))

        plt.close(fig)

    imageio.mimsave(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\results\Concentration_plots_gifs\horizontal_field_0.gif', images, fps=5)





            















