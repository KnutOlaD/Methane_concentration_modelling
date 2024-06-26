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

    #example of usage
    #datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_test.nc'
    #particles = load_nc_data(datapath)



    return particles

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

def calc_atm_flux(C_o,C_a,Sc,u10=5,temperature=20,gas='methane'):
    ''' 
    Calculates the atmospheric flux of gas using the Wanninkhof (2014) formulation.

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

    #Calculate the gas transfer velocity
    k = 0.251 * u10**2 * (Sc/660)**(-0.5) #m/day

    #Calculate the atmospheric flux
    F = k * (C_o - C_a) #mol/m2/day

    return F



if __name__ == '__main__':

    #Just load the grid object to make it faster
    #with open('grid_object.pickle', 'rb') as f:
    #   GRID = pickle.load(f)

    datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_test.nc'
    particles = load_nc_data(datapath)

    #Add utm coordinates to the particles dictionary
    particles = add_utm(particles)

    #Create a zero grid
    GRID,bin_x,bin_y,bin_z,bin_time = create_grid(particles,
                                                savefile_path=False,
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
    vertical_profile = np.round(np.exp(np.arange(0,np.max(np.abs(particles['z'][:,0])+10),1)/44))
    #Create a matrix with the same size as particles['z'] and fill with the vertical profile depending
    #on the depth level where the particle was in its first active timestep

    #Create new dictionary entry with same size and mask as particles['z']:
    particles['weight'] = np.ma.zeros(particles['z'].shape)
    #add mask
    particles['weight'].mask = particles['z'].mask

    #Grid cell volume
    grid_resolution = [5000,5000,50]
    V_grid = grid_resolution[0]*grid_resolution[1]*grid_resolution[2]

    #Have a sparse matrix which keeps track of the number of particles in each grid cell. 
    GRID_part = GRID
    #Establish a matrix for atmospheric flux which is the same size as GRID only with one depth layer
    GRID_atm_flux = GRID[1][:]

    #Atmospheric background concentration
    atmospheric_conc = ((44.64*2)/1000000) #mol/m3

    #Oswald spøiboøotu coeffocient
    oswald_solu_coeff = 0.28 #(for methane)

    #Set wind speed and temperature to constant value for now
    U_constant = 5 #m/s
    T_constant = 10 #degrees celcius

    gas_transfer_vel = calc_gt_vel(u10=U_constant,temperature=T_constant,gas='methane')

    #WATCH OUT: PARTICLES['Z'] IS NEGATIVE DOWNWARDS
    #Fill the GRID with the horizontal field at all timesteps
    for j in range(0,len(particles['time']-1)): 
        #Calculate gas transfer velocity:
        gas_transf_vel = 0.31 * (u10**2 + 0.5 * (Sc/660)**(-2/3)) #m/day
        print(j)
        #Get the utm coordinates of the particles at time step j but only non-masked values
        bin_x_number = np.digitize(particles['UTM_x'][:,j].compressed(),bin_x)
        bin_y_number = np.digitize(particles['UTM_y'][:,j].compressed(),bin_y)
        bin_z_number = np.digitize(np.abs(particles['z'][:,j]).compressed(),bin_z)
        for i in range(0,len(bin_z_number)): #This essentially loops over all particles
            #Give initial weight to particles that just became active
            if particles['z'].mask[i,j] == False and particles['z'].mask[i,j-1] == True or j == 0:
                #use the round of the depth
                particles['weight'][i,j] = vertical_profile[
                    int(np.abs(particles['z'][i,j]))]
            
            #Set the weight to the average of the weights of the particles in the previous grid cell. 
            if j != 0 and GRID_part[j-1][bin_z_number[i]][bin_x_number[i],bin_y_number[i]] > 1:
                particles['weight'][i,j] = (V_grid * 
                                            GRID[j][bin_z_number[i]][bin_x_number[i],bin_y_number[i]])/(
                                                GRID_part[j-1] [bin_z_number[i]][bin_x_number[i],bin_y_number[i]]
                                            )
            
            #Apply loss and calculate and store atmospheric flux if particle was in the surface layer.
            if bin_z_number[i] == 1:
                #Calculate atmoshperic flux
                GRID_atm_flux[j][1][bin_x_number[i],bin_y_number[i]] = calc_atm_flux(
                    C_o=GRID[j][1][bin_x_number[i],bin_y_number[i]],
                    C_a=atmospheric_conc,
                    Sc=calc_schmidt_number(T=20,gas='methane'),
                    u10=U_constant,
                    temperature=temp_constant,
                    gas='methane')
                #Calculate the loss. Loss occurs at the same timestep as flux.
                GRID[j][1][bin_x_number[i],bin_y_number[i]] = (GRID[j][1][bin_x_number[i],bin_y_number[i]] - 
                                                                GRID_atm_flux[j][1][bin_x_number[i],bin_y_number[i]])
            
            #Add the vertical profile to the GRID
            #calculate the concentration of the grid cell. 
            GRID[j][bin_z_number[i]][bin_x_number[i],bin_y_number[i]] += particles['weight'][i,j]/V_grid
            #Add the number of particles to the particles matrix. 
            GRID_part[j][bin_z_number[i]][bin_x_number[i],bin_y_number[i]] += 1
        
        #Calculate the atmospheric flux. 
        GRID_atm_flux[j][1][:,:] = (GRID[j][1][:,:]-atm_background)*0.4

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





            