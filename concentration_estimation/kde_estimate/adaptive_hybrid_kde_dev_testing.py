'''
Script for testing the adaptive hydrid Kernel Density estimator

Author: Knut Ola Dølven

'''

import numpy as np
import matplotlib.pyplot as plt
#import sns to make use of the colormaps there
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import numba
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.ndimage import generic_filter

#Triggers
create_test_data = False
load_test_data = True
plotting = True

# Parameters
grid_size = 150
grid_size_plot = 100
x_grid = np.linspace(0, grid_size, grid_size)
y_grid = np.linspace(0, grid_size, grid_size)
frac_diff = 1000 #the fractinoal difference between the full dataset and the test dataset

num_particles_per_timestep = 5000 #seeded one after the other
time_steps = 380
dt = 0.1
stdev = 1.4 #Stochastic std
U_a = np.array([2.5, 2.5]) #Advection velocity
num_particles = num_particles_per_timestep*time_steps

#Change the default colormap to something bright other than viridis
#use rocket from sns
plt.rcParams['image.cmap'] = 'rocket'


# ------------------------------------------------------- #
###########################################################
##################### FUNCTIONS ###########################
###########################################################
# ------------------------------------------------------- #

from scipy.signal import correlate2d

def calc_integral_length_scale(data):
    '''
    Calculates the integral length scale of a 2D dataset using spatial autocorrelation.
    '''
    # Subtract the mean to get anomalies
    data_anomaly = data - np.mean(data)
    
    # Calculate the 2D autocorrelation function (ACF)
    acf = correlate2d(data_anomaly, data_anomaly, mode='full', boundary='wrap')
    
    # Normalize the ACF
    acf /= np.max(acf)
    
    # Find the distance from the center of the ACF to where it first falls below 1/e
    center = np.array(acf.shape) // 2
    distance = np.linalg.norm(np.indices(acf.shape).T - center, axis=-1)
    
    # Estimate the integral length scale as the area under the ACF
    integral_length_scale = np.sum(acf * distance) / np.sum(acf)
    
    return integral_length_scale

@numba.jit(nopython=True)#, parallel=True)
def histogram_estimator_numba(particles, grid_size, weights=None, bandwidths=None):
    '''
    Input:
    particles: np.array of shape (num_particles, 2)
    grid_size: int
    weights: np.array of shape (num_particles,)
    bandwidths: np.array of shape (num_particles,)
    
    Output:
    particle_count: np.array of shape (grid_size, grid_size)
    total_weight: np.array of shape (grid_size, grid_size)
    cell_bandwidths: np.array of shape (grid_size, grid_size)
    '''
    # Initialize the histograms
    particle_count = np.zeros((grid_size, grid_size), dtype=np.int32)
    total_weight = np.zeros((grid_size, grid_size), dtype=np.float64)
    cell_bandwidths = np.zeros((grid_size, grid_size), dtype=np.float64)
    
    # Check if weights are provided
    if weights is None:
        weights = np.ones(len(particles), dtype=np.float64)
    # Check if bandwidths are provided
    if bandwidths is None:
        bandwidths = np.ones(len(particles), dtype=np.float64)
    
    # Create a 2D histogram of particle positions
    for i in numba.prange(len(particles)):
        x, y = particles[i, :]
        if np.isnan(x) or np.isnan(y):
            continue
        x = int(x)
        y = int(y)
        if x >= grid_size or y >= grid_size or x < 0 or y < 0:
            continue
        total_weight[y, x] += weights[i]
        particle_count[y, x] += 1
        cell_bandwidths[y,x] += bandwidths[i]
    
    #Divide cell_bandwidth with particle count to obtain the average bandwidth
    cell_bandwidths = cell_bandwidths/particle_count

    return total_weight, particle_count, cell_bandwidths

#Function to calculate the grid projected kernel density estimator
def grid_proj_kde(grid_x, grid_y, kde_pilot, gaussian_kernels, kernel_bandwidths, cell_bandwidths):
    """
    Projects a kernel density estimate (KDE) onto a grid using Gaussian kernels.

    Parameters:
    grid_x (np.array): Array of grid cell boundaries in the x-direction.
    grid_y (np.array): Array of grid cell boundaries in the y-direction.
    kde_pilot (np.array): The pilot KDE values on the grid.
    gaussian_kernels (list): List of Gaussian kernel matrices.
    kernel_bandwidths (np.array): Array of bandwidths associated with each Gaussian kernel.
    cell_bandwidths (np.array): Array of bandwidths of the particles.

    Returns:
    np.array: The resulting KDE projected onto the grid.

    Notes:
    - This function only works with a simple histogram estimator as the pilot KDE.
    - The function assumes that the Gaussian kernels are symmetric around their center.
    - The grid size is determined by the lengths of grid_x and grid_y.
    - The function iterates over non-zero values in the pilot KDE and applies the corresponding Gaussian kernel.
    - The appropriate Gaussian kernel is selected based on the bandwidth of each particle.
    - The resulting KDE is accumulated in the output grid n_u.
    """
    # ONLY WORKS WITH SIMPLE HISTOGRAM ESTIMATOR ESTIMATE AS PILOT KDE!!!
    
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
        kernel_index = np.argmin(np.abs(kernel_bandwidths - cell_bandwidths[i, j]))
        # kernel_index = kernel_indices[i * grid_size + j]
        kernel = gaussian_kernels[kernel_index]
        kernel_size = len(kernel) // 2  # Because it's symmetric around the center.

        # Define the window boundaries
        i_min = max(i - kernel_size, 0)
        i_max = min(i + kernel_size + 1, gridsize_x)
        j_min = max(j - kernel_size, 0)
        j_max = min(j + kernel_size + 1, gridsize_y)

        # Calculate the weighted kernel
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
    Adaptive: if True, the kernel grid will be adaptive to the particle density at each point

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
        #Adapt the bandwidth if adaptive is not False    
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
        kernel_matrix = ((1/(2*np.pi*bw[i]*bw[i]))*np.exp(-0.5*((a/bw[i])**2+(b/bw[i])**2)))/np.sum(((1/(2*np.pi*bw[i]*bw[i]))*np.exp(-0.5*((a/bw[i])**2+(b/bw[i])**2))))
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

# Function to update particle positions
def update_positions(particles, U_a, stdev, dt):
    '''
    Input:
    particles: np.array of shape (num_particles, 2)
    U_a: np.array of shape (2,)

    '''
    # Advective term
    advective_displacement = U_a * dt
    
    # Stochastic term
    stochastic_displacement = np.random.normal(0, stdev, particles.shape) * np.sqrt(dt)
    
    # Update positions
    particles += advective_displacement + stochastic_displacement
    
    #particles = np.mod(particles, grid_size)
    
    return particles

    
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


#
#Create test data function
#
def create_test_data(stdev=1.4, num_particles_per_timestep=5000, time_steps=380, dt=0.1, grid_size=150, num_particles=5000):
    #Creates a bunch of particles and the time dependant bandwidth parameter. 
    #returns the particle trajectories and the bandwidth vector
    # Release position
    release_position = np.array([10, 10])
    #Make U_a a periodic function with size time_steps
    U_a = [0,5] #Initial value
    #Initial magnitude
    magU = np.sqrt(U_a[0]**2+U_a[1]**2)
    U_a = np.tile(U_a, (time_steps, 1))
    for i in range(1, time_steps):
        U_a[i][:][0] = 2*magU+ np.sin(i/50)*2*magU
        #make it a bit more complex by adding another sine function with different
        #frequency
        #U_a[i][:][1] = 2*magU+ np.sin(i/50)*2*magU + np.sin(i/10)*2*magU
        print(np.sin(i/10))
        #L2 normalize the velocity
        U_a[i] = (U_a[i]/(np.sqrt(U_a[i][0]**2+U_a[i][1]**2)))*magU #Concervation of mass

    # Simulate particle trajectories
    trajectories = np.zeros((time_steps-1, num_particles, 2))*np.nan
    #create the bandwidth vector for each particle
    bw = np.ones(num_particles)*0

    for t in range(time_steps-1):
        if t == 0:
            # Initialize particle matrix at first timestep
            particles = np.ones([num_particles_per_timestep, 2])*release_position
        else:
            particles_old = particles

            #REMOVING PARTICLES THAT HAVE LEFT THE GRID#
            #particles_old[np.where(particles_old[:,0] >= grid_size)] = np.nan
            #particles_old[np.where(particles_old[:,1] >= grid_size)] = np.nan
            #particles_old[np.where(particles_old[:,0] < 0)] = np.nan
            #particles_old[np.where(particles_old[:,1] < 0)] = np.nan
            #if np.sum(np.isnan(particles_old)) > 0:
            #    print('Particles have left the domain')
            #    particles_left += np.sum(np.isnan(particles_old))
            #############################################

            # add particles to the particle array
            particles = np.ones([num_particles_per_timestep*(t+1), 2])*release_position
            # add in the old particle positions to the new array
            particles[:num_particles_per_timestep*(t)] = particles_old
            #Set particles that has left the dodmain to nan
            #Update the bw vector

        particles = update_positions(particles, U_a[t], stdev, dt)
        trajectories[t,:len(particles)] = particles
        bw[:len(particles)] = bw[:len(particles)] + np.sqrt(stdev*0.001)
        #limit bw to a maximum value
        #bw[bw > 20] = 20
    with open('trajectories_full.pkl', 'wb') as f:
        pickle.dump(trajectories[-1], f)
    with open('bw.pkl', 'wb') as f:
        pickle.dump(bw, f)
    
    return trajectories, bw

def silvermans_simple_2d(n,dim):
    '''
    Calculate the Silverman's multiplication factor using the number of datapoints and the dimensionality of the data
    '''
    silvermans_factor = (4/(dim+2))**(1/(dim+4))*n**(-1/(dim+4))
    return silvermans_factor

def silvermans_h(data,dim):
    '''
    Calculate the Silverman's multiplication factor using the covariance matrix and 
    the number o datapoints
    '''
    #remove nans
    data = data[~np.isnan(data).any(axis=1)]
    # Calculate the covariance matrix - this represents the spread of the data
    cov_matrix = np.cov(data, rowvar=False)
    # Calculate the determinant of the covariance matrix - this represents the volume of the data
    det_cov = np.linalg.det(cov_matrix)
    # Calculate the number of data points
    n = len(data)
    # Calculate neff
    silvermans_factor = (n*(dim+2)/4.)**(-1./(dim+4))
    #multiply with the covariance matrix 
    silvermans_factor = silvermans_factor
    return silvermans_factor

def get_test_data(load_test_data=True,frac_diff = 1000,weights = 'log_weights'):
    if load_test_data == True:
        # Load test data
        with open(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\src\Dispersion_modelling\concentration_estimation\kde_estimate\trajectories_full.pkl', 'rb') as f:
            trajectories_full = pickle.load(f)
        with open(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\src\Dispersion_modelling\concentration_estimation\kde_estimate\bw.pkl', 'rb') as f:
            bw = pickle.load(f)
    else:
        # Create test data
        trajectories_full, bw = create_test_data()

    trajectories = trajectories_full[::frac_diff,:]
    #pick only the right data from the bw vector
    bw = bw[::frac_diff]

    if weights == 'log_weights':
        weights = 1-np.log(np.linspace(1,100,len(trajectories_full)))/(np.log(100)*2)
        weights_test = weights[::frac_diff]

    return trajectories, trajectories_full, bw, weights, weights_test


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
    np.add.at(total_weight, (y_indices, x_indices), weights)
    np.add.at(particle_count, (y_indices, x_indices), 1)
    np.add.at(cell_bandwidth, (y_indices, x_indices), bandwidths * weights)

    cell_bandwidth = np.divide(cell_bandwidth, total_weight, out=np.zeros_like(cell_bandwidth), where=total_weight!=0)

    return particle_count, total_weight, cell_bandwidth

#create random binned data
binned_data = np.random.randint(0,10,(10,10))
window_size = 2

def histogram_variance(binned_data, window_size):
    '''
    Calculate the simple variance of the binned data using ...
    '''
    #check that there's data in the binned data
    if np.sum(binned_data) == 0:
        return 0
    #get the central value of all bins
    grid_size = len(binned_data)
    #Central point of all grid cells
    X = np.arange(0,grid_size*window_size,window_size)
    Y = np.arange(0,grid_size*window_size,window_size)
    #Calculate the average position in the binned data
    mu_x = np.sum(binned_data*X)/np.sum(binned_data)
    mu_y = np.sum(binned_data*Y)/np.sum(binned_data)
    #Calculate the variance
    var_y = np.sum(binned_data*(X-mu_x)**2)/np.sum(binned_data)
    var_x = np.sum(binned_data*(Y-mu_y)**2)/np.sum(binned_data)
    #Calculate the covariance
    cov_xy = np.sum(binned_data*(X-mu_x)*(Y-mu_y))/np.sum(binned_data)
    #Calculate the total variance
    variance_data = var_x+var_y+2*cov_xy
    return variance_data

# ------------------------------------------------------- #
###########################################################
##################### INITIATION ##########################
###########################################################
# ------------------------------------------------------- #
# Add folder path
#get the test data
trajectories, trajectories_full, bw, weights, weights_test = get_test_data(load_test_data=load_test_data,frac_diff=frac_diff)
bw_full = np.ones(len(trajectories_full))
weights_full = weights

grid_size = 100
#Get the grid
x_grid = np.linspace(0, grid_size, grid_size)
y_grid = np.linspace(0, grid_size, grid_size)
X,Y = np.meshgrid(x_grid,y_grid)
#GROUND TRUTH POSITIONS
p_full_x = trajectories_full[:,0] #the full x particle positions positions
p_full_y = trajectories_full[:,1]
#TEST POSITIONS
p_x = trajectories[:,0] #the test x particle positions positions
p_y = trajectories[:,1]
#OBTAIN GAUSSIAN KERNELS
num_kernels = 25
ratio = 1/3
gaussian_kernels, kernel_bandwidths = generate_gaussian_kernels(num_kernels, ratio)

######################################
############ GROUND TRUTH ############
######################################

ground_truth,count_truth,bandwidths_placeholder = histogram_estimator(p_full_x,p_full_y, x_grid,y_grid,bandwidths=bw_full,weights=weights_full
)/np.sum(histogram_estimator(p_full_x,p_full_y, x_grid,y_grid,bandwidths=bw_full,weights=weights_full))   

#make a plot of the ground truth
plt.figure()
plt.imshow(ground_truth)
plt.colorbar()
plt.title('Ground truth')
plt.show()

################################################
########### NAIVE HISTOGRAM ESTIMATE ###########
################################################

naive_estimate,count_naive,cell_bandwidths = histogram_estimator(p_x,p_y, x_grid,y_grid,bandwidths=bw,weights=weights_test
)/np.sum(histogram_estimator(p_x,p_y, x_grid,y_grid,bandwidths=bw,weights=weights_test))

#make a plot of the naive estimate
plt.figure()
plt.imshow(naive_estimate)
plt.colorbar()
plt.title('Naive histogram estimate')
plt.show()

#########################################################
########### TIME VARYING BANDWIDTH ESTIMATE #############
#########################################################

#We already have a time varying bandwidth in the preloaded data (bw) and the naive prebin estimate
#we also have gaussian_kernels and associated kernel_bandwidths. We can now use
#the grid_proj_kde directly to get the kde estimate but we need to divide the cell_bandwidths
#with the particle count to get the average bandwidth in each cell using np.divide

pre_estimate,count_pre,cell_bandwidths = histogram_estimator(p_x,p_y, x_grid,y_grid,bandwidths=bw,weights=weights_test)
kde_time_bw = grid_proj_kde(x_grid, y_grid, naive_estimate, gaussian_kernels, kernel_bandwidths, cell_bandwidths)

#make a plot of the time dependent bandwidth estimate
plt.figure()
plt.imshow(kde_time_bw)
plt.colorbar()
plt.title('Time dependent bandwidth estimate')
plt.show()

########################################################
########### DATA DRIVEN BANDWIDTH ESTIMATE #############
########################################################

#Apply the histogram_variance function to all cells in the naive estimate



#bin all particles
histogram_prebinned,counts,cell_bandwidths = histogram_estimator(trajectories, grid_size,weights=weights_test)

#histogram_prebinned, x_edges, y_edges = np.histogram2d(trajectories[:,1],trajectories[:,0],bins=[x_grid,y_grid])

#calculate density and standard deviation in each cell
from scipy.ndimage import generic_filter
import numpy.random as npr

def local_counts(data):
    return np.sum(data)

def local_std(data):
    #data = data[data>0]
    window_mean = np.mean(data[data>0])
    return np.sqrt(np.sum((data - window_mean)**2))



def local_numcounts(data):
    return np.sum(data>0)
            
    

p_size = 7

local_counts_est = np.zeros(np.shape(histogram_prebinned))    
local_counts_est[counts>0] = generic_filter(counts, local_counts, size=(p_size,p_size))[counts>0]
#plt.imshow(local_counts_est)
print(len(local_counts_est[local_counts_est>0]))

local_std_est = np.zeros(np.shape(histogram_prebinned))
local_std_est[counts>0] = (generic_filter(counts, local_std, size=(p_size,p_size))[counts>0])/local_counts_est[counts>0]
print(len(local_std_est[local_std_est>0]))
#add noise lost during the binning process
#grid cell size
dxy = x_grid[1]-x_grid[0]
#add noise to the std



plt.imshow(local_std_est)


#calculate the silvermans factor for each non-zero value cell
h = np.zeros(np.shape(local_counts_est))

h[local_counts_est>0] = silvermans_simple_2d(local_counts_est[local_counts_est>0],2)*local_std_est[local_counts_est>0]


plt.imshow(h*2)
plt.colorbar()
plt.title('Adaptive bandwidth estimate')
plt.show()

#calculate the bandwidth for each cell


############################################################
################## CALCULATE KERNEL DENSITY ################
############################################################

#...using grid_projected_kde
kde_pilot = histogram_prebinned/np.sum(histogram_prebinned)
x_grid = np.linspace(0, 10, grid_size)
ratio = 1/3
num_kernels = 15
gaussian_kernels_test, kernel_bandwidths, kernel_origin = generate_gaussian_kernels(x_grid, num_kernels, ratio)

cell_bandwidths = h*10

n_u = grid_proj_kde(grid_size, 
                    kde_pilot, 
                    gaussian_kernels_test,
                    kernel_bandwidths,
                    cell_bandwidths)

plt.figure()
plt.imshow(n_u)
plt.colorbar()
plt.title('Grid projected KDE using adaptive ...')



### CALCULATE USING THE SCIPY GAUSSIAN KDE FUNCTION ###
kde_scipy_silverman = gaussian_kde(trajectories.T,bw_method = 'silverman',weights=weights_test)
kde_scipy_silverman = np.reshape(kde_scipy_silverman(np.vstack([X.ravel(), Y.ravel()])), X.shape)

plt.figure()
plt.imshow(kde_scipy_silverman)
plt.colorbar()
plt.title('Gaussian KDE using scipy Silverman constant bandwidth')

ground_truth,count_truth,bandwidths_placeholder = histogram_estimator(trajectories_full, grid_size,weights=weights)



######################################################################
######## GRID PROJECTED NON-ADAPTIVE KERNEL DENSITY ESTIMATOR ########
######################################################################

#make an array of arbitrary bandwidths for testing
particle_bandwidths = np.ones(num_particles)*1
for i in range(1,num_particles):
    particle_bandwidths[i] = particle_bandwidths[i-1]+0.00001

#flipt it upside down
particle_bandwidths = particle_bandwidths[::-1]


############################################################
##### GRID PROJECTED ADAPTIVE KERNEL DENSITY ESTIMATOR #####
############################################################

grid_size = 100  # Example grid size

histogram_est,kde_pilot,cell_bandwidths = histogram_estimator(trajectories, grid_size,weights=weights_test,bandwidths=particle_bandwidths[::frac_diff])

x_grid = np.linspace(0, 10, grid_size)
ratio = 1/3
gaussian_kernels_test, kernel_bandwidths, kernel_origin = generate_gaussian_kernels(x_grid, num_kernels, ratio)
cell_bandwidths = cell_bandwidths

n_u = grid_proj_kde(grid_size, 
                    kde_pilot, 
                    gaussian_kernels_test,
                    kernel_bandwidths,
                    cell_bandwidths)

plt.figure()
plt.imshow(n_u)
plt.title('Grid projected KDE using time dependent bandwidth')
plt.colorbar()


#plot the ground truth
plt.figure()
plt.imshow(ground_truth)
plt.title('Ground truth')
plt.colorbar()


#Histogram estimator estimate
histogram_estimator_est,counts,cell_bandwidths = histogram_estimator(trajectories, grid_size,weights=weights_test)
#Kernel density estimator with time dependent bandwidth estimate
kernel_density_estimator_est = kernel_matrix_2d_NOFLAT(trajectories[:,0].T,trajectories[:,1].T,x_grid,y_grid,bw,weights_test)
kernel_density_estimator_est = kernel_density_estimator_est.T

#Normalize everything
histogram_estimator_est = histogram_estimator_est/np.sum(histogram_estimator_est)
kernel_density_estimator_est = kernel_density_estimator_est/np.sum(kernel_density_estimator_est)
ground_truth,count_truth,bandwidths_placeholder = histogram_estimator(trajectories_full, grid_size,weights=weights)/np.sum(histogram_estimator(trajectories_full, grid_size,weights=weights))


#####################################################
##### CALCULATE PILOT KDE USING PREMADE PACKAGE #####
#####################################################

#create a 2d grid
X, Y = np.meshgrid(x_grid, y_grid)
#...using vstack
positions = np.vstack([X.ravel(), Y.ravel()])

# Sample particle positions
values = np.array(trajectories.T)  
#remove nans in values
weights = weights_test[~np.isnan(values).any(axis=0)]
values = values[:,~np.isnan(values).any(axis=0)]
kde_pilot = gaussian_kde(values,bw_method = 'silverman',weights = weights)
#with constant bandwidth
kde_pilot = gaussian_kde(values,bw_method = 0.1,weights = np.ones(len(values[0])))
kde_pilot = np.reshape(kde_pilot(positions).T, X.shape)

#try to use the FFT method from kdepy ... maybe later...

#############################################################################
####### CALCULATE BANDWIDTH H USING THE HISTOGRAM ESTIMATOR ESTIMATE ########
#############################################################################

#The bandwidth must be h = n*del_x, i.e. a whole number times the grid resolution
#this means that h is 1,2,3,4,5, etc corresponding to a 3x3, 5x5, 7x7, 9x9, 11x11, etc approximation
# to a gaussian kernel. 

# Create kernels:
x_grid = np.linspace(0, 10, 100)
num_kernels = 20
ratio = 1/3
gaussian_kernels, bandwidths_h, kernel_origin = generate_gaussian_kernels(x_grid, num_kernels, ratio)

#plot the kernels in a 3x4 plot
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()
for i in range(12):
    a = np.arange(-i,i+1,1)
    b = np.arange(-i,i+1,1)
    #make it a 3x3 kernel giving the distance in each dimension (3x3 matrix)
    #a = a.reshape(-1,1)
    #b = b.reshape(1,-1)
    #plot the kernel centered at kernel_origin
    axes[i].pcolor(kernel_origin[i][0]+a,kernel_origin[i][1]+b,gaussian_kernels[i])
    
    axes[i].set_title('Gaussian Kernel, h='+str(i+1))
    #set the same limit for everyone
    axes[i].set_xlim(-12, 12)
    axes[i].set_ylim(-12, 12)

plt.tight_layout()
plt.show()

######################################################################
######## GRID PROJECTED NON-ADAPTIVE KERNEL DENSITY ESTIMATOR ########
######################################################################

#make an array of arbitrary bandwidths for testing
particle_bandwidths = np.ones(num_particles)*1
for i in range(1,num_particles):
    particle_bandwidths[i] = particle_bandwidths[i-1]+0.00001

#flipt it upside down
particle_bandwidths = particle_bandwidths[::-1]


############################################################
##### GRID PROJECTED ADAPTIVE KERNEL DENSITY ESTIMATOR #####
############################################################

grid_size = 100  # Example grid size

histogram_est,kde_pilot,cell_bandwidths = histogram_estimator(trajectories, grid_size,weights=weights_test,bandwidths=particle_bandwidths[::frac_diff])

x_grid = np.linspace(0, 10, grid_size)
ratio = 1/3
gaussian_kernels_test, kernel_bandwidths, kernel_origin = generate_gaussian_kernels(x_grid, num_kernels, ratio)
cell_bandwidths = cell_bandwidths

n_u = grid_proj_kde(grid_size, 
                    kde_pilot, 
                    gaussian_kernels_test,
                    kernel_bandwidths,
                    cell_bandwidths)

plt.figure()
plt.imshow(n_u)
plt.title('Grid projected KDE using time dependent bandwidth')
plt.colorbar()

#plot the ground truth
plt.figure()
plt.imshow(ground_truth)
plt.title('Ground truth')
plt.colorbar()



#####################################################
#####################################################
#####################################################
#################### PLOTTING #######################
#####################################################
#####################################################
#####################################################

plotting = True
if plotting == True:
    #stop the script here

    ##########################################################################
    # Create a 1x2 subplot of all the particles in the full and test dataset #
    ##########################################################################

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot final particle positions for trajectories_full
    axes[0].scatter(trajectories_full[:,0], trajectories_full[:,1], s=1)
    axes[0].set_xlim(0, grid_size_plot)
    axes[0].set_ylim(0, grid_size_plot)
    axes[0].set_title('Final Particle Positions (Full)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    # Plot final particle positions for trajectories
    axes[1].scatter(trajectories[:,0], trajectories[:,1], s=1)
    axes[1].set_xlim(0, grid_size_plot)
    axes[1].set_ylim(0, grid_size_plot)
    axes[1].set_title('Final Particle Positions (Sampled)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    plt.tight_layout()
    plt.show()

    ########################################################
    ### Create 2x3 color plots with data and differences ###
    ########################################################

    # Set font size
    font_size = 14
    # Create a 3x2 subplot
    fig, axes = plt.subplots(3, 2, figsize=(16, 24))
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    # Plot each matrix in the first column
    matrices = [ground_truth, histogram_estimator_est, kernel_density_estimator_est]
    titles = ['Histogram est, N='+str(num_particles), 
            'Histogram Estimator, N='+str(num_particles/frac_diff), 
            'Kernel Density Estimator, N='+str(num_particles/frac_diff)]
    colormap = 'plasma'

    #Get color limits from the ground_truth matrix
    vmin = np.min(ground_truth)
    vmax = np.max(ground_truth)

    for ax, matrix, title in zip(axes[::2], matrices, titles):
        im = ax.pcolor(matrix, cmap=colormap,vmin=vmin,vmax=vmax)
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel('x', fontsize=font_size)
        ax.set_ylabel('y', fontsize=font_size)
        ax.axis('square')
        ax.set_xlim(0, grid_size_plot)
        ax.set_ylim(0, grid_size_plot)
        
        # Create a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    #Plot the Z matrix (the default gaussian_kde solution) in the upper right hand side plot
    #using the same plotting routine as above
    im = axes[1].pcolor(kde_pilot, cmap=colormap,vmin=vmin,vmax=vmax)
    axes[1].set_title('Gaussian KDE using Scott assumption', fontsize=font_size)
    axes[1].set_xlabel('x', fontsize=font_size)
    axes[1].set_ylabel('y', fontsize=font_size)
    axes[1].axis('square')
    axes[1].set_xlim(0, grid_size_plot)
    axes[1].set_ylim(0, grid_size_plot)
    #and same color range and colorbar
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    #Plot the grid projected kernel density estimate in the middle right figure

    im = axes[3].pcolor(A, cmap=colormap,vmin=vmin,vmax=vmax)
    axes[3].set_title('Grid projected KDE', fontsize=font_size)
    axes[3].set_xlabel('x', fontsize=font_size)
    axes[3].set_ylabel('y', fontsize=font_size)
    axes[3].axis('square')
    axes[3].set_xlim(0, grid_size_plot)
    axes[3].set_ylim(0, grid_size_plot)
    #and same color range and colorbar
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    #



    '''
    # Plot the differences in the second column
    diff_matrices = [np.abs(ground_truth - histogram_estimator_est), np.abs(ground_truth - kernel_density_estimator_est)]
    diff_titles = ['Difference Histogram Estimator', 'Difference Kernel Density Estimator']

    for ax, matrix, title in zip(axes[3::2], diff_matrices, diff_titles):
        im = ax.pcolor(matrix, cmap=colormap)
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel('x', fontsize=font_size)
        ax.set_ylabel('y', fontsize=font_size)
        ax.axis('square')
        ax.set_xlim(0, grid_size_plot)
        ax.set_ylim(0, grid_size_plot)
        
        # Create a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    '''

    plt.tight_layout()
    plt.show()

    # Set font size
    font_size = 14
    # Create a 3x2 subplot
    fig, axes = plt.subplots(3, 2, figsize=(16, 24))
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    # Plot each matrix in the first column
    matrices = [ground_truth, histogram_estimator_est, kernel_density_estimator_est]
    titles = ['Histogram est, N='+str(num_particles), 
            'Histogram Estimator, N='+str(num_particles/frac_diff), 
            'Kernel Density Estimator, N='+str(num_particles/frac_diff)]
    colormap = 'plasma'
    # Get color limits from the ground_truth matrix
    vmin = np.min(ground_truth)
    vmax = np.max(ground_truth)
    # Get color limits for the difference matrices
    diff_vmin = min(np.min(np.abs(ground_truth - histogram_estimator_est)), np.min(np.abs(ground_truth - kernel_density_estimator_est)))
    diff_vmax = max(np.max(np.abs(ground_truth - histogram_estimator_est)), np.max(np.abs(ground_truth - kernel_density_estimator_est)))
    levels = np.linspace(vmin, vmax, 40)
    for ax, matrix, title in zip(axes[::2], matrices, titles):
        im = ax.contourf(matrix, cmap=colormap, levels=levels)
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel('x', fontsize=font_size)
        ax.set_ylabel('y', fontsize=font_size)
        ax.axis('square')
        ax.set_xlim(0, grid_size_plot)
        ax.set_ylim(0, grid_size_plot)
        
        # Create a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    #Plot the Z matrix (the default gaussian_kde solution) in the upper right hand side plot
    #using the same plotting routine as above
    im = axes[1].contourf(kde_pilot, cmap=colormap, levels=levels)
    axes[1].set_title('Gaussian KDE using Scott assumption', fontsize=font_size)
    axes[1].set_xlabel('x', fontsize=font_size)
    axes[1].set_ylabel('y', fontsize=font_size)
    axes[1].axis('square')
    axes[1].set_xlim(0, grid_size_plot)
    axes[1].set_ylim(0, grid_size_plot)
    #and same color range and colorbar
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    #Plot the grid projected kernel density estimate in the middle right figure
    im = axes[3].contourf(A, cmap=colormap, levels=levels)
    axes[3].set_title('Grid projected NON-WEIGHTED KDE', fontsize=font_size)
    axes[3].set_xlabel('x', fontsize=font_size)
    axes[3].set_ylabel('y', fontsize=font_size)
    axes[3].axis('square')
    axes[3].set_xlim(0, grid_size_plot)
    axes[3].set_ylim(0, grid_size_plot)
    #and same color range and colorbar
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    '''
    # Plot the differences in the second column
    diff_matrices = [np.abs(ground_truth - histogram_estimator_est), np.abs(ground_truth - kernel_density_estimator_est)]
    diff_titles = ['Difference Histogram Estimator', 'Difference Kernel Density Estimator']

    for ax, matrix, title in zip(axes[3::2], diff_matrices, diff_titles):
        im = ax.contourf(matrix, cmap=colormap, vmin=diff_vmin, vmax=diff_vmax)
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel('x', fontsize=font_size)
        ax.set_ylabel('y', fontsize=font_size)
        ax.axis('square')
        ax.set_xlim(0, grid_size_plot)
        ax.set_ylim(0, grid_size_plot)
        
        # Create a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    '''

    plt.tight_layout()
    plt.show()
    #print the summed differences
    print('Summed differences between the two estimators')
    print('Difference, histogram estimator')
    print(np.sum(np.abs(ground_truth - histogram_estimator_est)))
    print('Difference, kernel density estimator')
    print(np.sum(np.abs(ground_truth - kernel_density_estimator_est)))
    print('Difference, grid projected KDE')
    print(np.sum(np.abs(ground_truth - A)))

    ###############
    #TEST WHAT'S FASTEST GRID PROJECTED VS kernel_matrix_2d_NOFLAT

    #Create a bw v
    trajectories_test = trajectories_full[::10000]

    #Test on the big particle dataset
    import time
    start = time.time()
    kernel_density_estimator_est = kernel_matrix_2d_NOFLAT(trajectories_test[:,0],trajectories_test[:,1],x_grid,y_grid,bw_full,weights_full)
    kernel_density_estimator_est = kernel_density_estimator_est.T
    end = time.time()

    print('Time for kernel_matrix_2d_NOFLAT')
    print(end-start)

    start = time.time()
    #Precompute the pilot KDE
    kde_pilot = histogram_estimator(trajectories_test, grid_size,weights=weights_full)[1]
    n_u = grid_proj_kde(grid_size, kde_pilot, gaussian_kernels)
    end = time.time()

    print('Time for grid projected KDE')
    print(end-start)

    #Plot to see if they are the same
    # Create a 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    # Plot each matrix in the first column
    matrices = [kernel_density_estimator_est, n_u]
    titles = ['kernel_matrix_2d_NOFLAT', 'Grid projected KDE']
    colormap = 'plasma'
    # Get color limits from the ground_truth matrix
    vmin = np.min(kernel_density_estimator_est)
    vmax = np.max(kernel_density_estimator_est)
    levels = np.linspace(vmin, vmax, 40)

    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.contourf(matrix, cmap=colormap, levels=levels)
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel('x', fontsize=font_size)
        ax.set_ylabel('y', fontsize=font_size)
        ax.axis('square')
        ax.set_xlim(0, grid_size_plot)
        ax.set_ylim(0, grid_size_plot)
        
        # Create a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    plt.tight_layout()

    plt.show()



