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

#set plotting style
plt.style.use('dark_background')
#set the plotting style back to default
plt.style.use('default')

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

    return total_weight,particle_count, cell_bandwidth

def histogram_std(binned_data, effective_samples = None, bin_size=1):
    '''
    Calculate the simple variance of the binned data using ...
    '''
    #set integral length scale to the size of the grid if not provided
    if effective_samples == None:
        effective_samples = np.sum(binned_data)
    #get 
    #check that there's data in the binned data
    if np.sum(binned_data) == 0:
        return 0
    #get the central value of all bins
    grid_size = len(binned_data)
    #Central point of all grid cells
    X = np.arange(0,grid_size*bin_size,bin_size)
    Y = np.arange(0,grid_size*bin_size,bin_size)
    #Calculate the weigthed average position in the binned data
    mu_x = np.sum(binned_data*X)/np.sum(binned_data)
    mu_y = np.sum(binned_data*Y)/np.sum(binned_data)
    #Calculate the variance
    std_y = np.sqrt(np.sum(binned_data*(X-mu_x)**2)/np.sum(binned_data)*(effective_samples/(effective_samples-1)))
    std_x = np.sqrt(np.sum(binned_data*(Y-mu_y)**2)/np.sum(binned_data)*(effective_samples/(effective_samples-1)))
    #Calculate the covariance
    std_xy = np.sqrt(np.sum(binned_data*(X-mu_x)*(Y-mu_y))/np.sum(binned_data)*(effective_samples/(effective_samples-1)))
    #Calculate the total variance
    std_data = (std_x+std_y+2*std_xy)/4
    #https://towardsdatascience.com/on-the-statistical-analysis-of-rounded-or-binned-data-e24147a12fa0
    #Sheppards correction
    std_data = std_data - 1/12*(3*bin_size**2)
    return std_data

def window_sum(data):
    # Filter out zero values
    non_zero_data = data[data != 0]
    return np.sum(non_zero_data)

def calculate_autocorrelation_vectorized(data):
    '''
    Calculates the autocorrelation for all lags along rows and columns, separately, as an average.
    
    Input:
    data: 2D matrix with equal number of rows and columns
    
    Output:
    autocorr_rows: 1D array with the average autocorrelation for each lag along rows
    autocorr_cols: 1D array with the average autocorrelation for each lag along columns
    '''
    num_rows, num_cols = data.shape
    lags = num_rows - 1
    autocorr_rows = np.zeros((num_rows, lags))
    autocorr_cols = np.zeros((num_cols, lags))
    
    for k in range(1, lags + 1):
        autocorr_rows[:, k - 1] = np.sum(data[:, :-k] * data[:, k:], axis=1) / (num_cols - k)
        autocorr_cols[:, k - 1] = np.sum(data[:-k, :] * data[k:, :], axis=0) / (num_rows - k)
    
    autocorr_rows = np.nanmean(autocorr_rows, axis=0)
    autocorr_cols = np.nanmean(autocorr_cols, axis=0)
    
    return autocorr_rows, autocorr_cols

def get_integral_length_scale_vectorized(histogram_prebinned, window_size):
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
    
    for i in range(0, rows - window_size + 1):
        for j in range(0, cols - window_size + 1):
            window = histogram_prebinned[i:i + window_size, j:j + window_size]
            if np.any(window != 0):  # Check if there are any non-zero elements in the window
                autocorr_rows, autocorr_cols = calculate_autocorrelation(window)
                autocorr = (autocorr_rows + autocorr_cols) / 2
                integral_length_scale = np.sum(autocorr) / autocorr[0]
                integral_length_scale_matrix[i:i + window_size, j:j + window_size] = integral_length_scale
    
    return integral_length_scale_matrix
    
def calculate_autocorrelation(data):
    '''
    Calculates the autocorrelation for all lags along rows and columns, separately, as an average.
    
    Input:
    data: 2D matrix with possibly unequal number of rows and columns
    
    Output:
    autocorr_rows: 1D array with the average autocorrelation for each lag along rows
    autocorr_cols: 1D array with the average autocorrelation for each lag along columns
    '''
    num_rows, num_cols = data.shape
    max_lag = min(num_rows, num_cols) - 1
    
    # Initialize the autocorrelation arrays
    autocorr_rows = np.zeros(max_lag)
    autocorr_cols = np.zeros(max_lag)
    
    # Precompute denominators for efficiency
    row_denominators = np.array([1 / (num_cols - k) for k in range(1, max_lag + 1)])
    col_denominators = np.array([1 / (num_rows - k) for k in range(1, max_lag + 1)])
    
    # Calculate the autocorrelation for all lags along rows
    for k in range(1, max_lag + 1):
        autocorr_rows[k - 1] = np.mean([row_denominators[k - 1] * np.sum(data[i, :num_cols - k] * data[i, k:]) for i in range(num_rows)])
    
    # Calculate the autocorrelation for all lags along columns
    for k in range(1, max_lag + 1):
        autocorr_cols[k - 1] = np.mean([col_denominators[k - 1] * np.sum(data[:num_rows - k, j] * data[k:, j]) for j in range(num_cols)])
    
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
        data_subset = histogram_prebinned_padded[i:i + window_size, j:j + window_size]
        if np.any(data_subset != 0):
            autocorr_rows, autocorr_cols = calculate_autocorrelation(data_subset)
            autocorr = (autocorr_rows + autocorr_cols) / 2
            integral_length_scale = np.sum(autocorr) / autocorr[0]
            integral_length_scale_matrix[i, j] = integral_length_scale


    return integral_length_scale_matrix


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
#Set weights_full and weights to 1
weights_full = np.ones(len(trajectories_full))
weights_test = np.ones(len(trajectories))


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
kde_time_bw = grid_proj_kde(x_grid, y_grid, pre_estimate, gaussian_kernels, kernel_bandwidths, cell_bandwidths)

#make a plot of the time dependent bandwidth estimate
plt.figure()
plt.imshow(kde_time_bw)
plt.colorbar()
plt.title('Time dependent bandwidth estimate (old method)')
plt.show()

########################################################
########### DATA DRIVEN BANDWIDTH ESTIMATE #############
########################################################

#MAKE PRE BINNED ESTIMATE
histogram_prebinned,count_prebinned,cell_bandwidths = histogram_estimator(p_x,p_y, x_grid,y_grid,bandwidths=bw,weights=weights_test)

#Calculate the integral length scale for the whole prebinned data to get the window size
data_subset = histogram_prebinned
autocorr_rows, autocorr_cols = calculate_autocorrelation(data_subset)
autocorr = (autocorr_rows + autocorr_cols) / 2
integral_length_scale = np.sum(autocorr) / autocorr[0]
window_size = int(np.ceil(np.mean(integral_length_scale)))

#Define window size, i.e. the size the adaptation is applied to
window_size = 17
pad_size = window_size // 2
#pad the naive estimate with zeros (reflective padding) to avoid problems at the edges.
histogram_prebinned_padded = np.pad(histogram_prebinned, pad_size, mode='reflect')
naive_estimate_padded = histogram_prebinned_padded
count_prebinned_padded = np.pad(count_prebinned, pad_size, mode='reflect')

###
#ESTIMATE THE STANDARD DEVIATION IN EACH WINDOW ASSOCIATED WITH EACH NON-ZERO CELL.
###

variance_estimate = np.zeros(np.shape(naive_estimate))
weight_estimate = np.zeros(np.shape(naive_estimate))
integral_length_scale_matrix = np.zeros(np.shape(naive_estimate))
h_matrix = np.zeros(np.shape(naive_estimate))
#get non_zero indices
non_zero_indices = np.argwhere(histogram_prebinned != 0)
N_eff_advanced = np.zeros(np.shape(naive_estimate))
N_eff_simple = np.zeros(np.shape(naive_estimate))
std_estimate = np.zeros(np.shape(naive_estimate))
N_silv = np.zeros(np.shape(naive_estimate))


#calculate variances, weights, integral length scales and hs for all non-zero cells
for idx in non_zero_indices:
    i,j = idx
    data_subset = naive_estimate_padded[i:i+window_size,j:j+window_size] #using the padded matrix, so no dividing here...
    subset_indices = np.argwhere(data_subset != 0)

    weight_estimate[i,j] = np.sum(data_subset)#CALCULATE N, I.E. NUMBER OF PARTICLES IN ALL ALL NON-ZERO CELLS
    autocorr_rows, autocorr_cols = calculate_autocorrelation(data_subset)
    autocorr = (autocorr_rows + autocorr_cols) / 2
    integral_length_scale = np.sum(autocorr) / autocorr[0]
    integral_length_scale_matrix[i, j] = integral_length_scale
    N_eff_simple[i,j] = weight_estimate[i,j]/window_size #CALCULATE ADVANCED EFFECTIVE N_eff_advanced simply
    N_eff_advanced_ij = weight_estimate[i,j]/integral_length_scale
    N_eff_advanced[i,j] = N_eff_advanced_ij
    std_estimate[i,j] = histogram_std(data_subset/np.sum(data_subset),effective_samples=N_eff_advanced_ij,bin_size=1)
    h_matrix[i,j] = silvermans_simple_2d(weight_estimate[i,j], 2)*(std_estimate[i,j])
'''
#plot the variance estimate
plt.figure()
plt.imshow(std_estimate)
plt.colorbar()
plt.title('Standard deviation estimate')
plt.show()

#plot it.
plt.imshow(weight_estimate)
plt.colorbar()
plt.title('Sum estimate (N pure)')
plt.show()

#plot the integral length scale
plt.imshow(integral_length_scale_matrix)
plt.colorbar()
plt.title('Integral length scale')
plt.show()

#calculate the advanced N_eff
#N_eff_advanced = weight_estimate/integral_length_scale_matrix

#plot the advanced N_eff
plt.imshow(N_eff_advanced)
plt.colorbar()
plt.title('Advanced effective N_eff using integral length scale')
plt.show()

#plot Silvermans factor
plt.imshow(N_silv)
plt.colorbar()
plt.title('Silverman factor estimate')
plt.show()

#plot the bandwidth estimate
plt.figure()
plt.imshow(h)
plt.colorbar()
plt.title('Bandwidth estimate')
plt.show()

'''

###
#CALCULATE THE KERNEL DENSITY ESTIMATE
###
h_grid = h_matrix
#h_grid[std_estimate == 0] = 1000
kde_data_driven = grid_proj_kde(x_grid, y_grid, histogram_prebinned, gaussian_kernels, kernel_bandwidths, h_grid)
#normalize
kde_data_driven = kde_data_driven/np.sum(kde_data_driven)

#plot the kernel density estimate
plt.figure()
plt.imshow(kde_data_driven)
plt.colorbar()
plt.title('Data driven kde estimate using NeffNeff! Full pakke!')
plt.show()

#set all nan values in h to 0
h[np.isnan(h)] = 0
#plot everything:

# Determine the global min and max values for the lower three plots
vmin = min(np.min(kde_data_driven), np.min(naive_estimate), np.min(ground_truth))
vmax = max(np.max(kde_data_driven), np.max(naive_estimate), np.max(ground_truth))

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Data driven kde estimate using NeffNeff! Full pakke!
im1 = axs[1, 2].imshow(kde_data_driven, vmin=vmin, vmax=vmax)
axs[1, 2].set_title('Data driven estimator using "NeffNeffNeff!')
fig.colorbar(im1, ax=axs[1, 2])

# Plot 2: Bandwidth estimate
im2 = axs[0, 2].imshow(h)
axs[0, 2].set_title('Bandwidth estimate')
fig.colorbar(im2, ax=axs[0, 2])

# Plot 3: Integral length scale
im3 = axs[0, 1].imshow(integral_length_scale_matrix)
axs[0, 1].set_title('Integral length scale')
fig.colorbar(im3, ax=axs[0, 1])

# Plot 4: Standard deviation estimate
im4 = axs[0, 0].imshow(std_estimate)
axs[0, 0].set_title('Standard deviation estimate')
fig.colorbar(im4, ax=axs[0, 0])

# Plot 5: Naive histogram estimate
im5 = axs[1, 1].imshow(naive_estimate, vmin=vmin, vmax=vmax)
axs[1, 1].set_title('Naive histogram estimate')
fig.colorbar(im5, ax=axs[1, 1])

# Plot 6: Ground truth
im6 = axs[1, 0].imshow(ground_truth, vmin=vmin, vmax=vmax)
axs[1, 0].set_title('Ground truth')
fig.colorbar(im6, ax=axs[1, 0])

plt.tight_layout()
plt.show()

#plot a histogram of the non-zero elements of the standard deviation matrix and N_eff matrix in a subplot
plt.figure()
plt.subplot(1,2,1)
plt.hist(std_estimate[std_estimate != 0].flatten(),bins=100)
plt.title('Standard deviation histogram')
plt.subplot(1,2,2)
plt.hist(N_eff_advanced[N_eff_advanced != 0].flatten(),bins=100)
plt.title('N_eff histogram')
plt.show()








### CALCULATE INTEGRAL LENGTH SCALE FOR ALL DATA ###

#autocorr_rows, autocorr_cols = calculate_autocorrelation(histogram_prebinned)
#autocorr = (autocorr_rows + autocorr_cols) / 2
#integral_length_scale = np.sum(autocorr) / autocorr[0]
