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
from matplotlib.ticker import ScalarFormatter
#import gridspec
from matplotlib.gridspec import GridSpec as GridSpec
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.spatial import KDTree


#set plotting style
plt.style.use('dark_background')
#set the plotting style back to default
plt.style.use('default')

#Triggers
create_test_data_this_run = False
load_test_data = False
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
plt.rcParams['image.cmap'] = 'mako_r'


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

#reflect_kernel_contribution(kernel, x, y, legal_grid, density_grid)
#Function to calculate the grid projected kernel density estimator
def grid_proj_kde(grid_x, 
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
    #else:
        #illegal_cells = np.zeros((len(grid_x), len(grid_y)), dtype=bool)
        #check if any of the illegal cell positions are within the kernel area
    
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
def create_test_data(stdev=1.4, 
                     num_particles_per_timestep=5000, 
                     time_steps=380, 
                     dt=0.1, 
                     grid_size=100,
                     illegal_positions=None):
    # Define illegal cells
    if illegal_positions is None:
        illegal_positions = np.zeros((grid_size, grid_size), dtype=bool)
        a = 35
        b = 20
        x0 = 55
        y0 = 95
        for i in range(grid_size):
            for j in range(grid_size):
                if ((i - x0) / a)**2 + ((j - y0) / b)**2 <= 1:
                    illegal_positions[i, j] = 1

        # Make also every number where y is larger than 80 and x smaller than 50 illegal
        illegal_positions[:50, 80:] = 1

        #add a narrow obstacle in the middle of the grid oriented 30 degrees from the x-axis
        #illegal_positions[0:60, 20:30] = 1

    # Create a true/false mask of illegal cells
    legal_cells = ~illegal_positions
    # Indices of legal cells
    legal_indices = np.argwhere(legal_cells)
    # Coordinates
    x_grid = np.arange(illegal_positions.shape[0])
    y_grid = np.arange(illegal_positions.shape[1])
    legal_coordinates = np.array([x_grid[legal_indices[:, 0]], y_grid[legal_indices[:, 1]]]).T
    from scipy.spatial import KDTree
    tree = KDTree(legal_coordinates)

    # Release position
    release_position = np.array([10, 10])
    # Make U_a a periodic function with size time_steps
    U_a = [0, 5]  # Initial value
    # Initial magnitude
    magU = np.sqrt(U_a[0]**2 + U_a[1]**2)
    U_a = np.tile(U_a, (time_steps, 1))
    for i in range(1, time_steps):
        U_a[i][:][0] = 2 * magU + np.sin(i / 50) * 2 * magU
        # make it a bit more complex by adding another sine function with different frequency
        # U_a[i][:][1] = 2*magU+ np.sin(i/50)*2*magU + np.sin(i/10)*2*magU
        print(np.sin(i / 10))
        # L2 normalize the velocity
        U_a[i] = (U_a[i] / (np.sqrt(U_a[i][0]**2 + U_a[i][1]**2))) * magU  # Conservation of mass

    # Simulate particle trajectories
    trajectories = np.zeros((num_particles_per_timestep * time_steps, 2)) * np.nan
    # Create the bandwidth vector for each particle
    bw = np.ones(num_particles_per_timestep * time_steps) * 0

    for t in range(time_steps - 1):
        if t == 0:
            # Initialize particle matrix at first timestep
            particles = np.ones([num_particles_per_timestep, 2]) * release_position
        else:
            particles_old = particles

            # Add particles to the particle array
            particles = np.ones([num_particles_per_timestep * (t + 1), 2]) * release_position
            # Add in the old particle positions to the new array
            particles[:num_particles_per_timestep * t] = particles_old
            # Set particles that have left the domain to nan
            # Update the bw vector

        print(np.shape(particles))
        particles = update_positions(particles, U_a[t], stdev, dt)

        # Reposition illegal particles
        p_x, p_y = particles[:, 0], particles[:, 1]
        valid_indices = ~np.isnan(p_x) & ~np.isnan(p_y) & (p_x >= 0) & (p_x < grid_size) & (p_y >= 0) & (p_y < grid_size)
        is_illegal = np.zeros(p_x.shape, dtype=bool)
        is_illegal[valid_indices] = ~legal_cells[p_x[valid_indices].astype(int), p_y[valid_indices].astype(int)]
        illegal_positions = particles[is_illegal]
        _, nearest_indices = tree.query(illegal_positions)
        mapped_positions = legal_coordinates[nearest_indices]
        particles[is_illegal, 0] = mapped_positions[:, 0]
        particles[is_illegal, 1] = mapped_positions[:, 1]

        trajectories[:len(particles)] = particles
        bw[:len(particles)] = bw[:len(particles)] + np.sqrt(stdev * 0.001)
        # Limit bw to a maximum value
        # bw[bw > 20] = 20

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

def get_test_data(load_test_data=True,frac_diff = 1000,weights = 'log_weights',illegal_positions=None):
    if load_test_data == True:
        # Load test data
        with open(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\src\Dispersion_modelling\concentration_estimation\kde_estimate\trajectories_full.pkl', 'rb') as f:
            trajectories_full = pickle.load(f)
        with open(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\src\Dispersion_modelling\concentration_estimation\kde_estimate\bw.pkl', 'rb') as f:
            bw = pickle.load(f)
    else:
        # Create test data
        trajectories_full, bw = create_test_data(illegal_positions=illegal_positions)
        # Save test data to the same locations
        with open(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\src\Dispersion_modelling\concentration_estimation\kde_estimate\trajectories_full.pkl', 'wb') as f:
            pickle.dump(trajectories_full, f)
        with open(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\src\Dispersion_modelling\concentration_estimation\kde_estimate\bw.pkl', 'wb') as f:
            pickle.dump(bw, f)

    trajectories = trajectories_full[::frac_diff,:]
    #pick only the right data from the bw vector
    bw = bw[::frac_diff]

    if weights == 'log_weights':
        weights = 1-np.log(np.linspace(1,100,len(trajectories_full)))/(np.log(100)*2)
        weights_test = weights[::frac_diff]

    return trajectories, trajectories_full, bw, weights, weights_test


def histogram_estimator(x_pos, y_pos, grid_x, grid_y, bandwidths=None, weights=None, illegal_cells=None):
    '''
    Input:
    x_pos (np.array): x-coordinates of the particles
    y_pos (np.array): y-coordinates of the particles
    grid_x (np.array): grid cell boundaries in the x-direction
    grid_y (np.array): grid cell boundaries in the y-direction
    bandwidths (np.array): bandwidths of the particles
    weights (np.array): weights of the particles
    illegal_cells: list of tuples with illegal cells where no particles are allowed

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
    [mu_x,mu_y] = np.sum(binned_data*X)/np.sum(binned_data),np.sum(binned_data*Y)/np.sum(binned_data)
    #Calculate the variance
    variance = (np.sum(binned_data*((X-mu_x)**2+(Y-mu_y)**2))/(np.sum(binned_data)-1))-1/12*bin_size*bin_size#*(effective_samples/(effective_samples-1))
    std_data = np.sqrt(variance)
    #std_data = np.sqrt(std_x**2+std_y**2+2*std_xy**2)/4
    #std_data = (std_x+std_y+2*std_xy)/4
    #https://towardsdatascience.com/on-the-statistical-analysis-of-rounded-or-binned-data-e24147a12fa0
    #Sheppards correction
    std_data = std_data #- 1/12*bin_size*bin_size
    return std_data

def histogram_std_sep(binned_data, effective_samples = None, bin_size=1):
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
    std_y = np.sqrt(np.sum(binned_data*(X-mu_x)**2)/(np.sum(binned_data)-1))#*(effective_samples/(effective_samples-1)))
    std_x = np.sqrt(np.sum(binned_data*(Y-mu_y)**2)/(np.sum(binned_data)-1))#*(effective_samples/(effective_samples-1)))
    #Calculate the covariance
    std_xy = np.sqrt(np.sum(binned_data*(X-mu_x)*(Y-mu_y))/(np.sum(binned_data)-1))#*(effective_samples/(effective_samples-1)))
    #Calculate the total variance
    std_data = np.sqrt(std_x**2+std_y**2+2*std_xy**2)
    #std_data = (std_x+std_y+2*std_xy)/4
    #https://towardsdatascience.com/on-the-statistical-analysis-of-rounded-or-binned-data-e24147a12fa0
    #Sheppards correction
    std_data = std_data - 1/12*(3*bin_size**2)
    return std_data


def histogram_std_matrix(binned_data, effective_samples = None, bin_size=1):
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
    #Calculate the variance for each variable
    var_x = np.sum(binned_data*(X-mu_x)**2)#/(np.sum(binned_data)-1)
    var_y = np.sum(binned_data*(Y-mu_y)**2)#/(np.sum(binned_data)-1)
    #Calculate the covariance
    cov_xy = np.sum(binned_data*(X-mu_x)*(Y-mu_y))#/(np.sum(binned_data)-1)
    cov_yx = np.sum(binned_data*(Y-mu_y)*(X-mu_x))#/(np.sum(binned_data)-1)
    #define the covariance matrix
    cov_matrix = np.array([[var_x,cov_xy],[cov_yx,var_y]])
    #Calculate the determinant of the covariance matrix - this represents the volume of the data
    det_cov = np.linalg.det(cov_matrix)
    #use the square root and add the 1/12 factor to get the standard deviation
    std_data = np.sqrt(det_cov) + 1/12*bin_size*bin_size
    #https://towardsdatascience.com/on-the-statistical-analysis-of-rounded-or-binned-data-e24147a12fa0
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


#make the illegal cells an elliptic shape
def create_illegal_dataset(grid_size,p_x,p_y,x_grid,y_grid):

    #Define illegal grid cells
    illegal_cells = np.zeros((grid_size,grid_size))
    #illegal_cells[55:65,70:85] = 1

    a = 30
    b = 15
    x0 = 55
    y0 = 95
    for i in range(grid_size):
        for j in range(grid_size):
            if ((i-x0)/a)**2 + ((j-y0)/b)**2 <= 1:
                illegal_cells[i,j] = 1

    #make also eevry number where y is larger than 80 and x smaller than 50 illegal
    illegal_cells[:50,80:] = 1
    #make a plot of the illegal cells

    #create a true/false mask of illegal cells
    legal_cells = illegal_cells == 0
    #indices of legal cells
    legal_indices = np.argwhere(legal_cells)
    #coordinates
    legal_coordinates = np.array([x_grid[legal_indices[:,0]],y_grid[legal_indices[:,1]]]).T
    from scipy.spatial import KDTree
    tree = KDTree(legal_coordinates)
    #get only p_x and p_ys that are within the grid
    #p_x[p_x < 0] = 0
    #p_x[p_x >= grid_size] = grid_size-1
    #p_y[p_y < 0] = 0
    #p_y[p_y >= grid_size] = grid_size-1
    #remove nans
    #p_x = p_x[~np.isnan(p_x)]
    #p_y = p_y[~np.isnan(p_y)]
    #do the same with particle positions
    particle_positions = np.array([p_x,p_y]).T
    # Initialize is_illegal with False values
    is_illegal = np.zeros(p_x.shape, dtype=bool)
    # Filter out NaN values and ensure positions are within the grid domain
    valid_indices = ~np.isnan(p_x) & ~np.isnan(p_y) & (p_x >= 0) & (p_x < grid_size) & (p_y >= 0) & (p_y < grid_size)
    # Update is_illegal only for valid indices
    is_illegal[valid_indices] = ~legal_cells[p_x[valid_indices].astype(int), p_y[valid_indices].astype(int)]
    # Find nearest legal cells only for illegal particles
    illegal_positions = particle_positions[is_illegal]
    _, nearest_indices = tree.query(illegal_positions)  # get the nearest legal cell using the KDTree
    mapped_positions = legal_coordinates[nearest_indices]  # get the mapped positions
    # Insert the mapped positions into the illegal positions in p_x
    p_x[is_illegal] = mapped_positions[:,0]
    p_y[is_illegal] = mapped_positions[:,1]

    return p_x,p_y

#@njit
def reflect_kernel_contribution(kernel, x, y, legal_grid, density_grid):
    kernel_size = kernel.shape[0]
    half_k = kernel_size // 2
    
    # Iterate over each cell within the kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            xi, yj = x + i - half_k, y + j - half_k
            
            # Skip if the kernel cell is out of bounds
            #if xi < 0 or yj < 0 or xi >= legal_grid.shape[0] or yj >= legal_grid.shape[1]:
            #    continue
            
            # Check if the path from particle to this cell is blocked
            if not legal_grid[xi, yj]:  # "illegal" cell found
                # Reflect across the closest "legal" boundary cell
                x_reflect, y_reflect = reflect_with_shadow(x, y, xi, yj, legal_grid)
                
                # Apply reflection if reflected position is legal and within bounds
                if x_reflect is not None and y_reflect is not None:
                    density_grid[x_reflect, y_reflect] += kernel[i, j]
            else:
                # If no barrier, add kernel contribution directly
                density_grid[xi, yj] += kernel[i, j]

#@njit
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
    
def bresenham(x0, y0, x1, y1):
    """
    Bresenham's Line Algorithm to generate points between (x0, y0) and (x1, y1)
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    # checking the direction of the line and which octant it is in
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy #This is the difference between the distances in the two directions

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        if 2*err > -dy: #This means that ||dx|| > ||dy||
            err -= dy
            x0 += sx #...which means that we should move in the x direction
        if 2*err < dx:
            err += dx
            y0 += sy

    return points
    
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


# ------------------------------------------------------- #
###########################################################
##################### INITIATION ##########################
###########################################################
# ------------------------------------------------------- #
# Add folder path
# Define illegal positions
### For the plotting we need the illegal cells ###
illegal_positions = np.zeros((grid_size_plot, grid_size_plot), dtype=bool)
a = 35
b = 20
x0 = 55
y0 = 95
for i in range(grid_size_plot):
    for j in range(grid_size_plot):
        if ((i - x0) / a)**2 + ((j - y0) / b)**2 <= 1:
            illegal_positions[i, j] = 1
# Make also every number where y is larger than 80 and x smaller than 50 illegal
illegal_positions[:50, 80:100] = 1
#add a narrow obstacle in the middle of the grid
#illegal_positions[0:60, 20:22] = 1
illegal_positions = illegal_positions.T
#get the test data
if create_test_data_this_run == True:
    trajectories,trajectories_full,bw,weights,weights_test = get_test_data(load_test_data=False,frac_diff=frac_diff,illegal_positions=illegal_positions.T)
else:
    trajectories, trajectories_full, bw, weights, weights_test = get_test_data(load_test_data=True,frac_diff=frac_diff,illegal_positions=illegal_positions.T)   

#reduce size of test data by picking only every 10th particle
trajectories = trajectories[::1]
bw = bw[::1]
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

#trajectories_full[:,0],trajectories_full[:,1] = create_illegal_dataset(grid_size,p_full_x,p_full_y,x_grid,y_grid)
#and corresponding test data
#trajectories = trajectories_full[::frac_diff]
#weights = weights_full[::frac_diff]

plt.figure()
plt.scatter(trajectories[:,0],trajectories[:,1])

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

p_x = trajectories[:,0]
p_y = trajectories[:,1]

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

print('Window size:',window_size)

#Define window size, i.e. the size the adaptation is applied to
#window_size = 17
pad_size = window_size // 2
#pad the naive estimate with zeros (reflective padding) to avoid problems at the edges.
histogram_prebinned_padded = np.pad(histogram_prebinned, pad_size, mode='reflect')
naive_estimate_padded = histogram_prebinned_padded
count_prebinned_padded = np.pad(count_prebinned, pad_size, mode='reflect')

###
#ESTIMATE THE STANDARD DEVIATION IN EACH WINDOW ASSOCIATED WITH EACH NON-ZERO CELL.
###

#set a threshold where the statistics are not calculated
stats_threshold = window_size

variance_estimate = np.zeros(np.shape(naive_estimate))
weight_estimate = np.zeros(np.shape(naive_estimate))
integral_length_scale_matrix = np.zeros(np.shape(naive_estimate))
h_matrix = np.zeros(np.shape(naive_estimate))*np.nan
#get non_zero indices
non_zero_indices = np.argwhere(histogram_prebinned != 0)
N_eff_advanced = np.zeros(np.shape(naive_estimate))
N_eff_simple = np.zeros(np.shape(naive_estimate))
std_estimate = np.zeros(np.shape(naive_estimate))*np.nan
N_silv = np.zeros(np.shape(naive_estimate))
small_n_eff = np.zeros(np.shape(naive_estimate))

#calculate variances, weights, integral length scales and hs for all non-zero cells
for idx in non_zero_indices:
    i,j = idx
    data_subset = naive_estimate_padded[i:i+window_size,j:j+window_size] #using the padded matrix, so no dividing here...
    data_subset_counts = count_prebinned_padded[i:i+window_size,j:j+window_size]
    subset_indices = np.argwhere(data_subset != 0)
    #normalize the data subset to psi
    data_subset = (data_subset/np.sum(data_subset))*np.sum(data_subset_counts)
    weight_estimate[i,j] = np.sum(data_subset)
    

    #Calculate (depending on the number of particles in the window)
    if np.sum(data_subset) < stats_threshold:
        #print('Too few particles in window, using window size and simple N_eff')
        #the standard deviation is set tohalf the window_size if there are too few particles in the window
        #to robustly estimate the standard deviation
        std_estimate[i,j] = window_size/2#np.sqrt(window_size)*(1/12) #assuming uniform distribution within the window
        N_eff_advanced[i,j] = np.sum(data_subset)/window_size
    else:
        std_estimate[i,j] = histogram_std(data_subset,effective_samples=None,bin_size=1)
        autocorr_rows, autocorr_cols = calculate_autocorrelation(data_subset)
        autocorr = (autocorr_rows + autocorr_cols) / 2

        integral_length_scale = np.sum(autocorr) / autocorr[np.argwhere(autocorr != 0)[0]] #just finding first non_zero element in autocorr
        integral_length_scale_matrix[i, j] = integral_length_scale
        #get small_n_eff
        small_n_eff[i,j] = (np.sum(data_subset.flatten())**2)/np.sum(data_subset.flatten()**2)
        small_n_eff[i,j] = np.sum(data_subset)
        N_eff_simple[i,j] = small_n_eff[i,j]/window_size #CALCULATE ADVANCED EFFECTIVE N_eff_advanced simply
        N_eff_advanced_ij = small_n_eff[i,j]/integral_length_scale
        N_eff_advanced[i,j] = N_eff_advanced_ij

        #print idx if the standard deviation is zero
        if std_estimate[i,j] == 0:
            print('Zero standard deviation at index:',idx)

    h_matrix[i,j] = silvermans_simple_2d(N_eff_advanced[i,j], 2)*(std_estimate[i,j])#**2
    if h_matrix[i,j] == 0:
        print('Zero bandwidth at index:',idx)

#h_matrix = h_matrix**2

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
#pad x_grid and y_grid to avoid edge effects
x_grid_padded = np.pad(x_grid, pad_size, mode='reflect')
y_grid_padded = np.pad(y_grid, pad_size, mode='reflect')
#pad h_grid
h_grid_padded = np.pad(h_matrix, pad_size, mode='reflect')
#pad illegal positions
illegal_positions_padded = np.pad(illegal_positions, pad_size, mode='reflect')

h_grid = h_matrix
h = h_matrix
#DO a data driven estimate without boundary control
kde_data_driven_naive = grid_proj_kde(x_grid_padded, y_grid_padded, histogram_prebinned_padded, gaussian_kernels, kernel_bandwidths, h_grid_padded, illegal_cells = np.zeros(np.shape(histogram_prebinned_padded)))
# normalize
kde_data_driven_naive = kde_data_driven_naive / np.sum(kde_data_driven_naive)
#Do a data driven estimate with boundary control
#h_grid[std_estimate == 0] = 1000
kde_data_driven = grid_proj_kde(x_grid_padded, y_grid_padded, histogram_prebinned_padded, gaussian_kernels, kernel_bandwidths, h_grid_padded, illegal_cells = illegal_positions_padded)
#normalize
kde_data_driven = kde_data_driven/np.sum(kde_data_driven)
#Calculate estimate using time dependent bandwidth
cell_bandwidths_padded = np.pad(cell_bandwidths, pad_size, mode='reflect')
kde_time_bw = grid_proj_kde(x_grid_padded, y_grid_padded, histogram_prebinned_padded, gaussian_kernels, kernel_bandwidths, cell_bandwidths_padded, illegal_cells = illegal_positions_padded)



# Compute 2D Silverman estimate using Gaussian KDE with trajectory data
data = trajectories[~np.isnan(trajectories).any(axis=1)]
kde = gaussian_kde(data.T, bw_method='silverman')
x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)
Z = kde(np.vstack([X.flatten(), Y.flatten()])).reshape(X.shape)
kde_silverman_naive = Z / np.sum(Z)

#set all nan values in h to 0
#h[np.isnan(h)] = 0

########################################################
################### PLOT EVERYTHING ####################
########################################################

# Replace NaN values with zeros
kde_data_driven = np.nan_to_num(kde_data_driven, nan=0.0)
naive_estimate = np.nan_to_num(naive_estimate, nan=0.0)
ground_truth = np.nan_to_num(ground_truth, nan=0.0)
h = np.nan_to_num(h, nan=0.0)
N_eff_advanced = np.nan_to_num(N_eff_advanced, nan=0.0)
std_estimate = np.nan_to_num(std_estimate, nan=0.0)

#normalize the time dependent bandwidth estimate
kde_time_bw = kde_time_bw/np.sum(kde_time_bw)

# Determine the global min and max values for the lower three plots
vmin = min(np.min(kde_data_driven), np.min(naive_estimate), np.min(ground_truth))
vmax = max(np.max(kde_data_driven), np.max(naive_estimate), np.max(ground_truth))

    ######### PLOT FOR ADAPTATION PARAMETERS #########

if plotting == True:

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Data driven kde estimate using NeffNeff! Full pakke!
    im1 = axs[1, 2].imshow(kde_data_driven, vmin=vmin, vmax=vmax)
    axs[1, 2].set_title('Data driven estimator')
    fig.colorbar(im1, ax=axs[1, 2], shrink=0.8)

    # Plot 2: Bandwidth estimate
    im2 = axs[0, 2].imshow(h)
    axs[0, 2].set_title('Bandwidth estimate')
    fig.colorbar(im2, ax=axs[0, 2], shrink=0.8)

    # Plot 3: Integral length scale
    im3 = axs[0, 1].imshow(N_eff_advanced)
    axs[0, 1].set_title('N_eff_advanced')
    fig.colorbar(im3, ax=axs[0, 1], shrink=0.8)

    # Plot 4: Standard deviation estimate
    im4 = axs[0, 0].imshow(std_estimate)
    axs[0, 0].set_title('Standard deviation estimate')
    fig.colorbar(im4, ax=axs[0, 0], shrink=0.8)

    # Plot 5: Naive histogram estimate
    im5 = axs[1, 1].imshow(naive_estimate, vmin=vmin, vmax=vmax)
    axs[1, 1].set_title('Naive histogram estimate')
    fig.colorbar(im5, ax=axs[1, 1], shrink=0.8)

    # Plot 6: Ground truth
    im6 = axs[1, 0].imshow(ground_truth, vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Ground truth')
    fig.colorbar(im6, ax=axs[1, 0], shrink=0.8)

    plt.tight_layout()
    plt.show()

    #plot a histogram of the non-zero elements of the standard deviation matrix and N_eff matrix in a subplot
    plt.figure()
    plt.subplot(1,3,1)
    plt.hist(std_estimate[std_estimate != 0].flatten(),bins=100)
    plt.title('Standard deviation')
    plt.subplot(1,3,2)
    plt.hist(N_eff_advanced[N_eff_advanced != 0].flatten(),bins=100)
    plt.title('N_eff')
    plt.subplot(1,3,3)
    plt.hist(h[h != 0].flatten(),bins=100)
    plt.title('Bandwidth')
    plt.show()

    #Plot weight estimate, small_n_eff and N_eff_advanced
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    # Plot the histogram for N_eff_advanced
    axes[2].hist(N_eff_advanced[N_eff_advanced > 0].flatten(), bins=30, color='blue', edgecolor='black')
    axes[2].set_title('Histogram of N_eff_advanced')
    axes[2].set_xlabel('N_eff_advanced')
    axes[2].set_ylabel('Frequency')
    # Plot the histogram for weight_estimate
    axes[0].hist(weight_estimate[weight_estimate > 0].flatten(), bins=30, color='green', edgecolor='black')
    axes[0].set_title('Histogram of Weight Estimate')
    axes[0].set_xlabel('Weight Estimate')
    axes[0].set_ylabel('Frequency')
    # Plot the histogram for small_n_eff
    axes[1].hist(small_n_eff[small_n_eff > 0].flatten(), bins=30, color='red', edgecolor='black')
    axes[1].set_title('Histogram of Small N_eff')
    axes[1].set_xlabel('Small N_eff')
    axes[1].set_ylabel('Frequency')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()


    #Set colormap to be used for the pcolor and contour plots
    cmap = 'mako_r'


    ### MAKE PLOT OF ADAPTATION PARAMETERS ###
    # Replace NaN values with zeros
    std_estimate = np.nan_to_num(std_estimate, nan=0.0)
    N_eff_advanced = np.nan_to_num(N_eff_advanced, nan=0.0)
    h = np.nan_to_num(h, nan=0.0)

    # Create the first 2x2 figure with GridSpec
    fig1 = plt.figure(figsize=(12.5, 10))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # Plot Standard deviation estimate using pcolor
    ax1 = fig1.add_subplot(gs[0, 0])
    pc1 = ax1.pcolor(std_estimate, cmap=cmap)
    ax1.set_title('Standard Deviation Estimate', fontsize=14)
    fig1.colorbar(pc1, ax=ax1)

    # Plot N_eff_advanced estimate using pcolor
    ax2 = fig1.add_subplot(gs[0, 1])
    pc2 = ax2.pcolor(N_eff_advanced, cmap=cmap)
    ax2.set_title('N_eff Estimate', fontsize=14)
    fig1.colorbar(pc2, ax=ax2)

    # Plot Bandwidth estimate using pcolor
    ax3 = fig1.add_subplot(gs[1, 0])
    pc3 = ax3.pcolor(h, cmap=cmap)
    ax3.set_title('Bandwidth Estimate', fontsize=14)
    fig1.colorbar(pc3, ax=ax3)

    # Create a histogram with tighter x-limits
    ax_hist = fig1.add_subplot(gs[1, 1])

    ax_hist.hist(std_estimate[std_estimate != 0].flatten(), bins=100, color='blue', edgecolor='black', alpha=0.5, label='Standard Deviation')
    ax_hist.hist(h[h != 0].flatten(), bins=100, color='red', edgecolor='black', alpha=0.5, label='Bandwidth')
    ax_hist.hist(N_eff_advanced[N_eff_advanced != 0].flatten()**(1/6), bins=100, color='green', edgecolor='black', alpha=0.5, label='N_eff$^{1/6}$')
    #ax_hist.set_xlabel('Value', fontsize=14)
    #ax_hist.set_ylabel(' fontsize=14)
    ax_hist.legend(fontsize=12)
    # Adjust the size of subplot(gs[1,1]) such that it has the same width and height as the other three plots
    pos1 = ax_hist.get_position()  # get the original position
    pos2 = [pos1.x0, pos1.y0, pos1.width , pos1.height]
    ax_hist.set_position(pos2)  # set a new position

    # Manually remove whitespace between subplots
    plt.tight_layout()

    plt.show()

    ### PLOT THE GROUND TRUTH ###
    # Assuming ground_truth, naive_estimate, kde_data_driven, kde_time_bw, and kde_silverman_naive are defined

    vmin = ground_truth.min()
    vmax = ground_truth.max()
    levels = np.linspace(vmin, vmax, 50)

    # Adjust for padding
#    ground_truth = ground_truth[pad_size:-pad_size, pad_size:-pad_size]
#    naive_estimate = naive_estimate[pad_size:-pad_size, pad_size:-pad_size]
    kde_data_driven = kde_data_driven[pad_size:-pad_size, pad_size:-pad_size]
    kde_time_bw = kde_time_bw[pad_size:-pad_size, pad_size:-pad_size]
    kde_data_driven_naive = kde_data_driven_naive[pad_size:-pad_size, pad_size:-pad_size]
#    kde_silverman_naive = kde_silverman_naive[pad_size:-pad_size, pad_size:-pad_size]
#    illegal_positions = illegal_positions[pad_size:-pad_size, pad_size:-pad_size]

    cmap = 'mako_r'

    # Create a figure with a GridSpec layout
    fig = plt.figure(figsize=(12.5, 15))
    gs = GridSpec(3, 2, figure=fig)

    # Plot the ground truth in the center top position
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = ax1.contourf(ground_truth, levels=levels, cmap=cmap, extend='max')
    ax1.set_title('Ground truth', fontsize=14)
    ax1.contour(ground_truth, levels=levels[::2], colors='black', linewidths=0.25)
    cbar1 = fig.colorbar(cf1, ax=ax1, extend='max', location='right')
    cbar1.formatter = ScalarFormatter(useMathText=True)
    cbar1.formatter.set_scientific(True)
    cbar1.formatter.set_powerlimits((0, 0))
    cbar1.update_ticks()

    # Add a black line around the illegal cells
    #ax1.contour(illegal_positions, levels=[0.5], colors='black', linewidths=1)

    # Hide the top right subplot
    fig.delaxes(fig.add_subplot(gs[0, 1]))

    # Plot Naive estimate using contourf
    ax2 = fig.add_subplot(gs[1, 0])
    cf2 = ax2.contourf(naive_estimate, levels=levels, cmap=cmap, extend='max')
    #cf2 = ax2.pcolor(naive_estimate, vmin = np.min(levels),vmax = np.max(levels))
    ax2.set_title('Histogram estimate', fontsize=14)
    ax2.contour(naive_estimate, levels=levels[::2], colors='black', linewidths=0.25)
    cbar2 = fig.colorbar(cf2, ax=ax2, extend='max', location='right')
    cbar2.formatter = ScalarFormatter(useMathText=True)
    cbar2.formatter.set_scientific(True)
    cbar2.formatter.set_powerlimits((0, 0))
    cbar2.update_ticks()

    # Plot Data driven estimate using NeffNeffNeff using contourf
    ax3 = fig.add_subplot(gs[1, 1])
    cf3 = ax3.contourf(kde_data_driven_naive, levels=levels, cmap=cmap, extend='max')
    ax3.set_title('AKDE', fontsize=14)
    ax3.contour(kde_data_driven_naive, levels=levels[::2], colors='black', linewidths=0.25)
    cbar3 = fig.colorbar(cf3, ax=ax3, extend='max', location='right')
    cbar3.formatter = ScalarFormatter(useMathText=True)
    cbar3.formatter.set_scientific(True)
    cbar3.formatter.set_powerlimits((0, 0))
    cbar3.update_ticks()

    # Plot time dependent bandwidth estimate using contourf
    ax4 = fig.add_subplot(gs[2, 0])
    cf4 = ax4.contourf(kde_time_bw, levels=levels, cmap=cmap, extend='max')
    ax4.set_title('TKDE and boundary control', fontsize=14)
    ax4.contour(kde_time_bw, levels=levels[::2], colors='black', linewidths=0.25)
    cbar4 = fig.colorbar(cf4, ax=ax4, extend='max', location='right')
    cbar4.formatter = ScalarFormatter(useMathText=True)
    cbar4.formatter.set_scientific(True)
    cbar4.formatter.set_powerlimits((0, 0))
    cbar4.update_ticks()

    # Plot Silverman (naive) estimate using contourf
    ax5 = fig.add_subplot(gs[2, 1])
    cf5 = ax5.contourf(kde_silverman_naive, levels=levels, cmap=cmap, extend='max')
    ax5.set_title('Silverman (non-adaptive) KDE', fontsize=14)
    ax5.contour(kde_silverman_naive, levels=levels[::2], colors='black', linewidths=0.25)
    cbar5 = fig.colorbar(cf5, ax=ax5, extend='max', location='right')
    cbar5.formatter = ScalarFormatter(useMathText=True)
    cbar5.formatter.set_scientific(True)
    cbar5.formatter.set_powerlimits((0, 0))
    cbar5.update_ticks()

    # Plot the data adaptive data driven estimate with boundary control in the remaining subplot
    ax6 = fig.add_subplot(gs[0, 1])
    cf6 = ax6.contourf(kde_data_driven, levels=levels, cmap=cmap, extend='max')
    ax6.set_title('AKDE and boundary control', fontsize=14)
    ax6.contour(kde_data_driven, levels=levels[::2], colors='black', linewidths=0.25)
    cbar6 = fig.colorbar(cf6, ax=ax6, extend='max', location='right')
    cbar6.formatter = ScalarFormatter(useMathText=True)
    cbar6.formatter.set_scientific(True)
    cbar6.formatter.set_powerlimits((0, 0))
    cbar6.update_ticks()

    illegal_positions_0_1 = np.logical_not(illegal_positions).astype(int)

    # Add a black line around the illegal cells for each plot
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.contour(illegal_positions, levels=[0,1], colors='black', linewidths=2)
        ax.pcolormesh(illegal_positions_0_1[1:,:], cmap='gray', alpha=0.2, vmin=0, vmax=1,shading='flat')
        #fill the illegal positions with a transparent color
        #ax.imshow(illegal_positions, cmap='gray')
        
        #ax.imshow(illegal_positions_0_1, cmap='gray', alpha=0.3, vmin=0, vmax=1, extent=ax6.get_xlim() + ax6.get_ylim(), origin='lower')
        #Do this in a different way
        #ax.imshow(illegal_positions, cmap='gray', alpha=0.5)


    plt.tight_layout()
    plt.show()



    ########################################

    ###### testing reflection stuff #######

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Example usage
x0, y0 = 5, 5
xi = np.arange(11)
yj = np.arange(11)
legal_grid = np.ones((11, 11), dtype=bool)
legal_grid[7, 8] = False  # Example of an illegal cell
legal_grid[8,7] = False
legal_grid[7,7] = False

#creatte an island in some other octant
legal_grid[:3,4:6] = False
legal_grid[2,3] = False
legal_grid[2,2] = False


shadowed_cells = identify_shadowed_cells(x0, y0, xi, yj, legal_grid)
# Plot all shadowed cells in yellow, the illegal cell in red, and the origin as a green dot
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(0, 11, 1))
ax.grid(False)  # Disable the grid

# Plot a Gaussian centered at the origin
x, y = np.mgrid[0:11, 0:11]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
rv = multivariate_normal([5, 5], [[4, 0], [0, 4]])
density = ax.pcolor(x, y, rv.pdf(pos), label='Kernel density')

# Plot the shadowed cells
for cell in shadowed_cells:
    ax.add_patch(plt.Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, color='yellow', alpha=1))

# Plot the legal and illegal cells
for m in range(11):
    for n in range(11):
        if legal_grid[m, n]:
            ax.add_patch(plt.Rectangle((m - 0.5, n - 0.5), 1, 1, fill=None, edgecolor='black'))
        else:
            ax.add_patch(plt.Rectangle((m - 0.5, n - 0.5), 1, 1, color='black'))

# Calculate the line from 5,5 to 8,1 and plot it
cells = bresenham(x0, y0, 8, 1)
for cell in cells:
    ax.add_patch(plt.Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, color='grey', alpha=0.7))

# Plot the line
ax.plot([x0, 8], [y0, 1], color='white', marker='o')

# Create legend patches
shadowed_patch = mpatches.Patch(color='yellow', label='Blocked cells')
illegal_patch = mpatches.Patch(color='black', label='Bathymetry/land')
plt.legend(handles=[illegal_patch, shadowed_patch])

# Plot the origin
ax.plot(x0, y0, color='green', marker='o')
ax.text(x0+2.3, y0, 'x_0, y_0', color='white', fontsize=10, ha='right',
        bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0'))

# Plot the endpoint
ax.plot(8, 1, color='white', marker='o')
ax.text(8+0.34, 1, 'x_1, y_1', color='white', fontsize=10, ha='left',
        bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0'))

plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.xlabel('x')
plt.ylabel('y')

plt.show()

# Plot the grid and the shadowed cells for each iteration
not_now = 1
if not_now == 0:
    for i in xi:
        for j in yj:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xticks(np.arange(0, 10, 1))
            ax.set_yticks(np.arange(0, 10, 1))
            ax.grid(True)

            # Plot the legal and illegal cells
            for m in range(10):
                for n in range(10):
                    if legal_grid[m, n]:
                        ax.add_patch(plt.Rectangle((m - 0.5, n - 0.5), 1, 1, fill=None, edgecolor='black'))
                    else:
                        ax.add_patch(plt.Rectangle((m - 0.5, n - 0.5), 1, 1, color='red'))

            # Get the intersecting cells
            cells = bresenham(x0, y0, i, j)

            # Plot the intersecting cells
            for cell in cells:
                ax.add_patch(plt.Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, color='blue', alpha=0.5))

            # Identify shadowed cells
            shadowed_cells = identify_shadowed_cells(x0, y0, [i], [j], legal_grid)

            # Plot the shadowed cells
            for cell in shadowed_cells:
                ax.add_patch(plt.Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, color='yellow', alpha=0.5))

            # Plot the line
            ax.plot([x0, i], [y0, j], color='green', marker='o')

            plt.xlim(-1, 10)
            plt.ylim(-1, 10)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Line from ({x0}, {y0}) to ({i}, {j})')
            plt.show()
    
