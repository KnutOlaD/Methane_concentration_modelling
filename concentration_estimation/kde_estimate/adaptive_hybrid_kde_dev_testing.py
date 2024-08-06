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

#plt.plot(U_a[:][:,0])


#Let the current be periodic in the x-direction
@numba.jit(parallel=True)
def histogram_estimator(particles, grid_size,weights = None):
    '''
    Input:
    particles: np.array of shape (num_particles, 2)
    grid_size: int
    
    Output:
    particle_count: np.array of shape (grid_size, grid_size)
    '''



    # Create a 2D histogram of particle positions
    particle_count = np.zeros((grid_size, grid_size))
    total_weight = np.zeros((grid_size, grid_size))
    for i in range(len(particles)):
        x, y = particles[i,:]
        if np.isnan(x) or np.isnan(y):
            continue
        x = int(x)
        y = int(y)
        if x >= grid_size or y >= grid_size or x < 0 or y < 0:
            continue
        total_weight[x, y] += 1*weights[i]
        particle_count[x,y] += 1

    particle_count = particle_count.T
    total_weight = total_weight.T

    return total_weight,particle_count


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

# ------------------------------------------------------- #
###########################################################
##################### INITIATION ##########################
###########################################################
# ------------------------------------------------------- #

if create_test_data == True:
    trajectories, bw = create_test_data()

if load_test_data == True:
    # Load test data
    #add directory
    with open(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\src\Dispersion_modelling\concentration_estimation\kde_estimate\trajectories_full.pkl', 'rb') as f:
        trajectories_full = pickle.load(f)
    with open(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\src\Dispersion_modelling\concentration_estimation\kde_estimate\bw.pkl', 'rb') as f:
        bw = pickle.load(f) #This is just the time dependent a priori bandwidth parameter
else:
    # Create test data
    trajectories_full = trajectories[-1]
#load test data

### CREATE TEST DATA ###
trajectories = trajectories_full[::frac_diff,:]
#pick only the right data from the bw vector
bw = bw[::frac_diff]
weights = np.ones(len(bw))

####################################################
####### MAKE COLOR PLOT OF PARTICLE DENSITY ########
####################################################

#Create a weight function that decreases logarithmically with time
weights = 1-np.log(np.linspace(1,100,len(trajectories_full)))/(np.log(100)*2)
#flip weights 
weights = weights[::-1]

weights_test = weights[::frac_diff]

#Histogram estimator estimate
histogram_estimator_est,counts = histogram_estimator(trajectories, grid_size,weights=weights_test)
#Kernel density estimator with time dependent bandwidth estimate
kernel_density_estimator_est = kernel_matrix_2d_NOFLAT(trajectories[:,0],trajectories[:,1],x_grid,y_grid,bw,weights_test)
kernel_density_estimator_est = kernel_density_estimator_est.T

#Normalize everything
histogram_estimator_est = histogram_estimator_est/np.sum(histogram_estimator_est)
kernel_density_estimator_est = kernel_density_estimator_est/np.sum(kernel_density_estimator_est)
ground_truth,count_truth = histogram_estimator(trajectories_full, grid_size,weights=weights)/np.sum(histogram_estimator(trajectories_full, grid_size,weights=weights))

#####################################################
##### CALCULATE PILOT KDE USING PREMADE PACKAGE #####
#####################################################

#create a 2d grid
X, Y = np.meshgrid(x_grid, y_grid)
#...using vstack
positions = np.vstack([X.ravel(), Y.ravel()])

# Sample particle positions
values = np.array(trajectories.T)  # Replace with your actual data
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

#First, calculate Gaussian kernels for h = 1,2,3,4,5,6,7,8,9,10,11,12

del_grid = x_grid[1]-x_grid[0]

gaussian_kernels = []
gaussian_kernels.append(np.array([[1]]))
num_ker = 12 #The number of kernels to calculate
bandwidths_h = np.zeros(num_ker) #Array to store the bandwidth ranges associated with each kernel
#
kernel_origin = []
kernel_origin.append(np.array([0,0]))
for i in range(1,12):
    a = np.arange(-i,i+1,1)
    b = np.arange(-i,i+1,1)
    #make it a 3x3 kernel giving the distance in each dimension (3x3 matrix)
    a = a.reshape(-1,1)
    b = b.reshape(1,-1)
    #2 times the standard deviation 
    h = i*(del_grid*0.5)+0.5*del_grid
    #kernel_matrix[i,:] = #gaussian_kernel_2d_sym(a,b,bw=1, norm='l2norm')
    kernel_matrix = ((1/(2*np.pi*h**2))*np.exp(-0.5*((a/h)**2+(b/h)**2))/np.sum(((1/(2*np.pi*h*h))*np.exp(-0.5*((a/h)**2+(b/h)**2)))))
    gaussian_kernels.append(kernel_matrix)
    bandwidths_h[i] = h #can use just find nearest value here.
    kernel_origin.append(np.array([0,0]))
    
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

# Initialize the result matrix
A = np.zeros((grid_size, grid_size))

kernel_dim = 4 #The half width dimension of the kernels in grid cells (also sets h = kernel_dim*0.5)

# Iterate over the grid points
for i in range(grid_size):
    for j in range(grid_size):
        # Determine the kernel size
        kernel_size = len(gaussian_kernels[kernel_dim]) // 2

        kernel = gaussian_kernels[kernel_dim]
        # Define the window boundaries
        i_min = (i - kernel_size)
        i_max = (i + kernel_size + 1)
        j_min = (j - kernel_size)
        j_max = (j + kernel_size + 1)
        #Mirror the kernel if the kernel goes outside the grid
        # crop the kernel and mirror the part that leaked out of the grid
        if i_min < 0:
            #add the leaked mass to the edge of the kernel
            kernel[-i_min:-i_min-i_min] + np.flipud(kernel[0:-i_min])
            #crop the kernel by distance i_min:0
            kernel = kernel[-i_min:]  
            i_min = 0          
        if i_max > grid_size:
            kernel[:i_max-grid_size] + np.flipud(kernel[-i_max+grid_size:])
            kernel = kernel[:grid_size-i_max]
            i_max = grid_size
        if j_min < 0:
            kernel[:,-j_min:-j_min-j_min] + np.fliplr(kernel[:,0:-j_min])
            kernel = kernel[:,-j_min:]
            j_min = 0
        if j_max > grid_size:
            kernel[:,:j_max-grid_size] + np.fliplr(kernel[:,-j_max+grid_size:])
            kernel = kernel[:,:grid_size-j_max]
            j_max = grid_size
        # Calculate the weighted kernel

        weighted_kernel = kernel * counts[i, j]
        # Add the contribution to the result matrix
        A[i_min:i_max, j_min:j_max] += weighted_kernel

# Normalize the result matrix
A = A / np.sum(A)

############################################################
##### GRID PROJECTED ADAPTIVE KERNEL DENSITY ESTIMATOR #####
############################################################

# Initialize the result matrix
A_adaptive = np.zeros((grid_size, grid_size))

# Find the optimal bandwidth h for each grid point
#Find all non-zero elements in the histogram estimator
non_zero_elements_x,non_zero_elements_y = np.where(counts > 0)
#Loop over the non-zero elements and calculate the optimal bandwidth for each grid point


d2Z_dx2 = np.gradient(np.gradient(counts, axis=0), axis=0)
d2Z_dy2 = np.gradient(np.gradient(counts, axis=1), axis=1)

#compute roughness of the kernel (isotropic Gaussian kernel)
R_k = 1/(2*np.pi)

N = counts[12,12]

h_opt = np.zeros((grid_size, grid_size))

h_opt[12,12] = (R_k/(N*(d2Z_dx2[12,12]+d2Z_dy2[12,12])))**(1/6)

# Iterate over the non_zero_elements and calculate the optimal bandwidth for each grid point
for i, j in zip(non_zero_elements_x,non_zero_elements_y):
    # Calculate the optimal bandwidth
    N = counts[i,j]
    h_opt[i,j] = (R_k/(N*(d2Z_dx2[i,j]+d2Z_dy2[i,j])))**(1/5)
    #find nearest discrete value of h_opt
    h_opt[i,j] = bandwidths_h[np.argmin(np.abs(bandwidths_h-h_opt[i,j]))]

#Different approach 

Z = counts

def get_local_roughness_second_derivative(Z, x_grid, y_grid):
    d2Z_dx2 = np.gradient(np.gradient(Z, axis=0), axis=0)
    d2Z_dy2 = np.gradient(np.gradient(Z, axis=1), axis=1)
    R_fpp_local = d2Z_dx2**2 + d2Z_dy2**2
    return R_fpp_local

R_fpp_local = get_local_roughness_second_derivative(Z, x_grid, y_grid)

R_K = 1 / (2 * np.sqrt(np.pi))

# Define the bandwidths to test
bandwidths_h = np.linspace(0, 10.0, num=12)

# Initialize the array to store the optimal bandwidths
optimal_bandwidths = np.zeros_like(Z)

# Iterate over non-zero elements in the counts matrix
for i in range(len(x_grid)):
    for j in range(len(y_grid)):
        best_h = None
        best_imse = np.inf
        for h in bandwidths_h:
            # Calculate the IMSE for the current bandwidth
            imse = R_K / (Z[i, j] * h**2) + (1/4) * h**4 * R_fpp_local[i, j]
            # Update the best bandwidth if the current IMSE is lower
            if imse < best_imse:
                best_imse = imse
                best_h = h
        # Store the optimal bandwidth for the current grid cell
        optimal_bandwidths[i, j] = best_h


# Iterate over the grid points and calculate the kernel density estimate
for i in range(grid_size):
    for j in range(grid_size):
        # determine the kernel size depending on the optimal bandwidth
        kernel_dim = int(h_opt[i,j])
        # Determine the kernel size
        kernel_size = len(gaussian_kernels[kernel_dim]) // 2

        kernel = gaussian_kernels[kernel_dim]
        # Define the window boundaries
        i_min = (i - kernel_size)
        i_max = (i + kernel_size + 1)
        j_min = (j - kernel_size)
        j_max = (j + kernel_size + 1)
        #Mirror the kernel if the kernel goes outside the grid
        # crop the kernel and mirror the part that leaked out of the grid
        if i_min < 0:
            #add the leaked mass to the edge of the kernel
            kernel[-i_min:-i_min-i_min] + np.flipud(kernel[0:-i_min])
            #crop the kernel by distance i_min:0
            kernel = kernel[-i_min:]  
            i_min = 0          
        if i_max > grid_size:
            kernel[:i_max-grid_size] + np.flipud(kernel[-i_max+grid_size:])
            kernel = kernel[:grid_size-i_max]
            i_max = grid_size
        if j_min < 0:
            kernel[:,-j_min:-j_min-j_min] + np.fliplr(kernel[:,0:-j_min])
            kernel = kernel[:,-j_min:]
            j_min = 0
        if j_max > grid_size:
            kernel[:,:j_max-grid_size] + np.fliplr(kernel[:,-j_max+grid_size:])
            kernel = kernel[:,:grid_size-j_max]
            j_max = grid_size
        # Calculate the weighted kernel

        weighted_kernel = kernel * counts[i, j]
        # Add the contribution to the result matrix
        A_adaptive[i_min:i_max, j_min:j_max] += weighted








#####################################################
#####################################################
#####################################################
#################### PLOTTING #######################
#####################################################
#####################################################
#####################################################

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

