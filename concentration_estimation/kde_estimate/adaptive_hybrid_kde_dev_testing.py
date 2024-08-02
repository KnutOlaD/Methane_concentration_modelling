'''
Script for testing the adaptive hydrid Kernel Density estimator

Author: Knut Ola Dølven

'''

import numpy as np
import matplotlib.pyplot as plt
#import sns to make use of th ecolormaps there
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

#Triggers
create_test_data = False
load_test_data = True
plotting = True

# Parameters
grid_size = 150
grid_size_plot = 100
x_grid = np.linspace(0, grid_size, grid_size)
y_grid = np.linspace(0, grid_size, grid_size)

num_particles_per_timestep = 5000 #seeded one after the other
time_steps = 380
dt = 0.1
stdev = 1.4 #Stochastic std
U_a = np.array([2.5, 2.5]) #Advection velocity
num_particles = num_particles_per_timestep*time_steps
# Release position
release_position = np.array([10, 10])
#Make U_a a periodic function with size time_steps
U_a = [0,5] #Initial value
#Initial magnitude
magU = np.sqrt(U_a[0]**2+U_a[1]**2)
U_a = np.tile(U_a, (time_steps, 1))
for i in range(1, time_steps):
    U_a[i][:][0] = 2*magU+ np.sin(i/50)*2*magU
    print(np.sin(i/10))
    #L2 normalize the velocity
    U_a[i] = (U_a[i]/(np.sqrt(U_a[i][0]**2+U_a[i][1]**2)))*magU #Concervation of mass

#plt.plot(U_a[:][:,0])


#Let the current be periodic in the x-direction
def histogram_estimator(particles, grid_size):
    '''
    Input:
    particles: np.array of shape (num_particles, 2)
    grid_size: int
    
    Output:
    particle_count: np.array of shape (grid_size, grid_size)
    '''
    # Create a 2D histogram of particle positions
    particle_count = np.zeros((grid_size, grid_size))
    for particle in particles:
        x, y = particle
        if np.isnan(x) or np.isnan(y):
            continue
        x = int(x)
        y = int(y)
        if x >= grid_size or y >= grid_size or x < 0 or y < 0:
            continue
        particle_count[x, y] += 1

    # Flip the x-axis to match the orientation of the plot
    particle_count = particle_count.T
    

    return particle_count


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

def compute_rdf(particles, bin_width=1.0, max_distance=50):
    num_particles = particles.shape[0]
    distances = np.zeros((num_particles, num_particles))
    
    # Calculate distances between all pairs of particles
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            dist = np.linalg.norm(particles[i] - particles[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Bin the distances to create the RDF
    bins = np.arange(0, max_distance + bin_width, bin_width)
    rdf, _ = np.histogram(distances, bins=bins)
    
    # Normalize the RDF
    rdf = rdf / num_particles
    return rdf, bins[:-1]

if create_test_data == True:
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


if load_test_data == True:
    # Load test data
    with open('trajectories_full.pkl', 'rb') as f:
        trajectories_full = pickle.load(f)
    with open('bw.pkl', 'rb') as f:
        bw = pickle.load(f)
else:
    # Create test data
    trajectories_full = trajectories[-1]
#load test data

### CREATE TEST DATA ###
#fractional difference
frac_diff = 1000
trajectories = trajectories_full[::frac_diff,:]
#pick only the right data from the bw vector
bw = bw[::frac_diff]
weights = np.ones(len(bw))

####################################################
####### MAKE COLOR PLOT OF PARTICLE DENSITY ########
####################################################

histogram_estimator_est = histogram_estimator(trajectories, grid_size)/np.sum(histogram_estimator(trajectories, grid_size))
#transpose
kernel_density_estimator_est = kernel_matrix_2d_NOFLAT(trajectories[:,0],trajectories[:,1],x_grid,y_grid,bw,weights)
kernel_density_estimator_est = kernel_density_estimator_est.T
#normalize
kernel_density_estimator_est = kernel_density_estimator_est/np.sum(kernel_density_estimator_est)

ground_truth = histogram_estimator(trajectories_full, grid_size)/np.sum(histogram_estimator(trajectories_full, grid_size))


#####################################################
#####################################################
#################### PLOTTING #######################
#####################################################
#####################################################
#####################################################

if plotting == False:
    #stop the script here
    exit()

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
levels = np.linspace(vmin, vmax, 20)
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

plt.tight_layout()
plt.show()
#print the summed differences
print('Summed differences between the two estimators')
print('Difference, histogram estimator')
print(np.sum(np.abs(ground_truth - histogram_estimator_est)))
print('Difference, kernel density estimator')
print(np.sum(np.abs(ground_truth - kernel_density_estimator_est)))