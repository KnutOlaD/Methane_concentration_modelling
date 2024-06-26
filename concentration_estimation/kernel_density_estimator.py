'''
Self made kernel density estimation

author: Knut Ola Dølven
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
import time as time
from numba import jit, prange, njit

########################################
############# FUNCTIONS ################
########################################

#Run scipy.stats.gaussian_kde on a set of points for comparison
test = 0
if test == 1:

    # Generate data
    mu = [0., 0.]
    sigma = [[1, 0], [0, 5]]
    n = 1000

    x, y = np.random.multivariate_normal(mu, sigma, n).T
    # Plot data
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.2, s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Scatter plot of data')
    plt.show()

    # Estimate density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Plot density
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=1, alpha=0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Scatter plot of data with density')
    plt.show()

    #Create a grid
    #Define the grid
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    #Define the resolution
    dxy_grid = 0.1
    #Create a grid
    x_grid = np.arange(xmin,xmax,dxy_grid)
    y_grid = np.arange(ymin,ymax,dxy_grid)
    #Create a grid of points
    xx, yy = np.meshgrid(x_grid,y_grid)

    time_start = time.time()
    #evaluate the kernel on the datapoints and grid
    grid_points = np.vstack([xx.ravel(),yy.ravel()])
    kde = gaussian_kde(xy)
    z = kde.evaluate(grid_points)
    #Reshape the z values to the grid
    z = z.reshape(xx.shape)
    time_end = time.time()
    print('Time elapsed: ',time_end-time_start)

    #Plot the kernel density estimate
    #number of contours
    N = 8
    fig, ax = plt.subplots()
    ax.contour(xx, yy, z, N, linewidths=0.8, colors='k')
    ax.contourf(xx, yy, z, N,cmap='RdBu_r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Kernel density estimate')
    plt.show()

################################################
############# CREATE MY OWN KDE ################
################################################

#use gaussian kernel
#Define the kernel
def gaussian_kernel(x,bw=1):
    #create a gaussian kernel
    return (1/(bw*np.sqrt(2*np.pi)))*np.exp(-0.5*(x/bw)**2)


def gaussian_kernel_2d(x,y,bw=[[1,0],[0,1]],norm='l2norm'):
    #create a 2d gaussian kernel with bandwidth (standard deviation) bw
    #located at (x,y)
    #normalizes the kernel according to norm which has options
    #'l2norm', 'l1norm', 'maxnorm' or 'none' (no normalization)
    #default is 'l2norm'
    #bw is a 2x2 matrix with the bandwidths on the diagonal
    #and the covariance on the off-diagonal

    #make it quick if the kernel is symmetric (diagonal bandwidth with the same value)
    dens_kernel = np.zeros((len(x),len(y)))
    
#    if bw[0,1] == 0 and bw[1,0] == 0:
#    dens_kernel = ((1/(2*np.pi*bw**2))*np.exp(-0.5*((x/bw)**2+(y/bw)**2)))

    for i in range(len(x)):
        for j in range(len(y)):
            dens_kernel[i,j] = multivariate_normal.pdf([x[i],y[j]],mean=[0,0],cov=bw)

    return normalize_kernel(dens_kernel,norm=norm)

def gaussian_kernel_2d_sym(x,y,bw=1,norm='l2norm'):
    #create a 2d gaussian kernel with bandwidth (standard deviation) bw
    #located at (x,y)
    #normalizes the kernel according to norm which has options
    #'l2norm', 'l1norm', 'maxnorm' or 'none' (no normalization)
    #default is 'l2norm'
    #bw is a 2x2 matrix with the bandwidths on the diagonal
    #and the covariance on the off-diagonal

    #make it quick if the kernel is symmetric (diagonal bandwidth with the same value)
    dens_kernel = ((1/(2*np.pi*bw**2))*np.exp(-0.5*((x/bw)**2+(y/bw)**2)))

    #normalize the kernel
    dens_kernel = dens_kernel/np.sum(dens_kernel)

    return normalize_kernel(dens_kernel,norm=norm)

def normalize_kernel(kernel,norm='l2norm'):
    #normalize the kernel
    if norm == 'l2norm':
        return kernel/np.sum(kernel)
    elif norm == 'l1norm':
        return kernel/np.sum(kernel)
    elif norm == 'maxnorm':
        return kernel/np.max(kernel)
    else:
        return kernel


def add_elements(z_kernelized, ix, iy, kernel_matrix):
    for i in range(ix.shape[0]):
        for j in range(ix.shape[1]):
            z_kernelized[ix[i, j], iy[i, j]] += kernel_matrix[i, j]
    return z_kernelized


    return z_kernelized

#create a function of this
#@jit(nopython=True) THIS ONE DOES NOT HAVE THE ALIASING PROBLEM
#USE THIS. STILL QUITE FAST. 
def kernel_matrix_2d_NOFLAT(x,y,x_grid,y_grid,bw,weights):
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

    Output:
    z_kernelized: a 3d matrix with the kernel values for each datapoint
    '''

    #calculate the grid resolution
    dxy_grid = x_grid[1]-x_grid[0]

    #create a grid for z values
    z_kernelized = np.zeros((len(x_grid),len(y_grid)))

    for i in range(len(x)):
        #calculate the kernel for each datapoint
        #kernel_matrix[i,:] = gaussian_kernel_2d(grid_points[0]-x[i],grid_points[1]-y[i],bw=bw)
        #create a matrix for the kernel that makes sure the kernel resolution fits
        #within the grid resolution (adaptive kernel size)
        ker_size = int(np.ceil((2*bw[i])/dxy_grid)*3)*2
        a = np.linspace(-2*bw[i],2*bw[i],ker_size)
        b = np.linspace(-2*bw[i],2*bw[i],ker_size)
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
        #add the kernel values to the grid
        # Use the function in your code
        z_kernelized[ix,iy] += kernel_matrix*weights[i]
        #z_kernelized[ix,iy] += kernel_matrix
    #reshape z_kernelized to the grid

    return z_kernelized

#create a function of this
@jit(nopython=True)#This can be run in parallell but for N=10000 it was slower, parallel=True)
def kernel_matrix_2d(x,y,x_grid,y_grid,bw,weights):
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
    
    Output:
    z_kernelized: a 2d matrix with the kernel values for each datapoint
    '''

    dxy_grid = x_grid[1]-x_grid[0]
    z_kernelized = np.zeros((len(x_grid),len(y_grid)))
    z_kernelized = z_kernelized.ravel()
    
    for i in range(len(x)):
        #create kernel, first determine the kernel resolution
        ker_size = int(np.ceil((3*bw[i])/dxy_grid))*4
        #ker_size = int(np.ceil((2*bw[i])/dxy_grid)*3)*2
        #ker_size = 28
        #make sure the kernel size is even
        #if ker_size%2 == 0:
        #    ker_size += 1
        #print(ker_size)
        
        #ker_size = 100
        #create grud for the kernel matrix
        a = np.linspace(-3*bw[i],3*bw[i],ker_size)
        b = np.linspace(-3*bw[i],3*bw[i],ker_size)
        a = a.reshape(-1,1)
        b = b.reshape(1,-1)
        kernel_matrix = ((1/(2*np.pi*bw[i]**2))*np.exp(-0.5*((a/bw[i])**2+(b/bw[i])**2)))/np.sum(((1/(2*np.pi*bw[i]**2))*np.exp(-0.5*((a/bw[i])**2+(b/bw[i])**2))))
        #add zeroes to the edges of the kernel matrix
        lx = a+x[i]+dxy_grid/2
        ly = b+y[i]+dxy_grid/2
        ix = np.digitize(lx,x_grid)-1 #CHECK FOR OFF BY ONE ERROR. This is correct due to python indexing startint at zero.
        iy = np.digitize(ly,y_grid)-1
        
        #ixm,iym = np.meshgrid(ix,iy)
        #make a meshgrid without using meshgrid
        ixm = np.zeros((ker_size,ker_size))
        iym = np.zeros((ker_size,ker_size))
        for j in range(ker_size):
            ixm[j,:] = ix.flatten()
            iym[:,j] = iy.flatten()
        
        # Flatten the arrays
        ix_flat = ixm.ravel()
        iy_flat = iym.ravel()
        kernel_matrix_flat = kernel_matrix.ravel()*weights[i] 

        #store indices
        indices = np.zeros((len(ix_flat)))

        # Perform the addition
        for j in range(len(ix_flat)):
            index = int(ix_flat[j]*len(y_grid) + iy_flat.transpose()[j])
            indices[j] = index
            z_kernelized[index] += kernel_matrix_flat[j]

    # Reshape z_kernelized back to 2D
    z_kernelized = z_kernelized.reshape(len(x_grid), len(y_grid))
    #shift the grid to the correct position

    return z_kernelized

###########################
#TESTIN
##########################
'''
x_grid = np.array([-2,-2,-0,1,2])
y_grid = np.array([-1,0,1])

ix = np.digitize(-1.1,x_grid)-1
iy = np.digitize(1.1,y_grid)-1

print(ix,iy)

z_mat = np.zeros((len(x_grid),len(y_grid)))

z_mat[ix,iy] += 1

print(z_mat)

z_mat_flat = np.zeros((len(x_grid),len(y_grid)))
z_mat_flat = z_mat_flat.ravel()

flatindex = ix*len(y_grid)+iy

z_mat_flat[flatindex] += 1

z_mat_flat = z_mat_flat.reshape(len(x_grid),len(y_grid))

print(z_mat_flat)
'''

############################
##########################
#########################

# n IS THE NUMBER OF DATAPOINTS AND N IS THE NUMBER OF CONTOURS IN THE PLOT

#run the following only if script is main
if __name__ == '__main__':

    #set plotting style
    plt.style.use('dark_background')

    #generate data
    mu = [0., 0.]
    sigma = [[1, 0], [0, 5]]
    n = 100
    x,y = np.random.multivariate_normal(mu, sigma, n).T

    #plot the points
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=1, s=10, c='w')

    #kde parameters
    bw = 0.5*np.ones(len(x))
    weights = np.ones(len(x))#*np.linspace(100,1,len(x))

    #assign a gaussian 2d kernel to each datapoint and compute the sum on a grid
    #Define the grid

    #choose new values for x and y randomly from within the domain
    x_old = x
    y_old = y

    xmin = np.min(x_old)
    xmax = np.max(x_old)
    ymin = np.min(y_old)
    ymax = np.max(y_old)

    x = np.random.uniform(xmin,xmax,n)
    y = np.random.uniform(ymin,ymax,n)

    xmin = np.min(x_old)-3
    xmax = np.max(x_old)+3
    ymin = np.min(y_old)-3
    ymax = np.max(y_old)+3

    #Define the resolution
    dxy_grid = 0.1
    #Create a grid
    x_grid = np.arange(xmin,xmax,dxy_grid)
    y_grid = np.arange(ymin,ymax,dxy_grid)
    #create a grid for z values
    #Create a grid of points
    xx, yy = np.meshgrid(x_grid,y_grid)
    #create a grid for z values
    z = np.zeros((len(x_grid),len(y_grid)))

    #evaluate the kernel on the datapoints and grid
    grid_points = np.vstack([xx.ravel(),yy.ravel()])
    #Create a kernel matrix for each datapoint
    kernel_matrix_sum = grid_points*0
    #craete a matrix that can hold 3x3 kernel matrices
    kernel_matrix = np.zeros((n,9,9))
    #create a 3dN vector for holding all the N=nx9 kernel positions and values
    kernel_matrix_locs = np.zeros((n,9,9))

    z = kernel_matrix_2d(x,y,x_grid,y_grid,bw=np.ones(n)*bw,weights=weights)

    Z_NOFLAT = kernel_matrix_2d_NOFLAT(x,y,x_grid,y_grid,bw=np.ones(n)*bw*0.9,weights=weights)
    #make sure z matches xx and yy
    zz = z.transpose()

    #Plot the kernel density estimate
    N = 8 #number of contours
    fig, ax = plt.subplots()
    #ax.contour(xx, yy, zz, N, linewidths=0.8, colors='k')
    #ax.contourf(xx, yy, zz, N,cmap='inferno')
    ax.pcolor(xx, yy, zz, cmap='RdBu_r')

    #plot the datapoints
    ax.scatter(x, y, alpha=1, s=10, c='w')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Kernel density estimate')
    plt.show()

    #MAKE MESHGRID WITH NOFLAT
    #Plot the kernel density estimate
    zz = Z_NOFLAT.transpose()

    #plot with NOFLAT
    fig, ax = plt.subplots()
    #ax.contour(xx, yy, zz, N, linewidths=0.8, colors='k')
    #ax.contourf(xx, yy, zz, N,cmap='inferno')
    ax.pcolor(xx, yy, zz, cmap='RdBu_r')
    ax.scatter(x, y, alpha=1, s=10, c='w')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Kernel density estimate NOFLAT')
    plt.show()





