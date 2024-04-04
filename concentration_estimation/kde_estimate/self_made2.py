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


#generate data
mu = [0., 0.]
sigma = [[1, 0], [0, 5]]
n = 10
x,y = np.random.multivariate_normal(mu, sigma, n).T

#plot the points
fig, ax = plt.subplots()
ax.scatter(x, y, alpha=1, s=10, c='k')

#kde parameters
bw = 0.1 #bandwidth

#assign a gaussian 2d kernel to each datapoint and compute the sum on a grid
#Define the grid
xmin = np.min(x)-3
xmax = np.max(x)+3
ymin = np.min(y)-3
ymax = np.max(y)+3
#Define the resolution
dxy_grid = 0.1
#Create a grid
x_grid = np.arange(xmin,xmax,dxy_grid)
y_grid = np.arange(ymin,ymax,dxy_grid)
#Create a grid of points
xx, yy = np.meshgrid(x_grid,y_grid)

#evaluate the kernel on the datapoints and grid
grid_points = np.vstack([xx.ravel(),yy.ravel()])
#Create a kernel matrix for each datapoint
kernel_matrix_sum = np.zeros((len(grid_points[0])))
#craete a matrix that can hold 3x3 kernel matrices
kernel_matrix = np.zeros((n,3,3))
for i in range(n):
    #calculate the kernel for each datapoint
    #kernel_matrix[i,:] = gaussian_kernel_2d(grid_points[0]-x[i],grid_points[1]-y[i],bw=bw)
    #create a matrix for creating the 3x3 kernel matrix
    a = np.array([[-bw,0,bw],[-bw,0,bw],[-bw,0,bw]])
    b = np.array([[bw,bw,bw],[0,0,0],[-bw,-bw,-bw]])
    #kernel_matrix[i,:] = #gaussian_kernel_2d_sym(a,b,bw=1, norm='l2norm')
    kernel_matrix[i,:,:] = ((1/(2*np.pi*bw**2))*np.exp(-0.5*((a/bw)**2+(b/bw)**2)))/np.sum(((1/(2*np.pi*bw**2))*np.exp(-0.5*((a/bw)**2+(b/bw)**2))))
    #add the kernel_matrix values by binning them into the grid using digitize
    #get the indices of the grid points that are closest to the datapoints
    #locaatio of kernel function using a, b and x/y
    lx = a+x[i]
    ly = b+y[i]
    #get the indices of the grid points that are closest to the datapoints
    ix = np.digitize(lx,x_grid)
    iy = np.digitize(ly,y_grid)
    #add the kernel values to the kernel_matrix_sum
    kernel_matrix_sum[ix,iy] += kernel_matrix[i,:,:].ravel()  

    



    #check if the kernle is normalized
    print(np.sum(kernel_matrix[i,:]))

#Sum the kernel matrices
z = np.sum(kernel_matrix,axis=0)
#Reshape the z values to the grid
z = z.reshape(xx.shape)

#Plot the kernel density estimate
N = 8 #number of contours
fig, ax = plt.subplots()
ax.contour(xx, yy, z, N, linewidths=0.8, colors='k')
ax.contourf(xx, yy, z, N,cmap='RdBu_r')
#plot the datapoints
ax.scatter(x, y, alpha=1, s=10, c='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Kernel density estimate')
plt.show()





