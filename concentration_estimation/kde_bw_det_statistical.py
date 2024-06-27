'''
Determine kernel bandwidth using statistics instead of physics
Author: Knut Ola Dølven
'''
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import netCDF4 as nc
import utm
from scipy.sparse import csr_matrix as csr_matrix
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from numpy.ma import masked_invalid
import imageio
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import seaborn as sns

#

#################
### FUNCTIONS ###
#################

#Gaussian kernel
def gaussian_kernel(x, x0, bw):
    return (1/(bw*np.sqrt(2*np.pi)))*np.exp(-((x-x0)**2)/(2*bw**2))

#calculate the energy of the Gaussian kernel
def energy_gaussian_kernel(bw):
    return 1/(bw*np.sqrt(2*np.pi))

#calculate the bandwidth using the statistics
def bandwidth_statistics(data):
    #calculate the mean
    mean = np.mean(data)
    #calculate the standard deviation
    std = np.std(data)
    #calculate the bandwidth
    bw = 1.06*std*(len(data)**(-1/5))
    return bw

resolution = 0.1
np.sum(gaussian_kernel(np.arange(0, 10, resolution), 5, 1))*resolution
#It's normalized!

