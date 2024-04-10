'''
Estimates the aging constant for particles in the model.

Needs the right data to work. 

'''        
import numpy as np
import matplotlib.pyplot as plt
from numba import jit



@jit(nopython=True)
def calc_diff_dist_2d(x_meas,y_meas):
    #calculate the difference between all points
    x_diff = x_meas[:, np.newaxis] - x_meas
    y_diff = y_meas[:, np.newaxis] - y_meas
    dist = np.sqrt(x_diff**2 + y_diff**2)

    return dist

#create a variable that goes from 1:200,201:400,401:600 etc.
#for each timestep, calculate the distance between all particles
i=0
kk=0
distancemat = np.zeros((6200,len(particles['time'])))
medianmat = np.zeros((6200,len(particles['time'])))
time_mat = np.zeros((6200,len(particles['time']))) #This just stores time after activation
while i < particles['time'].shape[0]:
    idx = np.arange(((kk))*200,((kk+1))*200,1).astype(int)

    UTM_x = particles['UTM_x'][idx,i].compressed()
    UTM_y = particles['UTM_y'][idx,i].compressed()
    #remove nans in the UTM_x/UTM_y data
    UTM_x = UTM_x[~np.isnan(UTM_x)]
    UTM_y = UTM_y[~np.isnan(UTM_y)]
    time_step = particles['time'][i]
    dist = calc_diff_dist_2d(UTM_x,UTM_y)
    distancemat[idx,i]   = np.nanmean(dist)
    medianmat[idx,i] = np.nanmedian(dist)
    i = i+1
    if i == particles['time'].shape[0] and idx[-1] < len(particles['UTM_x'])-200:
        #print(idx)
        kk = kk+1
        i = 0
        print(kk)

#find the first non-zero element in each column and store the index
for i in range(0,len(distancemat)):
    nonzero_elements = np.where(particles['UTM_x'][i,:] != 0)
    if len(nonzero_elements[0]) > 0:
        first_nonzero = nonzero_elements[0][0]
        time_mat[i,~particles['UTM_x'][i,:].mask] = np.arange(0,len(distancemat[i,~particles['UTM_x'][i,:].mask]),1)


distancemat[distancemat == 0] = np.nan
medianmat[medianmat == 0] = np.nan
#take the mean of the distancemat

tmpmat = np.zeros((len(distancemat),len(distancemat[0,:][~np.isnan(distancemat[0,:])])) )* np.nan
for i in range(0,len(distancemat)):
    tmpmat[i,:len(distancemat[i,:][~np.isnan(distancemat[i,:])])] = distancemat[i,:][~np.isnan(distancemat[i,:])]
#calculate average
avg_calc = np.nanmean(tmpmat,axis=0)


#plot all columns in the distancemat as a scatter plot
avg_calc_tmp = []
fig, ax = plt.subplots()
for i in range(0,len(distancemat)):
    tmp = distancemat[i,:][distancemat[i,:] != 0]
    #rempove nans as well
    tmp = tmp[~np.isnan(tmp)]
    ax.plot(tmp)
    ax.plot(avg_calc,linewidth=2,color='white')
    #plot y label (meters) and x labels (timesteps [hr])
    
ax.set_ylabel('Distance [m]')
ax.set_xlabel('Timestep [hr]')
ax.title.set_text('Mean distance between particles at each timestep')
#loop over to obtain average 
plt.show()


tmpmat = np.zeros((len(medianmat),len(medianmat[0,:][~np.isnan(medianmat[0,:])])) )* np.nan
for i in range(0,len(medianmat)):
    tmpmat[i,:len(medianmat[i,:][~np.isnan(medianmat[i,:])])] = medianmat[i,:][~np.isnan(medianmat[i,:])]
#calculate average
avg_calc = np.nanmean(tmpmat,axis=0)


fig, ax = plt.subplots()
for i in range(0,len(distancemat)):
    tmp = medianmat[i,:][medianmat[i,:] != 0]
    #rempove nans as well
    tmp = tmp[~np.isnan(tmp)]
    ax.plot(tmp)
    ax.plot(avg_calc,linewidth=2,color='white')
    #plot y label (meters) and x labels (timesteps [hr])
ax.set_ylabel('Distance [m]')
ax.set_xlabel('Timestep [hr]')
ax.title.set_text('Median distance between particles at each timestep')
plt.show()

#Loop over and obtain avera
