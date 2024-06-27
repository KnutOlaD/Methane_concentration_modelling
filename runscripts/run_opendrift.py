import sys
sys.path.append('/cluster/home/hes001/opendrift/')
sys.path.append('/cluster/home/hes001/Methane_dispersion_modelling/')
# clone opendrift here: https://github.com/OpenDrift/opendrift
import opendrift
import matplotlib.pyplot as plt
from opendrift.readers import reader_ROMS_native
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.models.oceandrift import OceanDrift
from datetime import datetime, timedelta
import pyproj
import os
import pandas as pd
import norkyst.roms_grid as rg
import numpy as np

RADIUS = 100

def run(hours_pr_release = 1):
    '''
    Release NorKyst particles
    - hours_pr_release : how often to release particles
    '''
    # Read files from KnutOla
    base = '/cluster/home/hes001/Methane_dispersion_modelling/particle_profiles/'
    files = [base+file for file in os.listdir(base) if '.txt' in file]
    for i, file in enumerate(files):
        if i == 0:
            frames = pd.read_csv(file, delimiter = '\t', header = 0)
        else:
            frames = pd.concat([frames, pd.read_csv(file, delimiter = '\t', header=0)])

    # initialize geod
    geod = pyproj.Geod(ellps='WGS84')
    
    # Initialize OpenDrift
    o = OceanDrift(loglevel=20)

    # Ok, and then we need to identify readable files
    start = datetime(2018, 5, 20)
    stop  = datetime(2018, 6, 20)
    dates = [start + timedelta(days = n) for n in range((stop+timedelta(days=2)-start).days)]

    M = rg.get_roms_grid('MET-NK', pyproj.Proj('EPSG:32633')) # her kan du også hente data fra andre havmodeller hos met, feks NorShelf - 'NS'
    M.load_grid()
    roms_files = []
    for d in dates:
        try:
            roms_files.append(M.test_day(d))
        except:
            print(f'- {d} is not available')

    # Read NorKyst data, add as reader
    reader_norkyst = reader_ROMS_native.Reader(roms_files)
    o.add_reader(reader_norkyst)

    # Configure particle behaviour
    o.set_config('general:coastline_action', 'previous') # This way, all particles that do not reach the open boundary, will stay active
    o.set_config('general:seafloor_action', 'lift_to_seafloor') # It makes sense to not let particles advect through the seafloor
    o.set_config('drift:max_age_seconds', timedelta(days = 7).total_seconds())
    
    # Define particle seed times
    stop_release = stop - timedelta(days=7)
    start_times = [start + timedelta(hours=n) for n in range((stop_release-start).days*24)]
    
    # Seed particles at each depth
    for depth, depth_group in frames.groupby('# depth [m]'):
        print(f'\nSeeding at {depth} meters')
        for particle_flux, flux_group in depth_group.groupby('particle flux [1/h]'):
            print(f'  - {int(particle_flux)} at each location')
            if particle_flux == 0:
                continue
                
            # Need to loop over each of location as well, and if needbe distribute over a larger area
            lon, lat = np.array([]), np.array([])
            for lon_loc, lat_loc in zip(flux_group['longitude [deg]'].to_numpy(), flux_group['latitude [deg]'].to_numpy()):
                if particle_flux == 1:
                    lon = np.atleast_1d(lon_loc)
                    lat = np.atleast_1d(lat_loc)
                elif particle_flux > 1:
                    lon_tmp = lon_loc * np.ones(int(particle_flux))
                    lat_tmp = lat_loc * np.ones(int(particle_flux))
                    points = np.array(sunflower(int(particle_flux)))
                    distance = np.sqrt(points[:,0]**2 + points[:,1]**2)*RADIUS
                    angle = 360*(np.arctan2(points[:,1], points[:,0])/(2*np.pi))
                    lon_tmp, lat_tmp, az = geod.fwd(lon_tmp, lat_tmp, angle, distance, radians=False)
                    lon = np.append(lon, lon_tmp)
                    lat = np.append(lat, lat_tmp)
            
            for t in start_times:
                o.seed_elements(
                    lon = lon, 
                    lat = lat,
                    z   = -depth,
                    wind_drift_factor = 0.0,
                    time = t,
                )

    # Run the particle model
    o.run(
        time_step=300,
        duration=timedelta(days=(stop-start).days), 
        time_step_output=3600,
        outfile=f'drift_norkyst_nohickup.nc',
        export_buffer_length=12,
        export_variables=['time', 'lon', 'lat', 'z'],
    )

# Functions used when seeding the particles
def sunflower(n, alpha=0, geodesic=False):
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    points = []
    angle_stride = 360 * phi if geodesic else 2 * np.pi / phi ** 2
    b = np.round(alpha * np.sqrt(n))  # number of boundary points
    for k in range(1, n + 1):
        r = radius(k, n, b)
        theta = k * angle_stride
        points.append((r * np.cos(theta), r * np.sin(theta)))
    return points

def radius(k, n, b):
    if k > n - b:
        return 1.0
    else:
        return np.sqrt(k - 0.5) / np.sqrt(n - (b + 1) / 2)