import sys
sys.path.append('/cluster/home/hes001/Methane_dispersion_modelling/opendrift/')
sys.path.append('/cluster/home/hes001/Methane_dispersion_modelling/fvtools/')

# clone opendrift here: https://github.com/OpenDrift/opendrift
import matplotlib.pyplot as plt
from opendrift.readers import reader_ROMS_native, reader_netCDF_CF_generic
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.models.oceandrift import OceanDrift
from datetime import datetime, timedelta
import pyproj
import os
import pandas as pd
#import norkyst.roms_grid as rg
import fvtools.grid.roms_grid as rg
import numpy as np

RADIUS = 100

def run(with_diffusion = False, zlevel_norkyst = False, hours_pr_release = 1, use_local = False):
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
    start = datetime(2018, 4, 20)
    stop  = datetime(2018, 6, 25)
    dates = [start - timedelta(days = 1) + timedelta(days = n) for n in range((stop + timedelta(days=1) - start).days)]

    # If using local files
    if not zlevel_norkyst:
        if use_local:
            o = get_local_reader(dates, o)
        else:
            o = get_thredds_reader(dates, o)

    else:
        o = get_thredds_zlevel_reader(dates, o)

    # Configure particle behaviour, environmental fallback values and vertical mixing parameterization parameters
    # ----
    print('Configure opendrift simulation')
    o = set_config(o)

    if with_diffusion and not zlevel_norkyst:
        o.set_config('environment:fallback:ocean_mixed_layer_thickness', 75) # Set the MLD to 75 meters.
        o.set_config('drift:vertical_mixing', True)
        o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
        o.set_config('vertical_mixing:timestep', 20.) # copied from the test cases, not sure if this is strict

    elif with_diffusion and zlevel_norkyst:
        o.set_config('drift:vertical_mixing', True)
        o.set_config('vertical_mixing:diffusivitymodel', 'environment')
        o.set_config('vertical_mixing:timestep', 20.) # copied from the test cases, not sure if this is strict

    elif not with_diffusion and zlevel_norkyst:
        o.set_config('drift:vertical_mixing', False)
        
    # Define particle seed times
    start_times = [start + timedelta(hours=n) for n in range((stop - start).days*24)]
    
    # Seed particles at each depth at each location
    for depth, depth_group in frames.groupby('# depth [m]'):
        print(f'\nSeeding at {depth} meters')
        for particle_flux, flux_group in depth_group.groupby('particle flux [1/h]'):
            print(f'  - {int(particle_flux)} particles at one location')
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

    if with_diffusion and not zlevel_norkyst:
        savename = 'drift_norkyst_unlimited_vdiff.nc'
        
    elif not with_diffusion and not zlevel_norkyst:
        savename = 'drift_norkyst_unlimited.nc'
        
    elif with_diffusion and zlevel_norkyst:
        savename = 'drift_norkyst_zlevel_unlimited_vdiff.nc'

    else:
        savename = 'drift_norkyst_zlevel_unlimited.nc'
        
    # Run the particle model
    o.run(
        time_step=300,
        duration=timedelta(days=(stop-start).days), 
        time_step_output=3600,
        outfile=savename,
        export_buffer_length=12,
        export_variables=['time', 'lon', 'lat', 'z'],
    )

# Functions to find forcing files
# ---- 
def get_local_reader(dates, o):
    '''
    Find local NorKyst files between start and stop stored in a directory
    '''
    nirdbase = '/nird/projects/NS9067K/apn_backup/ROMS/NK800_2018'
    betzybase = '/cluster/work/users/hes001/ROMS/NK800_2018'
    files = [f'{betzybase}/norkyst_800m_his.nc4_{date.strftime("%Y%m%d")}01-{(date+timedelta(days=1)).strftime("%Y%m%d")}00' 
             for date in dates]
    o.add_reader(reader_ROMS_native.Reader(files))
    return o

def get_thredds_zlevel_reader(dates, o):
    '''
    Get zlevel files stored on thredds
    '''
    # For å hente grid for ROMS-NK
    print('Link to MET servers to get z-level norkyst data')
    M = rg.get_roms_grid('MET-NK-Z', pyproj.Proj('EPSG:32633'))

    print('Load grid')
    M.load_grid()

    print('Link to MET servers to get NorShelf data')
    N = rg.get_roms_grid('H-NS', pyproj.Proj('EPSG:32633')) # Bruker NorShelf for å fylle hull
    N.load_grid()

    # Opening consequtive ROMS files using MFDataset. Making sure not to do so over gaps
    metnorkyst_files = []

    print('Finding ROMS files')
    for d in dates:
        try:
            fil = M.test_day(d)
            metnorkyst_files.append(fil)
            print(f'- Found {fil}')

        except:
            # Make a reader out of the available files stored to the roms_files list
            print(f'- {d} is not available from MET-NorKyst zlevels')
            if any(metnorkyst_files):
                o.add_reader(reader_netCDF_CF_generic.Reader(metnorkyst_files))
                roms_files = []

            # Fill date with data from NorShelf
            try:
                # This is not ideal, since it could end up making multiple readers for the same day
                present = N.test_day(d)
                past    = N.test_day(d-timedelta(days=1))
                future  = N.test_day(d+timedelta(days=1))
                o.add_reader(reader_ROMS_native.Reader([past, present, future]))
                print(f'  - Filled {d} with data from hourly NorShelf')

            except:
                basepath_local = '/cluster/work/users/hes001/ROMS/NK800_2018/'
                if d == datetime(2018, 4, 10):
                    print(f'  - using local files to fill data not covered by MET files at {d}')
                    o.add_reader(
                        reader_ROMS_native.Reader(
                            [f'{basepath_local}norkyst_800m_his.nc4_2018040901-2018041000',
                             f'{basepath_local}norkyst_800m_his.nc4_2018041001-2018041100']
                        )
                    )
                elif d == datetime(2018, 5, 6):
                    print(f'  - using local files to fill data not coverred by MET files at {d}')
                    o.add_reader(
                        reader_ROMS_native.Reader(
                            [f'{basepath_local}norkyst_800m_his.nc4_2018050501-2018050600',
                             f'{basepath_local}norkyst_800m_his.nc4_2018050601-2018050700']
                        )
                    )
                elif d == datetime(2018, 5, 15):
                    o.add_reader(
                        reader_ROMS_native.Reader(
                            [f'{basepath_local}norkyst_800m_his.nc4_2018051401-2018051500',
                             f'{basepath_local}norkyst_800m_his.nc4_2018051501-2018051600']
                        )
                    )
                else:
                    print(f'  - Could not find any data for {d} at MET-NO servers, consider downloading more IMR NorKyst files.')
                    raise NoAvailableData

    # Read NorKyst data, add as reader
    if any(metnorkyst_files):
        reader_norkyst = reader_netCDF_CF_generic.Reader(metnorkyst_files)
        o.add_reader(reader_norkyst)
        
    return o

def get_thredds_reader(dates, o):
    '''
    Get files stored on thredds (and locally for some problematic dates)
    '''
    M = rg.get_roms_grid('MET-NK', pyproj.Proj('EPSG:32633')) # her kan du også hente data fra andre havmodeller hos met, feks NorShelf - 'NS'
    M.load_grid()

    N = rg.get_roms_grid('H-NS', pyproj.Proj('EPSG:32633')) # Bruker NorShelf for å fylle hull
    N.load_grid()

    # Opening consequtive ROMS files using MFDataset. Making sure not to do so over gaps
    metnorkyst_files = []

    # Where to look for local files
    basepath_local = '/cluster/work/users/hes001/ROMS/NK800_2018/'
    
    print('Finding ROMS files')
    for d in dates:
        try:
            fil = M.test_day(d)
            metnorkyst_files.append(fil)
            print(f'- Found {fil}')

        except:
            # Make a reader out of the available files stored to the roms_files list
            print(f'- {d} is not available from MET-NorKyst')
            if any(metnorkyst_files):
                o.add_reader(reader_ROMS_native.Reader(metnorkyst_files))
                roms_files = []

            # Fill date with data from NorShelf
            try:
                # This is not ideal, since it could end up making multiple readers for the same day
                present = N.test_day(d)
                past    = N.test_day(d-timedelta(days=1))
                future  = N.test_day(d+timedelta(days=1))
                o.add_reader(reader_ROMS_native.Reader([past, present, future]))
                print(f'  - Filled {d} with data from hourly NorShelf')

            except:
                if d == datetime(2018, 4, 10):
                    print(f'  - using local files to fill data not covered by MET files at {d}')
                    o.add_reader(
                        reader_ROMS_native.Reader(
                            [f'{basepath_local}norkyst_800m_his.nc4_2018040901-2018041000',
                             f'{basepath_local}norkyst_800m_his.nc4_2018041001-2018041100']
                        )
                    )
                else:
                    print(f'  - Could not find any data for {d} at MET-NO servers, consider downloading more IMR NorKyst files.')
                    raise NoAvailableData

    # Read NorKyst data, add as reader
    if any(metnorkyst_files):
        reader_norkyst = reader_ROMS_native.Reader(metnorkyst_files)
        o.add_reader(reader_norkyst)

    return o

# Set standard configuration
# ----
def set_config(o):
    '''
    Set the configuration which is the same for both experiments
    '''
    o.set_config('drift:horizontal_diffusivity', 10) # Since this is value apparently is common for NorKyst applications
    o.set_config('general:coastline_action', 'previous') # This way, all particles that do not reach the open boundary, will stay active
    o.set_config('general:seafloor_action', 'lift_to_seafloor') # It makes sense to not let particles advect through the seafloor
    o.set_config('drift:max_age_seconds', timedelta(days = 4*7).total_seconds()) # 4 weeks of data
    o.set_config('drift:stokes_drift', False)
    o.set_config('seed:wind_drift_factor', 0.0)
    o.set_config('environment:fallback:x_sea_water_velocity', None)
    o.set_config('environment:fallback:y_sea_water_velocity', None)
    o.set_config('drift:stokes_drift', False)
    return o

# Functions used when seeding the particles
# -----
def sunflower(n, alpha=0, geodesic=False):
    '''
    Seeds the depth evenly
    '''
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

class NoAvailableData(Exception): pass
