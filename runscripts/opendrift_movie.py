import sys
sys.path.append('/home/hes/Methane_dispersion_modelling/opendrift/')
sys.path.append('/home/hes/Methane_dispersion_modelling/fvtools/')
sys.path.append('/home/hes/Methane_dispersion_modelling/')

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import xarray as xr
import numpy as np
import progressbar as pb

from shapely import Polygon
from functools import cached_property
from datetime import datetime, timedelta

def animate_opendrift(opendrift_file, figsize = [9,9], dpi = 100, fps = 12, frames = None, movie_name = None):
    '''
    Animate the particle drift simulation
    '''
    # Get the codec, set a movie name
    MovieWriter, codec = get_animator()
    if not movie_name:
        movie_name = opendrift_file.split('.nc')[0]
    movie_name = f'{movie_name}.{codec}'

    # Prepare the movie maker
    maker = MovieMaker(opendrift_file, figsize, dpi, frames)
    anim = manimation.FuncAnimation(maker.fig,
                                    maker.update,
                                    frames           = maker.frames if frames is None else frames,
                                    repeat           = False,
                                    blit             = False,
                                    cache_frame_data = False)
    writer = MovieWriter(fps = fps)
    
    # Write the movie
    write_movie(maker, anim, movie_name, writer)

class MovieMaker:
    def __init__(self, opendrift_file, figsize, dpi, frames):
        '''
        Assuming that you used NorKyst to force the experiment
        '''
        # Open the dataset
        self.d   = xr.open_dataset(opendrift_file)

        # Read the number of timesteps
        self.frames = self.d.time.shape[0]

        # Create the progressbar
        widget   = ['- Animating the particle drift', pb.Percentage(), pb.Bar(), pb.ETA()]
        self.bar = pb.ProgressBar(widgets = widget, maxval = self.frames if frames is None else frames)
        self.bar.start()

        # Create the figure, set the background
        self.fig = self.make_figure(figsize = figsize, dpi=dpi)
        self.set_context()

    @cached_property
    def extent(self):
        return [self.d.lon.minval, self.d.lon.maxval, self.d.lat.minval, self.d.lat.maxval]
    
    @property
    def proj(self):
        return ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))

    def update(self, t):
        '''
        Used by the animator
        '''
        self.bar.update(t)
        try:
            self.points.remove()
        except:
            pass
        self.ax.set_title(f'{self.date(t)}')
        self.points, = self.ax.plot(self.d.lon.isel(time=t), self.d.lat.isel(time=t), 'b.', markersize=0.05, transform = ccrs.PlateCarree())
        return self.points
    
    def date(self, index):
        '''
        Returns the date and time at the given timestep
        '''
        time = str(self.d.time.isel(time=index).data).split('T')
        dato = datetime(*[int(t) for t in time[0].split('-')])
        dag = [int(t) for t in time[1].split(':')[:-2]]
        if len(dag) == 1:
            dato = dato + timedelta(hours = dag[0])
        elif len(dag) == 2:
            dato = dato + timedelta(hours = dag[0], minutes = dag[1])
        elif len(dag) == 3:
            dato = dato + timedelta(hours = dag[0], minutes = dag[1], seconds = dag[2])

        return dato
    
    def set_context(self):
        '''
        Plot land, plot domain boundary
        '''
        self.ax = plt.axes(projection = ccrs.Orthographic(central_longitude = int(np.mean(self.extent[:2])), 
                                                          central_latitude  = int(np.mean(self.extent[2:]))))
        self.ax.add_feature(cfeature.LAND)
        self.ax.add_feature(cfeature.BORDERS)
        self.ax.set_extent(self.extent, self.proj)
        self.ax.add_geometries(self.get_norkyst_polygon(), ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=1.5)

        self.gl = self.ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        self.gl.top_labels = False
        self.gl.right_labels = False

    def get_norkyst_polygon(self):
        '''
        Build a polygon for the Norkyst boundary
        '''
        with xr.open_dataset('https://thredds.met.no/thredds/dodsC/fou-hi/new_norkyst800m/his/ocean_his.an.20180512.nc') as nk:
            coords = (
                (nk['lon_rho'][0, 0], nk['lat_rho'][0, 0]),
                (nk['lon_rho'][-1,0], nk['lat_rho'][-1,0]),
                (nk['lon_rho'][-1,-1], nk['lat_rho'][-1,-1]),
                (nk['lon_rho'][0, -1], nk['lat_rho'][0, -1]),
                (nk['lon_rho'][0, 0], nk['lat_rho'][0, 0]),
            )
        return Polygon(coords)
    
    def make_figure(self, figsize, dpi):
        '''
        Set up a figure
        '''
        return plt.figure(figsize = figsize, dpi = dpi)

# Helper functions
def get_animator():
    '''
    Check which animator is available, get going
    '''
    avail = manimation.writers.list()
    if 'ffmpeg' in avail:
        FuncAnimation = manimation.writers['ffmpeg']
        codec = 'mp4'
    elif 'imagemagick' in avail:
        FuncAnimation = manimation.writers['imagemagick']
        codec = 'gif'
    elif 'pillow' in avail:
        FuncAnimation = manimation.writers['pillow']
        codec = 'gif'         
    elif 'html' in avail:
        FuncAnimation = manimation.writers['html']
        codec = 'html'
    else:
        raise ValueError('None of the standard animators are available, can not make the movie')
    return FuncAnimation, codec

def write_movie(mmaker, anim, mname, writer):
    anim.save(mname, writer = writer)
    plt.close('all')
    mmaker.bar.finish()