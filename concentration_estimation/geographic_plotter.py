"""
Geographic Data Visualization Function

Provides a functino for creating publication-ready maps with customizable features
including contours, colorbars, progress bars, and points of interest.
Uses polar stereographic projection.
Part of the methane concentration modeling framework.

Requires:
- numpy
- matplotlib
- cartopy
- gc

See also function documentation for more details.

Author: Knut Ola Dølven
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter
import gc
from scipy.interpolate import RegularGridInterpolator
import pandas as pd


# Cache features for reuse
_CACHED_FEATURES = {}
def find_bathymetry_line(bathymetry, lon, lat):
    """
    Extract all bathymetry contour lines from a boolean/integer array.
    
    Parameters
    ----------
    bathymetry : np.ndarray
        2D boolean/integer array where 1s are above and 0s below the line
    lon : np.ndarray
        Longitude coordinates matrix
    lat : np.ndarray
        Latitude coordinates matrix
        
    Returns
    -------
    tuple
        (lines_x, lines_y) lists of coordinates for each contour line
    """
    from skimage import measure
    
    # Find contours in index space
    contours = measure.find_contours(bathymetry, level=0.5)
    
    if not contours:
        return None, None
    
    # Get the actual grid dimensions
    ny, nx = bathymetry.shape
    
    # Create regular grids for interpolation
    y_grid = np.arange(ny)
    x_grid = np.arange(nx)
    
    # Create interpolators for longitude and latitude
    lon_interp = RegularGridInterpolator((y_grid, x_grid), lon)
    lat_interp = RegularGridInterpolator((y_grid, x_grid), lat)
    
    # Process all contours
    lines_x = []
    lines_y = []
    
    for contour in contours:
        # Convert from index space to coordinate space
        points = np.column_stack((contour[:, 0], contour[:, 1]))
        line_x = lon_interp(points)
        line_y = lat_interp(points)
        
        lines_x.append(line_x)
        lines_y.append(line_y)
    
    return lines_x, lines_y

def plot_2d_data_on_map(data, lon, lat, projection, levels, timepassed,
                         colormap, title, unit, left_labels=True, 
                         bathymetry_line=None,**kwargs):

    """
    Creates a map visualization with contours, colorbar, and time progression.

    Parameters
    ----------
    data : np.ndarray
        2D array containing the data to plot
    lon : np.ndarray
        1D or 2D array of longitude coordinates
    lat : np.ndarray
        1D or 2D array of latitude coordinates
    projection : cartopy.crs
        Map projection to use
    levels : np.ndarray
        Contour levels for the plot
    timepassed : list
        [current_time, total_time] for progress bar
    colormap : str or matplotlib.colors.Colormap
        Colormap for the contour plot
    title : str
        Plot title
    unit : str
        Units for colorbar label
    bathymetry_line : tuple
        (line_x, line_y) coordinates of the bathymetry line
    **kwargs : dict
        Optional parameters:
            savefile_path : str
                Path to save the figure
            show : bool
                Whether to display the plot
            adj_lon : list [float, float]
                [min, max] longitude adjustments
            adj_lat : list [float, float]
                [min, max] latitude adjustments
            bar_position : list [float, float, float, float]
                [x, y, width, height] for progress bar
            dpi : int
                DPI for saved figure (default: 150)
            log_scale : bool
                Use logarithmic color scale (default: False)
            figuresize : list [float, float]
                Figure dimensions in inches (default: [14, 10])
            plot_model_domain : bool or list
                If True: automatically plot domain boundaries from data extent
                If list: [min_lon, max_lon, min_lat, max_lat, linewidth, color]
            contoursettings : list
                [stride, color, linewidth, decimal_places, fontsize]
                - stride: int, use every nth level for contour lines
                - color: str, contour line color
                - linewidth: float, contour line width
                - decimal_places: int or str, format for contour labels
                - fontsize: int, size of contour labels
            maxnumticks : int
                Maximum number of colorbar ticks (default: 10) or 'NoColorbar' to disable
            decimal_places : int
                Number of decimal places for colorbar labels (default: 2)
            plot_extent : list
                [lon_min, lon_max, lat_min, lat_max] to limit plot region
            starttimestring : str
                Start time label (default: 'May 20, 2021')
            endtimestring : str
                End time label (default: 'May 20, 2021')
            plot_sect_line : list 
                List of grid points (2d list) for cross section line
            poi : dict
                Point of interest with keys:
                - 'lon': longitude
                - 'lat': latitude
                - 'color': marker color (default 'red')
                - 'size': marker size (default 6)
                - 'label': text label (optional)
                - 'edgecolor': edge color (default 'black')

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """

    # Cache transform
    #transform = ccrs.PlateCarree() #the correct transform for lambert conformal
    # data transform
    data_transform = ccrs.PlateCarree() #the correct transform for lambert conformal

    # Input validation
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be 2D numpy array")
    
    # Clear previous plots and collect garbage
    plt.close('all')
    gc.collect()

    # Cache common values and pre-compute bounds
    if np.shape(lon) != np.shape(data):
        lon, lat = np.meshgrid(lon, lat)
    
    lon_bounds = np.min(lon), np.max(lon)
    lat_bounds = np.min(lat), np.max(lat)
    
    # Pre-calculate map extent with adjustments
    # Override extent if custom extent provided
    if plot_extent := kwargs.get('plot_extent', None):
        extent = plot_extent
    else:  
        adj_lon = kwargs.get('adj_lon', [0, 0])
        adj_lat = kwargs.get('adj_lat', [0, 0])
        extent = [
            lon_bounds[0] + adj_lon[0],
            lon_bounds[1] + adj_lon[1],
            lat_bounds[0] + adj_lat[0],
            lat_bounds[1] + adj_lat[1]
        ]
    
    # Setup figure - check if axis is provided
    if 'ax' in kwargs:
        ax = kwargs['ax']
        fig = ax.figure
    else:
        figsize = kwargs.get('figuresize', [14, 10])
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])
        ax = fig.add_subplot(gs[0], projection=projection)


    # Process data for log scale
    log_scale = kwargs.get('log_scale', False)
    if log_scale:
        data = np.ma.masked_invalid(data)
        positive_levels = levels[levels > 0]
        if len(positive_levels) > 0:
            min_level = np.min(positive_levels)
            max_level = np.max(levels)
            num_levels = len(levels)
            levels = np.logspace(np.log10(min_level), np.log10(max_level), num_levels)
            norm = LogNorm(vmin=min_level, vmax=max_level)
            masked_data = np.ma.masked_less_equal(data, 0)
        else:
            log_scale = False
            norm = None
            
    # Optimize contour plotting
    contourf_kwargs = {
        'transform': data_transform,
        'zorder': 0,
        'levels': levels,
        'cmap': colormap
    }
    
    contoursettings = kwargs.get('contoursettings', [2, '0.8', 0.1, None, None])

    try:
        if log_scale == True:
            contourf_kwargs.update({'norm': norm, 'extend': 'both'})
            contourf = ax.contourf(lon, lat, masked_data, **contourf_kwargs)
            
            # Add contour lines if contoursettings is provided
            if 'contoursettings' in kwargs:
                contoursettings = kwargs.get('contoursettings', [2, '0.8', 0.1, None, None])
                contour = ax.contour(lon, lat, masked_data,
                                levels=levels[::contoursettings[0]],
                                colors=contoursettings[1],
                                linewidths=contoursettings[2],
                                transform=data_transform,
                                norm=norm)
        else:
            contourf_kwargs['extend'] = 'max'
            contourf = ax.contourf(lon, lat, data, **contourf_kwargs)
            
            # Add contour lines
            if 'contoursettings' in kwargs:
                contoursettings = kwargs.get('contoursettings', [2, '0.8', 0.1, None, None])
                contour = ax.contour(lon, lat, data,
                            levels=levels[::contoursettings[0]],
                            colors=contoursettings[1],
                            linewidths=contoursettings[2],
                            transform=data_transform)
        # Add inline labels if requested
        if len(contoursettings) > 3 and contoursettings[3] is not None:
            # Handle both string format and decimal places
            if isinstance(contoursettings[3], str):
                fmt = contoursettings[3]
            else:
                fmt = f"%.{contoursettings[3]}f"
            ax.clabel(contour, inline=True, fontsize=contoursettings[4], fmt=fmt)

    except ValueError as e:
        raise ValueError(f"Contour plotting failed: {str(e)}")
    
    # Add after contour plotting section:
    # In plot_2d_data_on_map, replace the bathymetry plotting section with:
    if bathymetry_line is not None:
        lines_x, lines_y = bathymetry_line
        if lines_x is not None and lines_y is not None:
            for line_x, line_y in zip(lines_x, lines_y):
                ax.plot(line_x, line_y,
                    color='grey',
                    transform=data_transform,
                    linewidth=0.5,
                    zorder=5)

    maxnumticks = kwargs.get('maxnumticks', 10)
    decimal_places = kwargs.get('decimal_places', 2)

    # Optimize colorbar section
    if maxnumticks != 'NoColorbar':
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label(unit, fontsize=16, labelpad=10)

        # Set ticks based on scale
        if log_scale:
            # For log scale
            log_ticks = np.logspace(np.log10(levels[0]), np.log10(levels[-1]), maxnumticks)
            cbar.set_ticks(log_ticks)
            # Log norm is already set in contourf creation
        else:
            # For linear scale
            cbar.locator = MaxNLocator(nbins=min(maxnumticks, 6), prune='both', steps=[1, 2, 5, 10])
            linear_ticks = np.linspace(np.min(levels), np.max(levels), maxnumticks)
            cbar.set_ticks(linear_ticks)

        # Format tick labels
        with plt.style.context({'text.usetex': False}):
            def fmt(x, p):
                if log_scale:
                    if abs(x) < 1e-3 or abs(x) > 1e3:
                        return f"{x:.{decimal_places}e}"
                    return f"{x:.{decimal_places}f}"
                return f"{x:.{decimal_places}f}"
        
            cbar.formatter = FuncFormatter(fmt)
        cbar.update_ticks()
    else:
        cbar = None

    # Add features with caching
    def get_cached_feature(name):
        if name not in _CACHED_FEATURES:
            _CACHED_FEATURES[name] = getattr(cfeature, name)
        return _CACHED_FEATURES[name]
    
    ax.set_title(title, fontsize=16)
    ax.add_feature(get_cached_feature('LAND'), facecolor='0.2', zorder=2)
    ax.add_feature(get_cached_feature('COASTLINE'), zorder=3, color='0.5', linewidth=0.25)
    
    ax.set_extent(extent)
    
    #set the edgecolor parameter to the inverse color of the facecolor

    if poi := kwargs.get('poi'):  # Get poi with default None
        # Plot point
        ax.scatter(poi['lon'], 
                poi['lat'],
                poi.get('size', 6),
                color=poi.get('color', 'red'),
                edgecolors = poi.get('edgecolor', 'black'),
                label=poi.get('label', None),  # Add label parameter
                transform=data_transform,
                linewidths = 0.04*poi.get('size', 6),
                zorder=10)
        
        # Add label if provided
        #if 'label' in poi:
        #    ax.text(poi['lon']+0.1, 
        #            poi['lat']-0.05,
        #            poi['label'],
        #            color=poi.get('color', 'red'),
        #            fontsize=12,
        #            transform=data_transform,
        #            zorder=10)
        
        if poi.get('label'):
            ax.legend(prop={'size': 12})

    # Custom markers (cached transform)
    ax.plot(18.9553, 69.6496, marker='o', color='white', markersize=4, transform=data_transform)
    #ax.text(19.0553, 69.58006, 'Tromsø', transform=data_transform, color='white', fontsize=12)
    
    # Efficient gridlines
    gl = ax.gridlines(crs=data_transform, draw_labels=True, linewidth=0.5, 
                    color='grey', alpha=0.5, linestyle='--')
    gl.top_labels = False      # No labels on top
    gl.right_labels = False    # No labels on right
    if left_labels == True:
        gl.left_labels = True      # Labels on left
    gl.bottom_labels = True    # Labels on bottom
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    
    # Progress bar
    if kwargs.get('plot_progress_bar', True):
        pos = ax.get_position()
        # Adjust bar position to align with plot edges
        bar_position = kwargs.get('bar_position', [pos.x0, 0.12, pos.width, 0.03])
        
        ax2 = plt.subplot(gs[1])
        ax2.set_position([bar_position[0], bar_position[1], 
                        bar_position[2], bar_position[3]])
        
        # Ensure progress bar starts at left edge
        ax2.fill_between([0, timepassed[0]], [0, 0], [1, 1], 
                        color='grey')
        ax2.set_yticks([])
        
        # Align tick marks with bar edges
        ax2.set_xticks([0, timepassed[1]])
        ax2.set_xlim(0, timepassed[1])  # Force alignment
        
        ax2.set_xticklabels([
            kwargs.get('starttimestring', 'May 20, 2021'),
            kwargs.get('endtimestring', 'May 20, 2021')
        ], fontsize=16)
    
    #Plot model domain...
    plot_model_domain = kwargs.get('plot_model_domain', False)
    if plot_model_domain:
        if isinstance(plot_model_domain, bool):
            # Get boundary points from meshgrid
            left_edge = np.column_stack((lon[:,0], lat[:,0]))     # Western boundary
            right_edge = np.column_stack((lon[:,-1], lat[:,-1]))  # Eastern boundary
            bottom_edge = np.column_stack((lon[0,:], lat[0,:]))   # Southern boundary
            top_edge = np.column_stack((lon[-1,:], lat[-1,:]))    # Northern boundary
            
            # Combine edges with explicit ordering to avoid diagonals
            boundary = np.vstack([
                bottom_edge,        # South
                right_edge[1:],     # East (skip first point)
                top_edge[::-1],     # North (reversed)
                left_edge[::-1][1:] # West (reversed, skip first point)
            ])
            
            # Plot boundary
            ax.plot(boundary[:,0], boundary[:,1],
                    color='0.8',
                    linewidth=0.5,
                    transform=data_transform,
                    linestyle='--',
                    label='Model Domain')
            
    plot_sect_line = kwargs.get('plot_sect_line', None)

    #Plot a line along the grid points in plot_sect_line
    if plot_sect_line is not None and len(plot_sect_line) > 0:
    # Plot points using PlateCarree (these are fixed locations)
        for i in range(len(plot_sect_line)-1):
            #print(f"Processing cross section with {len(plot_sect_line)} points")
            x, y = plot_sect_line[i]
            ax.plot(x, y, 
                    marker='o', 
                    color='grey',
                    markersize=4, 
                    transform=data_transform,
                    zorder=10)

        # Save/show
    if savefile_path := kwargs.get('savefile_path', False):
        plt.savefig(savefile_path, 
                   dpi=kwargs.get('dpi', 150),
                   transparent=False,
                   bbox_inches='tight')
    if kwargs.get('show', False):
        plt.show()
    
    return fig


def plot_multiple_2d_data_on_map(data_list, lon, lat, projection, levels, timepassed,
                                colormap, unit, timestring = 'unknown', depthstring = 'unknown', 
                                bathymetry_lines=None,savefile_path=None,nrows=2, ncols=3, 
                                figsize=(18, 12),plot_progress_bar = False,**kwargs):
    """
    Creates multiple map visualizations in a grid layout.

    Parameters
    ----------
    data_list : list of np.ndarray
        List of 2D arrays containing the data to plot
    nrows : int
        Number of rows in the grid
    ncols : int
        Number of columns in the grid
    Other parameters same as plot_2d_data_on_map
    """

    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create main gridspec with adjusted ratios for thinner bars
    outer_gs = fig.add_gridspec(2, 2, 
                               height_ratios=[1, 0.03],
                               width_ratios=[1, 0.02],
                               hspace=0.01,
                               wspace=0.05)
    
    # Create nested gridspec for plots
    gs_maps = outer_gs[0, 0].subgridspec(nrows, ncols, 
                                        hspace=0.05, 
                                        wspace=0.05)
    
    contourfs = []  # Store contourf objects for colorbar

    # Loop through subplots
    for idx, data in enumerate(data_list):
        if idx >= nrows * ncols:
            break
            
        # Create subplot with projection
        ax = fig.add_subplot(gs_maps[idx//ncols, idx%ncols], projection=projection)
        
        data = data_list[idx]

        if bathymetry_lines is not None:
            bathymetry_line = bathymetry_lines[idx]

        # Call original plotting function with shared axis¨
        if idx % ncols != 0:
            plot_result = plot_2d_data_on_map(
                data=data.T,
                lon=lon,
                lat=lat,
                projection=projection,
                levels=levels,
                timepassed=timepassed,
                colormap=colormap,
                unit=unit,
                ax=ax,
                title = '',
                bathymetry_line = bathymetry_line,
                show=False,
                plot_progress_bar=False,
                plot_model_domain = False,
                left_labels=False,
                maxnumticks='NoColorbar',  # Disable individual colorbars
                **kwargs
            )
        else:
            plot_result = plot_2d_data_on_map(
                data=data.T,
                lon=lon,
                lat=lat,
                projection=projection,
                levels=levels,
                timepassed=timepassed,
                colormap=colormap,
                unit=unit,
                ax=ax,
                title = '',
                bathymetry_line = bathymetry_line,
                show=False,
                plot_progress_bar=False,
                plot_model_domain = False,
                maxnumticks='NoColorbar',  # Disable individual colorbars
                **kwargs
            )

        #plot grey shading for bathymetry (its only 1 and 0s for permissible and impermissible cells)
        #use pcolor for this
        #ax.pcolor(lon, lat, bathymetry[:,:,idx].T, transform=ccrs.PlateCarree(), cmap='Greys', alpha=0.5)

        # Store contourf for colorbar
        contourfs.append(ax.collections[0])

        # Remove labels as needed
        if idx % ncols != 0:
            ax.set_yticklabels([])
        if idx < (nrows-1)*ncols:
            ax.set_xticklabels([])

        # Plot the informatioun about depth as an inline textbox in the lower right corner
        ax.text(0.95, 0.05, depthstring[idx], transform=ax.transAxes, fontsize=16,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor=[0.1,0.1,0.1], alpha=0.9, edgecolor=[0.9,0.9,0.9], boxstyle='round,pad=0.5'))   


#        ax.text(0.95, 0.05, depthstring[idx], transform=ax.transAxes, fontsize=16,
#                verticalalignment='bottom', horizontalalignment='right',
#                bbox=dict(facecolor=[0.8,0.8,0.8], alpha=0.9, edgecolor=[0.2,0.2,0.2], boxstyle='round,pad=0.5'))   



    vmin = np.min(levels)
    vmax = np.max(levels)

    maxnumticks = 10

    # Add shared colorbar with customized settings
    cax = fig.add_subplot(outer_gs[0, 1])
    if len(contourfs) > 0:
        cbar = plt.colorbar(contourfs[0], cax=cax)
        cbar.set_label(unit, fontsize=16, labelpad=5)
        
        # Set colorbar ticks based on scale
        if kwargs.get('log_scale', False):
                        # Get the exponent range
            min_exp = np.floor(np.log10(vmin))
            max_exp = np.ceil(np.log10(vmax))
            
            # Generate base numbers for each order of magnitude
            base_numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
            tick_values = []
            
            # Generate tick values for each order of magnitude
            for exp in range(int(min_exp), int(max_exp + 1)):
                tick_values.extend(base_numbers * 10**exp)
            
            # Filter tick values to be within data range
            tick_values = np.array(tick_values)
            mask = (tick_values >= vmin) & (tick_values <= vmax)
            tick_values = tick_values[mask]
            
            # Set ticks
            cbar.set_ticks(tick_values)
            
            # Format tick labels
            def fmt(x, p):
                exp = np.floor(np.log10(abs(x)))
                base = x / 10**exp
                if abs(x) < 1e-3 or abs(x) > 1e3:
                    return f"{base:.0f}e{exp:.0f}"
                return f"{x:.1f}"
            
            cbar.formatter = FuncFormatter(fmt)

        cbar.ax.tick_params(labelsize=16, length=3, width=0.5)
        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_edgecolor('black')
        cbar.update_ticks()
        #increase the labelsize
        cbar.ax.yaxis.label.set_size(22)


    # Add progress bar with minimal height
    if kwargs.get('plot_progress_bar', True):
        ax_prog = fig.add_subplot(outer_gs[1, 0])
        ax_prog.fill_between([0, timepassed[0]], [0, 0], [1, 1], 
                           color='grey', alpha=0.7)  # Slightly transparent
        
        # Customize progress bar appearance
        ax_prog.set_yticks([])
        ax_prog.set_xticks([0, timepassed[1]])
        ax_prog.set_xlim(0, timepassed[1])
        
        # Adjust time labels
        ax_prog.set_xticklabels([
            kwargs.get('starttimestring', 'May 20, 2021'),
            kwargs.get('endtimestring', 'June 20, 2021')
        ], fontsize=14)  # Reduced font size
        
        # Minimize progress bar frame
        ax_prog.spines['top'].set_visible(False)
        ax_prog.spines['right'].set_visible(False)
        ax_prog.spines['left'].set_visible(False)
        ax_prog.tick_params(axis='x', length=3, width=0.5)

    #if savefile_path := kwargs.get('savefile_path', False):
        
    #make one title for the whole plot showing the time 
    fig.suptitle(timestring[0], fontsize=25)
    #remove whitespace between title and plots
    fig.subplots_adjust(top=0.96)
    
    # Adjust overall layout
    plt.tight_layout()

    return fig

def get_bathymetry_lines(impermissible_cells,lon_mesh,lat_mesh):
    """
    Create bathymetry contour lines from impermissible cells.
    
    Parameters
    ----------
    impermissible_cells : np.ndarray
        3D array of impermissible cells

    lon_mesh : np.ndarray
        2D array of longitude coordinates

    lat_mesh : np.ndarray
        2D array of latitude coordinates
        
    Returns
    -------
    list
        List of (line_x, line_y) tuples for each depth layer
    """
    bathymetry = np.zeros(np.shape(impermissible_cells))
    bathymetry[impermissible_cells == 0] = 0
    bathymetry[impermissible_cells > 0] = 1
    
    bathymetry_lines = []
    for i in range(bathymetry.shape[2]):
        line_x, line_y = gp.find_bathymetry_line(bathymetry[:,:,i].T, lon_mesh, lat_mesh)
        bathymetry_lines.append((line_x, line_y))
    
    bathymetry_lines[0] = ([],[])  # Insert empty line for first layer
    return bathymetry_lines

def plot_loss_analysis(times_totatm, total_atm_flux, ws_interp, particles_mox_loss, 
                      particles_mass_out, particles_mass_died, particle_mass_redistributed,
                      twentiethofmay, time_steps):
    """
    Create a 2x2 plot showing various loss mechanisms over time.
    
    Parameters
    ----------
    times_totatm : array-like
        Time vector
    total_atm_flux : array-like 
        Atmospheric flux data
    ws_interp : array-like
        Wind speed data
    particles_mox_loss : array-like
        Microbial oxidation loss data
    particles_mass_out : array-like
        Mass lost from domain
    particles_mass_died : array-like
        Mass of deactivated particles
    particle_mass_redistributed : array-like
        Mass redistributed
    """
    
    # Style settings
    colors = {
        'atm_flux': '#7e1e9c',
        'mox': '#014d4e', 
        'domain_loss': '#448ee4',
        'particle_death': '#b66325'
    }
    grid_alpha = 0
    linewidth = 2.5

    # Create figure and axes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Calculate time ticks
    tick_indices = np.linspace(twentiethofmay, time_steps - 1, 4).astype(int)
    tick_positions = times_totatm[tick_indices]
    tick_labels = pd.to_datetime(tick_positions).strftime('%d.%b')

    # Plot 1: Atmospheric Flux & Wind Speed
    _plot_atm_flux_wind(ax1, times_totatm, total_atm_flux, ws_interp,
                       twentiethofmay, time_steps, tick_positions, tick_labels,
                       colors['atm_flux'], grid_alpha, linewidth)

    # Plot 2: Microbial Oxidation
    _plot_mox(ax2, times_totatm, particles_mox_loss, twentiethofmay, time_steps,
              tick_positions, tick_labels, colors['mox'], grid_alpha, linewidth)

    # Plot 3: Domain Loss
    _plot_domain_loss(ax3, times_totatm, particles_mass_out, twentiethofmay, time_steps,
                     tick_positions, tick_labels, colors['domain_loss'], grid_alpha, linewidth)

    # Plot 4: Particle Death
    _plot_particle_death(ax4, times_totatm, particles_mass_died, particle_mass_redistributed,
                        twentiethofmay, time_steps, tick_positions, tick_labels,
                        colors['particle_death'], grid_alpha, linewidth)

    # Format all axes
    _format_axes([ax1, ax2, ax3, ax4], tick_positions, tick_labels)

    plt.tight_layout()
    return fig

def _plot_atm_flux_wind(ax, times, flux, wind, start, end, tick_pos, tick_labs,
                       color, grid_alpha, linewidth):
    """Plot atmospheric flux and wind speed."""
    ax.plot(times[start:end], flux[start:end], 
            color=color, linewidth=linewidth, label='Atmospheric Flux')
    
    ax.set_ylabel('Atmospheric Flux [mol hr$^{-1}$]', color=color)
    ax.grid(False, alpha=grid_alpha)
    ax.set_xlim([times[start], times[end-1]])
    ax.tick_params(axis='y', labelcolor=color)

    # Add wind speed on twin axis
    ax_twin = ax.twinx()
    ax_twin.plot(times[start:end], np.mean(np.mean(wind, axis=1), axis=1)[start:end],
                color='grey', linewidth=linewidth, linestyle='--', label='Wind Speed')
    ax_twin.set_ylabel('Wind Speed [m s$^{-1}$]', color='grey')
    #turn off grid for twin axis
    ax_twin.grid(False, alpha=grid_alpha)
    ax_twin.tick_params(axis='y', labelcolor='grey')
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=12)

# Add similar helper functions for other plots...

def _format_axes(axes, tick_positions, tick_labels):
    """Apply common formatting to all axes."""
    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, ha='center')
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(axis='both', labelsize=14)

def _plot_mox(ax, times, mox_loss, start, end, tick_pos, tick_labs,
              color, grid_alpha, linewidth):
    """Plot microbial oxidation loss."""
    ax.plot(times[start:end], mox_loss[start:end], 
            color=color, linewidth=linewidth, label='Microbial Oxidation')
    ax.set_ylabel('Microbial Oxidation [mol hr$^{-1}$]', color=color)
    ax.grid(True, alpha=grid_alpha)
    ax.set_xlim([times[start], times[end-1]])
    ax.tick_params(axis='y', labelcolor=color)

    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labels1, loc='upper center', fontsize=12) 

def _plot_domain_loss(ax, times, domain_loss, start, end, tick_pos, tick_labs,
                     color, grid_alpha, linewidth):
    """Plot mass lost from domain."""
    ax.plot(times[start:end], domain_loss[start:end], 
            color=color, linewidth=linewidth, label='Mass Lost from Domain')
    ax.set_ylabel('Mass Lost from Domain [mol hr$^{-1}$]', color=color)
    ax.grid(True, alpha=grid_alpha)
    ax.set_xlim([times[start], times[end-1]])
    ax.tick_params(axis='y', labelcolor=color)

    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labels1, loc='upper center', fontsize=12)

def _plot_particle_death(ax, times, mass_died, mass_redistributed, start, end,
                        tick_pos, tick_labs, color, grid_alpha, linewidth):
    """Plot mass of deactivated particles and redistributed mass."""
    ax.plot(times[start:end], mass_died[start:end], 
            color=color, linewidth=linewidth, label='Mass of Deactivated Particles')
    ax.plot(times[start:end], mass_redistributed[start:end], 
            color='grey', linewidth=linewidth, linestyle='--', label='Mass Redistributed')
    ax.set_ylabel('Mass of Deactivated Particles [mol hr$^{-1}$]', color=color)
    ax.grid(True, alpha=grid_alpha)
    ax.set_xlim([times[start], times[end-1]])
    ax.tick_params(axis='y', labelcolor=color)

    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labels1, loc='upper center', fontsize=12)


def plot_methane_fate(particle_lifespan_matrix, R_ox, show=True):
    """
    Creates plots showing the fate of methane over time.
    
    Parameters
    ----------
    particle_lifespan_matrix : np.ndarray
        3D array containing particle lifespan data with shape (time, particles, type)
        where type contains [methane amount, MOx loss, atmospheric loss]
    R_ox : float
        Oxidation rate constant [hr^-1]
    show : bool, optional
        Whether to display the plots (default: True)
        
    Returns
    -------
    tuple
        (fig1, fig2) containing the two created figure objects
    """
    
    # Calculate losses
    loss_atm = np.nansum(particle_lifespan_matrix[:,:,2], axis=1)
    loss_mox = (np.nansum(particle_lifespan_matrix[:,:,0], axis=1)) * R_ox * 3600
    
    # Trim last two timesteps (as in original code)
    loss_mox = loss_mox[:-2]
    loss_atm = loss_atm[:-2]
    meth_left = np.nansum(particle_lifespan_matrix[:-2,:,0], axis=1)
    
    # Calculate accumulated losses
    loss_mox_acc = np.cumsum(loss_mox)
    loss_atm_acc = np.cumsum(loss_atm)
    
    # Calculate total methane and fractions
    total_methane = meth_left + loss_mox_acc + loss_atm_acc
    frac_remain = meth_left / total_methane * 100
    frac_mox = loss_mox_acc / total_methane * 100
    frac_atm = loss_atm_acc / total_methane * 100
    
    # Convert time to days
    days = np.arange(len(total_methane))/24
    
    # Create first plot - stacked area
    fig1 = plt.figure(figsize=(8, 6), dpi=150)
    plt.fill_between(days, 0, frac_remain, 
                    label='Remains in water column', 
                    color='#069af3', alpha=0.7)
    plt.fill_between(days, frac_remain, frac_remain + frac_mox, 
                    label='Lost to MOx', 
                    color='#fe01b1', alpha=0.7)
    plt.fill_between(days, frac_remain + frac_mox, 100, 
                    label='Lost to atmosphere', 
                    color='#20f986', alpha=0.7)
    plt.ylabel('Fraction of total methane [%]', fontsize=14)
    plt.xlabel('Time [days]', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower left', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.xlim(0, max(days))
    
    # Create second plot - dual axis
    fig2, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(days, frac_atm, color='#20f986', 
                label='Lost to atmosphere', linewidth=2)
    ax1.set_xlabel('Time [days]', fontsize=14)
    ax1.set_ylabel('Fraction lost to atmosphere [%]', 
                color='#20f986', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#20f986', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    
    l2 = ax2.plot(days, frac_mox, color='#fe01b1', 
                label='Lost to MOx', linewidth=2)
    ax2.set_ylabel('Fraction lost to MOx [%]', 
                color='#fe01b1', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='#fe01b1', labelsize=12)
    
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if show:
        plt.show()
        
    return fig1, fig2


# Calculate the fraction of molecules at each depth level using the particle_lifespan_matrix[:,:,0] data
# run this if main script

loss_atm = np.nansum(particle_lifespan_matrix[:,:,2],axis=1)#/np.nansum(particle_lifespan_matrix[:,:,0],axis=1)
loss_mox = np.nansum(particle_lifespan_matrix[:,:,1],axis=1)
frac_loss_atm = np.nansum(particle_lifespan_matrix[:,:,2],axis=1)/np.nansum(particle_lifespan_matrix[:,:,0],axis=1)
frac_loss_mox = np.nansum(particle_lifespan_matrix[:,:,1],axis=1)/np.nansum(particle_lifespan_matrix[:,:,0],axis=1)

#accumulated fractional loss
frac_loss_atm_acc = np.cumsum(frac_loss_atm)
frac_loss_mox_acc = np.cumsum(frac_loss_mox)
loss_mox = (np.nansum(particle_lifespan_matrix[:,:,0],axis=1))*R_ox*3600

#set plot style to default
plt.style.use('default')
loss_mox = loss_mox[:-2]
loss_atm = loss_atm[:-2]
meth_left = np.nansum(particle_lifespan_matrix[:-2,:,0],axis=1)

#Calculate accumulated loss vectors
loss_mox_acc = np.cumsum(loss_mox)
loss_atm_acc = np.cumsum(loss_atm)

# Calculate total methane at each timestep
total_methane = meth_left + loss_mox_acc + loss_atm_acc

# Calculate fractions
frac_remain = meth_left / total_methane * 100
frac_mox = loss_mox_acc / total_methane * 100
frac_atm = loss_atm_acc / total_methane * 100

plt.figure(figsize=(8, 6), dpi=150)
days = np.arange(len(total_methane))/24  # Convert hours to days

plt.fill_between(days, 0, frac_remain, 
                label='Remains in water column', 
                color='#069af3', alpha=0.7)

plt.fill_between(days, frac_remain, frac_remain + frac_mox, 
                label='Lost to MOx', 
                color='#fe01b1', alpha=0.7)

plt.fill_between(days, frac_remain + frac_mox, 100, 
                label='Lost to atmosphere', 
                color='#20f986', alpha=0.7)
plt.ylabel('Fraction of total methane [%]', fontsize=14)
plt.xlabel('Time [days]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='lower left', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.xlim(0, max(days))
plt.show()

# 2. Dual axis plot for losses
fig, ax1 = plt.subplots(figsize=(8, 6),dpi=150)
ax2 = ax1.twinx()
# Plot atmospheric loss
l1 = ax1.plot(days, frac_atm, color='#20f986', 
            label='Lost to atmosphere', linewidth=2)
ax1.set_xlabel('Time [days]', fontsize=14)
ax1.set_ylabel('Fraction lost to atmosphere [%]', 
            color='#20f986', fontsize=14)
ax1.tick_params(axis='y', labelcolor='#20f986', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
# Plot MOx loss
l2 = ax2.plot(days, frac_mox, color='#fe01b1', 
            label='Lost to MOx', linewidth=2)
ax2.set_ylabel('Fraction lost to MOx [%]', 
            color='#fe01b1', fontsize=14)
ax2.tick_params(axis='y', labelcolor='#fe01b1', labelsize=12)
lns = l1 + l2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

