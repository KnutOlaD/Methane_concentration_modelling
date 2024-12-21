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


# Cache features for reuse
_CACHED_FEATURES = {}

def plot_2d_data_on_map(data, lon, lat, projection, levels, timepassed,
                         colormap, title, unit, **kwargs):

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
                Maximum number of colorbar ticks (default: 10)
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
    
    # Setup figure
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
    
    maxnumticks = kwargs.get('maxnumticks', 10)
    decimal_places = kwargs.get('decimal_places', 2)

    # Optimize colorbar section
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

    # Add features with caching
    def get_cached_feature(name):
        if name not in _CACHED_FEATURES:
            _CACHED_FEATURES[name] = getattr(cfeature, name)
        return _CACHED_FEATURES[name]
    
    ax.set_title(title, fontsize=16)
    ax.add_feature(get_cached_feature('LAND'), facecolor='0.2', zorder=2)
    ax.add_feature(get_cached_feature('COASTLINE'), zorder=3, color='0.5', linewidth=0.5)
    
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
    ax.text(19.0553, 69.58006, 'Tromsø', transform=data_transform, color='white', fontsize=12)
    
    # Efficient gridlines
    gl = ax.gridlines(crs=data_transform, draw_labels=True, linewidth=0.5, 
                    color='grey', alpha=0.5, linestyle='--')
    gl.top_labels = False      # No labels on top
    gl.right_labels = False    # No labels on right
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
