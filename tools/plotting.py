import tools
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import font_manager

import cartopy as cart
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

import h3

import sys
sys.path.append('/nethome/4302001/tracer_backtracking/tools')


def load_fonts():
    font_dirs = ['/nethome/4302001/.fonts']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)


def set_figure_NA(ax=None, figsize=(6,4)):
    """
    Set the figure for the North Atlantic.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection = cart.crs.PlateCarree())
        returnstuff = True
    else:
        returnstuff = False

    ax.add_feature(cart.feature.LAND, zorder=18, edgecolor='black')
    ax.gridlines(draw_labels=['left', 'bottom'], linewidth=0.5, color='gray', alpha=0.5)
    ax.set_extent([-85, -25, 10, 50])

    if returnstuff:
        return fig, ax


def field_cartopy_colorbar(da_field, x="nav_lon", y="nav_lat", levels=None, cbar_label=None, ax=None, contour=False, title=None, cbar_suppress=False, **kwargs):
    if ax is None:
        willreturn = True
        fig, ax = plt.subplots(subplot_kw={'projection': cart.crs.PlateCarree()})
    else:
        willreturn = False
        fig = ax.figure

    x = da_field[x]
    y = da_field[y]

    if contour:
        mapping = ax.contour(x, y, da_field, levels=levels, transform=cart.crs.PlateCarree(), **kwargs)
    elif levels is not None:
        mapping = ax.contourf(x, y, da_field, levels=levels, transform=cart.crs.PlateCarree(), **kwargs)
    else:
        mapping = ax.pcolormesh(x, y, da_field, transform=cart.crs.PlateCarree(), **kwargs)

    if cbar_label is None:
        try:
            cbar_label = da_field.name
        except:
            pass

    if not cbar_suppress:
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)
        cbar = plt.colorbar(mapping, cax=ax_cb, label=cbar_label)
        fig.add_axes(ax_cb)

    ax.add_feature(cart.feature.LAND, zorder=18, edgecolor='black')
    ax.gridlines(draw_labels=["left", "bottom"], linestyle='--', linewidth=0.5)
    ax.set_extent([-85, -25, 15, 50])

    if willreturn:
        return fig
    else:
        pass


def plot_trajectory_lon_lat_depth(ds_single_traj,
                                  event_indices=None,
                                  cmap=plt.cm.get_cmap("Set1"),
                                  figsize=(6, 6),
                                  min_extent=(-80, -30, 10, 60)):
    """
    Plot a single trajectory in lon-lat-depth space, using foldout maps

    Parameters
    ----------
    ds_single_traj : xarray.Dataset
        Dataset containing a single trajectory
    event_indices : list of tuples, optional
        List of tuples containing the start and end indices of each event, by default None
    cmap : matplotlib colormap, optional
        Colormap to use for the events, by default plt.cm.get_cmap("Set1")
    figsize : tuple, optional
        Figure size, by default (6, 6)

    Returns
    -------
    matplotlib figure
        Figure containing the plot
    """
    if type(event_indices) == type(None):
        event_indices = [(0, len(ds_single_traj['lon']))]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[2, 1],
                           height_ratios=[2, 1])

    # Create a GeoAxes in the left part of the plot
    ax1 = plt.subplot(gs[0, 0], projection=cart.crs.PlateCarree())
    ax1.add_feature(cart.feature.LAND, edgecolor='black')

    ax1.plot(ds_single_traj['lon'], ds_single_traj['lat'], color='grey', linewidth=0.5, transform=cart.crs.PlateCarree())
    for event_idx, event in enumerate(event_indices):
        color = cmap(event_idx % len(cmap.colors))
        ax1.scatter(x=ds_single_traj['lon'][event[0]:event[1]], y=ds_single_traj['lat']
                    [event[0]:event[1]], s=8, color=color, linewidths=0, edgecolors=None)
    ax1.scatter(x=ds_single_traj['lon'][0], y=ds_single_traj['lat'][0], color='green', marker='P', label='Start')
    ax1.scatter(x=ds_single_traj['lon'][-1], y=ds_single_traj['lat'][-1], color='red', marker='x', label='End')

    ax1.legend(loc='best')

    # Create a regular axes in the right part of the plot, sharing its y-axis
    ax2 = plt.subplot(gs[0, 1], sharey=ax1)

    ax2.plot(ds_single_traj['z'], ds_single_traj['lat'], color='grey', linewidth=0.5)
    for event_idx, event in enumerate(event_indices):
        color = cmap(event_idx % 9)
        ax2.scatter(x=ds_single_traj['z'][event[0]:event[1]], y=ds_single_traj['lat']
                    [event[0]:event[1]], s=8, color=color, linewidths=0, edgecolors=None)
    ax2.scatter(x=ds_single_traj['z'][0], y=ds_single_traj['lat'][0], color='green', marker='P', label='Start')
    ax2.scatter(x=ds_single_traj['z'][-1], y=ds_single_traj['lat'][-1], color='red', marker='x', label='End')

    ax2.tick_params(left=False, labelleft=False, right=True, labelright=True)  # Hide y-axis labels
    ax2.yaxis.set_major_formatter(cart.mpl.gridliner.LATITUDE_FORMATTER)  # Set x-axis labels to degree format
    ax2.set_xlabel('Depth (m)')

    # Create a regular axes in the bottom part of the plot, sharing its x-axis
    ax3 = plt.subplot(gs[1, 0], sharex=ax1)

    ax3.plot(ds_single_traj['lon'], ds_single_traj['z'], color='grey', linewidth=0.5)
    for event_idx, event in enumerate(event_indices):
        color = cmap(event_idx % 9)
        ax3.scatter(x=ds_single_traj['lon'][event[0]:event[1]], y=ds_single_traj['z']
                    [event[0]:event[1]], s=8, color=color, linewidths=0, edgecolors=None)
    ax3.scatter(x=ds_single_traj['lon'][0], y=ds_single_traj['z'][0], color='green', marker='P', label='Start')
    ax3.scatter(x=ds_single_traj['lon'][-1], y=ds_single_traj['z'][-1], color='red', marker='x', label='End')

    ax3.tick_params(top=False, labeltop=False)  # Hide x-axis labels
    ax3.xaxis.set_major_formatter(cart.mpl.gridliner.LONGITUDE_FORMATTER)  # Set y-axis labels to degree format

    ax3.invert_yaxis()
    ax3.set_ylabel('Depth (m)')

    gridliner = ax1.gridlines(draw_labels=True, xlocs=ax3.xaxis.get_major_locator(), ylocs=ax2.yaxis.get_major_locator())
    gridliner.bottom_labels = False
    gridliner.right_labels = False

    cur_extent = ax1.get_extent()
    new_extent = [min(cur_extent[0], min_extent[0]), max(cur_extent[1], min_extent[1]),
                  min(cur_extent[2], min_extent[2]), max(cur_extent[3], min_extent[3])]
    ax1.set_extent(new_extent, crs=cart.crs.PlateCarree())

    fig.suptitle(f'Trajectory {ds_single_traj.trajectory.values}')

    return fig
    # plt.show()

def plot_scatter_lon_lat_depth(lons,
                               lats, 
                               depths,
                                  figsize=(6, 6),
                                  min_extent=(-80, -30, 10, 60),
                                  **kwargs):
    """
    Plot a scatter in lon-lat-depth space, using foldout maps

    Parameters
    ----------
    ds_single_traj : xarray.Dataset
        Dataset containing a single trajectory
    event_indices : list of tuples, optional
        List of tuples containing the start and end indices of each event, by default None
    cmap : matplotlib colormap, optional
        Colormap to use for the events, by default plt.cm.get_cmap("Set1")
    figsize : tuple, optional
        Figure size, by default (6, 6)
    **kwargs
        Additional keyword arguments passed to ax.scatter

    Returns
    -------
    matplotlib figure
        Figure containing the plot
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[2, 1],
                           height_ratios=[2, 1])

    # Create a GeoAxes in the left part of the plot
    ax1 = plt.subplot(gs[0, 0], projection=cart.crs.PlateCarree())
    ax1.add_feature(cart.feature.LAND, edgecolor='black')

    ax1.scatter(lons, lats, transform=cart.crs.PlateCarree(), **kwargs)

    # Create a regular axes in the right part of the plot, sharing its y-axis
    ax2 = plt.subplot(gs[0, 1], sharey=ax1)

    ax2.scatter(depths, lats, **kwargs)

    ax2.tick_params(left=False, labelleft=False, right=True, labelright=True)  # Hide y-axis labels
    ax2.yaxis.set_major_formatter(cart.mpl.gridliner.LATITUDE_FORMATTER)  # Set x-axis labels to degree format
    ax2.set_xlabel('Depth (m)')

    # Create a regular axes in the bottom part of the plot, sharing its x-axis
    ax3 = plt.subplot(gs[1, 0], sharex=ax1)

    ax3.scatter(lons, depths, **kwargs)

    ax3.tick_params(top=False, labeltop=False)  # Hide x-axis labels
    ax3.xaxis.set_major_formatter(cart.mpl.gridliner.LONGITUDE_FORMATTER)  # Set y-axis labels to degree format

    ax3.invert_yaxis()
    ax3.set_ylabel('Depth (m)')

    gridliner = ax1.gridlines(draw_labels=True, xlocs=ax3.xaxis.get_major_locator(), ylocs=ax2.yaxis.get_major_locator())
    gridliner.bottom_labels = False
    gridliner.right_labels = False

    cur_extent = ax1.get_extent()
    new_extent = [min(cur_extent[0], min_extent[0]), max(cur_extent[1], min_extent[1]),
                  min(cur_extent[2], min_extent[2]), max(cur_extent[3], min_extent[3])]
    ax1.set_extent(new_extent, crs=cart.crs.PlateCarree())

    return fig
    # plt.show()


def pcolorhex(ax,
              hexagons,
              colors=None,
              draw_edges=True,
              fill_polygons=True,
              transform=cart.crs.PlateCarree(),
              **kwargs):
    """
    Draw a collection of hexagons colored by a value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw to
    hexes : list
        A list of `h3` hexagons (i.e. related to tools.countGrid.hexagons)
    colors : list, optional
        A list of colors to use for each hexagon, by default None
    draw_edges : bool, optional
        Whether to draw the edges of the hexagons, by default True
    fill_polygons : bool, optional
        Whether to fill the hexagons with color, by default True
    **kwargs
        Additional keyword arguments passed to `matplotlib.pyplot.fill`

    Returns
    -------
    None
        This function does not return anything, but it does draw to the axes
        that are passed in.
    """
    for i1, hex_ in enumerate(hexagons):
        # get latitude and longitude coordinates of hexagon
        lat_long_coords = np.array(h3.h3_to_geo_boundary(str(hex_)))
        x = lat_long_coords[:, 1]
        y = lat_long_coords[:, 0]

        # ensure that all longitude values are between 0 and 360
        x_hexagon = np.append(x, x[0])
        y_hexagon = np.append(y, y[0])
        if x_hexagon.max() - x_hexagon.min() > 25:
            x_hexagon[x_hexagon < 0] += 360

        # draw edges
        if draw_edges:
            ax.plot(x_hexagon, y_hexagon, 'k-', transform=transform, linewidth=.2)

        # fill polygons
        if fill_polygons:
            ax.fill(x_hexagon, y_hexagon, color=colors[i1], transform=transform, **kwargs)


def get_colornorm(vmin=None, vmax=None, center=None, linthresh=None, base=None):
    """"
    Return a normalizer

    Parameters
    ----------
    vmin : float (default=None)
        Minimum value of the data range
    vmax : float (default=None)
        Maximum value of the data range
    center : float (default=None)
        Center value for a two-slope normalization
    linthresh : float (default=None)
        Threshold for a symmetrical log normalization
    base : float (default=None)
        Base for a symmetrical log normalization

    Returns
    -------
    norm : matplotlib.colors.Normalize object
        A normalizer object
    """
    if type(base) is not type(None) and type(linthresh) is type(None):
        norm = colors.LogNorm(vmin=None, vmax=None)
    elif type(linthresh) is not type(None) and type(base) is not type(None):
        norm = colors.SymLogNorm(linthresh=linthresh, base=base, vmin=None, vmax=None)
    elif center:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    else:
        norm = colors.Normalize(vmin, vmax)
    return norm


def get_colors(inp, colormap, vmin=None, vmax=None, center=None, linthresh=None, base=0):
    """"
    Based on input data, minimum and maximum values, and a colormap, return color values

    Parameters
    ----------
    inp : array-like
        Input data
    colormap : matplotlib.colors.Colormap object
        Colormap to use
    vmin : float (default=None)
        Minimum value of the data range
    vmax : float (default=None)
        Maximum value of the data range
    center : float (default=None)
        Center value for a two-slope normalization
    linthresh : float (default=None)
        Threshold for a symmetrical log normalization
    base : float (default=None)
        Base for a symmetrical log normalization

    Returns
    -------
    colors : array-like
        Array of color values
    """
    norm = get_colornorm(vmin, vmax, center, linthresh, base)
    return colormap(norm(inp))


def plot_hex_hist(counts, grid, title=None, maxnorm=None, label='Particles per bin', ax=None, return_fig=False):
    """
    Plot a histogram of particle counts in a hexagonal grid

    Parameters
    ----------
    counts : array-like
        Array of particle counts. Should be the same length as the grid
    grid : tool.countGrid object
        Grid object containing the hexagonal grid
    title : str, optional
        Title of the plot, by default None
    maxnorm : int, optional
        Maximum value of the colorbar, by default None  
    label : str, optional
        Label of the colorbar, by default 'Particles per bin'
    ax : matplotlib.axes.Axes, optional
        Axes to plot to, by default None
    return_fig : bool, optional
        Whether to return the figure, by default False
    """
    if not maxnorm:
        maxnorm = counts.max()

    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': cart.crs.PlateCarree()})
    else:
        fig = ax.figure
        # ax.set_projection(cart.crs.PlateCarree())

    # Creating the histogram
    pcolorhex(ax, grid.hexagons, get_colors(counts, plt.cm.viridis, 0,
                                   maxnorm), draw_edges=False, alpha=1., label=' concentration')

    # Cartopy stuff
    ax.coastlines()
    ax.gridlines(draw_labels=["left", "bottom"])
    ax.set_extent((-85, -30, 20, 50), crs=cart.crs.PlateCarree())

    # Colorbar
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    cmap = plt.cm.viridis
    cmap.set_bad('w')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=maxnorm))
    cbar = plt.colorbar(sm, cax=ax_cb, label=label)
    fig.add_axes(ax_cb)

    # Title
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if return_fig:
        return fig

def plot_hex_hist_3d(lon,
                     lat,
                     depth,
                     horiz_grid,
                     figsize=(10, 6),
                     extent=(-80, -30, 10, 60),
                     depths = (0, 600)):
    """

    """

    hexCount = horiz_grid.count_2d(lon, lat)

    maxnorm = hexCount.max()
    colors = get_colors(hexCount, colormap=plt.cm.viridis, vmin=0, vmax=maxnorm)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 4,
                           width_ratios=[2, 0.7, 0.05, 0.1],
                           height_ratios=[2, 1])
    

    # Create a GeoAxes in the left part of the plot
    ax = plt.subplot(gs[0, 0], projection=cart.crs.PlateCarree())
    pcolorhex(ax, horiz_grid.hexagons, colors, draw_edges=False, alpha=1.)
    ax.add_feature(cart.feature.LAND, edgecolor='black', zorder=100)
    ax.set_extent(extent, crs=cart.crs.PlateCarree())
    cmap = plt.cm.viridis
    cmap.set_bad('w')

    normalizer=plt.Normalize(vmin=0, vmax=maxnorm)

    # Create a regular axes in the right part of the plot, sharing its y-axis
    ax_depth_lat = plt.subplot(gs[0, 1], sharey=ax)
    depht_lat_hexbin = ax_depth_lat.hexbin(depth, lat, 
                                           extent=(depths[0], depths[1], extent[2], extent[3]), 
                                           cmap=plt.cm.viridis,
                                           norm=normalizer,
                                           mincnt=1,
                                           gridsize=(10, 30))
    ax_depth_lat.tick_params(left=False, labelleft=False, right=True, labelright=True)  # Hide y-axis labels
    ax_depth_lat.yaxis.set_major_formatter(cart.mpl.gridliner.LATITUDE_FORMATTER)  # Set x-axis labels to degree format
    ax_depth_lat.set_xlabel('Depth (m)')
    ax_depth_lat.xaxis.set_ticks_position('top')
    ax_depth_lat.xaxis.set_label_position('top')


    # Create a regular axes in the bottom part of the plot, sharing its x-axis
    ax_lon_depth = plt.subplot(gs[1, 0], sharex=ax)
    lon_depth_hexbin = ax_lon_depth.hexbin(lon, depth, 
                                           extent=(extent[0], extent[1], depths[0], depths[1]), 
                                           cmap=plt.cm.viridis,
                                           norm=normalizer,
                                           mincnt=1,
                                           gridsize=(30, 10))
    ax_lon_depth.tick_params(top=False, labeltop=False)  # Hide x-axis labels
    ax_lon_depth.xaxis.set_major_formatter(cart.mpl.gridliner.LONGITUDE_FORMATTER)  # Set y-axis labels to degree format
    ax_lon_depth.invert_yaxis()
    ax_lon_depth.set_ylabel('Depth (m)')
    


    gridliner = ax.gridlines(draw_labels=True, xlocs=ax_depth_lat.xaxis.get_major_locator(),
                                     ylocs=ax_lon_depth.yaxis.get_major_locator())
    gridliner.bottom_labels = False
    gridliner.right_labels = False

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalizer)
    cbar_lon_lat = plt.colorbar(sm, cax=plt.subplot(gs[:, 3]), orientation='vertical')
    cbar_lon_lat.set_label('Particles per bin')


    return fig
    # plt.show()


def plot_2d_histogram(lons, lats, gridpoint_resolution=1.0, lognorm=True, cmap='viridis', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': cart.crs.PlateCarree()})
        return_fig = True
    else:
        fig = ax.figure
        return_fig = False
    
    ax.set_extent([-90, -30, 10, 50], cart.crs.PlateCarree())
    ax.add_feature(cart.feature.LAND, zorder=10, edgecolor='black')
    ax.gridlines(draw_labels=['bottom', 'left'], zorder=11, color='gray', alpha=0.5, linestyle='--')

    lon_bins = np.arange(-90, -20 + gridpoint_resolution, gridpoint_resolution)
    lat_bins = np.arange(10, 50 + gridpoint_resolution, gridpoint_resolution)

    Histogram, xedges, yedges = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])

    lon_2d, lat_2d = np.meshgrid(xedges, yedges)

    if lognorm:
        norm = colors.SymLogNorm(vmax=(Histogram + 1e-10).max(), vmin=0, linthresh=1)
        pcm = ax.pcolormesh(lon_2d, lat_2d, Histogram.T + 1e-10, cmap=cmap, transform=cart.crs.PlateCarree(), norm=norm, zorder=9)
    else:
        pcm = ax.pcolormesh(lon_2d, lat_2d, Histogram.T, cmap=cmap, transform=cart.crs.PlateCarree(), zorder=9)
    # set the colorbar to logartihmic scale

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = plt.colorbar(pcm, cax=ax_cb, label="Particle count")
    fig.add_axes(ax_cb)

    if return_fig:
        return fig
    


def plot_seasonal(positive_months, negative_months, title="", return_fig=False, counts=True):
    """
    Plot the montly distribution of positive and negative extreme events.

    Parameters
    ----------
    positive_months : array-like
        Array with month of each positive event (1-12)
    negative_months : array-like
        Array with month of each negative event (1-12)
    title : str, optional
        Title of the plot, by default ""
    return_fig : bool, optional
        If True, return the figure object, by default False
    counts : bool, optional
        If True, show the event counts in the title, by default True
    """

    bins_edges = np.linspace(0, 2*np.pi, 13)
    centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])

    hist_pos, _ = np.histogram(positive_months * 2*np.pi / 13, bins_edges)

    hist_neg, _ = np.histogram(negative_months * 2*np.pi / 13, bins_edges)

    angles = np.linspace(0, 2*np.pi, 13)
    month_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(6, 4))

    for ax, hist, title in zip([ax1, ax2], [hist_pos, hist_neg], [f'Positive {title}', f'Negative {title}']):
        bars = ax.bar(centers, hist, width=bins_edges[1] - bins_edges[0])

        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2.0)

        ax.set_xticks(angles)
        ax.set_xticklabels(month_names, rotation=0)
        # Create labels manually and set their positions
        for i, month_name in enumerate(month_names):
            angle = angles[i] + np.pi/12
            if angle >= 2*np.pi:
                angle -= 2*np.pi
            if angle >= np.pi/2 and angle < 1.5*np.pi:
                label_angle = -angle + np.pi
                va = 'center'
            else:
                label_angle = -angle
                va = 'center'
            ax.text(angle, ax.get_ylim()[1]*1.1, month_name,
                    rotation=np.degrees(label_angle),
                    ha="center", va=va)

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        if counts:
            title += '\n' + f' (N={hist.sum()})'

        ax.set_title(title, pad=20, fontsize=11, wrap=True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=2)

    if return_fig:
        return fig

    else:
        plt.show()
