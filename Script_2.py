import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from scipy.interpolate import griddata, Rbf
from matplotlib import ticker
import matplotlib.patches as mpatches
import matplotlib.markers as markers
from matplotlib.gridspec import GridSpec
from datetime import datetime
import os

# ================================
# === USER SETTINGS ============
# ================================

# === FILE PATHS & DATA COLUMNS ===
csv_path = "/path/to/file/air_quality_monthly_2023_2024.csv"  # csv path
shapefile_path = "/path/to/file/Cordillera Santiago.shp"  # shp path


X_COLUMN = "x"          # Column name for X coordinates (longitude)
Y_COLUMN = "y"          # Column name for Y coordinates (latitude)
DATA_COLUMN = "AQI"     # Column name for the data to interpolate and map
DATE_COLUMN = "Date"    # Column name for date information

# === SAVE SETTINGS ===
SAVE_MAP = True                    # Options: True, False - Enable/disable saving the map
SAVE_PATH = "/path/to/file/air_quality_map"    # Path and filename (without extension) for saving the map
SAVE_FORMAT = "png"                # Options: 'png', 'pdf', 'svg', 'eps', 'tiff', 'jpg', 'jpeg'
SAVE_DPI = 300                     # DPI for saved image (ignored for vector formats like PDF, SVG)
SAVE_BBOX_INCHES = 'tight'         # Options: 'tight', None - Bbox settings for saving
SAVE_PAD_INCHES = 0.1              # Padding around saved figure (when bbox_inches='tight')
SAVE_TRANSPARENT = False           # Options: True, False - Transparent background for saved image
SAVE_FACECOLOR = 'white'           # Background color for saved image (when transparent=False)
SAVE_EDGECOLOR = 'none'            # Edge color for saved image

# === INTERPOLATION SETTINGS ===
interpolation_resolution = 300  # Grid resolution for interpolation (higher = smoother but slower)
rbf_function = 'gaussian'      # Options: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'
rbf_smooth = 0.1               # Smoothing parameter (0 = exact interpolation, >0 = smoother)

# === MAP ELEMENTS TOGGLE ===
SHOW_SAMPLE_POINTS = False     # Options: True, False
SHOW_COLORBAR = True          # Options: True, False
SHOW_SCALEBAR = True          # Options: True, False
SHOW_NORTH_ARROW = True       # Options: True, False
SHOW_LEGEND = True            # Options: True, False
SHOW_POINT_LABELS = False     # Options: True, False (shows data values at sample points)

# === LAYOUT SETTINGS ===
PLOT_COLUMNS = 6              # Number of columns in the subplot grid
PLOT_ROWS = None              # Number of rows (None = auto-calculate)
FIGURE_SIZE = (20, 15)        # Figure size in inches (width, height)

HORIZONTAL_SPACING = 0.3      # Horizontal spacing between subplots
VERTICAL_SPACING = 0.2        # Vertical spacing between subplots
TOP_SPACING = 0.9             # Top margin for main title
BOTTOM_SPACING = 0.1          # Bottom margin

GRID_HEIGHT_RATIOS = None      # Custom height ratios for rows (None = equal heights)

# === LEGEND SETTINGS ===
LEGEND_SIDE = 'right'           # Options: 'right', 'left', 'top', 'bottom' (for figure-level legend placement)
LEGEND_FRAMEON = True           # Options: True, False (show legend frame)
LEGEND_FANCYBOX = True          # Options: True, False (rounded corners)
LEGEND_SHADOW = True            # Options: True, False (drop shadow)
LEGEND_FONTSIZE = 12            # Legend font size
LEGEND_MARKERSIZE = 8           # Legend marker size
LEGEND_BORDERPAD = 0.5          # Whitespace inside legend border
LEGEND_LABELSPACING = 0.5       # Vertical space between legend entries
LEGEND_HANDLELENGTH = 1.5       # Length of legend handles
LEGEND_BORDERAXESPAD = 0.5      # Pad between axes and legend border
LEGEND_TITLE = "Sample Points"  # Title for the legend (set to None to disable title)

# === COLOR SCHEME ===
INTERPOLATION_COLORMAP = 'inferno'  # Options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'Spectral', 'RdYlBu', 'RdYlGn', 'RdBu', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'Seismic'
INTERPOLATION_ALPHA = 0.75          # Transparency of interpolated surface (0 = transparent, 1 = opaque)
CONTOUR_LEVELS = 20                 # Number of contour levels
SHAPEFILE_EDGE_COLOR = 'black'      # Options: color names, hex codes, 'none'
SHAPEFILE_EDGE_WIDTH = 1            # Line width for shapefile boundaries
SHAPEFILE_FILL_COLOR = 'none'       # Options: color names, hex codes, 'none'

# === SAMPLE POINTS STYLING ===
POINT_COLOR = 'blue'           # Options: color names, hex codes
POINT_SIZE = 50                # Size of sample points
POINT_MARKER = 'o'             # Options: 'o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd'
POINT_EDGE_COLOR = 'black'     # Options: color names, hex codes, 'none'
POINT_EDGE_WIDTH = 1           # Width of point edges
POINT_LABEL_COLOR = 'white'    # Color of point labels
POINT_LABEL_SIZE = 8           # Font size of point labels
POINT_LABEL_OFFSET_X = 0       # Horizontal offset for point labels
POINT_LABEL_OFFSET_Y = 200     # Vertical offset for point labels

# === MAP ELEMENT POSITIONS ===
NORTH_ARROW_X = 0.95            # X position as fraction of plot width (0-1)
NORTH_ARROW_Y = 0.95            # Y position as fraction of plot height (0-1)
NORTH_ARROW_SIZE = 14          # Font size of north arrow
NORTH_ARROW_COLOR = 'black'    # Options: color names, hex codes

SCALEBAR_X = 0.10              # X position as fraction of plot width (0-1)
SCALEBAR_Y = 0.10              # Y position as fraction of plot height (0-1)
SCALEBAR_LENGTH = 1000         # Length of scalebar in map units (e.g., meters)
SCALEBAR_COLOR = 'black'       # Options: color names, hex codes
SCALEBAR_WIDTH = 2             # Line width of scalebar

# === COLORBAR SETTINGS ===
COLORBAR_POSITION = 'right'    # Options: 'bottom', 'top', 'left', 'right'
COLORBAR_SIZE = 0.5            # Size of colorbar as fraction of figure dimension
COLORBAR_PAD = 0.05            # Padding around colorbar
COLORBAR_ASPECT = 20           # Aspect ratio of colorbar
COLORBAR_HEIGHT = 0.03         # Height of colorbar as fraction of figure height (for top/bottom positions)
COLORBAR_WIDTH = 0.03          # Width of colorbar as fraction of figure width (for left/right positions)
COLORBAR_FONTSIZE = 10         # Font size for colorbar labels and title
COLORBAR_LABEL_SIZE = 12       # Font size for colorbar axis label
COLORBAR_TICK_SIZE = 8         # Font size for colorbar tick labels
COLORBAR_SHRINK = 0.8          # Shrink factor for colorbar size (0-1)
COLORBAR_EXTEND = 'neither'    # Options: 'neither', 'both', 'min', 'max' - Extend colorbar beyond data range

# === FIGURE SETTINGS ===
FIGURE_DPI = 100               # Figure resolution (dots per inch) for display
TITLE_TEXT = "Monthly Air Quality Index (2023-2024)"  # Main figure title
TITLE_SIZE = 16                # Main title font size
XLABEL_TEXT = "Longitude"      # X-axis label
YLABEL_TEXT = "Latitude"       # Y-axis label
LABEL_SIZE = 12                # Axis label font size
AXIS_ORIENTATION = 'normal'    # Options: 'normal', 'reverse_x', 'reverse_y', 'reverse_both'

# === DATE/TIME SETTINGS ===
DATE_FORMAT = '%b %Y'          # Date format for subplot titles (e.g., 'Jan 2023')

# ================================
# === HELPER FUNCTIONS =========
# ================================

def create_figure_layout(num_plots, plot_columns=None, plot_rows=None):
    """Create figure layout with proper spacing for legend placement."""
    if plot_columns is None and plot_rows is None:
        plot_columns = min(4, num_plots)
        plot_rows = int(np.ceil(num_plots / plot_columns))
    elif plot_columns is None:
        plot_columns = int(np.ceil(num_plots / plot_rows))
    elif plot_rows is None:
        plot_rows = int(np.ceil(num_plots / plot_columns))
    
    if GRID_HEIGHT_RATIOS is None:
        height_ratios = [1]*plot_rows
    else:
        height_ratios = GRID_HEIGHT_RATIOS
    
    # Adjust figure size and spacing based on legend position
    fig_width, fig_height = FIGURE_SIZE
    if SHOW_LEGEND and SHOW_SAMPLE_POINTS and LEGEND_SIDE in ['left', 'right']:
        fig_width += 2  # Add space for side legend
    elif SHOW_LEGEND and SHOW_SAMPLE_POINTS and LEGEND_SIDE in ['top', 'bottom']:
        fig_height += 1  # Add space for top/bottom legend
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=FIGURE_DPI)
    gs = GridSpec(plot_rows, plot_columns, 
                 height_ratios=height_ratios,
                 hspace=VERTICAL_SPACING, 
                 wspace=HORIZONTAL_SPACING)
    
    return fig, gs, plot_rows, plot_columns

def create_sample_points_legend(fig):
    """Create figure-level legend for sample points."""
    if not SHOW_LEGEND or not SHOW_SAMPLE_POINTS:
        return
    
    # Create legend element for sample points
    sample_point = mpatches.Circle((0, 0), 1, fc=POINT_COLOR, ec=POINT_EDGE_COLOR,
                                 linewidth=POINT_EDGE_WIDTH, label='Sample Points')
    
    # Position legend based on LEGEND_SIDE
    if LEGEND_SIDE == 'right':
        legend_x = 0.98
        legend_y = 0.5
        bbox_anchor = (legend_x, legend_y)
        loc = 'center right'
    elif LEGEND_SIDE == 'left':
        legend_x = 0.02
        legend_y = 0.5
        bbox_anchor = (legend_x, legend_y)
        loc = 'center left'
    elif LEGEND_SIDE == 'top':
        legend_x = 0.5
        legend_y = 0.98
        bbox_anchor = (legend_x, legend_y)
        loc = 'upper center'
    elif LEGEND_SIDE == 'bottom':
        legend_x = 0.5
        legend_y = 0.02
        bbox_anchor = (legend_x, legend_y)
        loc = 'lower center'
    else:
        bbox_anchor = None
        loc = 'upper right'
    
    # Create legend
    legend = fig.legend(handles=[sample_point],
                       title=LEGEND_TITLE,
                       loc=loc,
                       bbox_to_anchor=bbox_anchor,
                       frameon=LEGEND_FRAMEON,
                       fancybox=LEGEND_FANCYBOX,
                       shadow=LEGEND_SHADOW,
                       fontsize=LEGEND_FONTSIZE,
                       markerscale=LEGEND_MARKERSIZE/10,
                       borderpad=LEGEND_BORDERPAD,
                       labelspacing=LEGEND_LABELSPACING,
                       handlelength=LEGEND_HANDLELENGTH,
                       borderaxespad=LEGEND_BORDERAXESPAD)
    
    if LEGEND_FRAMEON:
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(0.8)
        frame.set_edgecolor('gray')
        frame.set_linewidth(0.5)

def add_simple_legend(ax, position='upper right'):
    """Add simple legend for sample points inside a subplot (fallback)."""
    if not SHOW_LEGEND or not SHOW_SAMPLE_POINTS:
        return
    
    proxy = mpatches.Circle((0, 0), 1, fc=POINT_COLOR, ec=POINT_EDGE_COLOR,
                           linewidth=POINT_EDGE_WIDTH, label='Sample Points')
    
    leg = ax.legend(handles=[proxy], 
                   title=LEGEND_TITLE,
                   loc=position,
                   frameon=LEGEND_FRAMEON,
                   fancybox=LEGEND_FANCYBOX,
                   shadow=LEGEND_SHADOW,
                   fontsize=LEGEND_FONTSIZE,
                   markerscale=LEGEND_MARKERSIZE/10,
                   borderpad=LEGEND_BORDERPAD,
                   labelspacing=LEGEND_LABELSPACING,
                   handlelength=LEGEND_HANDLELENGTH,
                   borderaxespad=LEGEND_BORDERAXESPAD)
    
    if LEGEND_FRAMEON:
        frame = leg.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(0.8)
        frame.set_edgecolor('gray')
        frame.set_linewidth(0.5)

def create_interpolation_grid(gdf_boundary, resolution):
    """Create interpolation grid within boundary."""
    bounds = gdf_boundary.total_bounds
    xi = np.linspace(bounds[0], bounds[2], resolution)
    yi = np.linspace(bounds[1], bounds[3], resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    polygon = gdf_boundary.geometry.iloc[0]
    points = np.column_stack((xi.ravel(), yi.ravel()))
    inside_mask = np.array([polygon.contains(Point(x, y)) for x, y in points]).reshape(xi.shape)
    
    return xi, yi, inside_mask

def perform_interpolation(points_df, xi, yi, inside_mask, data_column, rbf_function='gaussian', smooth=0.1):
    """Perform RBF interpolation on the data."""
    xi_inside = xi[inside_mask]
    yi_inside = yi[inside_mask]
    
    rbf = Rbf(points_df[X_COLUMN], points_df[Y_COLUMN], points_df[data_column], 
              function=rbf_function, smooth=smooth)
    zi_rbf = rbf(xi_inside, yi_inside)
    
    zi_full = np.full_like(xi, np.nan)
    zi_full[inside_mask] = zi_rbf
    
    return np.ma.masked_where(~inside_mask, zi_full)

def plot_map(ax, gdf_boundary, gdf_points, xi, yi, zi_masked, title_text, 
             show_points=True, show_scalebar=True, show_north_arrow=True):
    """Plot individual map with interpolated data."""
    gdf_boundary.plot(ax=ax, facecolor=SHAPEFILE_FILL_COLOR, 
                     edgecolor=SHAPEFILE_EDGE_COLOR, 
                     linewidth=SHAPEFILE_EDGE_WIDTH)
    
    if INTERPOLATION_COLORMAP not in plt.colormaps():
        cmap = 'viridis'
    else:
        cmap = INTERPOLATION_COLORMAP
    
    c = ax.contourf(xi, yi, zi_masked, levels=CONTOUR_LEVELS, 
                    cmap=cmap, alpha=INTERPOLATION_ALPHA, extend=COLORBAR_EXTEND)
    
    if show_points:
        gdf_points.plot(ax=ax, color=POINT_COLOR, markersize=POINT_SIZE, 
                       marker=POINT_MARKER, label="Sample Points", 
                       edgecolor=POINT_EDGE_COLOR, linewidth=POINT_EDGE_WIDTH)
    
    if SHOW_POINT_LABELS and show_points:
        for idx, row in gdf_points.iterrows():
            ax.text(row.geometry.x + POINT_LABEL_OFFSET_X, 
                   row.geometry.y + POINT_LABEL_OFFSET_Y, 
                   f"{row[DATA_COLUMN]:.0f}", color=POINT_LABEL_COLOR, 
                   fontsize=POINT_LABEL_SIZE, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=POINT_COLOR, alpha=0.7))
    
    if show_scalebar:
        bounds = gdf_boundary.total_bounds
        x0 = bounds[0] + (bounds[2] - bounds[0]) * SCALEBAR_X
        y0 = bounds[1] + (bounds[3] - bounds[1]) * SCALEBAR_Y
        
        ax.plot([x0, x0 + SCALEBAR_LENGTH], [y0, y0], 
                color=SCALEBAR_COLOR, lw=SCALEBAR_WIDTH)
        ax.text(x0 + SCALEBAR_LENGTH / 2, y0 - (bounds[3] - bounds[1]) * 0.02, 
                f'{SCALEBAR_LENGTH/1000:.0f} km', ha='center', va='top', 
                fontsize=10, color=SCALEBAR_COLOR)
    
    if show_north_arrow:
        bounds = gdf_boundary.total_bounds
        arrow_x = bounds[0] + (bounds[2] - bounds[0]) * NORTH_ARROW_X
        arrow_y = bounds[1] + (bounds[3] - bounds[1]) * NORTH_ARROW_Y
        
        ax.annotate('N', xy=(arrow_x, arrow_y),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', va='top', fontsize=NORTH_ARROW_SIZE, 
                    fontweight='bold', color=NORTH_ARROW_COLOR,
                    arrowprops=dict(arrowstyle='-|>', linewidth=2, color=NORTH_ARROW_COLOR))
    
    ax.set_title(title_text, fontsize=12)
    
    if AXIS_ORIENTATION == 'reverse_x':
        ax.invert_xaxis()
    elif AXIS_ORIENTATION == 'reverse_y':
        ax.invert_yaxis()
    elif AXIS_ORIENTATION == 'reverse_both':
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    return c

def save_figure(fig, save_path, file_format, dpi, bbox_inches, pad_inches, transparent, facecolor, edgecolor):
    """Save the figure with specified parameters."""
    if not SAVE_MAP:
        return
    
    try:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Add file extension if not present
        if not save_path.endswith(f'.{file_format}'):
            full_path = f"{save_path}.{file_format}"
        else:
            full_path = save_path
        
        # Save figure
        fig.savefig(full_path, 
                   format=file_format,
                   dpi=dpi,
                   bbox_inches=bbox_inches,
                   pad_inches=pad_inches,
                   transparent=transparent,
                   facecolor=facecolor,
                   edgecolor=edgecolor)
        
        print(f"✓ Map saved successfully: {full_path}")
        
    except Exception as e:
        print(f"✗ Error saving map: {e}")

# ================================
# === MAIN PROCESSING ==========
# ================================

def main():
    """Main processing function."""
    # Load shapefile
    try:
        gdf_boundary = gpd.read_file(shapefile_path)
        print(f"✓ Shapefile loaded: {shapefile_path}")
    except Exception as e:
        print(f"✗ Error loading shapefile: {e}")
        return
    
    # Load CSV data
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ CSV loaded: {csv_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return
    
    # Validate required columns
    required_columns = [X_COLUMN, Y_COLUMN, DATA_COLUMN, DATE_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"✗ Missing required columns: {missing_columns}")
        print(f"  Available columns: {list(df.columns)}")
        return
    
    # Process date information
    try:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df['month'] = df[DATE_COLUMN].dt.month
        df['year'] = df[DATE_COLUMN].dt.year
        print(f"✓ Date processing complete")
    except Exception as e:
        print(f"✗ Error processing dates: {e}")
        return
    
    # Create GeoDataFrame
    try:
        df["geometry"] = df.apply(lambda row: Point(row[X_COLUMN], row[Y_COLUMN]), axis=1)
        gdf_points = gpd.GeoDataFrame(df, geometry="geometry", crs=gdf_boundary.crs)
        print(f"✓ GeoDataFrame created with {len(gdf_points)} points")
    except Exception as e:
        print(f"✗ Error creating GeoDataFrame: {e}")
        return

    # Create interpolation grid
    try:
        xi, yi, inside_mask = create_interpolation_grid(gdf_boundary, interpolation_resolution)
        print(f"✓ Interpolation grid created ({interpolation_resolution}x{interpolation_resolution})")
    except Exception as e:
        print(f"✗ Error creating interpolation grid: {e}")
        return

    # Group data by month-year
    grouped = df.groupby(['year', 'month'])
    num_plots = len(grouped)
    print(f"✓ Found {num_plots} time periods to plot")
    
    # Create figure layout
    fig, gs, plot_rows, plot_columns = create_figure_layout(num_plots, PLOT_COLUMNS, PLOT_ROWS)
    ax_list = []
    contour_dict = {}
    all_data_values = []

    # Plot each month's data
    for i, ((year, month), month_df) in enumerate(grouped):
        row = i // plot_columns
        col = i % plot_columns
        
        month_gdf = gdf_points[(gdf_points['month'] == month) & (gdf_points['year'] == year)]
        all_data_values.extend(month_df[DATA_COLUMN].values)
        
        zi_masked = perform_interpolation(month_df, xi, yi, inside_mask, DATA_COLUMN, rbf_function, rbf_smooth)
        
        ax = fig.add_subplot(gs[row, col])
        ax_list.append(ax)
        
        title_text = datetime(year, month, 1).strftime(DATE_FORMAT)
        c = plot_map(ax, gdf_boundary, month_gdf, xi, yi, zi_masked, title_text,
                    show_points=SHOW_SAMPLE_POINTS, 
                    show_scalebar=SHOW_SCALEBAR,
                    show_north_arrow=SHOW_NORTH_ARROW)
        
        contour_dict[(row, col)] = c

    # Add main title
    fig.suptitle(TITLE_TEXT, fontsize=TITLE_SIZE, y=0.98)

    # Add colorbar
    if SHOW_COLORBAR and contour_dict:
        # Create dedicated axes for colorbar
        if COLORBAR_POSITION == 'bottom':
            cbar_ax = fig.add_axes([0.125, 0.05, 0.775, COLORBAR_HEIGHT])
            orientation = 'horizontal'
        elif COLORBAR_POSITION == 'top':
            cbar_ax = fig.add_axes([0.125, 0.95, 0.775, COLORBAR_HEIGHT])
            orientation = 'horizontal'
        elif COLORBAR_POSITION == 'left':
            cbar_ax = fig.add_axes([0.05, 0.125, COLORBAR_WIDTH, 0.775])
            orientation = 'vertical'
        elif COLORBAR_POSITION == 'right':
            cbar_ax = fig.add_axes([0.95, 0.125, COLORBAR_WIDTH, 0.775])
            orientation = 'vertical'
        
        last_key = list(contour_dict.keys())[-1]
        cbar = plt.colorbar(contour_dict[last_key], 
                          cax=cbar_ax,
                          orientation=orientation,
                          label=DATA_COLUMN,
                          aspect=COLORBAR_ASPECT,
                          shrink=COLORBAR_SHRINK,
                          pad=COLORBAR_PAD,
                          extend=COLORBAR_EXTEND)
        
        # Style the colorbar
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
        cbar.set_label(DATA_COLUMN, fontsize=COLORBAR_LABEL_SIZE)
        
        if orientation == 'horizontal':
            if COLORBAR_POSITION == 'top':
                cbar.ax.xaxis.set_label_position('top')
                cbar.ax.xaxis.set_ticks_position('top')
        else:  # vertical
            if COLORBAR_POSITION == 'left':
                cbar.ax.yaxis.set_label_position('left')
                cbar.ax.yaxis.set_ticks_position('left')

    # Add legend for sample points
    create_sample_points_legend(fig)

    plt.tight_layout()
    plt.subplots_adjust(top=TOP_SPACING)
    
    # Save the figure if enabled
    save_figure(fig, SAVE_PATH, SAVE_FORMAT, SAVE_DPI, SAVE_BBOX_INCHES, 
                SAVE_PAD_INCHES, SAVE_TRANSPARENT, SAVE_FACECOLOR, SAVE_EDGECOLOR)
    
    plt.show()
    
    print("✓ Map generation complete!")

if __name__ == "__main__":
    main()
