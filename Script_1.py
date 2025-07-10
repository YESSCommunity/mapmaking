import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from scipy.interpolate import Rbf
from matplotlib import ticker
import matplotlib.patches as mpatches
from matplotlib.ticker import StrMethodFormatter
import matplotlib.markers as markers

# ================================
# === USER SETTINGS ============
# ================================

# File paths
csv_path = "/path/to/file/air_quality_one_season.csv"  # csv path
shapefile_path = "/path/to/file/Cordillera Santiago.shp"  # shp path

# Data columns
x_col = 'x'  # Customize x column name
y_col = 'y'  # Customize y column name
data_column = 'AQI'  # Customize data column name
sample_name_column = 'SampleName'  # Customize sample name column name

# Interpolation settings
interpolation_resolution = 300  # Grid resolution for interpolation
rbf_function = 'gaussian'  # Options: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'
rbf_smooth = 0.1  # Smoothing parameter for RBF interpolation

# Display toggles
SHOW_SAMPLE_POINTS = True  # Show sample point markers
SHOW_COLORBAR = True  # Show colorbar
SHOW_SCALEBAR = True  # Show scale bar
SHOW_NORTH_ARROW = True  # Show north arrow
SHOW_LEGEND = True  # Show legend
SHOW_POINT_LABELS = True  # Show data value labels
SHOW_SAMPLE_NAMES = False  # Show sample name labels
SHOW_LABELS_IN_LEGEND = True  # Include label types in legend

# Coordinate formatting
COORD_FORMAT = 'default'  # Options: 'DMS', 'DD', or 'default'
COORD_DECIMAL_PLACES = 2  # Decimal places for coordinate values
LATITUDE_DIGITS = 6  # Number of digits to display for latitude labels
LONGITUDE_DIGITS = 6  # Number of digits to display for longitude labels
AXIS_ORIENTATION = 'standard'  # Options: 'standard', 'reverse_x', 'reverse_y', 'reverse_both'

# Interpolation visualization
INTERPOLATION_COLORMAP = 'inferno'  # Options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'RdYlBu', 'jet', 'rainbow', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter'
INTERPOLATION_ALPHA = 0.75  # Transparency of interpolation surface (0-1)
CONTOUR_LEVELS = 10  # Number of contour levels

# Shapefile appearance
SHAPEFILE_EDGE_COLOR = 'black'  # Options: color names, hex codes, RGB tuples
SHAPEFILE_EDGE_WIDTH = 1  # Line width for shapefile boundaries
SHAPEFILE_FILL_COLOR = 'none'  # Options: 'none', color names, hex codes, RGB tuples

# Sample points appearance
POINT_COLOR = 'blue'  # Options: color names, hex codes, RGB tuples
POINT_SIZE = 50  # Size of sample point markers
POINT_MARKER = 'o'  # Options: 'o', 's', '^', 'v', '<', '>', 'D', 'P', '*', 'X', 'h', 'H', '+', 'x', 'd', '|', '_'
POINT_EDGE_COLOR = 'black'  # Options: color names, hex codes, RGB tuples
POINT_EDGE_WIDTH = 1  # Width of point marker edges

# Data value labels
POINT_LABEL_COLOR = 'white'  # Options: color names, hex codes, RGB tuples
POINT_LABEL_SIZE = 8  # Font size for data value labels
POINT_LABEL_OFFSET_X = 0  # Horizontal offset for data value labels
POINT_LABEL_OFFSET_Y = 200  # Vertical offset for data value labels
POINT_LABEL_BACKGROUND_COLOR = None  # Options: None (uses POINT_COLOR), color names, hex codes, RGB tuples
POINT_LABEL_BACKGROUND_ALPHA = 0.7  # Transparency of label background (0-1)

# Sample name labels
SAMPLE_NAME_COLOR = 'red'  # Options: color names, hex codes, RGB tuples
SAMPLE_NAME_SIZE = 8  # Font size for sample name labels
SAMPLE_NAME_OFFSET_X = 0  # Horizontal offset for sample name labels
SAMPLE_NAME_OFFSET_Y = -200  # Vertical offset for sample name labels
SAMPLE_NAME_BACKGROUND_COLOR = 'white'  # Options: color names, hex codes, RGB tuples
SAMPLE_NAME_BACKGROUND_ALPHA = 0.7  # Transparency of label background (0-1)

# North arrow
NORTH_ARROW_X = 0.95  # X position as fraction of plot width (0-1)
NORTH_ARROW_Y = 0.95  # Y position as fraction of plot height (0-1)
NORTH_ARROW_SIZE = 14  # Font size for north arrow
NORTH_ARROW_COLOR = 'black'  # Options: color names, hex codes, RGB tuples

# Scale bar
SCALEBAR_X = 0.05  # X position as fraction of plot width (0-1)
SCALEBAR_Y = 0.05  # Y position as fraction of plot height (0-1)
SCALEBAR_LENGTH = 5000  # Scale bar length in map units
SCALEBAR_COLOR = 'black'  # Options: color names, hex codes, RGB tuples
SCALEBAR_WIDTH = 5  # Line width for scale bar

# Colorbar
COLORBAR_POSITION = 'right'  # Options: 'right', 'left', 'top', 'bottom'
COLORBAR_SIZE = 0.03  # Size of colorbar relative to plot
COLORBAR_PAD = 0.05  # Padding between colorbar and plot

# Legend
LEGEND_POSITION = 'upper left'  # Options: 'upper left', 'upper right', 'lower left', 'lower right', 'center left', 'center right', 'upper center', 'lower center', 'center', 'best'
LEGEND_FRAMEON = True  # Show legend frame
LEGEND_FANCYBOX = True  # Use rounded corners for legend
LEGEND_SHADOW = True  # Show shadow behind legend
LEGEND_ALPHA = 1.0  # Legend transparency (0-1)
LEGEND_FONTSIZE = 10  # Font size for legend text
LEGEND_TITLE = None  # Legend title (None for no title)
LEGEND_TITLE_FONTSIZE = 12  # Font size for legend title

# Legend entries for labels
POINT_LABEL_LEGEND_TEXT = f"{data_column} Values"  # Text for data value labels in legend
SAMPLE_NAME_LEGEND_TEXT = "Sample Names"  # Text for sample name labels in legend
POINT_LABEL_LEGEND_COLOR = 'white'  # Color for data value label legend entry
SAMPLE_NAME_LEGEND_COLOR = 'red'  # Color for sample name label legend entry

# Figure settings
FIGURE_SIZE = (8, 6)  # Figure size in inches (width, height)
FIGURE_DPI = 100  # Display DPI for figure
TITLE_TEXT = f"Interpolated {data_column}"  # Plot title
TITLE_SIZE = 16  # Font size for title
XLABEL_TEXT = "Longitude"  # X-axis label
YLABEL_TEXT = "Latitude"  # Y-axis label
LABEL_SIZE = 12  # Font size for axis labels

# ================================
# === SAVE FILE SETTINGS =======
# ================================

SAVE_FILE = False  # Set to True to save the figure
SAVE_PATH = "/path/to/save/air_quality_map.png"  # Customize save path
SAVE_FORMAT = 'png'  # Options: 'png', 'jpg', 'jpeg', 'pdf', 'svg', 'eps', 'tiff', 'ps'
SAVE_DPI = 300  # DPI for saved file (higher = better quality, larger file)
SAVE_BBOX_INCHES = 'tight'  # Options: 'tight', 'standard', or None
SAVE_PAD_INCHES = 0.1  # Padding around the figure when bbox_inches='tight'
SAVE_FACECOLOR = 'white'  # Options: 'white', 'black', 'transparent', color names, hex codes, RGB tuples
SAVE_EDGECOLOR = 'none'  # Options: 'none', color names, hex codes, RGB tuples
SAVE_TRANSPARENT = False  # Set to True for transparent background (works with png, pdf, svg)

# ================================
# === FUNCTIONS =================
# ================================

def decimal_to_dms(decimal_deg, is_longitude=True):
    abs_deg = abs(decimal_deg)
    degrees = int(abs_deg)
    minutes = int((abs_deg - degrees) * 60)
    seconds = (abs_deg - degrees - minutes / 60) * 3600
    hemisphere = ('E' if decimal_deg >= 0 else 'W') if is_longitude else ('N' if decimal_deg >= 0 else 'S')
    return f"{degrees}\u00b0{minutes:02d}'{seconds:04.1f}\"{hemisphere}"

def format_coordinates(value, coord_format, decimal_places, is_longitude=True, crs_info=None):
    if coord_format == 'DMS' and crs_info and crs_info.is_geographic:
        return decimal_to_dms(value, is_longitude)
    elif coord_format == 'DD' and crs_info and crs_info.is_geographic:
        return f"{value:.{decimal_places}f}\u00b0"
    else:
        return f"{value:,.{decimal_places}f}"

def format_coordinates_with_digits(value, coord_format, decimal_places, digits, is_longitude=True, crs_info=None):
    if coord_format == 'DMS' and crs_info and crs_info.is_geographic:
        return decimal_to_dms(value, is_longitude)
    elif coord_format == 'DD' and crs_info and crs_info.is_geographic:
        formatted = f"{value:.{decimal_places}f}\u00b0"
        return formatted[:digits] + "°" if len(formatted) > digits else formatted
    else:
        # Format without scientific notation and with full precision
        formatted = f"{value:.{decimal_places}f}"
        # Limit to specified number of digits
        if len(formatted) > digits:
            formatted = formatted[:digits]
        return formatted

def get_axis_labels(gdf, coord_format):
    crs_info = gdf.crs
    if coord_format in ['DMS', 'DD'] and crs_info and crs_info.is_geographic:
        return "Longitude", "Latitude"
    return "X Coordinate", "Y Coordinate"

# ================================
# === LOAD DATA =================
# ================================

gdf = gpd.read_file(shapefile_path)
df = pd.read_csv(csv_path)

required_columns = [x_col, y_col, data_column]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Check if sample name column exists when needed
if SHOW_SAMPLE_NAMES and sample_name_column not in df.columns:
    print(f"Warning: Sample name column '{sample_name_column}' not found. Sample names will not be displayed.")
    SHOW_SAMPLE_NAMES = False

df.dropna(subset=[x_col, y_col, data_column], inplace=True)
df["geometry"] = df.apply(lambda row: Point(row[x_col], row[y_col]), axis=1)
gdf_data = gpd.GeoDataFrame(df, geometry="geometry", crs=gdf.crs)

crs_info = gdf.crs
xlabel, ylabel = get_axis_labels(gdf, COORD_FORMAT)

# ================================
# === GRID CREATION =============
# ================================

bounds = gdf.total_bounds
xi = np.linspace(bounds[0], bounds[2], interpolation_resolution)
yi = np.linspace(bounds[1], bounds[3], interpolation_resolution)
xi, yi = np.meshgrid(xi, yi)

# FIX: Replace deprecated unary_union with union_all()
polygon = gdf.geometry.union_all()  # Updated to use the recommended method
points = np.column_stack((xi.ravel(), yi.ravel()))
inside_mask = np.array([polygon.contains(Point(x, y)) for x, y in points]).reshape(xi.shape)

xi_inside = xi[inside_mask]
yi_inside = yi[inside_mask]

# ================================
# === INTERPOLATION =============
# ================================

rbf = Rbf(df[x_col], df[y_col], df[data_column], function=rbf_function, smooth=rbf_smooth)
zi_rbf = rbf(xi_inside, yi_inside)

# Clip interpolated values to the actual data range
data_min = df[data_column].min()
data_max = df[data_column].max()
zi_rbf = np.clip(zi_rbf, data_min, data_max)

zi_full = np.full_like(xi, np.nan)
zi_full[inside_mask] = zi_rbf
zi_masked = np.ma.masked_where(~inside_mask, zi_full)

# ================================
# === PLOTTING ==================
# ================================

fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
gdf.plot(ax=ax, facecolor=SHAPEFILE_FILL_COLOR, edgecolor=SHAPEFILE_EDGE_COLOR, linewidth=SHAPEFILE_EDGE_WIDTH)

# Create contour plot with explicit data range (but don't show in legend)
c = ax.contourf(xi, yi, zi_masked, levels=CONTOUR_LEVELS, cmap=INTERPOLATION_COLORMAP, 
                alpha=INTERPOLATION_ALPHA, vmin=data_min, vmax=data_max)

# Create legend handles list (without contour range entry)
legend_handles = []

if SHOW_SAMPLE_POINTS:
    scatter = ax.scatter(gdf_data.geometry.x, gdf_data.geometry.y,
                        c=POINT_COLOR, s=POINT_SIZE, marker=POINT_MARKER,
                        edgecolor=POINT_EDGE_COLOR, linewidth=POINT_EDGE_WIDTH,
                        label="Sample Points")
    legend_handles.append(scatter)

if SHOW_POINT_LABELS:
    label_bg_color = POINT_LABEL_BACKGROUND_COLOR if POINT_LABEL_BACKGROUND_COLOR else POINT_COLOR
    for _, row in df.iterrows():
        ax.text(row[x_col] + POINT_LABEL_OFFSET_X, row[y_col] + POINT_LABEL_OFFSET_Y,
                f"{row[data_column]:.0f}", color=POINT_LABEL_COLOR, fontsize=POINT_LABEL_SIZE,
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=label_bg_color, alpha=POINT_LABEL_BACKGROUND_ALPHA))
    
    if SHOW_LABELS_IN_LEGEND:
        label_patch = mpatches.Patch(color=POINT_LABEL_LEGEND_COLOR, label=POINT_LABEL_LEGEND_TEXT)
        legend_handles.append(label_patch)

if SHOW_SAMPLE_NAMES:
    for _, row in df.iterrows():
        sample_name = row[sample_name_column] if sample_name_column in row else "N/A"
        ax.text(row[x_col] + SAMPLE_NAME_OFFSET_X, row[y_col] + SAMPLE_NAME_OFFSET_Y,
                str(sample_name), color=SAMPLE_NAME_COLOR, fontsize=SAMPLE_NAME_SIZE,
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=SAMPLE_NAME_BACKGROUND_COLOR, alpha=SAMPLE_NAME_BACKGROUND_ALPHA))
    
    if SHOW_LABELS_IN_LEGEND:
        name_patch = mpatches.Patch(color=SAMPLE_NAME_LEGEND_COLOR, label=SAMPLE_NAME_LEGEND_TEXT)
        legend_handles.append(name_patch)

if SHOW_COLORBAR:
    orientation = 'vertical' if COLORBAR_POSITION in ['right', 'left'] else 'horizontal'
    cbar = plt.colorbar(c, ax=ax, label=data_column, orientation=orientation,
                       shrink=1 - COLORBAR_SIZE, pad=COLORBAR_PAD)
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.update_ticks()

if SHOW_LEGEND and legend_handles:
    legend = ax.legend(handles=legend_handles, loc=LEGEND_POSITION, frameon=LEGEND_FRAMEON, 
                      fancybox=LEGEND_FANCYBOX, shadow=LEGEND_SHADOW,
                      fontsize=LEGEND_FONTSIZE, title=LEGEND_TITLE, title_fontsize=LEGEND_TITLE_FONTSIZE)
    legend.get_frame().set_alpha(LEGEND_ALPHA)

if SHOW_SCALEBAR:
    x0 = bounds[0] + (bounds[2] - bounds[0]) * SCALEBAR_X
    y0 = bounds[1] + (bounds[3] - bounds[1]) * SCALEBAR_Y
    ax.plot([x0, x0 + SCALEBAR_LENGTH], [y0, y0], color=SCALEBAR_COLOR, lw=SCALEBAR_WIDTH)
    ax.text(x0 + SCALEBAR_LENGTH / 2, y0 - (bounds[3] - bounds[1]) * 0.02,
            f'{SCALEBAR_LENGTH/1000:.0f} km', ha='center', va='top', fontsize=10, color=SCALEBAR_COLOR)

if SHOW_NORTH_ARROW:
    arrow_x = bounds[0] + (bounds[2] - bounds[0]) * NORTH_ARROW_X
    arrow_y = bounds[1] + (bounds[3] - bounds[1]) * NORTH_ARROW_Y
    ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(0, -20), textcoords='offset points',
                ha='center', va='top', fontsize=NORTH_ARROW_SIZE, fontweight='bold',
                color=NORTH_ARROW_COLOR,
                arrowprops=dict(arrowstyle='-|>', linewidth=2, color=NORTH_ARROW_COLOR))

# Coordinate formatting with digit control and disable scientific notation
if COORD_FORMAT in ['DD', 'DMS']:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: format_coordinates_with_digits(x, COORD_FORMAT, COORD_DECIMAL_PLACES, LONGITUDE_DIGITS, True, crs_info)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: format_coordinates_with_digits(y, COORD_FORMAT, COORD_DECIMAL_PLACES, LATITUDE_DIGITS, False, crs_info)))
else:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: f"{x:.{COORD_DECIMAL_PLACES}f}"[:LONGITUDE_DIGITS]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, pos: f"{y:.{COORD_DECIMAL_PLACES}f}"[:LATITUDE_DIGITS]))

# Disable scientific notation for axes
#ax.ticklabel_format(style='plain', axis='both')

# Axis orientation
if AXIS_ORIENTATION == 'reverse_x':
    ax.invert_xaxis()
elif AXIS_ORIENTATION == 'reverse_y':
    ax.invert_yaxis()
elif AXIS_ORIENTATION == 'reverse_both':
    ax.invert_xaxis()
    ax.invert_yaxis()

ax.set_title(TITLE_TEXT, fontsize=TITLE_SIZE)
ax.set_xlabel(XLABEL_TEXT, fontsize=LABEL_SIZE)
ax.set_ylabel(YLABEL_TEXT, fontsize=LABEL_SIZE)
ax.set_aspect('equal')

plt.tight_layout()

# ================================
# === SAVE FILE =================
# ================================

if SAVE_FILE:
    plt.savefig(SAVE_PATH, 
                format=SAVE_FORMAT,
                dpi=SAVE_DPI,
                bbox_inches=SAVE_BBOX_INCHES,
                pad_inches=SAVE_PAD_INCHES,
                facecolor=SAVE_FACECOLOR,
                edgecolor=SAVE_EDGECOLOR,
                transparent=SAVE_TRANSPARENT)
    print(f"Figure saved to: {SAVE_PATH}")

plt.show()
