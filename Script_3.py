
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from scipy.interpolate import Rbf
import matplotlib.patches as mpatches
from datetime import datetime
import imageio
import os

# ================================
# === USER SETTINGS ============
# ================================

# === FILE PATHS ===
csv_path = "/path/to/file/air_quality_monthly_2023_2024.csv" # csv path
shapefile_path = "/path/to/file/Cordillera Santiago.shp" # shp path
output_gif_path = "/path/to/file/aqi_monthly_animation.gif" # save animation


# === COLUMN NAMES ===
X_COLUMN = "x"             # Name of column for X coordinates
Y_COLUMN = "y"             # Name of column for Y coordinates
DATA_COLUMN = "AQI"        # Name of the data column to interpolate
DATE_COLUMN = "Date"       # Name of the date column

# === ANIMATION SETTINGS ===
FRAME_DURATION = 0.5  # seconds per frame
SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Fall': [9, 10, 11]
}
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Fall']

# === INTERPOLATION SETTINGS ===
interpolation_resolution = 300
rbf_function = 'gaussian'  # Options: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'
rbf_smooth = 0.1

# === MAP ELEMENTS ===
SHOW_SAMPLE_POINTS = True  # Options: True, False
SHOW_COLORBAR = True        # Options: True, False
SHOW_SCALEBAR = True        # Options: True, False
SHOW_NORTH_ARROW = True     # Options: True, False
SHOW_LEGEND = False         # Options: True, False
SHOW_POINT_LABELS = True    # Options: True, False

# === FIGURE SETTINGS ===
FIGURE_SIZE = (12, 10)      # Width, Height in inches
FIGURE_DPI = 100            
TITLE_SIZE = 16
LABEL_SIZE = 12

# === COLOR SCHEME ===
INTERPOLATION_COLORMAP = 'inferno'  # Options: any valid matplotlib colormap (check matplotlib documentation on the internet)
INTERPOLATION_ALPHA = 0.75
CONTOUR_LEVELS = 20
SHAPEFILE_EDGE_COLOR = 'black'
SHAPEFILE_EDGE_WIDTH = 1
SHAPEFILE_FILL_COLOR = 'none'

# === SAMPLE POINTS STYLING ===
POINT_COLOR = 'blue'
POINT_SIZE = 50
POINT_MARKER = 'o'            # Options: 'o', 's', '^', 'D', etc.
POINT_EDGE_COLOR = 'black'
POINT_EDGE_WIDTH = 1
POINT_LABEL_COLOR = 'white'
POINT_LABEL_SIZE = 8
POINT_LABEL_OFFSET_X = 0
POINT_LABEL_OFFSET_Y = 200

# === MAP ELEMENT POSITIONS ===
NORTH_ARROW_X = 0.95          # Relative position (0 to 1)
NORTH_ARROW_Y = 0.95
NORTH_ARROW_SIZE = 14
NORTH_ARROW_COLOR = 'black'

SCALEBAR_X = 0.05
SCALEBAR_Y = 0.05
SCALEBAR_LENGTH = 5000       # In map units (e.g., meters)
SCALEBAR_COLOR = 'black'
SCALEBAR_WIDTH = 4

# === COLORBAR SETTINGS ===
COLORBAR_ORIENTATION = 'vertical'  # Options: 'vertical', 'horizontal'
COLORBAR_LOCATION = 'right'        # Options: 'left', 'right', 'top', 'bottom'
COLORBAR_SIZE = 0.6                # Fraction of height or width (depending on orientation)
COLORBAR_PAD = 0.15

# === LEGEND SETTINGS ===
LEGEND_POSITION = 'upper right'     # Options: 'upper left', 'upper right', 'lower left', 'lower right'
LEGEND_FRAMEON = True
LEGEND_FANCYBOX = True
LEGEND_SHADOW = True
LEGEND_FONTSIZE = 12
LEGEND_MARKERSIZE = 10


# ================================
# === HELPER FUNCTIONS =========
# ================================

def create_interpolation_grid(gdf_boundary, resolution):
    bounds = gdf_boundary.total_bounds
    xi = np.linspace(bounds[0], bounds[2], resolution)
    yi = np.linspace(bounds[1], bounds[3], resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    polygon = gdf_boundary.geometry.iloc[0]
    points = np.column_stack((xi.ravel(), yi.ravel()))
    inside_mask = np.array([polygon.contains(Point(x, y)) for x, y in points]).reshape(xi.shape)
    
    return xi, yi, inside_mask

def perform_interpolation(points_df, xi, yi, inside_mask, data_column):
    xi_inside = xi[inside_mask]
    yi_inside = yi[inside_mask]
    
    rbf = Rbf(points_df["x"], points_df["y"], points_df[data_column], 
              function=rbf_function, smooth=rbf_smooth)
    zi_rbf = rbf(xi_inside, yi_inside)
    
    zi_full = np.full_like(xi, np.nan)
    zi_full[inside_mask] = zi_rbf
    
    return np.ma.masked_where(~inside_mask, zi_full)

def create_legend_patch():
    return mpatches.Circle((0, 0), 1, fc=POINT_COLOR, ec=POINT_EDGE_COLOR,
                          linewidth=POINT_EDGE_WIDTH, label='Sample Points')

def create_frame(gdf_boundary, gdf_points, xi, yi, zi_masked, title_text, temp_dir, frame_count, vmin, vmax):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    
    # Plot shapefile
    gdf_boundary.plot(ax=ax, facecolor=SHAPEFILE_FILL_COLOR, 
                     edgecolor=SHAPEFILE_EDGE_COLOR, 
                     linewidth=SHAPEFILE_EDGE_WIDTH)
    
    # Plot interpolation with global min/max for consistent colorbar
    if INTERPOLATION_COLORMAP not in plt.colormaps():
        cmap = 'viridis'
    else:
        cmap = INTERPOLATION_COLORMAP
    
    # Create contour levels based on global min/max to ensure consistency
    levels = np.linspace(vmin, vmax, CONTOUR_LEVELS)
    
    c = ax.contourf(xi, yi, zi_masked, levels=levels, 
                    cmap=cmap, alpha=INTERPOLATION_ALPHA, 
                    vmin=vmin, vmax=vmax, extend='both')  # Use global min/max with explicit levels
    
    # Plot sample points
    if SHOW_SAMPLE_POINTS:
        gdf_points.plot(ax=ax, color=POINT_COLOR, markersize=POINT_SIZE, 
                       marker=POINT_MARKER, edgecolor=POINT_EDGE_COLOR, 
                       linewidth=POINT_EDGE_WIDTH)
    
    # Add point labels
    if SHOW_POINT_LABELS and SHOW_SAMPLE_POINTS:
        for idx, row in gdf_points.iterrows():
            ax.text(row.geometry.x + POINT_LABEL_OFFSET_X, 
                   row.geometry.y + POINT_LABEL_OFFSET_Y, 
                   f"{row['AQI']:.0f}", color=POINT_LABEL_COLOR, 
                   fontsize=POINT_LABEL_SIZE, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=POINT_COLOR, alpha=0.7))
    
    # Add scalebar
    if SHOW_SCALEBAR:
        bounds = gdf_boundary.total_bounds
        x0 = bounds[0] + (bounds[2] - bounds[0]) * SCALEBAR_X
        y0 = bounds[1] + (bounds[3] - bounds[1]) * SCALEBAR_Y
        
        ax.plot([x0, x0 + SCALEBAR_LENGTH], [y0, y0], 
                color=SCALEBAR_COLOR, lw=SCALEBAR_WIDTH)
        ax.text(x0 + SCALEBAR_LENGTH / 2, y0 - (bounds[3] - bounds[1]) * 0.02, 
                f'{SCALEBAR_LENGTH/1000:.0f} km', ha='center', va='top', 
                fontsize=12, color=SCALEBAR_COLOR)
    
    # Add north arrow
    if SHOW_NORTH_ARROW:
        bounds = gdf_boundary.total_bounds
        arrow_x = bounds[0] + (bounds[2] - bounds[0]) * NORTH_ARROW_X
        arrow_y = bounds[1] + (bounds[3] - bounds[1]) * NORTH_ARROW_Y
        
        ax.annotate('N', xy=(arrow_x, arrow_y),
                    xytext=(0, -20), textcoords='offset points',
                    ha='center', va='top', fontsize=NORTH_ARROW_SIZE, 
                    fontweight='bold', color=NORTH_ARROW_COLOR,
                    arrowprops=dict(arrowstyle='-|>', linewidth=2, color=NORTH_ARROW_COLOR))
    
    # Add title
    ax.set_title(title_text, fontsize=TITLE_SIZE, pad=20)
    
    # Add colorbar with global scale - much larger and static
    if SHOW_COLORBAR:
        cbar = plt.colorbar(c, ax=ax, orientation=COLORBAR_ORIENTATION,
                           location=COLORBAR_LOCATION,
                           shrink=COLORBAR_SIZE, pad=COLORBAR_PAD)
        cbar.set_label('AQI', fontsize=LABEL_SIZE+4, fontweight='bold')
        cbar.ax.tick_params(labelsize=LABEL_SIZE+2, width=2, length=6)
        
        # Set explicit ticks at global min, max, and some intermediate values for static legend
        tick_values = np.linspace(vmin, vmax, 6)  # 6 evenly spaced ticks
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{val:.0f}' for val in tick_values])
        
        # Make colorbar outline thicker
        cbar.outline.set_linewidth(2)
    
    # Add legend
    if SHOW_LEGEND and SHOW_SAMPLE_POINTS:
        legend_patch = create_legend_patch()
        leg = ax.legend(handles=[legend_patch], loc=LEGEND_POSITION,
                       frameon=LEGEND_FRAMEON, fancybox=LEGEND_FANCYBOX,
                       shadow=LEGEND_SHADOW, fontsize=LEGEND_FONTSIZE,
                       markerscale=LEGEND_MARKERSIZE/8)
        
        if LEGEND_FRAMEON:
            frame = leg.get_frame()
            frame.set_facecolor('white')
            frame.set_alpha(0.8)
            frame.set_edgecolor('gray')
            frame.set_linewidth(0.5)
    
    # Remove axis ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    
    # Save frame
    frame_path = os.path.join(temp_dir, f"frame_{frame_count:03d}.png")
    plt.savefig(frame_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    return frame_path

# ================================
# === MAIN PROCESSING ==========
# ================================

def main():
    # Load data
    gdf_boundary = gpd.read_file(shapefile_path)
    df = pd.read_csv(csv_path)
    
    # Process date information
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    
    # Create GeoDataFrame
    df["geometry"] = df.apply(lambda row: Point(row["x"], row["y"]), axis=1)
    gdf_points = gpd.GeoDataFrame(df, geometry="geometry", crs=gdf_boundary.crs)

    # Calculate global min and max AQI values for consistent colorbar
    global_vmin = df['AQI'].min()
    global_vmax = df['AQI'].max()
    print(f"Global AQI range: {global_vmin:.1f} - {global_vmax:.1f}")

    # Create interpolation grid
    xi, yi, inside_mask = create_interpolation_grid(gdf_boundary, interpolation_resolution)
    
    # Create temporary directory for frames
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_paths = []
    frame_count = 0
    
    # Process data by season
    for season_name in SEASON_ORDER:
        season_months = SEASONS[season_name]
        
        # Get data for this season across all years
        season_data = gdf_points[gdf_points['month'].isin(season_months)]
        
        # Group by year and month to process each month separately
        grouped = season_data.groupby(['year', 'month'])
        
        for (year, month), month_df in sorted(grouped, key=lambda x: (x[0][0], x[0][1])):
            # Perform interpolation
            zi_masked = perform_interpolation(month_df, xi, yi, inside_mask, 'AQI')
            
            # Create title with only month and year (no season name)
            month_name = datetime(year, month, 1).strftime('%B %Y')
            title_text = f"{month_name}\nAQI Distribution"
            
            # Create and save frame with global min/max
            frame_path = create_frame(gdf_boundary, month_df, xi, yi, zi_masked, 
                                     title_text, temp_dir, frame_count, 
                                     global_vmin, global_vmax)
            frame_paths.append(frame_path)
            frame_count += 1
    
    # Create GIF from frames with correct duration - using fps for more reliable timing
    print(f"Creating GIF with {len(frame_paths)} frames...")
    fps = 1.0 / FRAME_DURATION  # Convert duration to fps (1 second = 1 fps)
    
    with imageio.get_writer(output_gif_path, mode='I', fps=fps, loop=0) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    # Clean up temporary files
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(temp_dir)
    
    print(f"Animation saved to {output_gif_path}")
    print(f"Total frames: {len(frame_paths)}")
    print(f"Frame duration: {FRAME_DURATION} seconds")
    print(f"FPS: {fps}")
    print(f"Total animation duration: {len(frame_paths) * FRAME_DURATION} seconds")

if __name__ == "__main__":
    main()

