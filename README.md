# YESS Mapmaking Course

This repository contains Python scripts and sample datasets for generating various types of maps, including single maps, multiple seasonal maps, and animated GIFs. These materials are part of the Mapmaking course hosted by the YESS Community.

We are actively updating the scripts, and you are welcome (and encouraged) to request specific features or suggest improvements. We'll do our best to accommodate your ideas.

Have fun, and happy mapping!

---

## Installation

To run the scripts in this repository, you will need the following Python packages:

- geopandas  
- pandas  
- matplotlib  
- numpy  
- shapely  
- scipy  
- imageio  

### Option 1: Install all at once

```bash
pip install -r requirements.txt
```

### Option 2: Install individually

```bash
pip install geopandas pandas matplotlib numpy shapely scipy imageio
```

### Note on GeoPandas

GeoPandas requires additional dependencies such as `fiona`, `pyproj`, and `gdal`, which can be difficult to install via `pip`. If you encounter installation issues, we recommend using `conda`:

```bash
conda install geopandas
```

This approach usually resolves all dependencies more reliably, especially on Windows and Linux systems.

Once all packages are installed, you should be able to run the scripts without any issues.

---

## Available Scripts

The repository currently includes the following main scripts:

- **Script_1**: Generates a single map with your sampling points interpolated within your target shapefile.  
- **Script_2**: Generates multiple maps (e.g., for different seasons or variables).  
- **Script_3**: Creates a GIF slideshow showing seasonal or time-series data for a location.  

Sample datasets are provided so you can get started right away. However, the scripts are designed to work with your own `.shp` and `.csv` files. Each script includes several customization options to adjust map elements and styles.

If you'd like to request additional features or encounter any issues, feel free to open a request or report a bug.

---

## Sample Data Format

**The sample data is structured as follows (first two rows shown):**

*For single interpolation and study area map:*
```
SampleName,x,y,AQI
S1,363626.6168816242,6279077.259971289,12.9
S2,362471.5586498214,6274457.027044078,29.4
```

*For multiple maps and grids:*
```
SampleName,x,y,Date,AQI
S1,363626.6168816242,6279077.259971289,2023-01-15,33.1
S2,362471.5586498214,6274457.027044078,2023-01-15,16.1
```
