# Mapmaking Course Repository

This repository contains Python scripts and sample datasets for generating a variety of maps, including single maps, multiple seasonal maps, and animated maps (GIFs). These materials are part of the Mapmaking course hosted by the YESS Community.

We are actively updating the scripts. You are welcome (and encouraged) to request specific features or suggest changes. We’ll do our best to accommodate them.

Have fun, and happy mapping!

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

`pip install -r requirements.txt`

### Option 2: Install individually

`pip install geopandas pandas matplotlib numpy shapely scipy imageio`

### Note on GeoPandas

GeoPandas has additional dependencies such as `fiona`, `pyproj`, and `gdal`, which can be tricky to install via pip. If you encounter issues, we recommend installing it via conda:

`conda install geopandas`

This method usually handles all dependencies more smoothly, especially on Windows and Linux systems.

Once the packages are installed, you should be able to run the scripts without any issues.

## Available Scripts

Currently, the repository includes three main scripts:

- Generate a single map with your sampling points interpolated inside your target shapefile.
- Generate multiple maps (e.g., for different seasons or variables).
- Create a GIF slideshow showing seasonal or time-series data for a location.

Dummy data is provided so you can get started right away, but the scripts are designed to work with your own `.shp` and `.csv` files. Each script includes several customization options to modify map elements and styles.

If you'd like additional features or run into any issues, feel free to make a request or report a bug.
