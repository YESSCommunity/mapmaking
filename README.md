# Mapmaking Course Repository

This repository contains Python scripts and sample datasets for creating different types of maps, including single maps, multiple maps, and animated maps. These materials are part of the Mapmaking course hosted by the YESS Community.

We are actively updating the scripts. You are welcome (and encouraged) to request specific changes or suggest new features. We will do our best to accommodate them.

Have fun, and happy mapping!


## Installation

To run the scripts in this repository, you will need to install the following Python packages:

- geopandas  
- pandas  
- matplotlib  
- numpy  
- shapely  
- scipy  
- imageio  

You can install them all at once using the following command:

`pip install -r requirements.txt`

Alternatively, you can install them one by one:

`pip install geopandas pandas matplotlib numpy shapely scipy imageio`

### Note on GeoPandas

GeoPandas has a few additional dependencies like `fiona`, `pyproj`, and `gdal`, which can sometimes be tricky to install via `pip`. If you run into issues, it's often easier to install GeoPandas using conda:

`conda install geopandas`

This will handle all underlying dependencies more smoothly, especially on Windows or Linux systems.

Once all packages are installed, you should be able to run the scripts without any issues.
