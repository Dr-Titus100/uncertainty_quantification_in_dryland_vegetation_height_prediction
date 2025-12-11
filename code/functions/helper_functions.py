# packages
import os
import laspy
import pylas
import pyproj
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
from sliderule import icesat2
from shapely.wkt import loads
from pyntcloud import PyntCloud
from scipy.spatial import cKDTree
from shapely.geometry import Point
from datetime import datetime, timedelta
from math import radians, cos, sin, sqrt, degrees
        
##################################################################
##################################################################
def reduce_dataframe(gdf, RGT=None, GT=None, track=None, pair=None, cycle=None, beam="", crs = 6340): # default crs for GEDI is crs=4326
    """
    A function for subsetting icesat2 data. The idea is to reduce the GeoDataFrame to plot a single beam.
    Convert coordinate reference system to compound projection
    This function was adopted from sliderule tutorial "https://slideruleearth.io/web/rtd/_static/html/grand_mesa_atl03_classification.html"
    """
    # convert coordinate reference system
    D3 = gdf.to_crs(crs)
    
    # reduce to reference ground track
    if RGT is not None:
        D3 = D3[D3["rgt"] == RGT]
    # reduce to ground track (gt[123][lr]), track ([123]), or pair (l=0, r=1) 
    gtlookup = {icesat2.GT1L: 1, icesat2.GT1R: 1, 
                icesat2.GT2L: 2, icesat2.GT2R: 2, 
                icesat2.GT3L: 3, icesat2.GT3R: 3}
    pairlookup = {icesat2.GT1L: 0, icesat2.GT1R: 1, 
                  icesat2.GT2L: 0, icesat2.GT2R: 1, 
                  icesat2.GT3L: 0, icesat2.GT3R: 1}

    if GT is not None:
        D3 = D3[(D3["track"] == gtlookup[GT]) & (D3["pair"] == pairlookup[GT])]
    if track is not None:
        D3 = D3[D3["track"] == track]
    if pair is not None:
        D3 = D3[D3["pair"] == pair]
    # reduce to weak or strong beams
    # tested on cycle 11, where the strong beam in the pair matches the spacecraft orientation.
    # need to check on other cycles
    if (beam == "strong"):
        D3 = D3[D3["sc_orient"] == D3["pair"]]
    elif (beam == "weak"):
        D3 = D3[D3["sc_orient"] != D3["pair"]]
    # reduce to cycle
    if cycle is not None:
        D3 = D3[D3["cycle"] == cycle]
    # otherwise, return both beams
    return D3

# ##################################################################
# ##################################################################
# def calculate_vegetation_height(row):
#     """
#     Calculate vegetation height for ICESat2 based on the criteria:
#     1. If canopy top height exists, vegetation height = canopytop_height - ground_height
#     2. Otherwise, if canopy height exists, vegetation height = canopy_height - ground_height
#     3. If neither exists, vegetation height = 0
#     """
#     if not pd.isna(row["canopytop_height"]):
#         return row["canopytop_height"] - row["ground_height"]
#     elif not pd.isna(row["canopy_height"]):
#         return row["canopy_height"] - row["ground_height"]
#     else:
#         return 0
    
##################################################################
##################################################################
def calculate_slope_utm(easting1, northing1, elev1, easting2, northing2, elev2):
    """
    Calculates slope using UTM coordinates (Easting, Northing in meters).

    Parameters:
        easting1, northing1: Coordinates of point 1 (in meters)
        easting2, northing2: Coordinates of point 2 (in meters)
        elev1, elev2: Elevation values for point 1 and 2 (in meters)

    Returns:
        Slope = rise/run (unitless, can be converted to degrees or percentage grade)
    """
    dx = easting2 - easting1
    dy = northing2 - northing1
    horizontal_distance = np.sqrt(dx**2 + dy**2)  # Euclidean distance. Base of the triangle
    elevation_difference = elev2 - elev1 # vertical side of the triangle

    if horizontal_distance == 0:
        return 0  # Avoid division by zero

    slope_ratio = elevation_difference/horizontal_distance  # Rise over run. This result is a unitless ratio (e.g., 0.1 = 10% slope).
    # slope_percent = slope_ratio*100
    
    # slope_degrees = degrees(np.arctan(slope_ratio))
    slope_degrees = degrees(np.arctan2(elevation_difference, horizontal_distance))
    return slope_ratio, slope_degrees

##################################################################
##################################################################
# Calculate slope
def calculate_distance_haversine(lat1, lon1, lat2, lon2):
    """
    Calculates horizontal distance between two points on the ground while taking into account the curvature of the Earth. 
    The haversine function computes the great-circle distance between two points on the Earth’s surface given their latitude and longitude. 
    This is effective for data in geographic coordinates (latitude and longitude).
    NB: Taking into account the Earth's curvature isn't that important when dealing with short distances. 
        Also, you can't apply Haversine directly to UTM because it is not a spherical geometry. 
        To use the Haversine formula, the input must be in latitude and longitude, because the formula is based on the spherical geometry of the Earth.
        It does not work with UTM (Easting/Northing), which are projected (flat) Cartesian coordinates.
    """
    R = 6371000  # earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(sqrt(a), sqrt(1 - a))
    return R * c

##################################################################
##################################################################
# Function to calculate slope between points
def calculate_slope_rise_run(gdf, prefix):
    """
    Calculates slope using vertical and horizontal difference between points.

    Here, we use the difference in elevation and coordinates between successive point to compute the slope between those points. 
    To calculate the slope between successive points,  we will calculate the rate of change in elevation over distance between points in the data. 
    Thus, we will compute the change in elevation (rise) and the change in horizontal distance (run) between each consecutive point. 
    Finally, the slope is the ratio of the two, rise/run. slope = (change in elevation(rise))/(change in horizontal distance(run)).
    """
    slopes = []
    for i in range(len(gdf) - 1):
        # get current and next points
        point1 = gdf.iloc[i].geometry
        point2 = gdf.iloc[i + 1].geometry
        
        # calculate horizontal distance in meters
        horizontal_distance = point1.distance(point2)
        
        # calculate elevation difference
        if prefix in ["GEDI01_B", "GEDI02_A", "GEDI02_B"]:
            elevation_diff = gdf.iloc[i + 1].elevation_NAVD88_gedi_from_crs - gdf.iloc[i].elevation_NAVD88_gedi_from_crs
        elif prefix == "ICESat2":
            elevation_diff = gdf.iloc[i + 1].elevation_NAVD88_icesat2_from_crs - gdf.iloc[i].elevation_NAVD88_icesat2_from_crs
        
        # calculate slope in degrees
        slope = degrees(np.arctan2(elevation_diff, horizontal_distance))
        slopes.append(slope)
    
    # append NaN for the last point (since there’s no slope calculation for the final point)
    slopes.append(np.nan)
    
    # add slope to the GeoDataFrame
    col_name = prefix+f'{"_slope_rise_run"}'
    gdf[col_name] = slopes
    # gdf[col_name] = np.abs(slopes)
    return gdf

##################################################################
##################################################################
def gps_to_utc(gps_seconds, leap_seconds=18):
    """
    Converts GPS time (seconds since 2019-01-01 10:17:16) to UTC time, standard datetime format.
    Thus, this function converts GPS time to a readable date.
    GEDI timestamps are usually stored in seconds since GPS epoch start date and time (2018-01-01 00:00:00 UTC). 
    Confirmed from the data attributes and description for `delta_time`.
    
    Parameters:
        gps_seconds (float): Time in GPS seconds
        leap_seconds (int): The GPS-UTC offset at the given time. As of 2024, GPS time is ahead of UTC by 18 seconds
        
    Returns:
        datetime: UTC datetime object
    """
    gps_epoch = datetime(2018, 1, 1, 0, 0, 0)  # GPS epoch start date. Format: YYYY, MM, DD, HH, MM, SS
    return gps_epoch + timedelta(seconds=gps_seconds - leap_seconds)

# # Apply conversion to GEDI dataset
# # Note: gps_to_utc is a function that converts GPS time to a readable date
# merged_gedi_gdf_L1B['datetime'] = merged_gedi_gdf_L2A['delta_time'].apply(gps_to_utc)

# # Extract only the date if needed
# merged_gedi_gdf_L1B['date'] = merged_gedi_gdf_L1B['datetime'].dt.date

##################################################################
##################################################################
# Function to convert YYYYDDDHHMMSS format to YYYY-MM-DD HH:MM:SS
def convert_to_datetime(filename):
    """
    Extracts the 13-digit acquisition date from the GEDI filename.
    The naming of GEDI version 2 data files follows a standard naming convention. 
    For instance, for the file named GEDI02_B_2021295182220_O16205_03_T09676_02_003_01_V002.h5 implies:
        - GEDI02_B: Product Short Name
        - 2021295182220: Julian Date and Time of Acquisition (YYYYDDDHHMMSS)
        - O16205: Orbit Number
        - 03: Sub-Orbit Granule Number (1-4)
        - T09676: Track Number (Reference Ground Track)
        - 02: Positioning and Pointing Determination System (PPDS) type (00 is predict, 01 rapid, 02 and higher is final)
        - 003: PGE Version Number
        - 01: Granule Production Version
        - V002: Product Version

    We used the same standard naming convention to name our GeoJSON files. Hence, we can use the file names to compute the acquisition date of the data.
    The GEDI data contains the GPS time `delta_time`. This GEDI timestamps are usually stored in seconds since GPS epoch. 
    Since we are not so sure of the GPS epoch start date and time, it is safer to compute the acquisition date and time from the file names.
    
    This functions helps us to convert the acquisition date from YYYYDDDHHMMSS format to YYYY-MM-DD HH:MM:SS format.
    """
    parts = filename.split("_")  # split filename by underscores
    if len(parts) == 10: # check if filename format is as expected.
        acquisition_date = parts[2]  # if filename format is as expected, The second part (index 2) contains the date
    else:
        acquisition_date = None # return None if filename format is unexpected
        
    if acquisition_date is None or len(acquisition_date) != 13:
        return None  # return None if acquisition_date is invalid

    # extract components
    year = int(acquisition_date[:4])      # YYYY
    day_of_year = int(acquisition_date[4:7])  # DDD
    hour = int(acquisition_date[7:9])     # HH
    minute = int(acquisition_date[9:11])  # MM
    second = int(acquisition_date[11:13]) # SS

    # convert DDD to month and day using datetime. Thus, converts Julian day (DDD) into a calendar date.
    date_obj = datetime(year, 1, 1) + timedelta(days=day_of_year - 1) #this automatically takes care of leap year effects.

    # return formatted string
    return acquisition_date, date_obj.replace(hour=hour, minute=minute, second=second).strftime("%Y-%m-%d %H:%M:%S")

##################################################################
##################################################################
"""
Computing Vegetation height

The approach here is to compute the difference between the ground elevation and the canopy top elevation.
With the GEDI data, it is quite straightforward. In other words, we use the elevations of the detected modes in the received waveform, `rxwaveform`, to compute the vegetation. We simply find the difference between the elevations of the highest and lowest detected modes to calculate the Relative Height(RH) metrics or simply the vegetation height. If the received waveform has only one detected mode, then it represents bare ground and the vegetation height is zero. Thus, for such a received waveform, the elevations of the highest and lowest detected modes are equal. 

```
Interpretation of RH Metrics
The GEDI L2A data product provides relative height (RH) metrics, which are “lidar perceived” metrics that have the following characteristics:
    1. RH100 = elev_highestreturn - elev_lowestmode
    2. The RH metrics are intended for vegetated surfaces. Results over bare/water surfaces are still valid but may present some confusing results.
    3. The lower RH metrics (e.g., RH10) will often have negative values, particularly in low canopy cover conditions. This is because a relatively high fraction of the waveform energy is from the ground and below elev_lowestmode. For example, if the ground return contains 30% of the energy, then RH1 through 15 are likely to be below 0 since half of the ground energy from the ground return is below the center of the ground return, which is used to determine the mean ground elevation in the footprint (elev_lowestmode).
```

`elev_highestreturn:` "Elevation of the highest detected return relative to reference ellipsoid". This is the elevation of the canopy top.
`elev_lowestmode:` "Elevation of center of lowest mode relative to reference ellipsoid". This is the ground elevation.

For the ICESat-2 data, we use a simple nearest neighbors approach. For each ground photon, we use the `cKDTree` algorithm to find the nearest canopy top photon spatially and temporally. We then take the difference between the heights of the canopy top photon and the ground photon as the vegetation height. In the absence of a canopy top photon we use the canopy photon.
"""

def calculate_icesat2_veg_height_wgs(icesat2_gdf): #compute vegetation height using WGS global coordinates (lon/lat)
    """
    The algorithm in this code:
        1. Finds all canopy top and canopy photons that were taken on the same date as each ground photon
        2. Selects the spatially closest canopy top photon within the radius threshold
        3. If no canopy top photon is found, selects the spatially closest canopy photon
        4. If neither is found, vegetation height is set to zero
        5. Ensures that each canopy top or canopy photon is assigned to only one ground photon (unique matching).

    Summary:
    1. Filters canopy top and canopy photons by the same date as each ground photon first
        a. `same_date_canopytop = canopytop_photons[canopytop_photons['date'] == ground_date]`
        b. `same_date_canopy = canopy_photons[canopy_photons['date'] == ground_date]`

    2. Uses `cKDTree` for spatial matching on same-date photons and maintains priority order
        a. Finds the closest canopy top photon within 30m
        b. If none are found, finds the closest canopy photon within 30m
        c. If neither are found, sets vegetation height to 0

    3. Stops searching after finding the best match
        a This avoids unnecessary distance calculations.

    4. Ensures that each canopy top or canopy photon is assigned to only one ground photon.
        a. Uses `used_canopytop_indices` and `used_canopy_indices` to keep track of assigned photons.
        b. A photon can only be used once as a nearest neighbor.

    5. Avoids duplicate matches by checking `if idx_top not in used_canopytop_indices` before assigning a match.
        a. Prevents multiple ground photons from using the same canopy top/canopy photon.

    Output
        1. Each ground photon gets matched to a unique canopy top or canopy photon that was captured on the same date.
        2. Only the spatially closest photon (within 30m) is used.
        3. If no match is found, the vegetation height is set to 0.

    Returns
        The function returns a geodataframe with ICESat2 vegetation heights computed
    """
    # convert dates to pandas datetime format
    icesat2_gdf["date"] = pd.to_datetime(icesat2_gdf["date"])

    # filter photons based on `atl08_class`
    # ATL08 classification: (0: noise, 1: ground, 2: canopy, 3: top of canopy, 4: unclassified)
    canopytop_photons = icesat2_gdf[icesat2_gdf["atl08_class"] == 3].copy()  # canopy top
    canopy_photons = icesat2_gdf[icesat2_gdf["atl08_class"] == 2].copy()  # canopy
    ground_photons = icesat2_gdf[icesat2_gdf["atl08_class"] == 1].copy()  # ground

    # convert latitude, longitude to radians for accurate spatial distance calculations
    # ground_photons['coords'] = np.radians(ground_photons[['Latitude', 'Longitude']].values)
    # ground_photons['coords'] = np.radians(ground_photons[['Latitude', 'Longitude']].values.tolist())
    ground_photons = ground_photons.copy()
    ground_photons["Latitude_rad"] = np.radians(ground_photons["Latitude"])
    ground_photons["Longitude_rad"] = np.radians(ground_photons["Longitude"])
    ground_photons["coords"] = ground_photons.apply(lambda row: (row["Latitude_rad"], row["Longitude_rad"]), axis=1)

    # distance threshold (approx. 30 meters in radians)
    distance_threshold = 4.71e-6  # ~30 meters

    # dictionary to store unique matches for canopy top and canopy photons
    used_canopytop_indices = set()
    used_canopy_indices = set()

    # initialize list for vegetation heights
    veg_heights = []

    # iterate over each ground photon
    for i, ground in ground_photons.iterrows():
        ground_date = ground["date"]

        # step 1: filter canopy top and canopy photons to the same date
        same_date_canopytop = canopytop_photons[canopytop_photons["date"] == ground_date]
        same_date_canopy = canopy_photons[canopy_photons["date"] == ground_date]

        # convert coordinates to radians
        canopytop_coords = np.radians(same_date_canopytop[["Latitude", "Longitude"]].values)
        canopy_coords = np.radians(same_date_canopy[["Latitude", "Longitude"]].values)

        # step 2: If any canopy top photons exist on the same date, find the spatially closest one
        if not same_date_canopytop.empty:
            canopytop_tree = cKDTree(canopytop_coords)
            dist_top, idx_top = canopytop_tree.query(ground["coords"], distance_upper_bound=distance_threshold)

            if dist_top != np.inf and idx_top not in used_canopytop_indices:  # unique match
                vegetation_height = same_date_canopytop.iloc[idx_top]["height"] - ground["height"]
                veg_heights.append(vegetation_height)
                used_canopytop_indices.add(idx_top)  # mark as used
                continue  # stop here as we found a match

        # step 3: If no canopy top photon was found, look for a canopy photon
        if not same_date_canopy.empty:
            canopy_tree = cKDTree(canopy_coords)
            dist_canopy, idx_canopy = canopy_tree.query(ground["coords"], distance_upper_bound=distance_threshold)

            if dist_canopy != np.inf and idx_canopy not in used_canopy_indices:  # unique match
                vegetation_height = same_date_canopy.iloc[idx_canopy]["height"] - ground["height"]
                veg_heights.append(vegetation_height)
                used_canopy_indices.add(idx_canopy)  # mark as used
                continue  # stop here as we found a match

        # step 4: if no valid canopy top or canopy photon was found, set vegetation height to 0
        veg_heights.append(0)

    # store vegetation height in the DataFrame
    icesat2_gdf_veg_ground = ground_photons.copy()
    icesat2_gdf_veg_ground["vegetation_height"] = veg_heights
    return icesat2_gdf_veg_ground

##################################################################
##################################################################
def calculate_icesat2_veg_height_utm(icesat2_gdf): #compute vegetation height using UTM coordinates in meters (easting/northing)
    """
    Computes vegetation height from ICESat-2 ATL03 data using ATL08 classification.
    Uses UTM (Easting/Northing) coordinates for accurate spatial distance calculation.
    
    For each ground photon:
        1. Filters canopy top and canopy photons from the same date.
        2. Finds the spatially closest canopy top photon within 30m.
        3. If none found, finds the closest canopy photon within 30m.
        4. Ensures one-to-one matching.
        5. If no match found, vegetation height = 0.

    Returns:
        A GeoDataFrame of ground photons with computed vegetation heights.
    """

    # Convert date to datetime
    icesat2_gdf["date"] = pd.to_datetime(icesat2_gdf["date"]).dt.strftime("%Y-%m-%d")

    # Filter photons by classification
    canopytop_photons = icesat2_gdf[icesat2_gdf["atl08_class"] == 3].copy()
    canopy_photons = icesat2_gdf[icesat2_gdf["atl08_class"] == 2].copy()
    ground_photons = icesat2_gdf[icesat2_gdf["atl08_class"] == 1].copy()

    # Use UTM coordinates for spatial calculations
    ground_photons = ground_photons.copy()
    ground_photons["coords"] = ground_photons[["Easting", "Northing"]].values.tolist()

    # Distance threshold in meters
    distance_threshold = 5 #30  # meters

    # Track used canopy and canopy top indices (for unique assignment)
    used_canopytop_indices = set()
    used_canopy_indices = set()

    # Output list
    veg_heights = []

    # Iterate through each ground photon
    for i, ground in ground_photons.iterrows():
        ground_date = ground["date"]
        ground_coord = np.array(ground["coords"])

        # Filter by date
        same_date_canopytop = canopytop_photons[canopytop_photons["date"] == ground_date]
        same_date_canopy = canopy_photons[canopy_photons["date"] == ground_date]

        # STEP 1: Try canopy top
        if not same_date_canopytop.empty:
            canopytop_coords = same_date_canopytop[["Easting", "Northing"]].values
            canopytop_tree = cKDTree(canopytop_coords)
            dist_top, idx_top = canopytop_tree.query(ground_coord, distance_upper_bound=distance_threshold)

            if dist_top != np.inf and idx_top not in used_canopytop_indices:
                vegetation_height = same_date_canopytop.iloc[idx_top]["height"] - ground["height"]
                veg_heights.append(vegetation_height)
                used_canopytop_indices.add(idx_top)
                continue

        # STEP 2: Try canopy
        if not same_date_canopy.empty:
            canopy_coords = same_date_canopy[["Easting", "Northing"]].values
            canopy_tree = cKDTree(canopy_coords)
            dist_canopy, idx_canopy = canopy_tree.query(ground_coord, distance_upper_bound=distance_threshold)

            if dist_canopy != np.inf and idx_canopy not in used_canopy_indices:
                vegetation_height = same_date_canopy.iloc[idx_canopy]["height"] - ground["height"]
                veg_heights.append(vegetation_height)
                used_canopy_indices.add(idx_canopy)
                continue

        # STEP 3: No match found
        veg_heights.append(0)

    # Assign vegetation height to output DataFrame
    icesat2_gdf_veg_ground = ground_photons.copy()
    icesat2_gdf_veg_ground["vegetation_height"] = veg_heights
    return icesat2_gdf_veg_ground

##################################################################
##################################################################
# Function to load GeoJSON files and add acquisition_date and date columns
def load_gedi_data(gedi_dir, prefix):#, slope_type):
    """
    Loads GEDI GeoJSON files from a directory, adding the acquisition_date and date columns and 
    subset for snow-off months and full power beams.
    We reproject the data to the same coordinate reference system as the ground truth (airborne lidar) data. We do so both horizontally and vertically.
    For the GEDI02_A product, compute slope using two approaches; haversine or rise/run approach.
    """
    
    #--------Create a transformer for vertical CRS transformation--------#
    #--------Method 1: NAVD88 (vertical transformation)--------#
    # Path to the geoid grid file for NAVD88
    grid_path = pyproj.datadir.get_data_dir() + "/us_noaa_g2018u0.tif" # best available transformation for CONUS (United States mainland) uses us_noaa_g2018u0.tif
    # grid_path = pyproj.datadir.get_data_dir() + "/us_noaa_vertconw.tif"

    # Define PROJ pipeline using GEOID12B (us_noaa_g2012ba0.tif)
    proj_pipeline = f"""
        +proj=pipeline 
        +step +proj=vgridshift +grids={grid_path}
    """
    # Create the transformer for vertical transformation only using the PROJ pipeline. The transformer uses the geoid model
    vertical_transformer1 = pyproj.Transformer.from_pipeline(proj_pipeline)

    #--------Method 2: NAVD88 (vertical transformation)--------#
    #--------Explicitly define source and target CRS--------#
    source_crs = "EPSG:7912" # Source CRS: WGS84 with ellipsoidal height (3D)
    target_crs = "EPSG:6340+5703" # Target CRS: UTM Zone 11N (horizontal) + NAVD88 (vertical) gravity-related height
    # Create the transformation with explicit CRS definition
    vertical_transformer2 = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    # list GEDI files
    gedi_files = [f for f in os.listdir(gedi_dir) if f.startswith(prefix) and f.endswith(".geojson")]
    gdf_list = []
    for file in gedi_files:
        gdf = gpd.read_file(os.path.join(gedi_dir, file))  # load GeoJSON
        if len(gdf) == 0: #check whether file is empty or not
            pass
        else:
            # Get acquisition and formatted date.
            acquisition_date, formatted_date = convert_to_datetime(file)
            gdf["acquisition_date"] = acquisition_date # add acquisition_date column
            gdf["date"] = formatted_date  # convert and store formatted date

            #--------reproject to UTM zone for Boise Idaho--------#
            # Horizontal transformation: The EPSG code for NAD83/UTM zone 11N is 26911 (NAD83 1983) or 6340 (NAD83 2011). 
            # utm_code = "epsg:26911"  # projected CRS for easier distance calculation
            utm_code = "epsg:6340"  # This CRS represent both horizontal (UTM Zone 11N) and vertical (NAVD88) coordinate systems together. projected CRS for easier distance calculation.
            gdf = gdf.to_crs(utm_code) # reproject data to desired CRS.
            gdf["Easting"] = gdf["geometry"].x
            gdf["Northing"] = gdf["geometry"].y 
            
            # Elevation columns
            if prefix in ["GEDI01_B", "GEDI02_B"]:
                gdf["elevation"] = gdf["geolocation_digital_elevation_model"]
            elif prefix == "GEDI02_A":
                gdf["elevation"] = gdf["digital_elevation_model"]
    
            #--------Vertical CRS transformation--------#
            #--------Method 1: NAVD88 (vertical transformation)--------#
            # # Apply vertical transformation (only changing elevation)
            gdf["elevation_NAVD88_gedi_from_pipe"] = gdf.apply(
                lambda row: vertical_transformer1.transform(row.Longitude, row.Latitude, row.elevation)[-1],
                axis=1)

            #--------Method 2: NAVD88 (vertical transformation)--------#
            #--------Explicitly define source and target CRS--------#
            # Apply the transformation (longitude, latitude remain unchanged)
            _, __, gdf["elevation_NAVD88_gedi_from_crs"] = zip(
                *gdf.apply(lambda row: vertical_transformer2.transform(row.Longitude, row.Latitude, row.elevation), 
                           axis=1))
            
            # extract UTM coordinates and elevation for GEDI data
            easting_gedi = gdf["Easting"][:]
            northing_gedi = gdf["Northing"][:]
            elevations_gedi = gdf["elevation_NAVD88_gedi_from_crs"][:]
            
            # We use L2A for calculating slope because the coordinates have been geo-corrected from the L1B product. 
            # Hence, the L1B coordiantes are slightly different from the L2A and L2B products.
            if prefix == "GEDI02_A": 
                ratio_col_name = prefix+f'{"_slope_ratio_utm"}'
                degrees_col_name = prefix+f'{"_slope_degrees_utm"}'
                # Calculate slope for GEDI using UTM coordinates. Rise over run
                gedi_slope_ratio_utm = [-999]
                gedi_slope_degrees_utm = [-999]
                for i in range(1, len(gdf)):
                    easting1, northing1, elev1 = easting_gedi.iloc[i-1], northing_gedi.iloc[i-1], elevations_gedi.iloc[i-1]
                    easting2, northing2, elev2 = easting_gedi.iloc[i], northing_gedi.iloc[i], elevations_gedi.iloc[i] 
                    slope_ratio_utm, slope_degrees_utm = calculate_slope_utm(easting1, northing1, elev1, easting2, northing2, elev2)
                    gedi_slope_ratio_utm.append(slope_ratio_utm)
                    gedi_slope_degrees_utm.append(slope_degrees_utm)
                gdf[ratio_col_name] = gedi_slope_ratio_utm
                gdf[degrees_col_name] = gedi_slope_degrees_utm
                # gdf[ratio_col_name] = np.abs(gedi_slope_ratio_utm)
                # gdf[degrees_col_name] = np.abs(gedi_slope_degrees_utm)

                # elif slope_type == 'rise_run': # use rise/run approach
                # calculate slope for GEDI data
                gdf = calculate_slope_rise_run(gdf, prefix) # Compute slope using the rise/run approach
            else:
                pass
                # elevations_gedi = gdf["digital_elevation_model"][:]
            gdf_list.append(gdf)
        
    # concatenate all GeoDataFrames into a single one
    merged_gdfs = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    
    # subset snow-off months, we consider months, from May to October as snow-off months
    merged_gdfs["date"] = pd.to_datetime(merged_gdfs["date"])
    merged_gdfs = merged_gdfs[(merged_gdfs["date"].dt.month >= 5)&(merged_gdfs["date"].dt.month <= 10)]

    # list of full power beams
    full_power_beams = ["BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"]
    # subset the dataframe to include only rows with full power beams
    merged_gdfs = merged_gdfs[merged_gdfs["BEAM"].isin(full_power_beams)]
    if prefix == "GEDI02_A":
        merged_gdfs = merged_gdfs.dropna()
        merged_gdfs = merged_gdfs[merged_gdfs[ratio_col_name] != -999]
        merged_gdfs = merged_gdfs[merged_gdfs[degrees_col_name] != -999]
    else:
        pass
    merged_gdfs = merged_gdfs.to_crs(utm_code)
    return merged_gdfs

##################################################################
##################################################################
def load_icesat2_data(icesat2_dir, prefix, process = False, filetype = "geojson"):
    """
    Loads ICESat2 GeoJSON/CSV files from a directory and 
    subset for snow-off months and full power beams.
    For the ATL03 product, compute slope using two approaches; haversine or rise/run approach.
    
    Return: this functions returns either ICESat2 ATL03 GeoDataFrame with slope computed or the ATLO8 GeoDataFrame.
    """
    # reproject to UTM zone for Boise Idaho.
    # Horizontal transformation: The EPSG code for NAD83/UTM zone 11N is 26911 (NAD83 1983) or 6340 (NAD83 2011). 
    # utm_code = "epsg:26911"  # projected CRS for easier distance calculation
    utm_code = "epsg:6340"  # This CRS represent both horizontal (UTM Zone 11N) and vertical (NAVD88) coordinate systems together. projected CRS for easier distance calculation.
    if process == True:
        if prefix == "ATL03": 
            if filetype == "geojson":
                icesat2_gdf_atl03 = gpd.read_file(icesat2_dir)
            elif filetype == "csv":
                icesat2_gdf_atl03 = pd.read_csv(icesat2_dir)
                # convert WKT (Well Known Text) to geometry
                icesat2_gdf_atl03["geometry"] = icesat2_gdf_atl03["geometry"].apply(loads)

            print("Columns:", icesat2_gdf_atl03.columns)
            # Convert to GeoDataFrame
            icesat2_gdf_atl03 = gpd.GeoDataFrame(icesat2_gdf_atl03, geometry="geometry", crs=utm_code)

            # the photon’s quality classification (0: nominal, 1: possible after pulse, 2: possible impulse responpse effect, 3: possible tep)
            icesat2_gdf_atl03 = icesat2_gdf_atl03[icesat2_gdf_atl03["quality_ph"] == 0]

            # This condition means we are only interested in strong beams
            icesat2_gdf_atl03 = icesat2_gdf_atl03[icesat2_gdf_atl03["pair"] == icesat2_gdf_atl03["sc_orient"]]

            # subset snow-off months, we consider months, from May to October as snow-off months
            icesat2_gdf_atl03["date"] = pd.to_datetime(icesat2_gdf_atl03["time"])
            icesat2_gdf_atl03 = icesat2_gdf_atl03[(icesat2_gdf_atl03["date"].dt.month >= 5)&(icesat2_gdf_atl03["date"].dt.month <= 10)]

            # filter photons based on `atl08_class`. Subset the atl03 product to eliminate photons classified as either `noise` or `unclassified`.
            # ATL08 classification: (0: noise, 1: ground, 2: canopy, 3: top of canopy, 4: unclassified)
            icesat2_gdf_atl03_photons = icesat2_gdf_atl03[icesat2_gdf_atl03["atl08_class"].isin([1, 2, 3])] 

            # compute vegetation height
            """
            We use a simpple nearest neighbors approach. 
            For each ground photon, we use the `cKDTree` algorithm to find the nearest canopy top photon spatially and temporally. 
            We then take the difference between the heights of the canopy top photon and the ground photon as the vegetation height. 
            In the absence of a canopy top photon we use the canopy photon.
            """
            # icesat2_gdf_atl03_veg_ground = calculate_icesat2_veg_height_utm(icesat2_gdf_atl03)
            # icesat2_data = icesat2_gdf_atl03_veg_ground
            
            # icesat2_gdf_atl03_veg_ground = calculate_icesat2_veg_height_wgs(icesat2_gdf_atl03)

            icesat2_data = icesat2_gdf_atl03_photons
        elif prefix == "ATL08":
            # load the ICESat2 GeoJSON file
            icesat2_gdf_atl08 = gpd.read_file(icesat2_dir)
            # icesat2_gdf_atl08 = icesat2_gdf_atl08.to_crs(utm_code) # reproject the ICESat2 geometries to the same CRS as the airborne lidar data
            # # this condition means we are only interested in strong beams
            # icesat2_gdf_atl08 = icesat2_gdf_atl08[icesat2_gdf_atl08["pair"] == icesat2_gdf_atl08["sc_orient"]]
            # subset snow-off months, we consider months, from May to October as snow-off months
            icesat2_gdf_atl08["date"] = pd.to_datetime(icesat2_gdf_atl08["time"])
            icesat2_data = icesat2_gdf_atl08[(icesat2_gdf_atl08["date"].dt.month >= 5)&(icesat2_gdf_atl08["date"].dt.month <= 10)] 
            # return icesat2_gdf_atl08
        
        ###########################################################################
        ###########################################################################
        # Calculate slope for ICESat2 using UTM coordinates. Rise over run
        print("Columns:", icesat2_data.columns)
        easting_icesat2 = icesat2_data["Easting"][:]
        northing_icesat2 = icesat2_data["Northing"][:]
        elevations_icesat2 = icesat2_data["elevation_NAVD88_icesat2_from_crs"][:]
        icesat2_slope_ratio_utm = [-999]
        icesat2_slope_degrees_utm = [-999]
        # print("Len(icesat2_data):", len(icesat2_data))
        for i in range(1, len(icesat2_data)):
            easting1, northing1, elev1 = easting_icesat2.iloc[i-1], northing_icesat2.iloc[i-1], elevations_icesat2.iloc[i-1]
            easting2, northing2, elev2 = easting_icesat2.iloc[i], northing_icesat2.iloc[i], elevations_icesat2.iloc[i] 
            slope_ratio_utm, slope_degrees_utm = calculate_slope_utm(easting1, northing1, elev1, easting2, northing2, elev2)
            icesat2_slope_ratio_utm.append(slope_ratio_utm)
            icesat2_slope_degrees_utm.append(slope_degrees_utm)
        # icesat2_slopes
        product = "ICESat2"
        ratio_col_name = product+f'{"_slope_ratio_utm"}'
        degrees_col_name = product+f'{"_slope_degrees_utm"}'
        icesat2_data[ratio_col_name] = icesat2_slope_ratio_utm
        icesat2_data[degrees_col_name] = icesat2_slope_degrees_utm
        # icesat2_gdf_atl03_veg_ground[ratio_col_name] = np.abs(icesat2_slope_ratio_utm)
        # icesat2_gdf_atl03_veg_ground[degrees_col_name] = np.abs(icesat2_slope_degrees_utm)
        ###########################################################################
        ###########################################################################

        # method 2: the rise/run approach
        """
        In this approach, we use the difference in elevation and coordinates between successive points to compute the slope between those points. 
        To calculate the slope between successive points, we will calculate the rate of change in elevation over distance between points in the data.
        Thus, we will compute the change in elevation (rise) and the change in horizontal distance (run) between each consecutive point.
        Finally, the slope is the ratio of the two, rise/run.
        """
        # change to a projected CRS for easier distance calculation
        # Horizontal transformation: The EPSG code for NAD83/UTM zone 11N is 26911 (NAD83 1983) or 6340 (NAD83 2011). 
        # icesat2_gdf3 = icesat2_gdf2.to_crs(utm_code)

        # Calculate slope for ICESat-2 data
        # col_name = product+f'{"_slope_rise_run"}'
        # icesat2_gdf = calculate_slope_rise_run(icesat2_gdf_atl03_veg_ground, product)
        # icesat2_gdf_atl03_veg_ground[col_name] = icesat2_gdf[col_name]

        icesat2_data = calculate_slope_rise_run(icesat2_data, product) # Compute slope using the rise/run approach
        # icesat2_data = icesat2_data.dropna()
        
        # icesat2_gdf_atl03_veg_ground = icesat2_gdf_atl03_veg_ground[icesat2_gdf_atl03_veg_ground[ratio_col_name] != -999]
        # icesat2_gdf_atl03_veg_ground = icesat2_gdf_atl03_veg_ground[icesat2_gdf_atl03_veg_ground[degrees_col_name] != -999]
        return icesat2_data

    elif process == False:
        if prefix == "ATL03": 
            # load the ICESat2 GeoJSON file
            icesat2_gdf_atl03_data = gpd.read_file(icesat2_dir)
            return icesat2_gdf_atl03_data
        elif prefix == "ATL08":
            # load the ICESat2 GeoJSON file
            icesat2_gdf_atl08_data = gpd.read_file(icesat2_dir)
            return icesat2_gdf_atl08_data

##################################################################
##################################################################
def merge_gedi_data(data_list, col_list, surfix_list, method):
    """
    Merges GEDI L1B, L2A, and L2B data products using either one of two methods:
    
    Method 1: Let us call this method the `merge` method. We merge the three data sets using their short numbers.
    We can use `gdf.merge()` since `shot_number` values perfectly match across data sets.
    
    Method 2: Let us call this method the `sjoin` method. Here we use spatial join instead of `merge`. 
    Assuming that `shot_number_x` doesn’t perfectly match `shot_number`, you can spatially join merged_gedi_gdf_l1b 
    with merged_gedi_gdf_l2a and merged_gedi_gdf_l2b based on geometry (lat/lon or northing/easting).

    We can use `gdf.sjoin_nearest()` if shot numbers do not match but geometries (lat/lon or northing/easting) are close.

    Note: In our specific case, both methods are equivalent. They work them same way. The results are exactly the same.
    This is beause the shot numbers do match and the geometries (lat/lon or northing/easting) are close. Hence, both methods 
    work equally the same way.

    Optional:
    We can use the steps below to check the accuracy of our spatial join. 

    1. Inspect distance values
        - Print summary statistics of distances.
        - Identify the maximum, mean, and standard deviation of distances.
        - Check for unusually high distances that may indicate incorrect joins.

    2. Set a distance threshold if necessary (max allowed distance). In our specific case this isn't really necessary, because the default threshold is already working fine.
        - Define a reasonable max_distance based on the expected spatial accuracy.
        - Re-run sjoin_nearest with this threshold.

    3. Visual Inspection (Optional)
        - Plot points before and after joining to see if they align correctly.
        
    Here, we assume: data1 = L1B, data2 = L2A, and data3 = L2B
    """
    
    data1, data2, data3 = data_list
    data1_col, data2_col, data3_col = col_list
    data1_surfix, data2_surfix, data3_surfix = surfix_list
    if method == "merge": #merge data on a common column
        # Merge data2 with data1 on data2_col and data1_col
        merged_data2_data1 = data2.merge(data1, left_on=data2_col, 
                                         right_on=data1_col, how="left", 
                                         suffixes=(data2_surfix, data1_surfix))

        # Merge the result with data3 on data3_col and data2_col
        final_merged_gdf = merged_data2_data1.merge(data3, left_on=data2_col, 
                                                    right_on=data3_col, how="left", 
                                                    suffixes=("", data3_surfix))
        return final_merged_gdf.dropna()
        
    elif method == "sjoin": # spatially join data using their spatial information, such as longitudes and latitudes
        # # projected CRS
        # projected_crs = "epsg:6340"  # Horizontal transformation: The EPSG code for NAD83/UTM zone 11N is 26911 (1983) or 6340 (2011).
        # data1 = data1.to_crs(projected_crs)
        # data2 = data2.to_crs(projected_crs)
        # data3 = data3.to_crs(projected_crs)

        # Perform nearest spatial join with data2
        spatial_merged_data2_data1 = data2.sjoin_nearest(data1, how="left", 
                                                         distance_col="distance_l1b",
                                                         lsuffix=data2_surfix, 
                                                         rsuffix=data1_surfix)

        # Perform nearest spatial join with data3
        final_spatial_gdf = spatial_merged_data2_data1.sjoin_nearest(data3, how="left", 
                                                                     distance_col="distance_l2b", 
                                                                     lsuffix="_merged", rsuffix=data3_surfix)
        
        # Set a reasonable threshold for max distance
        MAX_DISTANCE_ALLOWED = 20  # Adjust based on data distribution

        # Filter out spatial joins with unreasonable distances
        final_spatial_gdf_filtered = final_spatial_gdf[
            (final_spatial_gdf["distance_l1b"] <= MAX_DISTANCE_ALLOWED) &
            (final_spatial_gdf["distance_l2b"] <= MAX_DISTANCE_ALLOWED)]
        return final_spatial_gdf.dropna(), final_spatial_gdf_filtered.dropna()

##################################################################
##################################################################
# """
# Reprojecting to a Specific Coordinate System.

# This step may be optional in some cases, but necessary in our case since our analysis requires a particular projection. 
# Though our analysis does not strictly require any specific projection, we will reproject our data to the same coordinate reference system as the airborne lidar data. 
# We use the airborne lidar data as our ground truth data.
# """

##################################################################
##################################################################
def gdf_to_laz(x_col, y_col, z_col, output_dir, data = None, input_dir = None, load = False):
    """
    Converts a GeoDataFrame to a `.laz` file.
    Takes in a GeoDataFrame and converts it to a `laz` file.
    A LAZ file is a compressed LAS file format used for storing LiDAR data. 
    GeoPandas itself does not have built-in support for writing LAZ files. 
    However, you can convert the GeoDataFrame to a LAS format and then compress it to LAZ using the laspy and pyntcloud libraries.

    Steps to Save a GeoPandas DataFrame as a LAZ File:
        1. Convert GeoDataFrame to a Pandas DataFrame and extract the required x, y, z coordinates.
        2. Use `laspy` to create a LAS file.
        3. Convert it to LAZ using `laszip` or `pylas`.
    
    NB: You can leave the method as `None` or just pass any string to use the second method
    """
    
    if load == True:
        # Load csv file
        data_xyz = pd.read_csv(input_dir)
    else:
        data_xyz = data

    # Create a LAS file with the necessary header information
    las = laspy.LasData(header=laspy.LasHeader(point_format=3, version="1.4"))

    # Assign point cloud data to the LAS object
    las.x = data_xyz[x_col].values
    las.y = data_xyz[y_col].values
    las.z = data_xyz[z_col].values
    
    # Write to a LAS file
    las.write(output_dir)
    return laspy.read(laz_name) #open the laz file.

##################################################################
##################################################################