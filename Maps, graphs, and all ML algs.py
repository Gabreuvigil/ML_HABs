#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import ee
import geemap
import pandas as pd #read csv files
import numpy as np #numeric date(mean)
import seaborn as sns #visualization
import openpyxl as ap
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# ee.Authenticate()  # Only needed for the first time
ee.Initialize()


# In[2]:


study_lakes = pd.read_csv("C:\\Users\\gabea\\Dropbox (University of Oregon)\\nasa ccri\\python\\study_lakes.csv")

study_lakes


# In[3]:


# Load the counties shapefile using geopandas
folder_path_counties = "C:/Users/gabea/Dropbox (University of Oregon)/nasa ccri/python/State"
shapefile_name_counties = "Counties_Shoreline"
shapefile_path_counties = f"{folder_path_counties}/{shapefile_name_counties}.shp"
NYS = gpd.read_file(shapefile_path_counties)

# Reproject the counties shapefile to use WGS 84 (EPSG:4326)
NYS = NYS.to_crs(epsg=4326)

# Load the waterbodies shapefile using geopandas
folder_path_waterbodies = "C:/Users/gabea/Dropbox (University of Oregon)/nasa ccri/python/Waterbody_Inventory_List"
shapefile_name_waterbodies = "Priority_Waterbody_List___Lakes"
shapefile_path_waterbodies = f"{folder_path_waterbodies}/{shapefile_name_waterbodies}.shp"
gdf = gpd.read_file(shapefile_path_waterbodies)

# Reproject the waterbodies shapefile to use WGS 84 (EPSG:4326)
gdf = gdf.to_crs(epsg=4326)

# Assuming you have a DataFrame named 'onion_lakes' with columns 'lon' and 'lat' containing the longitude and latitude coordinates, respectively

# Convert 'onion_lakes' DataFrame to a GeoDataFrame with Point geometries
geometry = [Point(lon, lat) for lon, lat in zip(study_lakes['lon'], study_lakes['lat'])]
study_lakes_gdf = gpd.GeoDataFrame(study_lakes, crs=4326, geometry=geometry)

# Reproject the 'onion_lakes_gdf' to use WGS 84 (EPSG:4326)
study_lakes_gdf = study_lakes_gdf.to_crs(epsg=4326)

# Create the plot
fig, ax = plt.subplots()

# Plot the counties with customized border and fill colors
NYS.plot(ax=ax, edgecolor='black', facecolor='lightgrey')

# Plot the reprojected waterbodies with a different color
gdf.plot(ax=ax, color='blue')

# Plot the CSV data on the same plot using scatter plot
plt.scatter(study_lakes_gdf["geometry"].x, study_lakes_gdf["geometry"].y, color='red', marker='x', label='Stations')

# Show the plot
plt.show()


# In[4]:


unique_features_count = study_lakes['LOCATION_NAME'].nunique()

print("Number of unique features:", unique_features_count)


# In[5]:


unique_names = study_lakes['LAKE_WATERBODY_NAME'].unique()
sorted_unique_names = sorted(unique_names)

print("Unique names within the column (sorted alphabetically):")
print(sorted_unique_names)


# In[6]:


# Define the bands to select and their corresponding names
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
STD_NAMES = ['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2']

def maskS2sr(image):
    # Cloud mask
    cloudMask = image.select('QA60').bitwiseAnd(int('1111111000000000', 2)).eq(0)
    # Cloud shadow mask
    cloudShadowMask = image.select('QA60').bitwiseAnd(int('0000000111000000', 2)).eq(0)
    # Apply the cloud, cloud shadow, snow/ice, and water masks
    correctedImage = image.updateMask(cloudMask).updateMask(cloudShadowMask)
    return correctedImage

# Define the function to compute the mean reflectance values for the specified bands within the region of interest (lake)
def reflectance(img, point_geom):
    reflectance_values = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point_geom, scale=30).select(STD_NAMES)
    return img.set('DATE_SMP', img.date().format()).set('reflectance', reflectance_values)


# In[7]:


# Initialize an empty list to store the dataframes for each lake
dfs = []

# Loop through each observation in the 'cayuga' DataFrame and retrieve S2 imagery for each lake
for index, row in study_lakes.iterrows():
    LOCATION_NAME = row['LOCATION_NAME']
    lat = row['lat']
    lon = row['lon']

    # Create a point geometry for the lake
    point_geom = ee.Geometry.Point(lon, lat)

    # Retrieve Sentinel-2 imagery
    S2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filter(ee.Filter.calendarRange(5, 10, 'month')) \
        .filter(ee.Filter.calendarRange(2017, 2023, 'year')) \
        .filterBounds(point_geom) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(maskS2sr) \
        .select(S2_BANDS, STD_NAMES)
    
    # Map the reflectance function over the S2 ImageCollection for the specific lake
    map_reflectance = S2.map(lambda img: reflectance(img, point_geom))

    # Reduce the mapped image collection to get reflectance values for the specific lake
    list_reflectance = map_reflectance.reduceColumns(ee.Reducer.toList(2), ['DATE_SMP', 'reflectance']).values().get(0)

    # Convert the results to a pandas DataFrame
    df_reflectance = pd.DataFrame(list_reflectance.getInfo(), columns=['DATE_SMP', 'reflectance'])
    df_reflectance['DATE_SMP'] = pd.to_datetime(df_reflectance['DATE_SMP'])
    df_reflectance['DATE_SMP'] = df_reflectance['DATE_SMP'].dt.date
    df_reflectance['reflectance'] = df_reflectance['reflectance'].apply(lambda x: {k: v for k, v in x.items() if v is not None})

    # Unpack the 'reflectance' dictionary and create separate columns for each band
    df_reflectance = pd.concat([df_reflectance.drop('reflectance', axis=1),
                                df_reflectance['reflectance'].apply(pd.Series).astype('float64', errors='ignore')], axis=1)

    # Add a new column for the lake name
    df_reflectance['LOCATION_NAME'] = LOCATION_NAME

    # Add the DataFrame to the list
    dfs.append(df_reflectance)

# Concatenate all DataFrames into a single DataFrame
df_all_lakes = pd.concat(dfs, ignore_index=True)

# Sort the DataFrame by 'DATE_SMP' in ascending order
df_all_lakes.sort_values(by='DATE_SMP', inplace=True)

# Display
df_all_lakes


# In[280]:


# file_name = "study_lakes2.csv"  # Provide the desired file name
# df_all_lakes.to_csv(file_name, index=False)  # Set index=False to exclude row numbers in the output CSV

# print("DataFrame saved as CSV:", file_name)


# In[8]:


df_all_lakes['DATE_SMP'] = pd.to_datetime(df_all_lakes['DATE_SMP'])
study_lakes['DATE_SMP'] = pd.to_datetime(study_lakes['DATE_SMP'])


# In[9]:


df_all_lakes.dropna(inplace=True)
study_lakes.dropna(inplace=True)


# In[10]:


# Create time window
window_size = pd.Timedelta(days=7)

# Sort both dataframes by their respective date columns in ascending order
df_all_lakes.sort_values('DATE_SMP', inplace=True)
study_lakes.sort_values('DATE_SMP', inplace=True)


# In[11]:


merged_data = pd.merge_asof(study_lakes, df_all_lakes, on='DATE_SMP', by='LOCATION_NAME', tolerance=window_size)
merged_data.dropna(inplace=True)

merged_data


# In[12]:


merged_data['RedEdge1_minus_blue'] = merged_data['RedEdge1'] - merged_data['blue']
merged_data['RedEdge1_minus_red'] = merged_data['RedEdge1'] - merged_data['Red']
merged_data['G_R'] = merged_data['green'] - merged_data['Red']
merged_data['RedEdge1_minus_green'] = merged_data['RedEdge1'] - merged_data['green']
merged_data['green_minus_blue'] = merged_data['green'] - merged_data['blue']

# Display the updated DataFrame
merged_data


# In[19]:


# # Create new columns based on the NDCI algorithm
merged_data['NDCI'] = (merged_data['RedEdge1'] - merged_data['Red']) / (merged_data['RedEdge1'] + merged_data['Red'])

# # Create new columns based on the 3BDA algorithm
# merged_data['3BDA'] = ((merged_data['blue'] - merged_data['Red']) / (merged_data['blue'] + merged_data['Red'])) - merged_data['green']

# # Create new columns based on the 2BDA algorithm
# merged_data['2BDA'] = merged_data['NIR'] / merged_data['Red']

# # Create new columns based on the SABI algorithm
# merged_data['SABI'] = (merged_data['NIR'] - merged_data['Red']) / (merged_data['blue'] + merged_data['green'])

# # # Create new columns based on the NICK1 algorithm
# # merged_data['NICK'] = ((merged_data['blue'] + merged_data['green'] + merged_data['Red'] - merged_data['RedEdge4']) / (merged_data['RedEdge4'] + merged_data['SWIR1']))

# # # Create new columns based on the NICK2 algorithm
# # merged_data['NICK'] = (merged_data['green'] / (merged_data['Red'] * merged_data['blue'] + merged_data['green']))

# merged_data['NICK'] = ((merged_data['green'] * merged_data['SWIR1']) / merged_data['SWIR2']) / (merged_data['Red'] * merged_data['blue'] + merged_data['green'])

# # Display the updated DataFrame
# merged_data

# # # Create new columns based on the NDCI algorithm
# # merged_data['NDCI'] = (merged_data['NIR'] - merged_data['Red']) / (merged_data['NIR'] + merged_data['Red'])

# # # Create new columns based on the SABI algorithm
# # merged_data['SABI'] = (merged_data['NIR'] - merged_data['Red']) / (merged_data['blue'] + merged_data['green'])

# # # Create new columns based on the CIred-edge algorithm
# # merged_data['CIred_edge'] = (merged_data['RedEdge1'] - merged_data['Red']) / (merged_data['RedEdge1'] + merged_data['Red'])

# # # Create new columns based on the Red-edge NDVI algorithm
# # merged_data['Red_edge_NDVI'] = (merged_data['RedEdge1'] - merged_data['Red']) / (merged_data['RedEdge1'] + merged_data['Red'])

merged_data


# In[20]:


# Sorting in descending order (largest to smallest)
merged_data_sorted = merged_data.sort_values(by='Acres', ascending=False)

# Display the sorted DataFrame
merged_data_sorted


# In[21]:


merged_data_sorted['lake_code'] = pd.factorize(merged_data_sorted['LAKE_WATERBODY_NAME'])[0] + 1

merged_data_sorted


# In[22]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Assuming "merged_data" is your DataFrame
columns_to_scale = ['ChA', 'NIR', 'Red', 'RedEdge1', 'RedEdge2', 'blue', 'green', 'G_R', 'RedEdge1_minus_red', 'RedEdge1_minus_blue', 'RedEdge1_minus_green', 'green_minus_blue']

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit and transform the specified columns
merged_data_sorted[columns_to_scale] = scaler.fit_transform(merged_data_sorted[columns_to_scale])

# Drop NAs
merged_data_sorted.dropna(inplace=True)

merged_data_sorted


# In[24]:


# Plot scatterplot for NDCI
plt.scatter(merged_data['ChA'], merged_data['NDCI'])
plt.xlabel('Chlorophyll-a (ChA)')
plt.ylabel('NDCI')
plt.title('Chlorophyll-a vs NDCI')
plt.show()


# # All ML algorithms and statistic outputs

# In[25]:


# !pip install scikit-learn
# !pip install xgboost


# In[26]:


# Lets try a few of them here. All at once
import pandas as pd
import numpy as np
from itertools import product  # Add this import
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Different sets of varibales as inputs 

# RedEdge1 - RGB - BEST
X = merged_data_sorted[['RedEdge1_minus_blue', 'RedEdge1_minus_red', 'RedEdge1_minus_green', 'lake_code']]  # Features 
y = merged_data_sorted['ChA']  # Target variable

# # RedEdge1 - RGB - BEST
# X = merged_data[['RedEdge1_minus_blue', 'G_R', 'RedEdge1_minus_green', 'lake_code']]  # Features 
# y = merged_data['ChA']  # Target variable

# # RedEdge1 - RGB
# X = merged_data[['NICK', 'RedEdge1_minus_blue', 'RedEdge1_minus_Red', 'RedEdge1_minus_green', 'lake_code']]  # Features 
# y = merged_data['ChA']  # Target variable

# # Random lol
# X = merged_data[['RedEdge1_minus_blue', 'RedEdge1_minus_Red', 'RedEdge1_minus_green', 'green_minus_blue', 'lake_code']]  # Features 
# y = merged_data['ChA']  # Target variable

# # Individual bands
# X = merged_data[['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'RedEdge4', 'NIR', 'SWIR1', 'SWIR2']]  # Features 
# y = merged_data['ChA']  # Target variable

# # ChA algorithms 
# X = merged_data[['NDCI', '3BDA', '2BDA', 'SABI', 'lake_code']]  # Features 
# y = merged_data['ChA']  # Target variable

# X = merged_data[['NDCI', '3BDA', '2BDA', 'SABI', 'RedEdge1_minus_blue', 'RedEdge1_minus_Red', 'RedEdge1_minus_green', 'lake_code']]  # Features 
# y = merged_data['ChA']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# # Define the machine learning algorithms and their hyperparameter options
# models = [
#     ('Linear Regression', LinearRegression(), {}),
#     ('Random Forest', RandomForestRegressor(random_state=42), {'n_estimators': [50, 100, 200]}),
#     ('Gradient Boosting', GradientBoostingRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.05]}),
#     ('AdaBoost', AdaBoostRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [1.0, 0.5]}),
#     ('Neural Network', MLPRegressor(random_state=42), {'hidden_layer_sizes': [(100,), (100, 50)], 'max_iter': [500, 1000]}),
#     ('K Nearest Neighbors', KNeighborsRegressor(), {'n_neighbors': [5, 7]}),
#     ('XGBoost', xgb.XGBRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.05]}),
#     ('SVR', SVR(), {'kernel': ['rbf', 'linear', 'poly'], 'C': [1.0, 0.5]}),
# ]

# Define the machine learning algorithms and their hyperparameter options
models = [
    ('Linear Regression', LinearRegression(), {}),
    ('Random Forest', RandomForestRegressor(random_state=42), {'n_estimators': [38, 40, 42]}),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.05]}),
    ('AdaBoost', AdaBoostRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [1.0, 0.5]}),
    ('Neural Network', MLPRegressor(random_state=42), {'hidden_layer_sizes': [(100,), (100, 50)], 'max_iter': [500, 1000]}),
    ('K Nearest Neighbors', KNeighborsRegressor(), {'n_neighbors': [3, 5]}),
    ('XGBoost', xgb.XGBRegressor(random_state=42), {'n_estimators': [42], 'learning_rate': [0.1, 0.05]}),
    ('SVR', SVR(), {'kernel': ['rbf', 'linear', 'poly'], 'C': [1.0, 0.5]}),
]

# Store the results in a list of dictionaries
results = []
for name, model, hyperparameters in models:
    for param_combination in product(*hyperparameters.values()):
        params = dict(zip(hyperparameters.keys(), param_combination))
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)

        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results.append({
            'Model': name,
            'Hyperparameters': params,
            'Train Mean Squared Error': train_mse,
            'Train Root Mean Squared Error': train_rmse,
            'Test Mean Squared Error': test_mse,
            'Test Root Mean Squared Error': test_rmse,
            'Train R2 Score': train_r2,
            'Test R2 Score': test_r2
        })

# Create a DataFrame from the list of dictionaries
df_results = pd.DataFrame(results)

# Display the DataFrame
df_results


# In[27]:


# Test vs train scatterplot
import matplotlib.pyplot as plt

# Create the scatter plot
plt.scatter(df_results['Train R2 Score'], df_results['Test R2 Score'])

# Add labels and title
plt.xlabel('Train R2 Score')
plt.ylabel('Test R2 Score')
plt.title('Train R2 Score vs Test R2 Score')

# Show the plot
plt.show()


# In[29]:


# Test and train xcatterplot
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create the scatter plot
plt.scatter(X_train['RedEdge1_minus_blue'], y_train, label='RE1-Blue Train', color='darkblue', alpha=0.5)
plt.scatter(X_test['RedEdge1_minus_blue'], y_test, label='RE1-Blue Test', color='lightblue', alpha=0.5)

plt.scatter(X_train['RedEdge1_minus_green'], y_train, label='RE1-Green Train', color='darkgreen', alpha=0.5)
plt.scatter(X_test['RedEdge1_minus_green'], y_test, label='RE1-Green Test', color='lightgreen', alpha=0.5)

plt.scatter(X_train['RedEdge1_minus_Red'], y_train, label='RE1-Red Train', color='darkred', alpha=0.5)
plt.scatter(X_test['RedEdge1_minus_Red'], y_test, label='RE1-Red Test', color='pink', alpha=0.5)

# Fit linear regression models and plot the trend lines
for feature, color in [('RedEdge1_minus_blue', 'blue'), ('RedEdge1_minus_green', 'green'), ('RedEdge1_minus_Red', 'red')]:
    lr = LinearRegression()
    lr.fit(X_train[feature].values.reshape(-1, 1), y_train)
    y_pred_train = lr.predict(X_train[feature].values.reshape(-1, 1))
    y_pred_test = lr.predict(X_test[feature].values.reshape(-1, 1))
    plt.plot(X_train[feature], y_pred_train, color=color, linestyle='-', linewidth=2)
    plt.plot(X_test[feature], y_pred_test, color=color, linestyle='--', linewidth=2)

# Add labels and title
plt.xlabel('Band Ratio Values')
plt.ylabel('Chl-a')
plt.title('Normalized Chl-a vs Band Ratios')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[30]:


# Time series of insitu ChA observations  
plt.scatter(merged_data['DATE_SMP'], merged_data['ChA'], alpha=0.5, color='green')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Chl-a')
plt.title('Time Series of Normalized Chl-a from Stations')

# Rotate the date labels on the x-axis for better readability (optional)
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[31]:


# Map of sites 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Load the counties shapefile using geopandas
folder_path_counties = "C:/Users/gabea/Dropbox (University of Oregon)/nasa ccri/python/State"
shapefile_name_counties = "Counties_Shoreline"
shapefile_path_counties = f"{folder_path_counties}/{shapefile_name_counties}.shp"
NYS = gpd.read_file(shapefile_path_counties)

# Reproject the counties shapefile to use WGS 84 (EPSG:4326)
NYS = NYS.to_crs(epsg=4326)

# Load the waterbodies shapefile using geopandas
folder_path_waterbodies = "C:/Users/gabea/Dropbox (University of Oregon)/nasa ccri/python/Waterbody_Inventory_List"
shapefile_name_waterbodies = "Priority_Waterbody_List___Lakes"
shapefile_path_waterbodies = f"{folder_path_waterbodies}/{shapefile_name_waterbodies}.shp"
gdf = gpd.read_file(shapefile_path_waterbodies)

# Reproject the waterbodies shapefile to use WGS 84 (EPSG:4326)
gdf = gdf.to_crs(epsg=4326)
# Convert 'study_lakes' DataFrame to a GeoDataFrame with Point geometries
geometry = [Point(lon, lat) for lon, lat in zip(study_lakes['lon'], study_lakes['lat'])]
study_lakes_gdf = gpd.GeoDataFrame(study_lakes, crs='EPSG:4326', geometry=geometry)

# Reproject the 'study_lakes_gdf' to use WGS 84 (EPSG:4326)
study_lakes_gdf = study_lakes_gdf.to_crs(epsg=4326)

# Create the plot
fig, ax = plt.subplots()

# Plot the counties with customized border and fill colors
NYS.plot(ax=ax, edgecolor='black', facecolor='lightgrey', label='Counties')

# Plot the reprojected waterbodies with a different color
gdf.plot(ax=ax, color='blue', label='Waterbodies')

# Plot the study lakes data on the same plot using scatter plot
plt.scatter(study_lakes_gdf["geometry"].x, study_lakes_gdf["geometry"].y, color='green', marker='x', label='Lakes Stations')

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()


# In[32]:


# Best preforming stats from earlier, put into a neat csv file 
R_2 = pd.read_csv("R_2.csv")
R_2


# In[33]:


# Plotting statistical preformaces of ML  
import matplotlib.pyplot as plt

X = R_2['Model']
Ygirls = R_2['Test R^2 Score 1 Day']
Zboys = R_2['Test R^2 Score 3 Days']
Qtheythem = R_2['Test R^2 Score 7 Days']

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, Ygirls, 0.2, label = 'Test R² Score 1 Day', color='royalblue')
plt.bar(X_axis, Zboys, 0.2, label = 'Test R² Score 3 Days', color='mediumseagreen')
plt.bar(X_axis + 0.2, Qtheythem, 0.2, label = 'Test R² Score 7 Days', color='mediumorchid')

plt.xticks(X_axis, X)
plt.xlabel("Models")
plt.ylabel("Test R²")
plt.title("Test R² per Model")
plt.legend()
plt.show()


# In[34]:


# Colors!!!

from __future__ import division

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]

n = len(sorted_names)
ncols = 4
nrows = n // ncols + 1

fig, ax = plt.subplots(figsize=(8, 5))

# Get height and width
X, Y = fig.get_dpi() * fig.get_size_inches()
h = Y / (nrows + 1)
w = X / ncols

for i, name in enumerate(sorted_names):
    col = i % ncols
    row = i // ncols
    y = Y - (row * h) - h

    xi_line = w * (col + 0.05)
    xf_line = w * (col + 0.25)
    xi_text = w * (col + 0.3)

    ax.text(xi_text, y, name, fontsize=(h * 0.8),
            horizontalalignment='left',
            verticalalignment='center')

    ax.hlines(y + h * 0.1, xi_line, xf_line,
              color=colors[name], linewidth=(h * 0.6))

ax.set_xlim(0, X)
ax.set_ylim(0, Y)
ax.set_axis_off()

fig.subplots_adjust(left=0, right=1,
                    top=1, bottom=0,
                    hspace=0, wspace=0)
plt.show()

