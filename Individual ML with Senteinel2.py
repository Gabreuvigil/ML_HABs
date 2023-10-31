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


#Load in in-situ lake ChA data
onion_lakes = pd.read_csv("C:\\Users\\gabea\\Dropbox (University of Oregon)\\nasa ccri\\python\\onion_lakes.csv")

onion_lakes


# In[5]:


#Load in more in-situ lake ChA data
new_york_lakes = pd.read_csv("C:\\Users\\gabea\\Dropbox (University of Oregon)\\nasa ccri\\python\\new_york_lakes.csv")

new_york_lakes


# In[4]:


# Sort lakes by size (area)
lakes = pd.read_csv("C:\\Users\\gabea\\Dropbox (University of Oregon)\\nasa ccri\\python\\lakes.csv")
lakes['Lake Name'] = lakes['Lake Name'].str.upper()
lakes


# In[6]:


# Merge in-situ data with lake names
all_lakes = pd.merge(new_york_lakes, lakes, left_on='LAKE_WATERBODY_NAME', right_on='Lake Name', how='left')
all_lakes.dropna(inplace=True)
all_lakes


# In[7]:


# all_lakes.to_csv('all_lakes.csv', index=False)


# In[6]:


# Load the NY state counties shapefile using geopandas
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

# Convert 'onion_lakes' DataFrame to a GeoDataFrame with Point geometries
geometry = [Point(lon, lat) for lon, lat in zip(merged_df['lon'], merged_df['lat'])]
merged_df_gdf = gpd.GeoDataFrame(merged_df, crs=4326, geometry=geometry)

# Reproject the 'onion_lakes_gdf' to use WGS 84 (EPSG:4326)
merged_df_gdf = merged_df_gdf.to_crs(epsg=4326)

# Create the plot
fig, ax = plt.subplots()

# Plot the counties with customized border and fill colors
NYS.plot(ax=ax, edgecolor='black', facecolor='lightgrey')

# Plot the reprojected waterbodies with a different color
gdf.plot(ax=ax, color='blue')

# Plot the CSV data on the same plot using scatter plot
plt.scatter(merged_df_gdf["geometry"].x, merged_df_gdf["geometry"].y, color='red', marker='o', label='Stations')

# Show the plot
plt.show()


# In[7]:


merged_df['Square_M'] = pd.to_numeric(merged_df['Square_M'], errors='coerce')

# Create a new dataframe with rows where 'Acres' is greater than or equal to 1000
big_lakes = merged_df[merged_df['Square_M'] >= 4067094.3
]

# Display the filtered dataframe
big_lakes


# In[173]:


# Look at number of lakes
unique_features_count = onion_lakes['LAKE_WATERBODY_NAME'].nunique()

print("Number of unique features:", unique_features_count)


# In[175]:


# Print names of lakes 
unique_names = big_lakes['LAKE_WATERBODY_NAME'].unique()

print("Unique names within the column:")
print(unique_names)


# In[8]:


# Same thing as earlier, but with only the big lakes 
# Load the counties shapefile
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

# Convert 'big_lakes' DataFrame to a GeoDataFrame with Point geometries
geometry = [Point(lon, lat) for lon, lat in zip(big_lakes['lon'], big_lakes['lat'])]
filt_gdf = gpd.GeoDataFrame(big_lakes, crs=4326, geometry=geometry)

# Reproject the 'filt_gdf' to use WGS 84 (EPSG:4326)
filt_gdf = filt_gdf.to_crs(epsg=4326)

# Create the plot
fig, ax = plt.subplots()

# Plot the counties with customized border and fill colors
NYS.plot(ax=ax, edgecolor='black', facecolor='lightgrey')

# Plot the reprojected waterbodies with a different color
gdf.plot(ax=ax, color='blue')

# Plot the CSV data on the same plot using scatter plot
plt.scatter(filt_gdf["geometry"].x, filt_gdf["geometry"].y, color='red', marker='o', label='Stations')

# Show the plot of BIG lakes only 
plt.show()


# In[120]:


# load in the Lake Champlain data 
champ = pd.read_csv("C:\\Users\\gabea\\Dropbox (University of Oregon)\\nasa ccri\\python\\champ.csv")

champ


# In[121]:


# Define the S2 bands to select and their corresponding names
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
STD_NAMES = ['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2']

# cloud mask... need to re-do
def maskS2sr(image):
    # Cloud mask
    cloudMask = image.select('QA60').bitwiseAnd(int('1111111000000000', 2)).eq(0)
    # Cloud shadow mask
    cloudShadowMask = image.select('QA60').bitwiseAnd(int('0000000111000000', 2)).eq(0)
    # Apply the cloud, cloud shadow, snow/ice, and water masks
    correctedImage = image.updateMask(cloudMask).updateMask(cloudShadowMask)
    return correctedImage

# Define the function to compute the mean reflectance values for the specified bands within the region of interest (lake)
def reflectance(img, lake):
    reflectance_values = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=lake, scale=30).select(STD_NAMES)
    return img.set('DATE_SMP', img.date().format()).set('reflectance', reflectance_values)


# In[122]:


# Initialize an empty list to store the dataframes for each lake
dfs = []

# Loop through each observation in the in-situ DataFrame and retrieve S2 imagery for each lake
for index, row in champ.iterrows():
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


# In[8]:


# To datetime 
df_all_lakes['DATE_SMP'] = pd.to_datetime(df_all_lakes['DATE_SMP'])
champ['DATE_SMP'] = pd.to_datetime(champ['DATE_SMP'])


# In[124]:


# Drop NA values 
df_all_lakes.dropna(inplace=True)
champ.dropna(inplace=True)


# In[158]:


# Create the 3-day, 5-day, or 7-day time window between in-situ and satellite image 
window_size = pd.Timedelta(days=7)

# Sort both dataframes by their respective date columns in ascending order
df_all_lakes.sort_values('DATE_SMP', inplace=True)
champ.sort_values('DATE_SMP', inplace=True)


# In[159]:


# Merge them 
merged_data = pd.merge_asof(champ, df_all_lakes, on='DATE_SMP', by='LOCATION_NAME', tolerance=window_size)

merged_data


# In[160]:


# Drop NA valuessss
merged_data.dropna(inplace=True)

merged_data


# In[161]:


# Plot the scatter plot
plt.scatter(merged_data['DATE_SMP'], merged_data['ChA'])

# Add labels and title
plt.xlabel('Date')
plt.ylabel('ChA')
plt.title('Scatter Plot: ChA vs. Date')

# Rotate the date labels on the x-axis for better readability (optional)
plt.xticks(rotation=45)

# Show the plot
plt.show()


# # Random Forest

# In[162]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Preparing the training data
X_train = merged_data[['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2']]  # Features 
y_train = merged_data['ChA']  # Target variable

# Split the dataset into training and testing sets (70% training and 30% testing sets)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Standardize the input features (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict ChA values for both training and testing data
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Calculate Mean Squared Error (MSE) for training and testing data
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print("Training MSE:", mse_train)
print("Testing MSE:", mse_test)

# Create DataFrames for the training and testing data with all bands and the predicted values
data_train = pd.DataFrame(X_train, columns=['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2'])
data_train['original_ChA'] = y_train
data_train['predicted_ChA'] = y_pred_train

data_test = pd.DataFrame(X_test, columns=['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2'])
data_test['original_ChA'] = y_test
data_test['predicted_ChA'] = y_pred_test

# Feature Importances (optional)
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("Feature Importances:")
print(feature_importances)


# In[163]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Preparing the training data
X_train = merged_data[['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2']]  # Features 
y_train = merged_data['ChA']  # Target variable

# Split the dataset into training and testing sets (70% training and 30% testing sets)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Standardize the input features (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict ChA values for both training and testing data
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Calculate Mean Squared Error (MSE) for training and testing data
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

# Calculate R-squared (R2) for training and testing data
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("Training MSE:", mse_train)
print("Testing MSE:", mse_test)
print("Training R-squared:", r2_train)
print("Testing R-squared:", r2_test)

# Create DataFrames for the training and testing data with all bands and the predicted values
data_train = pd.DataFrame(X_train, columns=['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2'])
data_train['original_ChA'] = y_train
data_train['predicted_ChA'] = y_pred_train

data_test = pd.DataFrame(X_test, columns=['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2'])
data_test['original_ChA'] = y_test
data_test['predicted_ChA'] = y_pred_test

# Feature Importances (optional)
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("Feature Importances:")
print(feature_importances)


# # KNN (K nearest neighbors)

# In[164]:


# !pip install scikit-learn
import sklearn
from sklearn import neighbors
from sklearn import impute
from scipy import interpolate
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Assuming "merged_data" is your DataFrame
X = merged_data[['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2']]  # Features 
y = merged_data['ChA']  # Target variable

# Split the dataset into training and testing sets (80% training and 20% testing sets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features (use RobustScaler to handle outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN Regressor model
RegModel = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='kd_tree')
KNN = RegModel.fit(X_train_scaled, y_train)

# Predict ChA values for both training and testing data
y_pred_train = KNN.predict(X_train_scaled)
y_pred_test = KNN.predict(X_test_scaled)

# Measuring Goodness of fit in Training data and Testing data
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print('R2 Value (train):', r2_train)
print('R2 Value (test):', r2_test)

# Calculate Mean Squared Error (MSE) for testing data
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)

print("Mean Squared Error (Testing):", mse_test)
print("Root Mean Squared Error (Testing):", rmse_test)

print("Original ChA values (Testing):")
print(y_test)
print("Predicted ChA values (Testing):")
print(y_pred_test)


# # Ada Boosting

# In[165]:


import pandas as pd
import numpy as np  # Don't forget to import NumPy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score  # Import r2_score from sklearn.metrics

# Assuming "merged_data" is your DataFrame
X = merged_data[['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2']]  # Features 
y = merged_data['ChA']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_regressor = DecisionTreeRegressor(max_depth=1)
adaboost_regressor = AdaBoostRegressor(estimator=base_regressor, n_estimators=50, random_state=42)

adaboost_regressor.fit(X_train, y_train)

y_pred = adaboost_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

rmse = np.sqrt(mse)
print("Root Mean Squared Error (Testing):", rmse)

# Calculate R-squared (R2) for training and testing data
r2_train = r2_score(y_train, adaboost_regressor.predict(X_train))
r2_test = r2_score(y_test, y_pred)

print('R2 Value (train):', r2_train)
print('R2 Value (test):', r2_test)


# # Gradient Boosting

# In[166]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Assuming "merged_data" is your DataFrame
X = merged_data[['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2']]  # Features 
y = merged_data['ChA']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting Regressor model
gradient_boosting_regressor = GradientBoostingRegressor(n_estimators=50, random_state=42)

gradient_boosting_regressor.fit(X_train, y_train)

y_pred = gradient_boosting_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

rmse = np.sqrt(mse)
print("Root Mean Squared Error (Testing):", rmse)

# Calculate R-squared (R2) for training and testing data
r2_train = r2_score(y_train, gradient_boosting_regressor.predict(X_train))
r2_test = r2_score(y_test, y_pred)

print('R2 Value (train):', r2_train)
print('R2 Value (test):', r2_test)


# # Neural Network

# In[167]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Assuming "merged_data" is your DataFrame
X = merged_data[['blue', 'green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2']]  # Features 
y = merged_data['ChA']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Neural Network (MLP Regressor) model
neural_network_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

neural_network_regressor.fit(X_train, y_train)

y_pred = neural_network_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

rmse = np.sqrt(mse)
print("Root Mean Squared Error (Testing):", rmse)

# Calculate R-squared (R2) for training and testing data
r2_train = r2_score(y_train, neural_network_regressor.predict(X_train))
r2_test = r2_score(y_test, y_pred)

print('R2 Value (train):', r2_train)
print('R2 Value (test):', r2_test)


# In[168]:


# !pip install scikit-learn
# !pip install xgboost
# !pip install lightgbm


# In[169]:


# Lets try a few of them here. All at once
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Assuming "X" contains the features and "y" contains the target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the machine learning algorithms and their names
# Here, we can also adjust some of the hyperparameters 
models = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=42)),
    ('AdaBoost', AdaBoostRegressor(random_state=42)),
    ('Neural Network', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)),
    ('K Nearest Neighbors', KNeighborsRegressor()),
    ('XGBoost', xgb.XGBRegressor(random_state=42)),
    ('SVR', SVR())
]

# Store the results in a list of dictionaries
results = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'R2 Score': r2
    })

# Create a DataFrame from the list of dictionaries
df_results = pd.DataFrame(results)

# Display the DataFrame
print(df_results)

