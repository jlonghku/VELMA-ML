import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os,copy
import pandas as pd
from glob import glob
import xml.etree.ElementTree as ET
import subprocess,re
import concurrent.futures
import random
import uuid
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Dynamically parse the .asc file header and return raster data (NumPy array)
def load_asc_file(file_path):
    """
    Load an .asc file and return its header and data as a tuple.
    
    :param file_path: Path to the .asc file
    :return: A tuple containing the header (dictionary) and data (numpy array)
    """
    header = {}
    data_start_line = 0

    # Read the header information
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) == 2:
                key, value = parts[0].lower(), parts[1]
                try:
                    header[key] = int(value) if key in ['ncols', 'nrows'] else float(value)
                except ValueError:
                    data_start_line = i
                    break
            else:
                data_start_line = i
                break

        # Get the data dimensions
        ncols = int(header.get('ncols', 0))
        nrows = int(header.get('nrows', 0))

        # Determine data type based on content
        f.seek(0)
        for _ in range(data_start_line):
            next(f)
        first_data_line = next(f).strip().split()
        
        # Check for decimal points to infer data type
        dtype = np.float32 if any('.' in val for val in first_data_line) else np.int32

        # Read data with inferred data type
        f.seek(0)
        for _ in range(data_start_line):
            next(f)
        data = np.loadtxt(f, dtype=dtype)

        # If data is float type, replace nodata_value with NaN
        if dtype == np.float32:
            # Convert to float array (to handle missing values in float type)
            nodata_value = float(header.get("nodata_value", np.nan))
            if not np.isnan(nodata_value):
                data[data == nodata_value] = np.nan

        return data, header


def save_asc_file(data, header, file_path):
    """
    Save a numpy array as an .asc file with the specified header.

    :param data: Numpy array to be saved
    :param header: Dictionary containing header information for the .asc file
    :param file_path: Path to save the .asc file
    """
    # Get nodata value, defaulting to -9999 if not specified
    nodata_value = int(header.get("nodata_value", -9999))
    
    with open(file_path, 'w') as f:
        # Write header information
        for key, value in header.items():
            f.write(f"{key} {value}\n")
        
        # Write data rows, preserving integer format where applicable
        for row in data:
            row_str = ' '.join(
                str(val) if not np.isnan(val) else str(nodata_value)
                for val in row
            )
            f.write(row_str + '\n')
    
    print(f"Saved .asc file to {file_path}")


# Load image data from multiple paths, each containing multiple subfolders
def load_images_from_samples(sample_paths):
    """
    Load images from sample paths, with optional augmentation to reach num_samples.
    
    :param sample_paths: List of dictionaries with paths to the image channels
    """
    all_images = []

    # Load each image with three channels from the sample paths
    for sample_path in sample_paths:
        # Read the .asc files for three channels
        channel1 = load_asc_file(sample_path['dem_path'])[0]
        channel2 = load_asc_file(sample_path['cover_species_path'])[0]
        channel3 = load_asc_file(sample_path['soil_parameters_path'])[0]

        # Ensure all channels have the same shape
        assert channel1.shape == channel2.shape == channel3.shape, "Channel sizes do not match"

        # Combine the three channels into a sample with shape (3, nrows, ncols)
        sample_img = np.stack([channel1, channel2, channel3], axis=0)
        all_images.append(sample_img)

    # Stack all images into a single array
    all_images = np.stack(all_images, axis=0)
    all_images_tensor = torch.tensor(all_images, dtype=torch.float32).clone().detach()

    return all_images_tensor

# Load precipitation and temperature time series data from the sample path
def load_time_series_from_path(sample):
    """
    Load precipitation and temperature time series data from the given sample_path.
    Selects model based on the weather_model value in the sample_path dictionary.
    If the model is DefaultWeatherModel, it reads weather_Driver_path if available,
    otherwise reads rain_Driver_path and airTemperature_Driver_path.
    If the model is MultipleWeightedLocationWeatherModel, it reads multiple weighted location data.
    For weather_Locations_paths, it reads individual point CSV files and calculates the average.
    
    :param sample_path: Dictionary containing model paths and selection info
    :return: Average precipitation time series (precip), Average temperature time series (temp)
    """
    weather_model = sample['weather_model']
    if weather_model == "DefaultWeatherModel":
        if 'weather_Driver_path' in sample and sample['weather_Driver_path']:
            weather_data = pd.read_csv(sample['weather_Driver_path'])
            if len(weather_data.columns) == 4:
                weather_data.columns = ['Year', 'Jday', 'Precipitation', 'Temperature']
                weather_data.drop(columns=['Year', 'Jday'], inplace=True)
            elif len(weather_data.columns) == 2:
                weather_data.columns = ['Precipitation', 'Temperature']
            precip = weather_data['Precipitation']
            temp = weather_data['Temperature'] 
        else:
            rain_data = pd.read_csv(sample['rain_driver_path'])
            if len(rain_data.columns) == 3:
                rain_data.columns = ['Year', 'Jday', 'Precipitation']
                rain_data.drop(columns=['Year', 'Jday'], inplace=True)
            elif len(rain_data.columns) == 1:
                rain_data.columns = ['Precipitation']
            temp_data = pd.read_csv(sample['temp_driver_path'])
            if len(temp_data.columns) == 3:
                temp_data.columns = ['Year', 'Jday', 'Temperature']
                temp_data.drop(columns=['Year', 'Jday'], inplace=True)
            elif len(temp_data.columns) == 1:
                temp_data.columns = ['Temperature']
            precip = rain_data['Precipitation']
            temp = temp_data['Temperature']
    elif weather_model == "MultipleWeightedLocationWeatherModel":
        if 'weather_locations_path' in sample:
            locations_data = pd.read_csv(sample['weather_locations_path'], header=None, names=['x-coordinate', 'y-coordinate', 'uniqueName', 'driverDataFileName'])
            precip_list = []
            temp_list = []
            for _, row in locations_data.iterrows():
                driver_data_path = row['driverDataFileName']
                point_data = pd.read_csv(driver_data_path)
                if len(point_data.columns) == 4:
                    point_data.columns = ['Year', 'Jday', 'Precipitation', 'Temperature']
                    point_data.drop(columns=['Year', 'Jday'], inplace=True)
                elif len(point_data.columns) == 2:
                    point_data.columns = ['Precipitation', 'Temperature']
                precip_list.append(point_data['Precipitation'])
                temp_list.append(point_data['Temperature'])
            precip = pd.concat(precip_list, axis=1).mean(axis=1)
            temp = pd.concat(temp_list, axis=1).mean(axis=1) 
        else:
            raise ValueError("Missing path for weather locations data.")
    else:
        raise ValueError(f"Unsupported weather model: {weather_model}")
    
    return precip, temp


# Load time series data from multiple sample paths
def load_time_series_from_paths(samples, observed_end_year=None):
    """
    Load time series data from sample paths and filter them based on the specified year range.
    
    :param samples: List of sample dictionaries, each with paths and year filters
    :param observed_end_year: Optional; if provided, limits the observed_time_series to data up to this year
    :return: PyTorch tensor of time series data (precipitation and temperature) and optionally observed_time_series
    """
    def calculate_day_index(target_year, reference_year):
        """
        Calculate the day index for a given target year relative to the reference year, considering leap years.
        
        :param target_year: The year for which the day index is being calculated.
        :param reference_year: The starting reference year.
        :return: The calculated day index.
        """
        days = 0
        for year in range(reference_year, target_year):
            if is_leap_year(year):
                days += 366
            else:
                days += 365
        return days

    all_precips = []
    all_temps = []
    observed_precips = []
    observed_temps = []
    
    for sample in samples:
        precip, temp = load_time_series_from_path(sample)

        # Extract start and end years for the subset range
        start_year, end_year = sample['start_year'], sample['end_year']
        forcing_start_year = sample['forcing_start']
        
        # Calculate start and end indices for the given year range
        start_index = calculate_day_index(start_year, forcing_start_year)
        end_index = calculate_day_index(end_year + 1, forcing_start_year)  # +1 to include end year

        # Ensure end_index does not exceed actual data length
        end_index = min(end_index, len(precip), len(temp))

        # Subset data to the calculated indices
        all_precips.append(precip[start_index:end_index])
        all_temps.append(temp[start_index:end_index])

        # If observed_end_year is provided, calculate indices up to that year from the original data
        if observed_end_year is not None:
            observed_end_index = calculate_day_index(observed_end_year + 1, forcing_start_year)
            observed_end_index = min(observed_end_index, len(precip), len(temp))  # Ensure within original data length
            observed_precips.append(precip[start_index:observed_end_index])
            observed_temps.append(temp[start_index:observed_end_index])
    
    # Stack precipitation and temperature arrays and combine them
    all_precips = np.stack(all_precips, axis=0)
    all_temps = np.stack(all_temps, axis=0)
    time_series = np.stack([all_precips, all_temps], axis=-1)
    
    # Generate observed_time_series if observed_end_year is provided
    if observed_end_year is not None:
        observed_precips = np.stack(observed_precips, axis=0)
        observed_temps = np.stack(observed_temps, axis=0)
        observed_time_series = np.stack([observed_precips, observed_temps], axis=-1)
        return time_series, observed_time_series
    
    return time_series, time_series


def load_params_from_paths(xml_file, params_range, verbose=False):
    """
    Load parameters from specified XML files in the provided directory path, extracting numerical 
    and boolean parameters outside display tags and returning the XML and Velma parameters.
    
    :param output_paths: Path to the directory containing XML files.
    :param params_range: Dictionary with paths for Velma parameters to extract.
    :param verbose: Whether to print debugging information, default is False.
    :return: Tuple of arrays - (xml_params, velma_params)
    """   
    tree = ET.parse(xml_file)
    root = tree.getroot()
    xml_params = []    
    velma_params = []

    # Traverse the XML tree to find Velma parameters at specified paths
    for path in params_range.keys():
        elem = root.find(path)
        if elem is not None and elem.text:
            try:
                value = float(elem.text.strip())
                velma_params.append(value)
                if verbose:
                    print(f"Found velma_param: {path} = {value}")
            except ValueError:
                if verbose:
                    print(f"Warning: Could not convert value for {path}")

    # Recursive extraction function for other parameters
    def recursive_extract(elem, current_path=''):
        full_path = current_path + '/' + elem.tag
        if 'display' in full_path.lower():
            return

        if any(keyword in full_path.lower() for keyword in ['soil', 'cover', 'calibration']):
            if elem.text is not None:
                value = elem.text.strip()
                try:
                    xml_params.append(float(value))
                    if verbose:
                        print(f"Found xml_param: {full_path} = {value}")
                except ValueError:
                    if value.lower() == 'true':
                        xml_params.append(1.0)
                        if verbose:
                            print(f"Found xml_param: {full_path} = True (1.0)")
                    elif value.lower() == 'false':
                        xml_params.append(0.0)
                        if verbose:
                            print(f"Found xml_param: {full_path} = False (0.0)")

        for child in elem:
            recursive_extract(child, full_path)

    # Start recursive extraction for current XML file
    recursive_extract(root)

    return np.array(xml_params, dtype=np.float32), np.array(velma_params, dtype=np.float32)

# Load real output data from DailyResults.csv files
def load_outputs_from_paths(daily_results_paths, required_columns, year_range=None, plot_outputs=False):
    """
    Read the specified columns from the DailyResults.csv file in each sample path,
    and generate a 3D tensor of real outputs (real_outputs).
    This function will recursively search for DailyResults.csv files in the provided paths and filter
    data by the specified year range if provided.

    Parameters:
    - daily_results_paths: List of paths to DailyResults.csv files
    - required_columns: List of column names to extract from each file
    - year_range: Optional list [start_year, end_year] to filter data by year
    - plot_outputs: Whether to plot the output data, default is False

    Returns:
    - real_outputs: 3D tensor with shape [num_samples, num_timesteps, num_output_columns]
    """
    all_outputs = []   

    # Iterate through all provided DailyResults.csv paths
    for daily_results_path in daily_results_paths:
        # Read the CSV file
        data = pd.read_csv(daily_results_path)
        
        # Check if required columns are present
        if not all(col in data.columns for col in required_columns):
            print(f"Warning: One or more required columns are missing in {daily_results_path}, skipping.")
            continue
        
        # Filter data by the specified year range if provided
        if year_range is not None:
            start_year, end_year = year_range
            if 'Year' not in data.columns:
                print(f"Warning: 'Year' column not found in {daily_results_path}, skipping.")
                continue
            data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]

        # Extract the required columns and convert them to a numpy array, preserving the time steps
        selected_data = data[required_columns].astype(np.float32).values
        all_outputs.append(selected_data)
    
    if not all_outputs:
        raise ValueError("No valid data found from the provided sample paths.")
    
    # Stack all outputs into a 3D array, with shape [num_samples, num_timesteps, num_output_columns]
    real_outputs = np.stack(all_outputs, axis=0)

    # Optionally, plot the output data if plot_outputs is set to True
    if plot_outputs:
        plt.figure(figsize=(10, 6))
        for i in range(real_outputs.shape[0]):
            plt.plot(real_outputs[i, :, 0], label=f'Sample {i} Runoff')  # Only plot the first required column as an example
        plt.title('Real Outputs Over Samples')
        plt.xlabel('Time Step')
        plt.ylabel('Real Outputs')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return real_outputs


def load_observed_data(file_paths, required_columns):
    """
    Reads data from multiple files and combines them into a single DataFrame.
    
    Parameters:
    - file_paths: List of file paths to be read.
    - required_columns: List of column names for the combined DataFrame.

    Returns:
    - DataFrame with each file's data as a separate column.
    """
    # Check if the number of file paths matches the number of required columns
    if len(file_paths) != len(required_columns):
        raise ValueError("Number of file paths and required columns must be the same.")
    
    # Initialize an empty DataFrame to store combined data
    combined_data = pd.DataFrame()

    # Loop through each file and read the data
    for i, file in enumerate(file_paths):
        # Read the current file, assuming it has only one column
        data = pd.read_csv(file, header=0)  # Adjust header if necessary
        
        # Get the first (and only) column of data
        single_column = data.iloc[:, 0]
        
        # Assign a column name from required_columns list
        combined_data[required_columns[i]] = single_column

    return combined_data.values


def augment_and_save_images(sample, image_range, image_dir, image_augment_nums, idx):
    """
    Augments input images by applying random transformations and saves the modified versions.

    :param sample: Dictionary containing paths to input files.
    :param image_range: List of tuples specifying ranges for augmenting each component (DEM, cover, soil).
    :param image_dir: Directory where augmented images will be saved.
    :param image_augment_nums: Number of augmented images to generate.
    :param idx: Index used to label saved files.
    :return: List of dictionaries with paths to each augmented image component.
    """
    augmented_images = []
    if image_augment_nums <= 1:
        return [{}]
    
    input_path = sample['input_path']
    
    # Generate augmentations until the specified number is reached
    while len(augmented_images) < image_augment_nums:
        # Load DEM, cover, and soil data along with headers
        dem, header = load_asc_file(sample['dem_path'])
        cover_species = load_asc_file(sample['cover_species_path'])[0]
        soil_params = load_asc_file(sample['soil_parameters_path'])[0]
        
        # Handle no-data values in DEM
        nodata_value = header.get('nodata_value', -9999)
        nodata_value = np.array(nodata_value, dtype=dem.dtype)

        # Process DEM data by applying a random factor within the specified range
        dem_coef_range = image_range[0]
        random_factors = np.random.uniform(dem_coef_range[0], dem_coef_range[1])
        noisy_dem = np.where(dem == nodata_value, nodata_value, dem * random_factors)
        
        # Process cover and soil data by applying random perturbations within the specified range
        cover_perturbation_factor, cover_range_factor = image_range[1]
        soil_perturbation_factor, soil_range_factor = image_range[2]

        noisy_cover_species = randomize_ids_with_constraints(cover_species, cover_perturbation_factor, cover_range_factor)
        noisy_soil_params = randomize_ids_with_constraints(soil_params, soil_perturbation_factor, soil_range_factor)
        
        # Load headers separately for each component
        dem_header = load_asc_file(sample['dem_path'])[1]
        cover_species_header = load_asc_file(sample['cover_species_path'])[1]
        soil_params_header = load_asc_file(sample['soil_parameters_path'])[1]

        # Define file paths for saving augmented data
        dem_path = os.path.join(image_dir, f"augmented_{idx}_dem_{len(augmented_images)}.asc")
        cover_species_path = os.path.join(image_dir, f"augmented_{idx}_cover_species_{len(augmented_images)}.asc")
        soil_params_path = os.path.join(image_dir, f"augmented_{idx}_soil_params_{len(augmented_images)}.asc")

        # Save the augmented DEM, cover species, and soil parameters as .asc files
        save_asc_file(noisy_dem, dem_header, dem_path)
        save_asc_file(noisy_cover_species, cover_species_header, cover_species_path)
        save_asc_file(noisy_soil_params, soil_params_header, soil_params_path)
        
        # Append paths of augmented files to the results list
        augmented_images.append({
            'dem_path': dem_path,
            'cover_species_path': cover_species_path,
            'soil_parameters_path': soil_params_path
        })
        
        # Stop if the desired number of augmented images is reached
        if len(augmented_images) >= image_augment_nums:
            break
    
    return augmented_images


def randomize_ids_with_constraints(data, perturbation_factor, range_factor):
    unique_ids, counts = np.unique(data, return_counts=True)
    randomized_data = np.copy(data)
    
    # Obtain 2D coordinate matrix
    shape_x, shape_y = data.shape
    xv, yv = np.meshgrid(np.arange(shape_x), np.arange(shape_y), indexing='ij')
    coordinates = np.column_stack((xv.ravel(), yv.ravel()))
    
    # Create KDTree for efficient range search
    tree = cKDTree(coordinates)
    
    for unique_id, count in zip(unique_ids, counts):
        # Generate mask and indices for the current ID
        mask = (data == unique_id)
        id_positions = np.flatnonzero(mask)
        
        # Calculate target count with perturbation
        target_count = min(len(id_positions), int(count * (1 + np.random.uniform(-perturbation_factor, perturbation_factor))))
        
        # Obtain coordinates within range using KDTree
        center_coords = coordinates[id_positions]
        neighbors_within_range = tree.query_ball_point(center_coords, range_factor)
        
        # Filter valid positions based on range
        valid_positions = np.unique([idx for indices in neighbors_within_range for idx in indices])
        
        # Select new positions based on target count
        selected_positions = valid_positions if len(valid_positions) < target_count else np.random.choice(valid_positions, size=target_count, replace=False)
        
        # Convert selected positions to 2D indices
        selected_coords = coordinates[selected_positions]
        row_indices, col_indices = selected_coords[:, 0], selected_coords[:, 1]
        
        # Update randomized data with perturbed ID locations
        randomized_data[row_indices, col_indices] = unique_id
    
    return randomized_data

def augment_and_save_time_series(sample, ts_range, output_dir, num_augmentations, idx):
    """
    Apply random scaling and offset to the precipitation and temperature time series data, save each 
    augmented version to a new file, and return the new file paths.

    :param sample: Dictionary containing paths to time series files
    :param ts_range: List of tuples specifying (scale_range, offset_range) for Precipitation and Temperature
    :param output_dir: Directory to save the modified files
    :param num_augmentations: Number of augmented samples to generate
    :param idx: Integer index to identify the current sample in the augmented set
    :return: List of paths to the augmented data files
    """
    if num_augmentations<=1:
        return [{}]
    precip_scale_range, precip_offset_range = ts_range[0]
    temp_scale_range, temp_offset_range = ts_range[1]
    input_path=sample['input_path']
    def apply_random_transformations(data, column, scale_range, offset_range):
        if column in data.columns:
            scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
            offset_value = np.random.uniform(-offset_range, offset_range)
            data[column] = data[column] * scale_factor + offset_value

    output_files = []
    weather_model = sample['weather_model']

    for augment_idx in range(num_augmentations):
        if weather_model == "DefaultWeatherModel":
            if 'weather_Driver_path' in sample and sample['weather_Driver_path']:
                weather_data = pd.read_csv(sample['weather_Driver_path'])
                if len(weather_data.columns) == 4:
                    weather_data.columns = ['Year', 'Jday', 'Precipitation', 'Temperature']
                    weather_data.drop(columns=['Year', 'Jday'], inplace=True)
                elif len(weather_data.columns) == 2:
                    weather_data.columns = ['Precipitation', 'Temperature']
                apply_random_transformations(weather_data, 'Precipitation', precip_scale_range, precip_offset_range)
                apply_random_transformations(weather_data, 'Temperature', temp_scale_range, temp_offset_range)
                output_file = os.path.join(output_dir, f"augmented_{idx}_{augment_idx}_weather_driver.csv")
                weather_data.to_csv(os.path.join(input_path,output_file), index=False,header=None)
                output_files.append({'weather_Driver_path':output_file})
            else:
                rain_data = pd.read_csv(sample['rain_driver_path'])
                temp_data = pd.read_csv(sample['temp_driver_path'])
                if len(rain_data.columns) == 3:
                    rain_data.columns = ['Year', 'Jday', 'Precipitation']
                    rain_data.drop(columns=['Year', 'Jday'], inplace=True)
                elif len(rain_data.columns) == 1:
                    rain_data.columns = ['Precipitation']
                temp_data = pd.read_csv(sample['temp_driver_path'])
                if len(temp_data.columns) == 3:
                    temp_data.columns = ['Year', 'Jday', 'Temperature']
                    temp_data.drop(columns=['Year', 'Jday'], inplace=True)
                elif len(temp_data.columns) == 1:
                    temp_data.columns = ['Temperature']
                apply_random_transformations(rain_data, 'Precipitation', precip_scale_range, precip_offset_range)
                apply_random_transformations(temp_data, 'Temperature', temp_scale_range, temp_offset_range)
                output_file_rain = os.path.join(output_dir, f"augmented_{idx}_{augment_idx}_rain_driver.csv")
                output_file_temp = os.path.join(output_dir, f"augmented_{idx}_{augment_idx}_temp_driver.csv")
                rain_data.to_csv(os.path.join(input_path,output_file_rain), index=False,header=None)
                temp_data.to_csv(os.path.join(input_path,output_file_temp), index=False,header=None)
                output_files.append({'rain_driver_path':output_file_rain, 'temp_driver_path':output_file_temp})

        elif weather_model == "MultipleWeightedLocationWeatherModel":
            if 'weather_locations_path' in sample:
                locations_data = pd.read_csv(sample['weather_locations_path'],header=None, names=['x-coordinate', 'y-coordinate', 'uniqueName', 'driverDataFileName'])
                location_output_dir = os.path.join(output_dir, f"augmented_{idx}_{augment_idx}_weather_locations")
                os.makedirs(location_output_dir, exist_ok=True)
                location_files = []
                for _, row in locations_data.iterrows():
                    driver_data_path = row['driverDataFileName']
                    point_data = pd.read_csv(driver_data_path)
                    if len(point_data.columns) == 4:
                        point_data.columns = ['Year', 'Jday', 'Precipitation', 'Temperature']
                        point_data.drop(columns=['Year', 'Jday'], inplace=True)
                    elif len(point_data.columns) == 2:
                        point_data.columns = ['Precipitation', 'Temperature']
                    apply_random_transformations(point_data, 'Precipitation', precip_scale_range, precip_offset_range)
                    apply_random_transformations(point_data, 'Temperature', temp_scale_range, temp_offset_range)
                    point_output_file = os.path.join(location_output_dir, f"modified_{os.path.basename(driver_data_path)}")
                    point_data.to_csv(point_output_file, index=False,header=None)
                    location_files.append(point_output_file)
                # Save the location CSV to link the augmented files
                locations_data['drriverDataFileName'] = location_files
                locations_data.to_csv(os.path.join(output_dir, f"augmented_{idx}_{augment_idx}_weather_locations.csv"), index=False,header=None)
                output_files.append({'weather_locations_path':os.path.join(output_dir, f"augmented_{idx}_{augment_idx}_weather_locations.csv")})
            else:
                raise ValueError("Missing path for weather locations data.")
        else:
            raise ValueError(f"Unsupported weather model: {weather_model}")

    return output_files
def augment_samples(samples, augment_nums=(1,1,1), image_range=None, ts_range=None, params_range=None, LHS_samples=1, save_dir="augmented_samples"):
    """
    Augment samples based on target counts for images, time series, and parameters.
    Save the augmented data to the specified directory.

    :param samples: List of sample data to augment.
    :param augment_nums: Tuple specifying the number of augmentations for images, time series, and parameters.
    :param image_range: Range for image augmentation values.
    :param ts_range: Range for time series augmentation values.
    :param params_range: Range for parameter values to be sampled.
    :param LHS_samples: Number of samples to select using Latin Hypercube Sampling.
    :param save_dir: Directory where augmented samples will be saved.
    :return: List of paths to the generated XML files containing augmented parameters.
    """
    all_xml_files = []
    image_augment_nums = augment_nums[0]
    ts_augment_nums = augment_nums[1]
    param_augment_nums = augment_nums[2]

    # Augment DEM, cover species, and soil parameters (as .asc files)
    for idx, sample in enumerate(samples):
        input_path = sample['input_path']
        save_path = os.path.join(input_path, save_dir)
        os.makedirs(save_path, exist_ok=True)
        
        image_path = os.path.join(save_path, "images") 
        image_dir = os.path.join(save_dir, "images") 
        ts_path = os.path.join(save_path, "time_series")
        ts_dir = os.path.join(save_dir, "time_series")
        xml_path = os.path.join(save_path, "xmls")
        
        os.makedirs(image_path, exist_ok=True)
        os.makedirs(ts_path, exist_ok=True)
        os.makedirs(xml_path, exist_ok=True)
        
        # Augment and save images
        augmented_images = augment_and_save_images(sample, image_range, image_dir, image_augment_nums, idx)
        
        # Augment and save time series data
        augmented_ts = augment_and_save_time_series(sample, ts_range, ts_dir, ts_augment_nums, idx)
                
        # Augment parameters and store as dictionaries
        augmented_params = [
            {param: random.uniform(*range_values) for param, range_values in params_range.items()}
            for _ in range(param_augment_nums)
        ]
        
        # Calculate the total number of augmented sample combinations
        total_samples = len(augmented_images) * len(augmented_ts) * len(augmented_params)

        # Determine the final number of samples based on LHS_samples and total_samples
        if LHS_samples is None or LHS_samples > total_samples:
            LHS_samples = total_samples

        # Generate Latin Hypercube Sampling distribution
        lhs_design = latin_hypercube_sampling(3, LHS_samples)

        # Generate random sample indices based on LHS design
        indices_images = (lhs_design[:, 0] * len(augmented_images)).astype(int)
        indices_ts = (lhs_design[:, 1] * len(augmented_ts)).astype(int)
        indices_params = (lhs_design[:, 2] * len(augmented_params)).astype(int)
        
        # Create sampled combinations of augmented data
        sampled_combinations = [
            {**augmented_images[i], **augmented_ts[j], **augmented_params[k]}
            for i, j, k in zip(indices_images, indices_ts, indices_params)
        ]

        # Generate XML files for each sampled combination and store file paths
        for idy, params in enumerate(sampled_combinations):
            xml_file = os.path.join(xml_path, f"augmented_{idx}_time_series_{idy}.xml")
            all_xml_files.append(update_param(sample, params, xml_file))
            
    return all_xml_files

                            
def latin_hypercube_sampling(n_variables, n_samples):
    """
    Generate a Latin Hypercube Sampling (LHS) matrix.

    :param n_variables: Number of variables (columns) to sample.
    :param n_samples: Number of samples (rows) to generate.
    :return: A (n_samples, n_variables) numpy array representing the LHS samples.
    """
    # Initialize a matrix with zeros of shape (n_samples, n_variables)
    samples = np.zeros((n_samples, n_variables))
    
    # Process each variable individually
    for i in range(n_variables):
        # Divide the [0, 1) range into n_samples equal intervals
        cut = np.linspace(0, 1, n_samples + 1)
        
        # Randomly select a point within each interval
        points = np.random.uniform(low=cut[:-1], high=cut[1:], size=n_samples)
        
        # Shuffle the points to ensure randomness within the intervals
        np.random.shuffle(points)
        
        # Assign the shuffled points to the i-th column of the sample matrix
        samples[:, i] = points
    
    return samples


# Run the VELMA Java program and return the output data path and log file pathimport os
def run_java_jar(jar_file, xml_file, max_memory=None):
    """
    Run the VELMA Java program with an optional memory limit and retrieve the output data path from the log file.

    Parameters:
    - jar_file: Path to the Java JAR file.
    - xml_file: Path to the configuration XML file.
    - max_memory: Optional maximum memory allocation for the Java process (e.g., '2G'). If None, no limit is set.

    Returns:
    - output_path: Path to the output data extracted from the log file.
    """
    # Ensure the Velmalog directory exists
    os.makedirs('Velmalog', exist_ok=True)
    
    # Generate a unique log file name to avoid conflicts
    process_id = os.getpid()
    unique_id = uuid.uuid4().hex
    log_file = f'Velmalog/processor_{process_id}_{unique_id}.log'
    
    # Build the Java command with optional memory allocation
    cmd = ['java']
    if max_memory is not None:
        cmd.append(f'-Xmx{max_memory}')
    cmd.extend(['-cp', jar_file, 'gov.epa.velmasimulator.VelmaSimulatorCmdLine', xml_file])
    
    try:
        # Run the Java command and write output to the log file
        with open(log_file, 'w') as log:
            print(f"Java process started for XML file: {xml_file}")
            subprocess.run(cmd, check=True, stdout=log, stderr=log)
    except subprocess.TimeoutExpired as e:
        print(f"Error: {e}. Java process timed out for XML file: {xml_file}")
        return None 
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}. Java process failed for XML file: {xml_file}")
        return None 

    # Extract the output path from the log file
    keyword = 'Output Data Location'
    try:
        with open(log_file, 'r') as log:
            for line in log:
                match = re.search(rf'{keyword}.*"([^"]+)"', line)
                if match:
                    output_path = match.group(1)
                    return output_path 
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
    
    print(f"Keyword '{keyword}' not found in log file: {log_file}")
    return None


def get_samples(xml_files):
    """
    Extracts and organizes data from a list of XML files.

    :param xml_files: List of XML file paths to parse.
    :return: List of dictionaries containing parsed data for each XML file.
    """
    all_data = []
    for xml_file in xml_files:
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Create a dictionary to map parameter paths to XML elements and retrieve their text values
        paths = {
            'output_path': root.find("./calibration/VelmaInputs.properties/initializeOutputDataLocationRoot"),
            'input_data_location_root_name': root.find("./startups/VelmaStartups.properties/inputDataLocationRootName"),
            'input_data_location_dir_name': root.find("./startups/VelmaStartups.properties/inputDataLocationDirName"),
            'dem_path': root.find("./calibration/VelmaInputs.properties/input_dem"),
            'cover_species_path': root.find("./calibration/VelmaInputs.properties/coverSpeciesIndexMapFileName"),
            'soil_parameters_path': root.find("./calibration/VelmaInputs.properties/soilParametersIndexMapFileName"),
            'cover_age_path': root.find("./calibration/VelmaInputs.properties/coverAgeMapFileName"),
            'weather_driver_path': root.find("./weather/DefaultWeatherModel/weatherDriverDataFileName"),
            'rain_driver_path': root.find("./weather/DefaultWeatherModel/rainDriverDataFileName"),
            'temp_driver_path': root.find("./weather/DefaultWeatherModel/airTemperatureDriverDataFileName"),
            'weather_locations_path': root.find("./weather/MultipleWeightedLocationWeatherModel/weatherLocationsDataFileName"),
            'forcing_start': root.find("./calibration/VelmaInputs.properties/forcing_start"),
            'forcing_end': root.find("./calibration/VelmaInputs.properties/forcing_end"),
            'start_year': root.find("./calibration/VelmaInputs.properties/syear"),
            'end_year': root.find("./calibration/VelmaInputs.properties/eyear"),
        }

        # Extract text values and construct input data location path
        input_data_location_root_name = paths['input_data_location_root_name'].text if paths['input_data_location_root_name'] is not None else ""
        input_data_location_dir_name = paths['input_data_location_dir_name'].text if paths['input_data_location_dir_name'] is not None else ""
        input_path = os.path.join(input_data_location_root_name, input_data_location_dir_name)
        
        # Helper function to construct full paths for elements with relative paths
        def get_full_path(elem):
            return os.path.join(input_data_location_root_name, input_data_location_dir_name, elem.text.lstrip("./")) if elem is not None and elem.text else ""

        # Construct full paths for each specified element
        dem_path = get_full_path(paths['dem_path'])
        cover_species_path = get_full_path(paths['cover_species_path'])
        soil_parameters_path = get_full_path(paths['soil_parameters_path'])
        cover_age_path = get_full_path(paths['cover_age_path'])
        weather_driver_path = get_full_path(paths['weather_driver_path'])
        rain_driver_path = get_full_path(paths['rain_driver_path'])
        temp_driver_path = get_full_path(paths['temp_driver_path'])
        weather_locations_path = get_full_path(paths['weather_locations_path'])

        # Store XML elements in the mapping dictionary
        xml_mapping = {key: paths[key] for key in paths}

        # Parse the XML file to extract additional parameters
        output_path = paths['output_path'].text if paths['output_path'] is not None else ""
        xml_params, velma_params = load_params_from_paths(xml_file, params_range)

        # Collect data from XML elements
        data = {
            'name': xml_file,
            'xml_file': root,
            'forcing_start': int(paths['forcing_start'].text) if paths['forcing_start'] is not None else "",
            'forcing_end': int(paths['forcing_end'].text) if paths['forcing_end'] is not None else "",
            'start_year': int(paths['start_year'].text) if paths['start_year'] is not None else "",
            'end_year': int(paths['end_year'].text) if paths['end_year'] is not None else "",
            'dem_path': dem_path,
            'input_path': input_path,
            'output_path': output_path,
            'cover_species_path': cover_species_path,
            'soil_parameters_path': soil_parameters_path,
            'cover_age_path': cover_age_path,
            'weather_model': root.find("./weather")[0].tag if root.find("./weather") is not None and len(root.find("./weather")) > 0 else "",
            'weather_driver_path': weather_driver_path,
            'rain_driver_path': rain_driver_path,
            'temp_driver_path': temp_driver_path,
            'weather_locations_path': weather_locations_path,
            'xml_params': xml_params,
            'velma_params': velma_params,
            'velma_param_size': velma_params.shape[0],
            'xml_mapping': xml_mapping  # Include the mapping dictionary for reference
        }
        
        all_data.append(data)
    
    return all_data


def modify_years(xml_files, year_range):
    """
    Batch modify the start and end years in a list of XML files using the same year range.

    Parameters:
    - xml_files: List of paths to XML files to modify.
    - year_range: List containing [start_year, end_year] to be applied to each XML file.

    Returns:
    - modified_files: List of paths to modified XML files.
    """
    # Fixed paths for start and end year elements
    start_year_path = "./calibration/VelmaInputs.properties/syear"
    end_year_path = "./calibration/VelmaInputs.properties/eyear"
    
    start_year, end_year = year_range
    modified_files = []
    
    for xml_file in xml_files:
        # Load XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Modify start year
        start_elem = root.find(start_year_path)
        if start_elem is not None:
            start_elem.text = str(start_year)
        else:
            print(f"Warning: Start year element '{start_year_path}' not found in {xml_file}.")
        
        # Modify end year
        end_elem = root.find(end_year_path)
        if end_elem is not None:
            end_elem.text = str(end_year)
        else:
            print(f"Warning: End year element '{end_year_path}' not found in {xml_file}.")
        
        # Save changes to a new XML file or overwrite the original file
        modified_file = xml_file.replace(".xml", "_modified.xml")
        tree.write(modified_file)
        modified_files.append(modified_file)
    
    return modified_files


def update_param(data, param_updates, xml_file_path=None):
    """
    Update the parameter values in the data and synchronize the XML elements.
    :param data: Dictionary containing XML file and parsed data
    :param param_updates: Dictionary containing parameter keys (XPath) and new values
    :param xml_file_path: Path to the XML file to save changes, if provided
    """
    xml_mapping = data['xml_mapping']

    # Iterate over the parameter updates
    for param_key, new_value in param_updates.items():
        # Check if the parameter exists in the xml_mapping
        if param_key in xml_mapping:
            # Update the element directly from the mapping
            elem = xml_mapping[param_key]
        else:
            # If not in mapping, find it in the XML file
            elem = data['xml_file'].find(param_key)
            if elem is None:
                raise KeyError(f"The parameter key '{param_key}' was not found in the XML file.")

            # Add it to the mapping for future reference
            xml_mapping[param_key] = elem

        # Update the parameter value
        elem.text = str(new_value)
        data[param_key] = new_value

    # Save the updated XML file if the path is provided
    if xml_file_path:
        tree = ET.ElementTree(data['xml_file'])
        tree.write(xml_file_path)
        return xml_file_path
    else:
        return None


def scale_or_inverse(data, scalers=None, scaler_type='standard'):
    """
    If scalers are not provided, scale each feature of the data separately and return the scaled data and scalers.
    If scalers are provided, perform inverse transformation and return the original data.

    Parameters:
    - data: list or ndarray, input data, can have any number of dimensions (1D, 2D, 3D, 4D, or higher)
    - scalers: list of MinMaxScaler or None, scaler for each feature
    - scaler_type: str, type of scaler to use ('minmax' or 'standard')

    Returns:
    - scaled_data: ndarray, scaled data or inverse transformed data
    - scalers: list of MinMaxScaler or StandardScaler or None, list of scalers (if scaling operation)
    """
    # Convert data to numpy array if it's a list
    if isinstance(data, list):
        data = np.array(data)
    
    original_shape = data.shape
    
    # Handle different dimensionalities
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim == 2:
        n_samples, n_features = data.shape
        data = data.reshape(n_samples, 1, n_features)
    elif data.ndim >= 3:
        n_features = data.shape[-1]
    else:
        raise ValueError("Input data must have at least 1 dimension.")
    
    reshaped_data = data.reshape(-1, data.shape[-1])
    n_features = reshaped_data.shape[-1]

    if scalers is None:
        # Scale each feature separately
        if scaler_type == 'minmax':
            scalers = [MinMaxScaler() for _ in range(n_features)]
        elif scaler_type == 'standard':
            scalers = [StandardScaler() for _ in range(n_features)]
        else:
            raise ValueError("Invalid scaler_type. Use 'minmax' or 'standard'.")
        
        scaled_data = np.zeros_like(reshaped_data)
        for i in range(n_features):
            scaled_data[:, i] = scalers[i].fit_transform(reshaped_data[:, i].reshape(-1, 1)).flatten()
        return scaled_data.reshape(original_shape), scalers
    else:
        # Perform inverse transformation
        original_data = np.zeros_like(reshaped_data)
        for i in range(n_features):
            original_data[:, i] = scalers[i].inverse_transform(reshaped_data[:, i].reshape(-1, 1)).flatten()
        return original_data.reshape(original_shape), scalers


def scale_or_inverse_observed_data(observed_data, scaler_type='standard', scalers=None):
    """
    Scales or inverse scales the 'Variable' columns of each DataFrame in a list of DataFrames.

    Parameters:
    - observed_data: List of DataFrames, each containing 'Year', 'Jday', and 'Variable' columns.
    - scaler_type: str, type of scaler to use ('minmax' or 'standard') if scaling.
    - scalers: List of scalers or None; if provided, performs inverse scaling.

    Returns:
    - transformed_observed_data: List of DataFrames with scaled or inverse scaled 'Variable' columns.
    - observed_scaler: List of scalers used for each 'Variable' column (if scaling).
    """
    # Create a deep copy of observed_data to avoid modifying the original data
    observed_data_copy = copy.deepcopy(observed_data)
    
    # Find the maximum length of all 'Variable' columns and create a padded array with NaN values
    max_length = max(len(df) for df in observed_data_copy)
    variables_data = np.column_stack([np.pad(df['Variable'].values, 
                                             (0, max_length - len(df)), 
                                             constant_values=np.nan) 
                                      for i, df in enumerate(observed_data_copy)])
    
    # Determine whether to scale or inverse scale based on the presence of scalers
    if scalers is None:
        # Perform scaling using the specified scaler type
        scaled_variables, observed_scaler = scale_or_inverse(variables_data, scaler_type=scaler_type)
    else:
        # Perform inverse scaling using provided scalers
        scaled_variables, observed_scaler = scale_or_inverse(variables_data, scalers=scalers)
    
    # Place the transformed data back into each DataFrame's 'Variable' column
    for i, df in enumerate(observed_data_copy):
        df['Variable'] = scaled_variables[:len(df), i]
    
    return observed_data_copy, observed_scaler if scalers is None else scalers


# Create a dataloader containing images, time series, XML parameters, VELMA model parameters, and real outputs
def create_dataloader(num_samples, batch_size=64, velma_jar_path='Velma.jar', required_columns=None,
                      base_xml_files=None, params_range=None, year_range=None, max_workers=None, use_existing_data=False, generate_new_data=True,
                      output_dirs=None, LHS_samples=None, observed_data_path=None, max_memory=None): 
    """
    Create a data loader for model training, including data generation, augmentation, loading, and normalization.

    Parameters:
    - num_samples: Number of samples to augment.
    - batch_size: Batch size for data loading.
    - velma_jar_path: Path to the VELMA Java JAR file.
    - required_columns: List of columns required from the model output.
    - base_xml_files: List of base XML files for data augmentation.
    - params_range: Dictionary specifying parameter ranges for augmentation.
    - year_range: List specifying [start_year, end_year] for XML modification.
    - max_workers: Maximum number of parallel workers for VELMA runs.
    - use_existing_data: Whether to use existing data if specified.
    - generate_new_data: Whether to generate new data.
    - output_dirs: Directories containing existing output data.
    - LHS_samples: Number of samples for Latin Hypercube Sampling.
    - observed_data_path: Path to the observed data CSV file.
    - max_memory: Maximum memory allocation for Java processes (e.g., '10g').

    Returns:
    - dataloader: DataLoader object for PyTorch.
    - observed_data: Processed observed data.
    - scalers: Dictionary of scalers for outputs, observed data, and parameters.
    """
    
    output_paths = []
    
    # Generate and augment data if requested
    if generate_new_data and base_xml_files is not None:
        modified_xml_files = modify_years(base_xml_files, year_range)
        samples = get_samples(modified_xml_files)
        xml_files = augment_samples(samples, augment_nums=num_samples, image_range=[(0.8, 1.2), (0.2, 5), (0.2, 5)],
                                    ts_range=[(0.5, 0.1), (0.2, 0.3)], params_range=params_range, LHS_samples=LHS_samples)
        
        # Determine the number of parallel workers
        if max_workers is None:
            max_workers = min(len(xml_files), os.cpu_count())
       
        print(f"Using {max_workers} workers for parallel VELMA runs")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_java_jar, velma_jar_path, xml_file, max_memory=max_memory) for xml_file in xml_files]
            output_paths = [f.result() for f in concurrent.futures.as_completed(futures)]
        print(output_paths)

    # Use existing data if specified
    if use_existing_data and output_dirs is not None:
        output_paths = list(set(output_paths + output_dirs))
        print(f"Using existing outputs from {output_dirs}.")
    
    # Collect XML and output files from generated output paths
    xml_files, out_files = [], []
    for output_path in output_paths:
        for root, dirs, files in os.walk(output_path):
            if 'SimulationConfiguration.xml' in files and 'DailyResults.csv' in files:
                xml_files.append(os.path.join(root, 'SimulationConfiguration.xml'))
                out_files.append(os.path.join(root, 'DailyResults.csv'))
                
    modified_xml_files = modify_years(xml_files, year_range) 
    samples = get_samples(modified_xml_files)
    images = load_images_from_samples(samples)
    outputs = load_outputs_from_paths(out_files, required_columns, year_range=year_range)
    
    output_size = len(required_columns)
    xml_params = [sample['xml_params'] for sample in samples]
    velma_params = [sample['velma_params'] for sample in samples]

    # Load and preprocess observed data
    observed_data = pd.read_csv(observed_data_path, header=None)
    observed_end_year = int(observed_data.iloc[:, ::3].max().max())
    observed_data = [
        observed_data.iloc[:observed_data.iloc[:, i+2].last_valid_index() + 1, [i, i+1, i+2]]
        .dropna()
        .rename(columns={i: 'Year', i+1: 'Jday', i+2: 'Variable'})
        .assign(Year=lambda x: x['Year'].astype(int), Jday=lambda x: x['Jday'].astype(int))
        for i in range(0, observed_data.shape[1], 3)
    ]

    # Load time series data and observed time series data
    time_series, observed_time_series = load_time_series_from_paths(samples, observed_end_year=observed_end_year)

    # Normalize outputs, observed data, and parameters
    observed_data, observed_scaler = scale_or_inverse_observed_data(observed_data)
    outputs, output_scaler = scale_or_inverse(outputs)
    velma_params, param_scaler = scale_or_inverse(velma_params, scaler_type='minmax')
    scalers = {'output': output_scaler, 'observed': observed_scaler, 'param': param_scaler}
    
    # Convert data to PyTorch tensors
    time_series = torch.tensor(time_series, dtype=torch.float32).clone().detach()
    observed_time_series = torch.tensor(observed_time_series, dtype=torch.float32).clone().detach()
    xml_params = torch.tensor(np.array(xml_params), dtype=torch.float32).clone().detach()
    velma_params = torch.tensor(velma_params, dtype=torch.float32).clone().detach()
    outputs = torch.tensor(outputs, dtype=torch.float32).clone().detach()

    # Construct dataset and DataLoader
    dataset = TensorDataset(images, time_series, xml_params, outputs, velma_params, observed_time_series)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, observed_data, scalers


# 2. Multi-layer Convolutional Network for Image Feature Extraction
class MultiLayerImageFeatureExtractor(nn.Module):
    """
    A multi-layer convolutional neural network for extracting features from images.

    Parameters:
    - input_channels: Number of input channels in the images.
    - num_layers: Number of convolutional layers to add to the network.
    - base_num_filters: Base number of filters for the first layer. Each subsequent layer doubles the filters.
    - kernel_size: Size of the convolutional kernel.
    - pool_size: Size of the pooling kernel.
    - adaptive_pool_size: Output size of the adaptive average pooling layer.
    """
    def __init__(self, input_channels, num_layers, base_num_filters=16, kernel_size=3, pool_size=2, adaptive_pool_size=4):
        super(MultiLayerImageFeatureExtractor, self).__init__()

        layers = []
        in_channels = input_channels
        
        # Dynamically generate convolutional layers
        for i in range(num_layers):
            out_channels = base_num_filters * (2 ** i)  # Each layer has twice the filters of the previous one
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=2))  # Pooling layer
            in_channels = out_channels  # Update input channels for the next layer
        
        # Define the convolutional network using nn.Sequential
        self.conv_layers = nn.Sequential(*layers)

        # Use AdaptiveAvgPool2d to adjust the output feature map to a fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((adaptive_pool_size, adaptive_pool_size))

        # Calculate the size of the flattened feature vector
        self.flatten_size = adaptive_pool_size * adaptive_pool_size * out_channels

        # Fully connected layer
        self.fc = nn.Linear(self.flatten_size, 128)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: Input tensor with shape (batch_size, input_channels, height, width).

        Returns:
        - x: Output feature vector after convolution, pooling, and fully connected layers.
        """
        x = self.conv_layers(x)  # Pass through convolutional and pooling layers
        x = self.adaptive_pool(x)  # Pass through adaptive pooling layer
        x = x.view(-1, self.flatten_size)  # Flatten into a 1D vector
        x = torch.relu(self.fc(x))  # Pass through the fully connected layer
        return x


# Time series feature extraction network
class TimeSeriesFeatureExtractor(nn.Module):
    """
    A feature extractor for time series data using an LSTM network.

    Parameters:
    - input_size: Number of input features per time step.
    - hidden_size: Number of hidden units in each LSTM layer.
    - output_size: Size of the output feature vector.
    - num_layers: Number of LSTM layers.
    - dropout: Dropout probability for regularization.
    - bidirectional: Whether to use a bidirectional LSTM.
    """
    def __init__(self, input_size=2, hidden_size=64, output_size=128, num_layers=2, dropout=0.3, bidirectional=False):
        super(TimeSeriesFeatureExtractor, self).__init__()
        
        # Optional bidirectional LSTM with dropout for regularization
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        # Adjust input size for the fully connected layer if using a bidirectional LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Use Layer Normalization to stabilize training
        self.norm = nn.LayerNorm(lstm_output_size)
        
        # Fully connected layer to generate the output feature vector
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.contiguous()
        lstm_out, _ = self.lstm(x)
        
        # Extract the last time step's output from the LSTM
        x = lstm_out[:, -1, :]
        x = self.norm(x)  # Apply Layer Normalization
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Pass through the fully connected layer with ReLU activation
        x = torch.relu(self.fc(x))
        
        return x


# Main model: Combines image, time series, and XML features to generate a watershed feature vector
class MainModel(nn.Module):
    """
    Main model for combining image, time series, and XML features to generate a watershed feature vector
    and predict VELMA parameters.

    Parameters:
    - num_conv_layers: Number of convolutional layers in the image feature extractor.
    - xml_feature_size: Size of the XML feature vector.
    - activation_fn: Activation function to use.
    - use_dropout: Whether to apply dropout.
    - dropout_prob: Dropout probability.
    - velma_output_size: Size of the output layer in the VELMA prediction branch.
    - shared_hidden_size: Size of the hidden layer in the shared fully connected layer.
    - velma_hidden_layers: List of hidden layer sizes for the VELMA parameter prediction branch.
    """
    def __init__(self, num_conv_layers=4, xml_feature_size=128, activation_fn=nn.ReLU, use_dropout=True, dropout_prob=0.3, 
                 velma_output_size=3, shared_hidden_size=512, velma_hidden_layers=[128, 64]):
        super(MainModel, self).__init__()

        # Image and time series feature extractors
        self.image_feature_extractor = MultiLayerImageFeatureExtractor(input_channels=3, num_layers=num_conv_layers)
        self.time_series_feature_extractor = TimeSeriesFeatureExtractor()
        self.surrogate_feature_extractor = TimeSeriesFeatureExtractor()  # Surrogate model output feature extractor
        
        self.fc_xml = None
        self.xml_feature_size = xml_feature_size
        self.activation_fn = activation_fn()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_prob) if use_dropout else nn.Identity()

        # Fully connected layer for combined features
        combined_feature_size = 128 + 128 + 128 + self.xml_feature_size  # Updated feature size with surrogate features
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_feature_size, shared_hidden_size),
            self.activation_fn,
            self.dropout,
            nn.Linear(shared_hidden_size, 256),
            self.activation_fn,
            self.dropout
        )

        # Initialize the VELMA parameter prediction branch
        self.velma_branch = self._build_specific_branch(256, velma_output_size, velma_hidden_layers)
        

    # Method to build the VELMA parameter prediction branch
    def _build_specific_branch(self, input_size, output_size, hidden_layers):
        """
        Build the VELMA parameter prediction branch, a fully connected network with multiple layers and ReLU activation.
        """
        layers = []
        current_size = input_size
        
        # Dynamically build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(self.activation_fn)
            if self.use_dropout:
                layers.append(self.dropout)
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        
        return nn.Sequential(*layers)

    def forward(self, image, time_series, xml_params, surrogate_output):
        # Extract image features
        img_feat = self.image_feature_extractor(image)
        
        # Extract time series features
        ts_feat = self.time_series_feature_extractor(time_series)
        
        # Extract surrogate output features
        surrogate_feat = self.surrogate_feature_extractor(surrogate_output)
        
        # Initialize XML feature extraction layer lazily
        if self.fc_xml is None:
            input_size = xml_params.shape[1]
            self.fc_xml = nn.Linear(input_size, self.xml_feature_size).to(xml_params.device)
        
        # Extract XML parameter features
        xml_feat = self.activation_fn(self.fc_xml(xml_params))
        
        # Combine all feature vectors
        combined_feat = torch.cat((img_feat, ts_feat, surrogate_feat, xml_feat), dim=1)
        
        # Pass through the shared fully connected layer
        x = self.shared_fc(combined_feat)
        
        # Predict VELMA parameters using a sigmoid activation
        velma_params = torch.sigmoid(self.velma_branch(x))
        
        return velma_params


# Surrogate model: Used to approximate VELMA's outputs
class SurrogateModel(nn.Module):
    """
    Surrogate model for predicting outputs based on time series and VELMA parameters.

    Parameters:
    - time_series_input_size: Number of input features for the time series data.
    - lstm_hidden_size: Number of hidden units in the shared LSTM.
    - velma_param_size: Number of VELMA parameters.
    - shared_lstm_layers: Number of layers in the shared LSTM.
    - branch_lstm_hidden_size: Number of hidden units in each branch LSTM.
    - branch_lstm_layers: Number of layers in each branch LSTM.
    - fc_hidden_dims: List defining hidden layer sizes for the shared fully connected layers.
    - branch_fc_dims: List defining hidden layer sizes for each branch's fully connected layers.
    - output_sizes: List of output sizes for each branch.
    - use_dropout: Whether to use dropout in the layers.
    - dropout_prob: Dropout probability.
    """
    def __init__(self, time_series_input_size=2, lstm_hidden_size=64, velma_param_size=3, 
                 shared_lstm_layers=2, branch_lstm_hidden_size=32, branch_lstm_layers=1,
                 fc_hidden_dims=[256, 128, 64], branch_fc_dims=[128, 64, 32], output_sizes=[1, 1], 
                 use_dropout=True, dropout_prob=0.2):
        super(SurrogateModel, self).__init__()
        
        # Shared LSTM for processing time series data
        self.shared_lstm = nn.LSTM(input_size=time_series_input_size, hidden_size=lstm_hidden_size, 
                                   num_layers=shared_lstm_layers, batch_first=True)
        
        # Batch normalization layers for stabilizing training
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(lstm_hidden_size) for _ in range(shared_lstm_layers)])

        # Combined input size of shared LSTM output and VELMA parameters
        self.combined_input_size = lstm_hidden_size + velma_param_size
        
        # Define shared fully connected layers
        fc_layers = []
        for i, fc_dim in enumerate(fc_hidden_dims):
            in_dim = self.combined_input_size if i == 0 else fc_hidden_dims[i - 1]
            fc_layers.append(nn.Linear(in_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            if use_dropout:
                fc_layers.append(nn.Dropout(dropout_prob))
        
        # Stack shared fully connected layers
        self.fc_layers = nn.Sequential(*fc_layers)

        # Create deep structure for each output branch
        self.output_branches = nn.ModuleList([
            nn.Sequential(
                nn.LSTM(input_size=fc_hidden_dims[-1], hidden_size=branch_lstm_hidden_size,
                        num_layers=branch_lstm_layers, batch_first=True),
                nn.Linear(branch_lstm_hidden_size, branch_fc_dims[0]), nn.ReLU(),
                *(nn.Sequential(nn.Linear(branch_fc_dims[i], branch_fc_dims[i + 1]), nn.ReLU(), nn.Dropout(dropout_prob))
                  for i in range(len(branch_fc_dims) - 1)),
                nn.Linear(branch_fc_dims[-1], output_size)
            )
            for output_size in output_sizes
        ])

    def forward(self, time_series, velma_params):
        device = time_series.device
        velma_params = velma_params.to(device)

        # Process time series through the shared LSTM
        shared_lstm_out, _ = self.shared_lstm(time_series.to(device))
        
        # Apply batch normalization to LSTM output
        for i in range(len(self.batch_norms)):
            shared_lstm_out = self.batch_norms[i](shared_lstm_out.permute(0, 2, 1)).permute(0, 2, 1)

        # Expand and concatenate VELMA parameters with LSTM output
        velma_params_expanded = velma_params.unsqueeze(1).repeat(1, shared_lstm_out.size(1), 1)
        combined_input = torch.cat((shared_lstm_out, velma_params_expanded), dim=2)

        # Apply shared fully connected layers to each time step
        batch_size, seq_len, _ = combined_input.size()
        combined_input = combined_input.view(batch_size * seq_len, -1)
        fc_out = self.fc_layers(combined_input)  # Shape: [batch_size * seq_len, last_fc_dim]
        fc_out = fc_out.view(batch_size, seq_len, -1)  # Restore time series shape: [batch_size, seq_len, last_fc_dim]

        # Pass through each output branch's LSTM and fully connected network
        outputs = []
        for branch in self.output_branches:
            branch_lstm_out, _ = branch[0](fc_out)  # branch[0] is the branch-specific LSTM
            branch_output = branch[1:](branch_lstm_out)  # Subsequent fully connected layers
            outputs.append(branch_output)

        # Concatenate outputs from each branch along the output dimension
        output = torch.cat(outputs, dim=2)  # Shape: [batch_size, seq_len, -1]
        return output


# 3. Model Saving and Loading
def save_model(model, path):
    """
    Save the model state dictionary to a specified file path.

    Parameters:
    - model: The model to be saved.
    - path: Path where the model should be saved.
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save model state dictionary
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")

def load_model(model, path, device=None):
    """
    Load the model state dictionary from a specified file path.

    Parameters:
    - model: The model to load the state dictionary into.
    - path: Path where the model is saved.
    - device: Device to load the model onto, if specified.
    """
    try:
        # Check if the model file exists
        if os.path.exists(path):
            # Load model onto specified device (default to CUDA if available)
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Model loaded from {path} onto {device}")
        else:
            print(f"Error: The model file at {path} does not exist.")
    except Exception as e:
        print(f"Error occurred while loading the model: {e}")


def train_model(main_model, surrogate_model, optimizer_main, optimizer_surrogate, dataloader, 
                criterion=nn.MSELoss(), train_main=True, surrogate_epochs=None, main_epochs=None, epochs=100,
                save_path=None, grad_clip_value=None, device='cuda', plot_loss=True):
    """
    Unified training function to first train the surrogate model, then conditionally train the main model.

    Parameters:
    - main_model: The main model to be trained.
    - surrogate_model: The surrogate model to be trained first.
    - optimizer_main: Optimizer for the main model.
    - optimizer_surrogate: Optimizer for the surrogate model.
    - dataloader: DataLoader providing the training data.
    - criterion: Loss function for optimization (default is MSELoss).
    - train_main: Whether to train the main model after training the surrogate model.
    - surrogate_epochs: Number of epochs to train the surrogate model (default is same as `epochs`).
    - main_epochs: Number of epochs to train the main model (default is same as `epochs`).
    - epochs: Total epochs if surrogate_epochs and main_epochs are not specified.
    - save_path: Path to save model checkpoints.
    - grad_clip_value: Value for gradient clipping (if specified).
    - device: Device to perform training ('cuda' or 'cpu').
    - plot_loss: Whether to plot the training loss curve.

    Returns:
    - surrogate_losses: List of surrogate model losses per epoch.
    - main_losses: List of main model losses per epoch.
    """
    if surrogate_epochs is None:
        surrogate_epochs = epochs
    if main_epochs is None:
        main_epochs = epochs
    surrogate_losses = []
    main_losses = []
    
    # Train surrogate model
    surrogate_model.train()
    for epoch in range(surrogate_epochs):
        epoch_loss = 0.0

        for images, time_series, xml_params, real_outputs, velma_params, _ in dataloader:
            images, time_series, xml_params, real_outputs, velma_params = (
                item.to(device) for item in (images, time_series, xml_params, real_outputs, velma_params)
            )

            # Surrogate model training
            optimizer_surrogate.zero_grad()
            predicted_outputs = surrogate_model(time_series, velma_params)
            min_time_steps = min(predicted_outputs.size(1), real_outputs.size(1))
            predicted_outputs, real_outputs = (
                tensor[:, :min_time_steps, :] for tensor in (predicted_outputs, real_outputs)
            )
            loss_surrogate = criterion(predicted_outputs, real_outputs)
            loss_surrogate.backward()

            if grad_clip_value:
                torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), grad_clip_value)
            optimizer_surrogate.step()

            # Accumulate epoch loss for surrogate model
            epoch_loss += loss_surrogate.item()

        # Compute and record average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        surrogate_losses.append(avg_loss)
        print(f"Surrogate Model - Epoch {epoch+1}/{surrogate_epochs}, Loss: {avg_loss:.4f}")

        # Periodically save surrogate model
        if save_path and (epoch + 1) % 10 == 0:
            torch.save(surrogate_model.state_dict(), f'{save_path}_surrogate.pth')
    
    # Train main model if specified
    if train_main:
        main_model.train()
        for epoch in range(main_epochs):
            epoch_loss = 0.0

            for images, time_series, xml_params, real_outputs, velma_params, _ in dataloader:
                images, time_series, xml_params, real_outputs, velma_params = (
                    item.to(device) for item in (images, time_series, xml_params, real_outputs, velma_params)
                )

                # Main model training
                optimizer_main.zero_grad()
                predicted_outputs = surrogate_model(time_series, velma_params)  # Use trained surrogate model's output
                velma_params_pred = main_model(images, time_series, xml_params, predicted_outputs)
                loss_main = criterion(velma_params, velma_params_pred)
                loss_main.backward()

                if grad_clip_value:
                    torch.nn.utils.clip_grad_norm_(main_model.parameters(), grad_clip_value)
                optimizer_main.step()

                # Accumulate epoch loss for main model
                epoch_loss += loss_main.item()

            # Compute and record average loss for the epoch
            avg_loss = epoch_loss / len(dataloader)
            main_losses.append(avg_loss)
            print(f"Main Model - Epoch {epoch+1}/{main_epochs}, Loss: {avg_loss:.4f}")

            # Periodically save main model
            if save_path and (epoch + 1) % 10 == 0:
                torch.save(main_model.state_dict(), f'{save_path}_main.pth')

    # Plot the loss curve if specified
    if plot_loss:
        plt.figure()
        plt.plot(range(1, surrogate_epochs + 1), surrogate_losses, label='Surrogate Model Loss')
        if train_main:
            plt.plot(range(1, main_epochs + 1), main_losses, label='Main Model Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    return surrogate_losses, main_losses


def calculate_mse_loss(predicted_outputs, observed_data):
    """
    Calculate the MSE loss between predicted outputs and observed data, taking into account 
    irregular time intervals and leap years in the observed data.

    Parameters:
    - predicted_outputs: Tensor of shape (num_samples, timesteps, num_variables) representing model predictions.
    - observed_data: List of DataFrames, each containing 'Year', 'Jday', and 'Variable' columns for observed data.

    Returns:
    - mse_loss: The calculated mean squared error loss.
    - all_indices: List of sample indices for each variable, adjusted for leap years.
    """
    num_samples, timesteps, num_variables = predicted_outputs.shape

    # Calculate average intervals for each variable in observed data, considering leap years
    average_intervals = []
    for data in observed_data:
        years = data['Year']
        jdays = data['Jday']
        intervals = []
        for i in range(1, len(years)):
            year_diff = years.iloc[i] - years.iloc[i - 1]
            if year_diff == 0:
                interval = jdays.iloc[i] - jdays.iloc[i - 1]
            else:
                interval = (year_diff - 1) * 365 + jdays.iloc[i] + (366 if is_leap_year(years.iloc[i - 1]) else 365) - jdays.iloc[i - 1]
            intervals.append(interval)
        average_intervals.append(np.mean(intervals))

    # Assume the end date is day 365 of the maximum year
    max_year = max(df['Year'].max() for df in observed_data)
    max_length = max(len(data['Variable']) for data in observed_data)

    all_indices = []
    # Initialize padded tensors for selected and observed values to match max_length
    padded_selected_values = torch.full((num_samples, max_length, num_variables), float('nan'), device=predicted_outputs.device)
    padded_observed_values = torch.full((num_samples, max_length, num_variables), float('nan'), device=predicted_outputs.device)

    # Process each variable in observed data
    for var_idx in range(num_variables):
        years, jdays = observed_data[var_idx]['Year'].values, observed_data[var_idx]['Jday'].values
        sample_indices = timesteps - ((max_year - years) * 365 + (365 - jdays)) - 1
        # Adjust for leap years by shifting indices if after February 29
        sample_indices -= np.array([1 if is_leap_year(year) and jday > 59 else 0 for year, jday in zip(years, jdays)]) 
        all_indices.append(sample_indices)

        # Smooth predicted outputs if needed, based on average intervals
        kernel_size = max(1, int(average_intervals[var_idx]))
        if kernel_size > 1:
            smoothed = torch.nn.functional.avg_pool1d(predicted_outputs[:, :, var_idx].unsqueeze(1), kernel_size=kernel_size, stride=1).squeeze(1)
            selected_values = smoothed[:, sample_indices]
        else:
            selected_values = predicted_outputs[:, sample_indices, var_idx]

        # Convert observed data to tensor and expand to match sample count
        observed_values = torch.tensor(np.array(observed_data[var_idx]['Variable'].values), dtype=torch.float32).to(predicted_outputs.device)
        observed_values = observed_values.expand(num_samples, -1)

        # Pad selected and observed values to max_length
        padded_selected_values[:, :selected_values.shape[1], var_idx] = selected_values
        padded_observed_values[:, :observed_values.shape[1], var_idx] = observed_values

    # Calculate overall MSE loss, using a mask to ignore NaN values
    mask = ~torch.isnan(padded_selected_values)
    mse_loss = torch.nn.functional.mse_loss(padded_selected_values[mask], padded_observed_values[mask])

    return mse_loss, all_indices


def is_leap_year(year):
    """
    Determine if a given year is a leap year.
    
    :param year: The year to check.
    :return: True if the year is a leap year, False otherwise.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


# 7. Model evaluation and visualization
def optimize_and_visualize(main_model, surrogate_model, dataloader, observed_data, scalers, params_range, criterion=nn.MSELoss(), lr=0.001, epochs=100, use_main_model=False, device='cuda', plot_prediction=True):
    """
    Optimizes model parameters to make surrogate model predictions fit observed data, and optionally visualizes the comparison.
    Optionally uses the main model to initialize parameters.

    Parameters:
    - main_model: The main model (optional, only used if use_main_model is True).
    - surrogate_model: The surrogate model for predictions.
    - dataloader: DataLoader providing images, time_series, and xml_params.
    - observed_data: Observed data in a list format.
    - scalers: Dictionary of scalers for inverse transforming predictions and observed data.
    - params_range: Dictionary specifying the range for each parameter to be optimized.
    - criterion: Loss function for optimization (default is MSELoss).
    - lr: Learning rate for the optimizer.
    - epochs: Number of epochs for optimization.
    - use_main_model: Whether to use main_model to initialize parameters.
    - device: Device to perform optimization ('cuda' or 'cpu').
    - plot_prediction: Whether to plot predicted vs observed data after optimization.

    Returns:
    - predicted_outputs: Predicted values from the surrogate model after optimization.
    - best_params: Optimized model parameters.
    - indices: List of indices used for aligning observed data with predictions.
    """
    
    # Freeze parameters of surrogate and main models to prevent updates
    for param in surrogate_model.parameters():
        param.requires_grad = False
    if use_main_model and main_model is not None:
        for param in main_model.parameters():
            param.requires_grad = False

    # Extract a single batch of data from the dataloader
    images, time_series, xml_params, outputs, velma_params, observed_time_series = next(iter(dataloader))
    images, time_series, xml_params, outputs, observed_time_series = (
        item.to(device) for item in (images, time_series, xml_params, outputs, observed_time_series)
    )

    # Initialize parameters
    if use_main_model and main_model is not None:
        with torch.no_grad():
            initial_params = main_model(images, observed_time_series, xml_params, outputs)
            best_params = scale_or_inverse(initial_params.detach().cpu().numpy(), scalers=scalers['param']) 
            print('Predicted parameters from main model:', best_params.mean(axis=0))
            return best_params
    else:
        initial_params = torch.tensor(
            [np.random.uniform(low, high, size=(images.size(0),)) for low, high in params_range.values()],
            device=device, dtype=torch.float32, requires_grad=True
        ).t()

    # Set up parameters for optimization
    optimal_params = initial_params.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([optimal_params], lr=lr)

    # Optimization loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted_outputs = surrogate_model(observed_time_series, optimal_params)
          
        # Align timesteps between predictions and observed data
        loss, indices = calculate_mse_loss(predicted_outputs, observed_data)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Clamp parameters to stay within specified ranges
        with torch.no_grad():
            for idx, (low, high) in enumerate(params_range.values()):
                optimal_params[:, idx] = torch.clamp(optimal_params[:, idx], min=low, max=high)

        # Print progress every 10 epochs or at the last epoch
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Obtain optimized parameters
    best_params, _ = scale_or_inverse(optimal_params.detach().cpu().numpy(), scalers=scalers['param'])  
    print("Optimized surrogate model parameters:", best_params.mean(axis=0))

    # Visualize predictions vs. observed data
    predicted_outputs, _ = scale_or_inverse(predicted_outputs.detach().cpu().numpy(), scalers=scalers['output'])
    observed_data, _ = scale_or_inverse_observed_data(observed_data, scalers=scalers['observed'])
    
    if plot_prediction:
        for i in range(predicted_outputs.shape[2]):
            plt.figure(figsize=(12, 6))
            plt.plot(predicted_outputs[0, :, i], label="Surrogate Model Predicted (Best Fit)", color='green')
            plt.scatter(indices[i], observed_data[i]['Variable'], label="Observed Data", color='blue')
            
            plt.title(f"Best Fit: Surrogate Model Prediction vs Observed Data for Variable {i + 1}")
            plt.xlabel("Time Step")
            plt.ylabel("Output Value")
            plt.legend()
            plt.show()
    
    return predicted_outputs, best_params, indices


def run_and_validate_model(best_params, params_range, base_xml_file, new_xml_path="modified_config.xml", jar_file="Velma.jar", required_columns=None, observed_data=None, scalers=None, indices=None, predicted_outputs=None, plot_comparison=True,max_memory=None):
    """
    Modify an XML configuration with optimized parameters, run a model, and compare its outputs to observed data.

    Parameters:
    - best_params: Optimized parameters as a numpy array.
    - params_range: Dictionary specifying parameter paths and ranges.
    - base_xml_file: Path to the base XML file template.
    - new_xml_path: Path to save the modified XML file.
    - jar_file: Path to the Java JAR file for running the model.
    - required_columns: List of columns required from the model output.
    - observed_data: Observed data for comparison.
    - scalers: Dictionary of scalers for inverse-transforming predictions and observed data.
    - indices: List of indices for aligning predicted outputs with observed data.
    - predicted_outputs: Predicted values from the surrogate model.
    - plot_comparison: Whether to plot the comparison between model output, predictions, and observed data.

    Returns:
    - None
    """
    
    # Combine params_range and best_params to create param_dict
    observed_end_year=max(int(df['Year'].max()) for df in observed_data if 'Year' in df.columns)
    param_dict = {path: value for path, value in zip(params_range.keys(), best_params.mean(axis=0))} | {"./calibration/VelmaInputs.properties/eyear": observed_end_year}

    # Load the XML file and modify parameters based on param_dict
    tree = ET.parse(base_xml_file)
    root = tree.getroot()
    
    # Update XML parameters according to param_dict
    for param, value in param_dict.items():
        elem = root.find(param)
        if elem is not None:
            elem.text = str(value)
        else:
            raise ValueError(f"Parameter {param} not found in XML template.")
    
    # Save the modified XML file
    tree.write(new_xml_path)
    
    # Run the Java JAR file with the modified XML configuration
    output_path = run_java_jar(jar_file, new_xml_path,max_memory=max_memory)
    
    # Load model outputs from the specified path
    outputs = load_outputs_from_paths([output_path + "/DailyResults.csv"], required_columns)
    observed_data, _ = scale_or_inverse_observed_data(observed_data, scalers=scalers['observed'])
    
    # Plot each variable from model outputs against observed data and surrogate predictions
    if plot_comparison:
        for i in range(outputs.shape[2]):
            plt.figure(figsize=(12, 6))
            plt.plot(predicted_outputs[0, :, i], label="Surrogate Model Predicted (Best Fit)", color='green')
            plt.plot(outputs[0, :, i], label="Model Output", color='red')
            plt.scatter(indices[i], observed_data[i]['Variable'], label="Observed Data", color='blue')    
            plt.title(f"Comparison: Model Output vs Observed Data for Variable {i}")
            plt.xlabel("Time Step")
            plt.ylabel("Output Value")
            plt.legend()
            plt.show()



# Main script in __main__ block to allow command line input
if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate the model for VELMA simulations")
    parser.add_argument('--epochs', type=int, default=100, help='Total number of training epochs for the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Size of each batch for training and data loading')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--load_existing', action='store_true', help='Load pre-trained models if available')
    parser.add_argument('--save_path', type=str, default="./trained_models/model", help='Directory path to save trained models')
    parser.add_argument('--jar_file', type=str, default="Velma.jar", help='Path to the VELMA simulation JAR file')
    parser.add_argument('--max_memory', type=str, default='2g', help='Maximum memory allocation for the Java process (e.g., "2g" for 2 GB)')
    parser.add_argument('--seed', type=int, default=43, help='Random seed for ensuring reproducible results')

    args = parser.parse_args()


    # Dataset parameters
    use_existing_data = True
    generate_new_data = False
    num_samples = (1, 1, 5)
    LHS_samples = 10  
    base_xml_files = ['WS10/XMLs/1.xml']
    output_dirs = ['WS10/Results']
    required_columns = [
        'Runoff_All(mm/day)_Delineated_Average',
        'NH4_Loss(gN/day/m2)_Delineated_Average']
    observed_data_path = "WS10/DataInputs/m_7_Observed/Book2.csv"
    year_range = [2010, 2011]
    params_range = {
        './calibration/VelmaCalibration.properties/psm_q': (0.01, 0.02),
        './calibration/VelmaCalibration.properties/wet_nin': (0.1, 0.2),
        './calibration/VelmaInputs.properties/f_ksl': (0.001, 0.002)
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data loader
    dataloader, observed_data, scalers = create_dataloader(
        num_samples, args.batch_size, required_columns=required_columns, 
        base_xml_files=base_xml_files, params_range=params_range, year_range=year_range,
        use_existing_data=use_existing_data, generate_new_data=generate_new_data, 
        output_dirs=output_dirs, LHS_samples=LHS_samples, observed_data_path=observed_data_path,
        max_memory=args.max_memory
    )

    # Instantiate models
    main_model = MainModel().to(device)
    surrogate_model = SurrogateModel().to(device)

    # Optimizer and loss function
    optimizer_surrogate = torch.optim.Adam(surrogate_model.parameters(), lr=args.lr)
    optimizer_main = torch.optim.Adam(main_model.parameters(), lr=args.lr)
   
    os.makedirs(args.save_path, exist_ok=True)
    
    # Optionally load pre-trained models
    if args.load_existing:
        try:
            load_model(main_model, f'{args.save_path}_main.pth')
            print("Successfully loaded main model.")
        except FileNotFoundError:
            print("Main model file not found. Skipping load.")

        try:
            load_model(surrogate_model, f'{args.save_path}_surrogate.pth')
            print("Successfully loaded surrogate model.")
        except FileNotFoundError:
            print("Surrogate model file not found. Skipping load.")

    # Train the model  
    print("Training Model...")
    train_model(
        main_model, surrogate_model, optimizer_main, optimizer_surrogate, dataloader, 
        criterion=nn.MSELoss(), train_main=True, device=device, epochs=args.epochs, 
        main_epochs=50, save_path=args.save_path, plot_loss=False
    )

    # Reload the models for evaluation and visualization
    load_model(main_model, f'{args.save_path}_main.pth')
    load_model(surrogate_model, f'{args.save_path}_surrogate.pth')

    # Optimize parameters and visualize both models
    print("Optimize Parameters and Visualizing Results...")
    predicted_outputs, best_params, indices = optimize_and_visualize(
        main_model, surrogate_model, dataloader, observed_data, scalers, params_range, 
        use_main_model=False, device=device, plot_prediction=True
    )
    
    # Run and validate model with optimized parameters
    print("Run and Validate Model with Optimize Parameters...")
    run_and_validate_model(
        best_params, params_range, base_xml_files[0], new_xml_path="modified_config.xml",  
        required_columns=required_columns, observed_data=observed_data, scalers=scalers, 
        indices=indices, predicted_outputs=predicted_outputs, plot_comparison=True, 
        max_memory=args.max_memory
    )

    


