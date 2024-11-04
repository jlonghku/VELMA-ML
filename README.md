# User Manual for the VELMA-ML Code

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Code Structure Overview](#2-code-structure-overview)
- [3. Key Functions](#3-key-functions)
  - [3.1 Data Loading and Processing Functions](#31-data-loading-and-processing-functions)
  - [3.2 Data Augmentation Functions](#32-data-augmentation-functions)
  - [3.3 Model Training and Evaluation Functions](#33-model-training-and-evaluation-functions)
  - [3.4 Optimization and Visualization Functions](#34-optimization-and-visualization-functions)
  - [3.5 XML Handling and Modification Functions](#35-xml-handling-and-modification-functions)
  - [3.6 Java Process Management Functions](#36-java-process-management-functions)
  - [3.7 Utility Functions](#37-utility-functions)
- [4. Adjustable Parameters in Main Function](#4-adjustable-parameters-in-main-function)
- [5. Usage Example](#5-usage-example)

## 1. Introduction

This user manual describes the structure, key functions, and usage of the code for the VELMA-ML project. It will help users understand how to adjust parameters for their own purposes and how to use the functions effectively. The code is primarily designed to train machine learning models, specifically a main model and a surrogate model, for simulating hydrological outputs and estimating parameters.

## 2. Code Structure Overview

The code consists of several Python functions and classes, primarily focused on:

- Loading various types of input data (raster images, time series, XML parameters).
- Augmenting data for model training.
- Creating models for image, time series, and parameter prediction.
- Training the models, evaluating their performance, and optimizing parameters.

Key components:

- Data Loading Functions
- Image and Time Series Feature Extractors
- Model Definition (Main Model and Surrogate Model)
- Training and Evaluation Functions

## 3. Key Functions

### 3.1 Data Loading and Processing Functions

- **`load_asc_file`**: Reads an `.asc` file and returns its header and data.
- **`load_images_from_samples`**: Loads images from given sample paths, each containing multiple channels (DEM, land cover, soil parameters).
- **`load_time_series_from_path`** and **`load_time_series_from_paths`**: Load precipitation and temperature time series data for specified years, handling both single-location and multi-location data models.

### 3.2 Data Augmentation Functions

- **`augment_and_save_images`**: Augments image data by applying transformations, then saves and returns paths to the augmented files.
- **`augment_and_save_time_series`**: Augments precipitation and temperature data by applying random scaling and offsets, then saves and returns paths to the files.
- **`augment_samples`**: Combines image, time series, and parameter augmentations to generate XML files with new configurations.

### 3.3 Model Training and Evaluation Functions

- **`MainModel`**: Combines image, time series, and XML features to generate a watershed feature vector and predict VELMA parameters.
- **`SurrogateModel`**: Used to approximate VELMA's outputs using time series data and VELMA parameters as input.
- **`train_model`**: Trains both the surrogate and main models sequentially on a given data loader.
- **`calculate_mse_loss`**: Calculates the MSE loss between predicted outputs and observed data, accounting for leap years and varying intervals.

### 3.4 Optimization and Visualization Functions

- **`optimize_and_visualize`**: Optimizes model parameters to fit observed data using the surrogate model and visualizes predictions vs. observed data.
- **`run_and_validate_model`**: Modifies an XML configuration with optimized parameters, runs the VELMA model, and visualizes outputs vs. observed data.

### 3.5 XML Handling and Modification Functions

- **`modify_years`**: Batch updates the start and end years in a list of XML files.
- **`update_param`**: Updates XML parameters and synchronizes them with an internal mapping.
- **`get_samples`**: Parses XML files and extracts sample information, including file paths and model settings.

### 3.6 Java Process Management Functions

- **`run_java_jar`**: Runs the VELMA Java program with optional memory limit, captures output in a log, and retrieves the output data path.

### 3.7 Utility Functions

- **`scale_or_inverse`** and **`scale_or_inverse_observed_data`**: Normalize or reverse-normalize data using specified scalers.
- **`is_leap_year`**: Checks if a year is a leap year.

## 4. Adjustable Parameters in Main Function

In the `__main__` section, users can adjust the following parameters:

- **`--epochs`**: Total number of training epochs (default: 100).
- **`--batch_size`**: Size of each batch for training and data loading (default: 64).
- **`--lr`**: Learning rate for the optimizer (default: 1e-3).
- **`--load_existing`**: Load pre-trained models if available (default: False).
- **`--save_path`**: Path to save trained models (default: `./trained_models/model`).
- **`--jar_file`**: Path to the VELMA simulation JAR file (default: `Velma.jar`).
- **`--max_memory`**: Maximum memory allocation for Java process (e.g., `2g` for 2 GB).
- **`--seed`**: Random seed for reproducibility (default: 43).

Additionally, the following dataset parameters can be adjusted:

- **`use_existing_data`**: Boolean flag to indicate whether to use existing data.
- **`generate_new_data`**: Boolean flag to indicate whether to generate new data.
- **`num_samples`**: Tuple specifying the number of augmentations for images, time series, and parameters.
- **`LHS_samples`**: Number of samples to select using Latin Hypercube Sampling.
- **`base_xml_files`**: List of XML files used as templates for generating samples.
- **`output_dirs`**: List of directories containing existing output data.
- **`required_columns`**: List of columns from `DailyResults.csv` required for model output.
- **`observed_data_path`**: Path to the CSV file containing observed data for comparison. The CSV file should be formatted with columns in the following order: `['Year', 'Jday', 'Variable']`. Specifically:
  - `Year`: Represents the calendar year of each observation (e.g., 2020, 2021). This helps to track the temporal sequence across multiple years.
  - `Jday`: Stands for Julian day, indicating the specific day of the year (from 1 to 365 or 366 for leap years). This allows for daily granularity in tracking observations.
  - `Variable`: Represents the observed values of interest, such as runoff, soil moisture, or other environmental measurements. This column contains the data points for the specific parameter being analyzed, ensuring a consistent format across all observations.
- **`year_range`**: List specifying the start and end year for data filtering.
- **`params_range`**: Dictionary specifying the range of values for XML parameters to be modified during augmentation.

Users can modify these parameters to change the behavior of the data generation, augmentation, and training processes.

## 5. Usage Example

Below is an example of how to run the script from the command line:

```bash
python your_script.py --epochs 200 --batch_size 32 --lr 0.0005 --load_existing --save_path "./trained_models/model" --jar_file "path/to/Velma.jar" --max_memory "4g"
```

In this example, the model will train for 200 epochs, using a batch size of 32 and a learning rate of 0.0005. Pre-trained models (if available) will be loaded, and model checkpoints will be saved in the specified path.

