# Guide for the VELMA-ML

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Code Structure Overview](#2-code-structure-overview)
- [3. Main Function Workflow](#3-main-function-workflow)
- [4. Key Functions](#4-key-functions)
  - [4.1 Data Loading and Processing Functions](#41-data-loading-and-processing-functions)
  - [4.2 Data Augmentation Functions](#42-data-augmentation-functions)
  - [4.3 Model Training and Evaluation Functions](#43-model-training-and-evaluation-functions)
  - [4.4 Optimization and Visualization Functions](#44-optimization-and-visualization-functions)
  - [4.5 XML Handling and Modification Functions](#45-xml-handling-and-modification-functions)
  - [4.6 Java Process Management Functions](#46-java-process-management-functions)
  - [4.7 Utility Functions](#47-utility-functions)
- [5. Adjustable Parameters in Main Function](#5-adjustable-parameters-in-main-function)
- [6. Usage Example](#6-usage-example)

## 1. Introduction

This guide describes the structure, key functions, and usage of the code for the VELMA-ML project. It will help users understand how to adjust parameters for their own purposes and how to use the functions effectively. The code is primarily designed to train machine learning models, specifically a main model and a surrogate model, for simulating hydrological outputs and estimating parameters.

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

## 3. Main Function Workflow

The main function of the script follows the workflow below:

1. **Parse Command-Line Arguments**: The script starts by parsing command-line arguments, which determine various settings such as the number of training epochs, batch size, learning rate, etc.

2. **Set Dataset Parameters**: Next, the dataset parameters are set, either from default values or based on user inputs. These include flags for using existing data or generating new data, specifying XML files, directories, required columns, and parameter ranges.

3. **Set Random Seed for Reproducibility**: A random seed is set to ensure reproducible results, both for CPU and GPU computations.

4. **Select Device for Computation**: The script selects a computation device (CPU or GPU) depending on availability.

5. **Create DataLoader**: The create_dataloader function is called to load and preprocess the data, including augmentation and normalization. This step returns the data loader, observed data, and scalers for normalization.

6. **Initialize Models**: Both the MainModel and SurrogateModel are instantiated. These models are designed to predict VELMA parameters and approximate VELMA's outputs, respectively.

7. **Set Up Optimizers**: Optimizers for both models are created, using the learning rate specified in the command-line arguments.

8. **Load Pre-trained Models (Optional)**: If specified by the user, pre-trained models are loaded from the designated save path.

9. **Train Models**: The train_model function is called to train both the main and surrogate models. The surrogate model is trained first, followed by the main model, using the training data provided by the data loader.

10. **Optimize Parameters**: Once training is complete, the optimize_and_visualize function is used to optimize model parameters to fit the observed data. This involves adjusting the surrogate model's parameters to minimize the error between predicted outputs and observed data.

11. **Run and Validate Model**: Finally, the run_and_validate_model function modifies the XML configuration with optimized parameters, runs the VELMA model using the Java JAR file, and compares the model outputs to observed data.

## 4. Key Functions

### 4.1 Data Loading and Processing Functions

- **load_asc_file**: Reads an .asc file and returns its header and data.
- **load_images_from_samples**: Loads images from given sample paths, each containing multiple channels (DEM, land cover, soil parameters).
- **load_time_series_from_path** and **load_time_series_from_paths**: Load precipitation and temperature time series data for specified years, handling both single-location and multi-location data models.

### 4.2 Data Augmentation Functions

- **augment_and_save_images**: Augments image data by applying transformations, then saves and returns paths to the augmented files.
- **augment_and_save_time_series**: Augments precipitation and temperature data by applying random scaling and offsets, then saves and returns paths to the files.
- **augment_samples**: Combines image, time series, and parameter augmentations to generate XML files with new configurations.

### 4.3 Model Training and Evaluation Functions

- **MainModel**: Combines image, time series, and XML features to generate a watershed feature vector and predict VELMA parameters.
- **SurrogateModel**: Used to approximate VELMA's outputs using time series data and VELMA parameters as input.
- **train_model**: Trains both the surrogate and main models sequentially on a given data loader.
- **calculate_mse_loss**: Calculates the MSE loss between predicted outputs and observed data, accounting for leap years and varying intervals.

### 4.4 Optimization and Visualization Functions

- **optimize_and_visualize**: Optimizes model parameters to fit observed data using the surrogate model and visualizes predictions vs. observed data.
- **run_and_validate_model**: Modifies an XML configuration with optimized parameters, runs the VELMA model, and visualizes outputs vs. observed data.

### 4.5 XML Handling and Modification Functions

- **modify_years**: Batch updates the start and end years in a list of XML files.
- **update_param**: Updates XML parameters and synchronizes them with an internal mapping.
- **get_samples**: Parses XML files and extracts sample information, including file paths and model settings.

### 4.6 Java Process Management Functions

- **run_java_jar**: Runs the VELMA Java program with optional memory limit, captures output in a log, and retrieves the output data path.

### 4.7 Utility Functions

- **scale_or_inverse** and **scale_or_inverse_observed_data**: Normalize or reverse-normalize data using specified scalers.
- **is_leap_year**: Checks if a year is a leap year.

## 5. Adjustable Parameters in Main Function

In the __main__ section, users can adjust the following parameters:

- **--epochs**: Total number of training epochs (default: 100).
- **--batch_size**: Size of each batch for training and data loading (default: 64).
- **--lr**: Learning rate for the optimizer (default: 1e-3).
- **--load_existing**: Load pre-trained models if available (default: False).
- **--save_path**: Path to save trained models (default: ./trained_models/model).
- **--jar_file**: Path to the VELMA simulation JAR file (default: Velma.jar).
- **--max_memory**: Maximum memory allocation for Java process (e.g., 2g for 2 GB).
- **--seed**: Random seed for reproducibility (default: 43).

Additionally, the following dataset parameters can be adjusted:

- **use_existing_data**: Boolean flag to indicate whether to use existing data.
- **generate_new_data**: Boolean flag to indicate whether to generate new data.
- **num_samples**: Tuple specifying the number of augmentations for images, time series, and parameters.
- **LHS_samples**: Number of samples to select using Latin Hypercube Sampling.
- **base_xml_files**: List of XML files used as templates for generating samples.
- **output_dirs**: List of directories containing existing output data.
- **required_columns**: List of columns from DailyResults.csv required for model output.
- **observed_data_path**: Path to the CSV file containing observed data for comparison. The CSV file should be formatted with columns in the following order: ['Year', 'Jday', 'Variable']. Specifically:
  - Year: Represents the calendar year of each observation (e.g., 2020, 2021). This helps to track the temporal sequence across multiple years.
  - Jday: Stands for Julian day, indicating the specific day of the year (from 1 to 365 or 366 for leap years). This allows for daily granularity in tracking observations.
  - Variable: Represents the observed values of interest, such as runoff, soil moisture, or other environmental measurements. This column contains the data points for the specific parameter being analyzed, ensuring a consistent format across all observations.
- **year_range**: List specifying the start and end year for data filtering.
- **params_range**: Dictionary specifying the range of values for XML parameters to be modified during augmentation.

Users can modify these parameters to change the behavior of the data generation, augmentation, and training processes.

## 6. Usage Example

Below is an example of how to run the script from the command line:

```bash
python your_script.py --epochs 200 --batch_size 32 --lr 0.0005 --load_existing --save_path "./trained_models/model" --jar_file "path/to/Velma.jar" --max_memory "4g"
