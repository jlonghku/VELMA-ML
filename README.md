# User Manual for the VELMA Simulation Code

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

This user manual describes the structure, key functions, and usage of the code for the VELMA simulation project. It will help users understand how to adjust parameters for their own purposes and how to use the functions effectively. The code is primarily designed to train machine learning models, specifically a main model and a surrogate model, for simulating hydrological outputs and estimating parameters.

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

- ``: Reads an `.asc` file and returns its header and data.
- ``: Loads images from given sample paths, each containing multiple channels (DEM, land cover, soil parameters).
- `` and ``: Load precipitation and temperature time series data for specified years, handling both single-location and multi-location data models.

### 3.2 Data Augmentation Functions

- ``: Augments image data by applying transformations, then saves and returns paths to the augmented files.
- ``: Augments precipitation and temperature data by applying random scaling and offsets, then saves and returns paths to the files.
- ``: Combines image, time series, and parameter augmentations to generate XML files with new configurations.

### 3.3 Model Training and Evaluation Functions

- ``: Combines image, time series, and XML features to generate a watershed feature vector and predict VELMA parameters.
- ``: Used to approximate VELMA's outputs using time series data and VELMA parameters as input.
- ``: Trains both the surrogate and main models sequentially on a given data loader.
- ``: Calculates the MSE loss between predicted outputs and observed data, accounting for leap years and varying intervals.

### 3.4 Optimization and Visualization Functions

- ``: Optimizes model parameters to fit observed data using the surrogate model and visualizes predictions vs. observed data.
- ``: Modifies an XML configuration with optimized parameters, runs the VELMA model, and visualizes outputs vs. observed data.

### 3.5 XML Handling and Modification Functions

- ``: Batch updates the start and end years in a list of XML files.
- ``: Updates XML parameters and synchronizes them with an internal mapping.
- ``: Parses XML files and extracts sample information, including file paths and model settings.

### 3.6 Java Process Management Functions

- ``: Runs the VELMA Java program with optional memory limit, captures output in a log, and retrieves the output data path.

### 3.7 Utility Functions

- `` and ``: Normalize or reverse-normalize data using specified scalers.
- ``: Checks if a year is a leap year.

## 4. Adjustable Parameters in Main Function

In the `__main__` section, users can adjust the following parameters:

- ``: Total number of training epochs (default: 100).
- ``: Size of each batch for training and data loading (default: 64).
- ``: Learning rate for the optimizer (default: 1e-3).
- ``: Load pre-trained models if available (default: False).
- ``: Path to save trained models (default: `./trained_models/model`).
- ``: Path to the VELMA simulation JAR file (default: `Velma.jar`).
- ``: Maximum memory allocation for Java process (e.g., `2g` for 2 GB).
- ``: Random seed for reproducibility (default: 43).

Additionally, the following dataset parameters can be adjusted:

- ``: Boolean flag to indicate whether to use existing data.
- ``: Boolean flag to indicate whether to generate new data.
- ``: Tuple specifying the number of augmentations for images, time series, and parameters.
- ``: Number of samples to select using Latin Hypercube Sampling.
- ``: List of XML files used as templates for generating samples.
- ``: List of directories containing existing output data.
- ``: List of columns from `DailyResults.csv` required for model output.
- ``: Path to the CSV file containing observed data for comparison.
- ``: List specifying the start and end year for data filtering.
- ``: Dictionary specifying the range of values for XML parameters to be modified during augmentation.

Users can modify these parameters to change the behavior of the data generation, augmentation, and training processes.

## 5. Usage Example

Below is an example of how to run the script from the command line:

```bash
python your_script.py --epochs 200 --batch_size 32 --lr 0.0005 --load_existing --save_path "./trained_models/model" --jar_file "path/to/Velma.jar" --max_memory "4g"
```

In this example, the model will train for 200 epochs, using a batch size of 32 and a learning rate of 0.0005. Pre-trained models (if available) will be loaded, and model checkpoints will be saved in the specified path.
