# **Model Usage Manual**

This manual provides a detailed explanation of the model architecture, the custom functions defined in the code, and instructions for usage. The main focus is on understanding how to adjust parameters, run the code, and customize it for your purposes.

---

## **Table of Contents**

1. [Model Architecture](#model-architecture)
   - [MainModel](#mainmodel)
     - [Image Feature Extraction](#image-feature-extraction-cnn)
     - [Time Series Feature Extraction](#time-series-feature-extraction-lstm)
     - [XML Feature Extraction](#xml-feature-extraction)
     - [Fully Connected Layers](#fully-connected-layers)
     - [VELMA Parameter Prediction](#velma-parameter-prediction)
   - [SurrogateModel](#surrogatemodel)
     - [LSTM for Time Series](#lstm-for-time-series)
     - [Fully Connected Layers](#fully-connected-layers-surrogatemodel)
2. [Custom Functions](#custom-functions)
   - [load_asc_file](#load_asc_file)
   - [load_images_from_paths](#load_images_from_paths)
   - [augment_images](#augment_images)
   - [load_spatial_model_data](#load_spatial_model_data)
   - [parse_xml_for_params](#parse_xml_for_params)
   - [run_java_jar](#run_java_jar)
3. [Main Function Parameters](#main-function-parameters)
4. [Usage Instructions](#usage-instructions)

---

## **Model Architecture**

The code contains two primary models: **MainModel** and **SurrogateModel**. These models work together to predict the VELMA parameters and approximate the output of the VELMA hydrological model. Below is a detailed breakdown of the architecture.

### **MainModel**

The `MainModel` is responsible for predicting VELMA parameters using three different inputs: images, time series, and XML parameters. Each type of input is processed separately by feature extraction networks, which are then combined to predict the desired parameters.

#### **1. Image Feature Extraction (CNN)**

- **Function**: `MultiLayerImageFeatureExtractor`
- **Purpose**: Extracts features from raster images (e.g., elevation, soil type) using a multi-layer Convolutional Neural Network (CNN).
- **Key Functions**:
  - `nn.Conv2d`: Performs 2D convolution to detect spatial patterns in the image.
  - `nn.ReLU`: Introduces non-linearity.
  - `nn.MaxPool2d`: Reduces spatial dimensions.
  - `nn.AdaptiveAvgPool2d`: Ensures the final output feature map has a consistent size.
  - `nn.Linear`: Flattens and maps the image features into a fixed-size vector.

#### **2. Time Series Feature Extraction (LSTM)**

- **Function**: `TimeSeriesFeatureExtractor`
- **Purpose**: Processes time series data (e.g., rainfall, temperature) using a Long Short-Term Memory (LSTM) network to capture temporal dependencies.
- **Key Functions**:
  - `nn.LSTM`: Captures short- and long-term dependencies in sequential data.
  - `nn.LayerNorm`: Normalizes the LSTM output.
  - `nn.Linear`: Maps the LSTM output to a fixed-size feature vector.

#### **3. XML Feature Extraction**

- **Function**: `nn.Linear` (within `MainModel`)
- **Purpose**: Extracts features from XML parameters by passing them through a fully connected layer.
- **Key Functions**:
  - `nn.Linear`: Maps the raw XML parameters to a feature vector.

#### **4. Fully Connected Layers**

- **Function**: `nn.Sequential` (within `MainModel`)
- **Purpose**: Combines the feature vectors extracted from the image, time series, and XML inputs into a single vector. This vector is then passed through a series of fully connected layers to predict the VELMA parameters.
- **Key Functions**:
  - `nn.Linear`: Performs linear transformations.
  - `nn.ReLU`: Introduces non-linearity.
  - `nn.Dropout`: Prevents overfitting by randomly deactivating neurons during training.

#### **5. VELMA Parameter Prediction**

- **Function**: `_build_specific_branch`
- **Purpose**: Predicts the VELMA parameters using a fully connected network with multiple layers.
- **Key Functions**:
  - `nn.Linear`: Predicts the VELMA parameters.
  - `torch.sigmoid`: Ensures the predicted parameters are within a normalized range (0-1), which are later scaled to the required ranges for the specific parameters.

---

### **SurrogateModel**

The `SurrogateModel` simulates the behavior of the VELMA hydrological model by approximating its output. It takes the predicted VELMA parameters and time series inputs to predict hydrological variables such as runoff, soil moisture, and other outputs.

#### **1. LSTM for Time Series**

- **Function**: `nn.LSTM`
- **Purpose**: Processes the input time series data (rainfall and temperature) and captures the temporal dependencies.
- **Key Functions**:
  - `nn.LSTM`: Captures temporal dependencies in the time series data.

#### **2. Fully Connected Layers (SurrogateModel)**

- **Function**: `nn.Sequential`
- **Purpose**: After the time series and predicted VELMA parameters are combined, the concatenated features are passed through fully connected layers to generate the model's output.
- **Key Functions**:
  - `nn.Linear`: Transforms the concatenated feature vector into the output predictions.
  - `nn.ReLU`: Introduces non-linearity.

---

## **Custom Functions**

### **load_asc_file**
- **Purpose**: Dynamically parses the header of `.asc` files and returns the raster data as a NumPy array.
- **How It Works**: This function reads the header information, processes metadata (e.g., number of rows/columns, no-data values), and loads the raster data into memory.

### **get_latest_asc_file**
- **Purpose**: Retrieves the latest `.asc` file from a specified folder.
- **How It Works**: Searches through a folder for `.asc` files and returns the file with the latest modification time.

### **load_images_from_paths**
- **Purpose**: Loads image data from multiple sample paths, each containing multiple subfolders with `.asc` files for different image channels (e.g., DEM, soil coverage, soils).
- **How It Works**: This function loads the `.asc` files, checks that all channels have matching shapes, and stacks them into a multi-channel image tensor.

### **augment_images**
- **Purpose**: Augments image datasets by adding noise or duplicating images to increase the number of samples.
- **How It Works**: Depending on the user's preference, Gaussian noise is added to the images, or the samples are simply duplicated to increase the dataset size.

### **load_spatial_model_data**
- **Purpose**: Loads spatial model data from CSV files (e.g., Site files) and calculates the average precipitation and temperature time series.
- **How It Works**: This function parses CSV files to extract time series data (precipitation and temperature) and returns the average values across all sites.

### **parse_xml_for_params**
- **Purpose**: Parses an XML file to extract numerical and boolean parameters used for the VELMA model.
- **How It Works**: This function recursively traverses the XML tree, retrieves relevant parameters, and returns them as NumPy arrays.

### **run_java_jar**
- **Purpose**: Runs the VELMA Java program and returns the output data path and log file path.
- **How It Works**: This function executes a Java command-line program (`VelmaSimulatorCmdLine`) and captures the output path from the generated log file.

---

## **Main Function Parameters**

The main function supports several command-line arguments that control the model’s behavior. Below is a breakdown of the key parameters:

- **`--epochs`** (default: 100):  
  Number of training epochs. Increasing this value will train the model for more iterations but will increase training time.

- **`--batch_size`** (default: 32):  
  Number of samples to process in each batch during training.

- **`--lr`** (default: `1e-3`):  
  Learning rate for the optimizer. Lower values slow down training but can lead to better convergence.

- **`--save_path`** (default: `"./trained_models/model"`):  
  Directory where the trained models will be saved.

- **`--num_conv_layers`** (default: 4):  
  Number of convolutional layers used in the image feature extraction network.

- **`--visualize_samples`** (default: 5):  
  Number of samples to visualize during evaluation.

- **`--seed`** (default: 43):  
  Random seed for reproducibility. Using the same seed ensures that results are consistent across different runs.

---

## **Usage Instructions**

### **Setup**

1. **Prepare Dependencies**:  
   Ensure that all necessary libraries such as `torch`, `numpy`, and `pandas` are installed. You also need to have access to the `velma.jar` file, which should be placed in the same directory as the code.

2. **Run the Main Script**:  
   The main script is designed to be run from the command line. You can use the following command to start training:
   ```bash
   python main.py --epochs 100 --batch_size 32 --lr 0.001 --save_path "./trained_models/model"
