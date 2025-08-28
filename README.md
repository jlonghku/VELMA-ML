# VELMA-ML Model User Manual

This manual provides instructions for preparing data, training the surrogate model, evaluating performance, and optimizing parameters using the machine learning model `SurrogateModel` within the VELMA-ML framework.

---

## 1. Requirements

- PyTorch
- NumPy
- Pandas
- Matplotlib

Install dependencies:

```bash
pip install torch numpy pandas matplotlib
```

---

## 2. Workflow Overview

The main script performs the following steps:

1. **Load and Slice Dataset**  
   Loads `dataset_velma_low.pt` and selects a sample slice.

2. **Preprocess Data**  
   Scales and splits the dataset into training and testing loaders.

3. **Instantiate Model**  
   Builds the `SurrogateModel` with sizes derived from the dataset.

4. **Train Model**  
   Trains the surrogate model with user-defined hyperparameters.

5. **Evaluate Model**  
   Compares model predictions with test data.

6. **Optimize Parameters [optional]**  
   Uses observed and climate data to optimize model parameters.

---

## 3. Data Preparation

- Input file: `dataset_velma_low.pt` (PyTorch `TensorDataset`)
  - Tensor order: `[date, climate_inputs, outputs, parameters]`
- Observed and climate CSVs for further prediction:  
  - `climate.csv`  
  - `observed.csv`

### Advanced Usage

In addition to the basic usage (train with a period of model data and then predict for longer time spans), the tool also supports **advanced strategies** depending on how data is prepared and how training is performed:

- **Residual Learning**  
  - Prepare paired outputs from high-resolution and low-resolution simulations.  
  - Compute the residual (difference) between high-resolution and low-resolution outputs.  
  - Train the surrogate model to predict this residual.  
  - During actual prediction, add the residual prediction back to the low-resolution output to reconstruct the original high-resolution result.  

- **Transfer Learning**  
  - Train the model on low-resolution data first.  
  - Then fine-tune the model with a smaller set of high-resolution data.  
  - This can be done by loading a previously trained model in the training function, or by manually freezing shared layers while retraining.  
  - The result is a surrogate model aligned with high-resolution outputs at much lower computational cost.  


---

## 4. Key Functions

### `pre_dataloader(dataset, batch_size, scaler_types, split_index)`
- Scales dataset with specified scalers (`'none'`, `'standard'`, `'minmax'`).
- Splits into training and test sets.
- Returns `train_loader`, `test_loader`, and `scalers`.

### `train_model(model, train_loader, lr, epochs, save, load, device, chunk_size)`
- Trains the surrogate model.
- Supports checkpoint saving/loading.
- Returns loss history.

### `evaluate_model(model, test_loader, scalers, device)`
- Evaluates model predictions vs. test data.
- Produces plots and saves results.

### `optimize_and_visualize(model, scalers, year_range, climate_path, observed_path, epochs, lr, required_columns, device)`
- Optimizes model parameters against observed data.
- Visualizes predictions vs. observations.
- Returns best predicted outputs and parameters.

---

## 5. Example Run

```python
# preprocess
train_loader, test_loader, scalers = pre_dataloader(
    dataset, batch_size=32,
    scaler_types=['none','none','standard','minmax'],
    split_index=0.5
)

# train
train_model(model, train_loader, lr=1e-4, epochs=10,
            save="./trained_models/base_model.pth",
            device=device, chunk_size=32)

# evaluate
evaluate_model(model, test_loader, scalers, device=device)

# optimize
predicted_outputs, best_params = optimize_and_visualize(
    model, scalers, [2010, 2019],
    "climate.csv", "observed.csv",
    epochs=5, lr=0.001,
    required_columns=[
        'Runoff_All(mm/day)_Delineated_Average',
        'NO3_Loss(gN/day/m2)_Delineated_Average',
    ],
    device=device
)
```

---

## 6. Output

- **Model checkpoints**: `./trained_models/base_model.pth`
- **Evaluation plots**: Predictions vs. Observations
- **CSV**: `output/predictions_vs_targets.csv`
- **Optimized parameters**: printed in console

---

## 7. Notes

- Ensure observed and climate CSV files cover the `year_range`.
- Adjust `scaler_types` according to data features:
  - `'none'` for dates/indices
  - `'standard'` for outputs
  - `'minmax'` for parameters
- GPU training is used if CUDA is available.

---
