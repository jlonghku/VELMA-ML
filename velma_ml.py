import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from utils import *  

# Scale -> split -> build DataLoaders
def pre_dataloader(dataset, batch_size, scaler_types=('none',), split_mode='time', split_index=0.5, shuffle_train=True, device=None):
    scaler = DatasetScalerSplitter(scaler_types, split_mode=split_mode, split_index=split_index, device=device)
    scaled = scaler.fit_transform(dataset)
    train_set, test_set = scaler.split(scaled)
    train_loader = DataLoader(TensorDataset(*train_set), batch_size=batch_size, shuffle=shuffle_train)
    test_loader  = DataLoader(TensorDataset(*test_set),  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, scaler.scalers  # return the list, not the wrapper


# Load model weights if shape matches
def load_model(model, path, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(path):
        print(f"Warning: weights file not found: {path}")
        return model
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning loading weights: {e}")
    return model


# Train loop with optional chunking and grad clipping
def train_model(model, train_data, lr=1e-3, epochs=50, save=None, load=None, grad_clip_value=None, device='cuda', plot_loss=True, chunk_size=None):
    def chunk_tensor(t, chunk, overlap=5):
        B, T, F_ = t.shape
        step = chunk - overlap
        return [t[:, i:i+chunk, :] for i in range(0, T - overlap, step) if i + chunk <= T]

    if load:
        model = load_model(model, load, device=device)

    model.train()
    losses, optim_student = [], torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_data:
            date, clim, outputs, params = (x.to(device) for x in batch)
            clim_chunks   = chunk_tensor(clim, chunk_size)   if chunk_size else [clim]
            output_chunks = chunk_tensor(outputs, chunk_size) if chunk_size else [outputs]

            for c_chunk, o_chunk in zip(clim_chunks, output_chunks):
                optim_student.zero_grad()
                pred = model(c_chunk, params)
                loss = F.mse_loss(pred, o_chunk)
                loss.backward()
                if grad_clip_value:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                optim_student.step()
                epoch_loss += loss.item()

        avg = epoch_loss / max(1, len(train_data))
        losses.append(avg)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg:.4f}")

        if save and ((epoch + 1) % 10 == 0):
            os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
            torch.save(model.state_dict(), save)

    if plot_loss:
        plt.figure()
        plt.plot(range(1, epochs + 1), losses, label='Training Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    return losses


# Evaluate and inverse-transform predictions/targets for plotting and CSV
def evaluate_model(model, test_data, scalers, device='cuda'):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for date, clim, outputs, params in test_data:
            clim, outputs, params = clim.to(device), outputs.to(device), params.to(device)
            pred = model(clim, params).cpu()
            all_preds.append(pred)
            all_targets.append(outputs.cpu())

    preds   = torch.cat(all_preds, dim=0)     # [B, T, F]
    targets = torch.cat(all_targets, dim=0)   # [B, T, F]

    # index 2 assumed to be output scaler
    preds_np   = scalers[2].inverse_transform(preds[0]).numpy()
    targets_np = scalers[2].inverse_transform(targets[0]).numpy()

    for i in range(preds.size(2)):
        plt.figure(figsize=(10, 4))
        plt.plot(targets_np[:, i], label='True')
        plt.plot(preds_np[:, i],   label='Predicted')
        plt.title(f'Output Feature {i}')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    combined = np.concatenate([targets_np, preds_np], axis=1)
    cols = [f"target_{i}" for i in range(preds.size(2))] + [f"pred_{i}" for i in range(preds.size(2))]
    os.makedirs("output", exist_ok=True)
    pd.DataFrame(combined, columns=cols).to_csv("output/predictions_vs_targets.csv", index=False)
    print("Saved to predictions_vs_targets.csv")


# Optimize params against observed series, visualize, and return best
def optimize_and_visualize(model, scalers, year_range, climate_path, observed_path,
                           epochs=5, lr=0.001, device='cuda', required_columns=None, plot_prediction=True):

    model.train()
    for p in model.parameters():
        p.requires_grad = False

    climate_tensor, _, _  = load_and_fill_observed(climate_path, year_range[0], year_range[1])
    observed_tensor, _, _ = load_and_fill_observed(observed_path, year_range[0], year_range[1], required_columns)

    climate_tensor  = climate_tensor.unsqueeze(0).to(device)
    observed_tensor = scalers[2].transform(observed_tensor.unsqueeze(0)).to(device)

    optimal_params = torch.rand((1, model.param_size), device=device, requires_grad=True)
    optimizer = optim.Adam([optimal_params], lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(climate_tensor, optimal_params)
        loss = mse_loss_with_mask(pred, observed_tensor).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([optimal_params], 0.5)
        optimizer.step()
        with torch.no_grad():
            optimal_params.clamp_(0.0, 1.0)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}")

    best_params = scalers[3].inverse_transform(optimal_params)
    observed_np  = scalers[2].inverse_transform(observed_tensor).detach().cpu().numpy()
    predicted_np = scalers[2].inverse_transform(pred).detach().cpu().numpy()

    if plot_prediction:
        best_j, all_r2 = select_best_prediction(observed_np, predicted_np, metric_fn=calculate_kge)
        plot_prediction_vs_observed(observed_np, predicted_np, best_j, required_columns, all_r2)
        print("Optimized surrogate model parameters:", best_params[best_j])

    return predicted_np[best_j, :, 0], best_params[best_j].detach().cpu().numpy()
