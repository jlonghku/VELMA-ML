import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, os


# pick best prediction index (sum of non-negative scores across features)
def select_best_prediction(observed, predicted, metric_fn):
    all_r2 = np.array([[metric_fn(observed[0, :, i], predicted[j, :, i])
                        for i in range(predicted.shape[2])]
                       for j in range(predicted.shape[0])])
    best_j = np.argmax(np.sum(np.maximum(all_r2, 0), axis=1))
    return best_j, all_r2


# quick plot per feature
def plot_prediction_vs_observed(observed, predicted, best_j, columns, all_r2):
    for i in range(predicted.shape[2]):
        obs = observed[0, :, i]
        pred = predicted[best_j, :, i]
        mask = ~np.isnan(obs)
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        s, e = idx[0], idx[-1]
        obs_crop = obs[s:e + 1]
        pred_crop = pred[s:e + 1]

        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(obs_crop)), obs_crop, label="Observed", color='blue')
        plt.plot(pred_crop, label="Predicted", color='green')
        plt.xlabel(f"Time (Score: {all_r2[best_j, i]:.3f})", fontsize=18, fontweight='bold')
        plt.ylabel(columns[i], fontsize=18, fontweight='bold')
        plt.tick_params(axis='both', labelsize=16)
        plt.legend()
        plt.tight_layout()
        os.makedirs("output", exist_ok=True)
        plt.savefig(f"output/plot_{i}.png", dpi=300, bbox_inches='tight')
        plt.show()


# KGE (handles NaN)
def calculate_kge(sim, obs):
    sim, obs = np.asarray(sim), np.asarray(obs)
    m = ~np.isnan(sim) & ~np.isnan(obs)
    sim, obs = sim[m], obs[m]
    if sim.size == 0:
        return np.nan
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-12)
    beta = (np.mean(sim) + 1e-12) / (np.mean(obs) + 1e-12)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


# MSE with NaN mask (mean over features)
def mse_loss_with_mask(pred, target):
    assert pred.shape == target.shape, "Shape mismatch"
    mask = ~torch.isnan(target)
    diff = torch.where(mask, pred - target, torch.zeros(1, device=pred.device, dtype=pred.dtype))
    mse_sum = (diff ** 2).sum(dim=(0, 1))
    cnt = mask.sum(dim=(0, 1)).clamp_min(1)
    return (mse_sum / cnt).mean()


# load CSV [datetime index, selected columns] -> tensor + mask
def load_and_fill_observed(file_path, start_year, end_year, required_columns=None):
    df = pd.read_csv(file_path, parse_dates=[0], index_col=0)
    cols = df.columns if required_columns is None else required_columns
    df = df.loc[f"{start_year}-01-01":f"{end_year}-12-31", cols]
    x = torch.from_numpy(df.values.astype(np.float32))
    return x, ~torch.isnan(x), cols


class TorchScaler:
    def __init__(self, scaler_type='standard', device=None):
        valid = {'standard', 'minmax', 'none'}
        if scaler_type not in valid:
            raise ValueError(f"Unsupported scaler_type: {scaler_type}. Choose from {valid}")
        self.scaler_type = scaler_type
        self.device = torch.device(device or 'cpu')
        self.mean = self.std = self.min = self.max = None

    def fit(self, data):
        if self.scaler_type == 'none':
            return self
        data = data.to(self.device)
        if data.dim() <= 1:
            data = data.view(-1, 1)
        mask = ~torch.isnan(data)
        if not mask.any():
            raise ValueError("No valid data points (all NaN)")
        valid = data[mask].reshape(-1, data.shape[-1])
        if valid.shape[0] == 1:
            self.scaler_type = 'none'
            return self

        if self.scaler_type == 'standard':
            self.mean = valid.mean(0, keepdim=True)
            self.std = valid.std(0, keepdim=True)
            self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
        elif self.scaler_type == 'minmax':
            self.min = valid.min(0, keepdim=True)[0]
            self.max = valid.max(0, keepdim=True)[0]
            self.max = torch.where(self.max == self.min, self.min + 1e-8, self.max)
        return self

    def transform(self, data):
        if self.scaler_type == 'none':
            return data
        orig_shape, orig_device = data.shape, data.device
        data = data.to(self.device)
        if data.dim() <= 1:
            data = data.view(-1, 1)

        if self.scaler_type == 'standard':
            out = (data - self.mean) / self.std
        elif self.scaler_type == 'minmax':
            out = (data - self.min) / (self.max - self.min)

        return out.reshape(orig_shape).to(orig_device)

    def inverse_transform(self, data):
        if self.scaler_type == 'none':
            return data
        orig_shape, orig_device = data.shape, data.device
        data = data.to(self.device)
        if data.dim() <= 1:
            data = data.view(-1, 1)

        if self.scaler_type == 'standard':
            out = data * self.std + self.mean
        elif self.scaler_type == 'minmax':
            out = data * (self.max - self.min) + self.min

        return out.reshape(orig_shape).to(orig_device)


class DatasetScalerSplitter:
    def __init__(self, scaler_types, split_mode='time', split_index=0.8, device=None):
        if not isinstance(scaler_types, (list, tuple)):
            scaler_types = [scaler_types]
        self.scalers = [TorchScaler(s, device) for s in scaler_types]
        if split_mode not in {'time', 'sample'}:
            raise ValueError("split_mode must be 'time' or 'sample'")
        self.split_mode, self.split_index = split_mode, split_index
        self.data_scaled = None

    def fit_transform(self, data):
        if isinstance(data, torch.utils.data.TensorDataset):
            tensors = data.tensors
        elif isinstance(data, (list, tuple)):
            tensors = data
        else:
            tensors = [data]
        if len(self.scalers) != len(tensors):
            raise ValueError(f"#scalers ({len(self.scalers)}) != #tensors ({len(tensors)})")

        self.data_scaled = tuple(s.fit(t).transform(t) for s, t in zip(self.scalers, tensors))
        return self.data_scaled

    def inverse_transform(self, scaled_tuple):
        if not isinstance(scaled_tuple, (list, tuple)):
            scaled_tuple = [scaled_tuple]
        if len(self.scalers) != len(scaled_tuple):
            raise ValueError(f"#scalers ({len(self.scalers)}) != #tensors ({len(scaled_tuple)})")
        return tuple(s.inverse_transform(t) for s, t in zip(self.scalers, scaled_tuple))

    def split(self, data=None):
        data = data or self.data_scaled
        if data is None:
            raise ValueError("No data to split. Call fit_transform first.")
        train, test = [], []
        for t in data:
            if self.split_mode == 'time':
                if t.dim() != 3 or self.split_index in (0, 1):
                    train.append(t); test.append(t); continue
                idx = int(t.shape[1] * self.split_index) if isinstance(self.split_index, float) else self.split_index
                train.append(t[:, :idx]); test.append(t[:, idx:])
            else:  # 'sample'
                idx = int(t.shape[0] * self.split_index) if isinstance(self.split_index, float) else self.split_index
                train.append(t[:idx]); test.append(t[idx:])
        return tuple(train), tuple(test)

