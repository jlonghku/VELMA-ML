from velma_ml import *
from models import SurrogateModel
import torch, time, warnings, os
warnings.filterwarnings("ignore", category=FutureWarning)
torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])

if __name__ == "__main__":
    # seed & device
    seed = 42
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.manual_seed(seed)

    t0 = time.time()

    # load & slice
    dataset = torch.load('dataset_velma_low.pt')
    sample = tuple(t[37:38] for t in dataset.tensors)  # pick one sample
    dataset = TensorDataset(*sample)

    # preprocess -> loaders + scalers
    train_loader, test_loader, scalers = pre_dataloader(
        dataset, batch_size=32,
        scaler_types=['none','none','standard','minmax'],
        split_index=0.5
    )

    # model
    param_size  = train_loader.dataset.tensors[3].shape[-1]
    output_size = train_loader.dataset.tensors[2].shape[-1]
    model = SurrogateModel(param_size=param_size, output_size=output_size).to(device)

    # train
    print("Training Model...")
    save_path = "./trained_models/base_model.pth"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    train_model(model, train_loader, lr=1e-4, epochs=10, save=save_path, load=save_path, device=device, chunk_size=32)

    # eval
    print("Evaluating Model...")
    evaluate_model(model, test_loader, scalers, device=device)

    # optimize + visualize
    print("Optimize Parameters and Visualizing Results...")
    year_range = [2010, 2019]
    required_columns = [
        'Runoff_All(mm/day)_Delineated_Average',
        'NO3_Loss(gN/day/m2)_Delineated_Average',
    ]
    predicted_outputs, best_params = optimize_and_visualize(
        model, scalers, year_range, "climate.csv", "observed.csv",
        epochs=5, lr=0.001, required_columns=required_columns, device=device
    )

    print(f"Total time: {time.time() - t0:.2f}s")
