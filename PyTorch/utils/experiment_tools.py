import os
import random 
import shutil
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from pathlib import Path
from typing import Iterator
import torch.optim as optim
import utils.losses as losses
import torchmetrics
from sklearn.model_selection import train_test_split
from utils.data import data_generator, load_npz
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.gol import DiffGoL

def set_random_seed(seed:int):
    """
    Fix the random number generetor seed to ensure reproducibility.

    Arguments:
        seed (int): random number generator seed.
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
def remove_directory(path:Path):
    """
    Removes the directory. If the directory does not exist, a warning
    message is printed.

    Args:
        path (Path): path of the directory to be removed.
    
    Warns:
        FileNotFoundErrors in case the path of the directory to be removed 
    is not found.
    """
    if not isinstance(path, Path):
        path = Path(path)

    try:
        shutil.rmtree(path) # ignore_errors = False --> Carefull with that, it may not delete
    except FileNotFoundError:
        print(f'⚠️ Directory not found, results could be saved at: {path}')

def create_directory(path:Path):
    """
    Creates the directory.

    Args:
        path (Path): path of the directory to be created.

    Raises:
        PermissionError if not allowed to create the path.
    """
    if not isinstance(path, Path):
        path = Path(path)

    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise PermissionError(f'❌ Permission error: you are not allowed to create {path}')
        

def build_dir_tree(parent_folder:str, seed:int, shape:tuple, delta:int, 
                         model_name:str, remove_model_data:bool=False, remove_test:bool=False, remove_all:bool=False):
    """
    Clean and creates the output directories related to the experiment.

    Args:
        parent_folder (str): path to the parent folder of the experiment.
        seed (int): random number generator seed of the experiment.
        shape (tuple[int,int]): Spatial dimensions (height, width) of 
            the input boards.
        delta (int): number of generations between the initial and final state.
        model_name (str): name of the model
        remove_model_data (bool): removes the model_name/ directories. Default to `False`.
        remove_test (bool): remove the Test/ directory. Default to `False`.
        remove_all (bool): remove the complete shape_delta_seed/ directory. Default to `False`

    Returns:
        A dictionary mapping directory names to their corresponding Path objects. 
        Keys are:
            - "experiment_folder": parent_folder/shape_delta_seed
            - "data_folder": root/ConwayStates
            - "training_data": ConwayStates/Train/model_name
            - "validation_data": ConwayStates/Validation/model_name
            - "test_data": ConwayStates/ Test
            - "results": root/Results/model_name

    Creates the following directories graph:

    parent_folder/
        └── {shape}_{delta}_{seed}/
            ├── ConwayStates/
            │   ├── Train/
            │   │   └── {model_name}/
            │   ├── Validation/
            │   │   └── {model_name}/
            │   └── Test/
            └── Results/
                └── {model_name}/
        
    """
    H,W = shape
    parent_folder = Path(parent_folder)

    shape_delta_seed_dir = parent_folder/f'{H}x{W}_{delta}_{seed}'

    boards_dir = shape_delta_seed_dir/f'ConwayStates'
    train_boards_dir = boards_dir/f'Train/{model_name}'

    val_boards_dir = boards_dir/f'Validation/{model_name}'
    test_boards_dir = boards_dir/f'Test'
    
    results_dir = shape_delta_seed_dir/f'Results/{model_name}'

    if remove_all:
        remove_directory(shape_delta_seed_dir)
    
    if remove_test:
        remove_directory(test_boards_dir)
        create_directory(test_boards_dir)
    else:
        create_directory(test_boards_dir)
    
    modeldata_dirs = [train_boards_dir, val_boards_dir, results_dir]
    if remove_model_data:
        for directory in modeldata_dirs:
            remove_directory(directory)
            create_directory(directory)
    else:
        for directory in modeldata_dirs:
            create_directory(directory)
    
    return {"experiment_folder":shape_delta_seed_dir, "data_folder":boards_dir,
            "training_data":train_boards_dir,"validation_data": val_boards_dir,
            "test_data": test_boards_dir,"results": results_dir}

def get_optimizer(name:str, params:Iterator[nn.Parameter], **kwargs):
    """
    Instantiates and returns a PyTorch optimizer by name.

    Args:
        name (stutilsr): Name of the optimizer class from torch.optim
            (e.g., 'Adam', 'SGD', 'AdamW').
        params (iterable): Model parameters to optimize,
            typically from model.parameters().
        **kwargs: Additional arguments passed to the optimizer
            (e.g., lr=0.001, weight_decay=1e-4).

    Returns:
        torch.optim.Optimizer: The instantiated optimizer.
    
    Raises:
        AttributeError: If the optimizer name does not exist in torch.optim.
    """
    if not hasattr(optim, name):
        available = [o for o in dir(optim) 
                    if isinstance(getattr(optim, o), type) and issubclass(getattr(optim, o), optim.Optimizer)]
        
        raise AttributeError(f'❌ Optimizer "{name}" not found in torch.optim. Available optimizers:\n'
                         f'{available}')
        
    optimizer_class = getattr(optim, name)

    return optimizer_class(params, **kwargs)

def get_loss(name:str, **kwargs):
    """
    Instantiates and returns a PyTorch loss function by name.

    Args:
        name (str): Name of the loss class from torch.nn
            (e.g., 'BCELoss', 'MSELoss', 'CrossEntropyLoss').
        **kwargs: Additional arguments passed to the loss function
            (e.g., reduction='mean', weight=class_weights).

    Returns:
        torch.nn.Module: The instantiated loss function.

    Raises:
        AttributeError: If the loss name does not exist in torch.nn.
    """
    if not hasattr(nn, name):
        available = [l for l in dir(nn)
                     if isinstance(getattr(nn, l), type) and issubclass(getattr(nn, l), nn.Module) and l.endswith('Loss')]
        
        raise AttributeError(f'❌ Loss "{name}" not found in torch.nn. Available loss functions:\n'
                         f'{available}')
        
    loss_class = getattr(nn, name)
    return loss_class(**kwargs)

def get_custom_loss(name:str, **kwargs):
    """
    Instantiates and returns a PyTorch customized loss function by name.

    Args:
        name (str): Name of the loss class from torch.nn
            (e.g., 'BinaryFocalCrossEntropy', 'FuzzyLoss', 'FFFuzz').
        **kwargs: Additional arguments passed to the loss function
            (e.g., lambda_phys, gamma, apply_class_balancing).

    Returns:
        torch.nn.Module: The instantiated custom loss function.

    Raises:
        AttributeError: If the loss name does not exist in torch.nn.
    """
    if not hasattr(losses, name):
        available = [l for l in dir(losses) 
                     if isinstance(getattr(losses, l), type) and issubclass(getattr(losses, l), nn.Module)]
        
        raise AttributeError(f'❌ Loss "{name}" not found in utils.losses. Available loss functions:\n'
                         f'{available}')
        
    loss_class = getattr(losses, name)
    return loss_class(**kwargs)

def get_metric(name:str, **kwargs):
    """
    Instantiates and returns a torchmetrics metric by name.

    Args:
        name (str): Name of the metric class from torchmetrics
            (e.g., 'Accuracy', 'F1Score', etc).
        **kwargs: Additional arguments passed to the loss function
            (e.g., task, num_classes, etc).

    Returns:
        torchmetrics.Metric: The instantiated metric function.

    Raises:
        AttributeError: If the metric name does not exist in torchmetrics.
    """
    if not hasattr(torchmetrics, name):
        available = [l for l in dir(torchmetrics) 
                     if isinstance(getattr(torchmetrics, l), type) and issubclass(getattr(torchmetrics, l), torchmetrics.Metric)]
        
        available_str = '\n'.join([', '.join(available[i:i+8]) for i in range(0, len(available), 8)])
        raise AttributeError(f'❌ Metric "{name}" not found in torchmetrics. Available metrics:\n{available_str}')
        
    metric_class = getattr(torchmetrics, name)
    return metric_class(**kwargs)

def training_loop(shape:tuple, delta:int, seed:int, output_dirs:dict[str,Path], device:torch.device,
                   model:nn.Module, num_outputs:int, loss_fn:nn.Module, optimizer:torch.optim.Optimizer, epochs:int, batch_size:int, epochs2hold:int,

                  tolerance_dts:int, tolerance_lr:int, factor:float, warmup_epochs:int, 
                  schedule_lambda_phys:dict[int, int]=None, schedule_lambda_bin:dict[int,int]=None, schedule_lambda_dice:dict[int,int]=None,

                  val_split:float=0.2, states2generate:int=60_000,  margin:int = 120_000, min_density:float=0.05, 
                  max_density:float=0.95, generator_batch_size:int=5_000):
    """
    Training loop for Conway's Game of Life inverse problem models.

    Handles dynamic dataset generation, model checkpointing, learning rate 
    scheduling, and optional lambda scheduling for custom loss functions.

    Args:
        shape (tuple[int, int]): Spatial dimensions (height, width) of the input boards.
        delta (int): Number of generations between the initial and final state.
        seed (int): Random number generator seed for reproducibility.
        output_dirs (dict[str, Path]): Dictionary mapping directory names to their
            corresponding Path objects. Must contain keys: "training_data",
            "validation_data" and "results".
        device (torch.device): Device to run the training on (CPU or CUDA).
        model (nn.Module): Model to train.
        num_outputs (int): Number of outputs of the model. If 2, the model returns
            (initial_state, final_state). If 1, returns only the initial state.
        loss_fn (nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        epochs2hold (int): Number of epochs between dataset regenerations. If <= 0,
            the dataset is generated once before training and held throughout.
        tolerance_dts (int): Number of epochs without improvement before
            regenerating the dataset.
        tolerance_lr (int): Patience for ReduceLROnPlateau scheduler. If <= 0,
            no learning rate scheduling is applied.
        factor (float): Factor by which the learning rate is reduced when
            ReduceLROnPlateau is triggered.
        warmup_epochs (int): Number of epochs before model checkpointing is enabled.
        schedule_lambda_phys (dict[int, float], optional): Dictionary mapping epoch
            numbers to lambda_phys values for the loss function. Default is None.
        schedule_lambda_bin (dict[int, float], optional): Dictionary mapping epoch
            numbers to lambda_bin values for the loss function. Default is None.
        val_split (float): Fraction of generated data used for validation. Default is 0.2.
        states2generate (int): Target number of states to generate per dataset. Default is 60_000.
        margin (int): Maximum number of boards generated before filtering. Default is 120_000.
        min_density (float): Minimum cell density of generated boards. Default is 0.05.
        max_density (float): Maximum cell density of generated boards. Default is 0.95.
        generator_batch_size (int): Batch size used during data generation. Default is 5_000.

    Returns:
        nn.Module: The best model found during training, loaded from the checkpoint.
            If no checkpoint was saved, returns the model from the last epoch.
    """

    # Making my work easier:
    H,W = shape
    train_boards_dir = output_dirs['training_data']
    val_boards_dir = output_dirs['validation_data']
    results_dir = output_dirs['results']
    file_train_history = results_dir/'train_history.csv'
    file_model = results_dir/'model.pt'
    lr0 = optimizer.param_groups[0]['lr']
    
    # Flux control params:
    best_val_loss_chkp, best_val_loss_dts, patience_dts = float('inf'), float('inf'), 0
    global_train_history = []
    change_dts_epoch, change_dts_patience = True, False

    # Apply ReduceLROnPlateau?
    if tolerance_lr > 0:
        scheduler_lr = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=tolerance_lr)
    
    # Hold dataset case:
    if epochs2hold <= 0:
        file_train = train_boards_dir/'train0.npz'
        file_val = val_boards_dir/'val0.npz'
        data_generator(seed, delta, shape, [train_boards_dir, val_boards_dir], 
                    ['train0', 'val0'], True, val_split, states2generate,
                    margin, min_density, max_density, generator_batch_size)
        _, x_train, y_train = load_npz(file_train, file_train, log=False)
        _, x_val, y_val = load_npz(file_val, file_val, log=False)

    for epoch in range(epochs):
        # Path to files:
        file_train = train_boards_dir/f'train{epoch}.npz'
        file_val = val_boards_dir/f'val{epoch}.npz'
        file_names = [f'train{epoch}', f'val{epoch}']

        # Change dataset due to epoch?
        if epochs2hold == 1 or (epochs2hold > 1 and epoch % (epochs2hold) == 0):
            change_dts_epoch = True
        
        # Change dataset:
        if (epochs2hold > 0) and (change_dts_epoch or change_dts_patience):
            # Reset values:
            best_val_loss_dts, best_val_loss_chkp = float('inf'), float('inf')
            change_dts_epoch, change_dts_patience = False, False

            for group_param in optimizer.param_groups:
                group_param['lr'] = lr0
            if tolerance_lr > 0:
                scheduler_lr = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=tolerance_lr)

            # Generate dataset:
            data_generator(seed, delta, shape, [train_boards_dir, val_boards_dir], file_names, True, val_split, states2generate, 
                           margin, min_density, max_density, generator_batch_size)
            # Load dataset:
            _, x_train, y_train = load_npz(file_train, file_train, log=False)
            _, x_val, y_val = load_npz(file_val, file_val, log=False)
        
        # Create the dataset iterators:
        init_train = torch.from_numpy(x_train.reshape((-1,1,H,W))).float()
        fin_train = torch.from_numpy(y_train.reshape((-1,1,H,W))).float()

        init_val = torch.from_numpy(x_val.reshape((-1,1,H,W))).float()
        fin_val = torch.from_numpy(y_val.reshape((-1,1,H,W))).float()

        train_dataset = TensorDataset(init_train, fin_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count()), pin_memory=True)

        val_dataset = TensorDataset(init_val, fin_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=min(6, os.cpu_count()), pin_memory=True)

        # Train model:
        step_losses = 0
        model.train()
        for initial, final in train_loader:
            # Reset:
            optimizer.zero_grad()
            # Send data to the device
            initial = initial.to(device)
            final = final.to(device)

            # Forward
            if num_outputs == 2:
                pred = model(final) # [initial, final]
                true = [initial, final]
            else:
                pred = model(final) # [initial]
                true = initial

            step_loss = loss_fn(pred,true)

            #Backprop
            step_loss.backward()
            optimizer.step()
            step_losses += step_loss.item()

        # Loss mean value:
        loss = step_losses / len(train_loader)

        # Validation:
        step_val_losses = 0
        model.eval()
        with torch.no_grad():
            for initial, final in val_loader:
                initial = initial.to(device)
                final = final.to(device)

                if num_outputs == 2:
                    pred = model(final)
                    true = [initial, final]
                else:
                    pred = model(final)
                    true = initial
                
                step_val_losses += loss_fn(pred,true).item()

            # Val loss mean value:   
            val_loss = step_val_losses/len(val_loader)
        
        # Callbacks:
        # Change dataset due to performance?
        if (val_loss < best_val_loss_dts):
            best_val_loss_dts = val_loss
            patience_dts = 0
            change_dts_patience = False
        else:
            patience_dts +=1
            if patience_dts == tolerance_dts:
                patience_dts = 0
                change_dts_patience = True
            else:
                change_dts_patience = False

        global_train_history.append({'loss':loss,'val_loss':val_loss, 'change_dts': change_dts_patience or change_dts_epoch})
        print(f'📑 Epoch {epoch}/{epochs-1} ---> loss:{loss:.4f} | val_loss: {val_loss:.4f}')    
        # Model Checkpoint:
        if (epoch >= warmup_epochs) and (val_loss < best_val_loss_chkp):

            print(f'🔥 Improvement found: {best_val_loss_chkp:.4f} --> {val_loss:.4f}. Saving model...')
            best_val_loss_chkp = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss_chkp,
                'scheduler_lr_state_dict': scheduler_lr.state_dict() if tolerance_lr > 0 else None,
                'patience_dts': patience_dts,
                'global_train_history': global_train_history}, file_model)
        
        # Check the ReduceLRonPlateau:
        if tolerance_lr > 0:
            scheduler_lr.step(val_loss)

        # Lambda_phys and lambda_bin:
        if schedule_lambda_phys:
            if epoch in schedule_lambda_phys:
                loss_fn.lambda_phys = schedule_lambda_phys[epoch]
        
        if schedule_lambda_bin:
            if epoch in schedule_lambda_bin:
                loss_fn.lambda_bin = schedule_lambda_bin[epoch]

        if schedule_lambda_dice:
            if epoch in schedule_lambda_dice:
                loss_fn.lambda_dice = schedule_lambda_dice[epoch]

        pd.DataFrame(global_train_history).to_csv(file_train_history, index=False)

    if file_model.exists():
        checkpoint = torch.load(file_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('✅ Best model loaded.')
    else:
        print('⚠️ No checkpoint was saved, returning last epoch model.')
    return model

def model_test(test_loader:DataLoader, model:nn.Module, num_outputs:int, metrics2compute:list[str], shape:tuple[int,int], 
            delta:int, epsilon:float, order:float, device:torch.device, results_dir:Path):
    """
    Evaluates the model on the test dataset, saving predictions and metrics.

    Args:
        test_loader (DataLoader): DataLoader containing the test dataset.
        model (nn.Module): Trained model to evaluate.
        num_outputs (int): Number of outputs of the model. If 2, the model returns
            (init_pred, fin_pred). If 1, only init_pred is returned and the final
            state is computed using DiffGoL.
        metrics2compute (list[str]): List of torchmetrics metric names to compute
            (e.g., ['Accuracy', 'F1Score', 'JaccardIndex']).
        shape (tuple[int, int]): Spatial dimensions (height, width) of the boards.
        delta (int): Number of generations between the initial and final state.
        epsilon (float): Sharpness parameter of the DiffGoL layer.
        order (float): Order parameter of the DiffGoL layer.
        device (torch.device): Device to run the evaluation on (CPU or CUDA).
        results_dir (Path): Path to the directory where results will be saved.

    Returns:
        None

    Saves:
        predictions.csv: Each row contains the flattened predicted initial state
            (start_0, ..., start_N) and predicted final state (stop_0, ..., stop_N)
            for each test sample.
        test_results.csv: Single row containing the computed metrics for both
            initial and final state predictions, with keys in the format
            {metric_name}_init and {metric_name}_fin.
    """

    # Files to save test results
    file_predictions = results_dir/'predictions.csv'
    file_test_results = results_dir/'test_results.csv'

    # Headers of predictions.csv
    cells = shape[0] * shape[1]
    init_cols = [f'start_{i}' for i in range(cells)]
    stop_cols = [f'stop_{i}' for i in range(cells)]

    gol = DiffGoL(delta, epsilon, order).to(device)

    # Instantiate metrics for both init and fin:
    metrics = {
        'init': {metric: get_metric(metric, task='binary').to(device) for metric in metrics2compute},
        'fin':  {metric: get_metric(metric, task='binary').to(device) for metric in metrics2compute}
    }

    # Evaluation:
    predictions = []
    model.eval()
    with torch.no_grad():
        for initial, final in test_loader:
            initial = initial.to(device)
            final = final.to(device)

            if num_outputs == 2:
                init_pred, fin_pred = model(final)
            else:
                init_pred = model(final)
                fin_pred = gol.forward(init_pred)

            # Update metrics:
            for name, metric_fn in metrics['init'].items():
                metric_fn.update(init_pred, initial)
            for name, metric_fn in metrics['fin'].items():
                metric_fn.update(fin_pred, final)

            # Save predictions:
            B = initial.shape[0]
            init_pred_np = init_pred.cpu().numpy().reshape(B, -1)
            fin_pred_np = fin_pred.cpu().numpy().reshape(B, -1)
            for i in range(B):
                row = dict(zip(init_cols, init_pred_np[i]))
                row.update(dict(zip(stop_cols, fin_pred_np[i])))
                predictions.append(row)

    # Compute and save test results:
    test_results = {}
    for split, metric_dict in metrics.items():
        for name, metric_fn in metric_dict.items():
            test_results[f'{name}_{split}'] = metric_fn.compute().item()
            metric_fn.reset()

    pd.DataFrame([test_results]).to_csv(file_test_results, index=False)
    pd.DataFrame(predictions, columns=init_cols + stop_cols).to_csv(file_predictions, index=False)


def training_loop2(model:nn.Module, optimizer:torch.optim, loss_fn:nn.Module, experiment_paths:dict[str:Path], files:list[str], 
                   val_split:float, seed:int, shape:tuple, epochs:int, batch_size:int, device:torch.device):
    """
    Training loop for a heatmap cleaner model for the Conway's Game of Life inverse problem.

    Loads precomputed heatmap predictions and ground truth states from a previous model,
    splits them into train and validation sets, saves them to disk, and trains the model
    with checkpointing.

    Args:
        model (nn.Module): Model to train. Must return (initial_state, final_state).
        optimizer (torch.optim): Optimizer to use for training.
        loss_fn (nn.Module): Loss function to optimize. Must accept
            ([init_pred, fin_pred], [init_true, fin_true]).
        experiment_paths (dict[str, Path]): Dictionary mapping directory names to their
            corresponding Path objects. Must contain keys: "training_data",
            "validation_data" and "results".
        files (list[str]): List containing two paths: [file_predictions, file_test].
            file_predictions is a .csv with heatmap predictions from the upstream model.
            file_test is a .npz with the ground truth initial and final states.
        val_split (float): Fraction of data used for validation.
        seed (int): Random number generator seed for reproducibility.
        shape (tuple[int, int]): Spatial dimensions (height, width) of the input boards.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        device (torch.Device): Device to run the training on (CPU or CUDA).

    Returns:
        nn.Module: The best model found during training, loaded from the checkpoint.
            If no checkpoint was saved, returns the model from the last epoch.
    """
    H, W = shape
    # Extract the data: (test and predictions from model2use)
    file_predictions, file_test = files
    
    start_cols = [f'start_{i}' for i in range(H*W)]
    heat_cols  = [f'heat_{i}'  for i in range(H*W)]
    fin_cols   = [f'stop_{i}'   for i in range(H*W)]

    X = pd.read_csv(file_predictions)[start_cols].values # (N, H*W)
    _, init_true, fin_true = load_npz(file_test, 'test.npz') # (N, H*W)

    X_train, X_val, Y_init_train, Y_init_val, Y_fin_train, Y_fin_val = train_test_split(X, init_true, fin_true, test_size=val_split, random_state=seed)

    # Save the data in the model's folder:
    file_training_states   = experiment_paths['training_data']/'train.npz'
    file_validation_states = experiment_paths['validation_data']/'val.npz'
    np.savez(file_training_states,
             **{col: X_train[:, i]      for i, col in enumerate(heat_cols)},
             **{col: Y_init_train[:, i] for i, col in enumerate(start_cols)},
             **{col: Y_fin_train[:, i]  for i, col in enumerate(fin_cols)})
    np.savez(file_validation_states,
             **{col: X_val[:, i]       for i, col in enumerate(heat_cols)},
             **{col: Y_init_val[:, i]  for i, col in enumerate(start_cols)},
             **{col: Y_fin_val[:, i]   for i, col in enumerate(fin_cols)})
    
    # DataLoaders:
    X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, H, W)
    X_val = torch.tensor(X_val, dtype=torch.float32).reshape(-1, 1, H, W)
    Y_init_train = torch.tensor(Y_init_train, dtype=torch.float32).reshape(-1, 1, H, W)
    Y_init_val = torch.tensor(Y_init_val, dtype=torch.float32).reshape(-1, 1, H, W)
    Y_fin_train = torch.tensor(Y_fin_train, dtype=torch.float32).reshape(-1, 1, H, W)
    Y_fin_val = torch.tensor(Y_fin_val, dtype=torch.float32).reshape(-1, 1, H, W)

    train_loader = DataLoader(TensorDataset(X_train, Y_init_train, Y_fin_train), batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(TensorDataset(X_val, Y_init_val, Y_fin_val), batch_size=batch_size, shuffle=False)

    # Training loop:
    best_val_loss = float('inf')
    file_model = experiment_paths['results']/'model.pt'
    file_train_history = experiment_paths['results']/'train_history.csv'
    global_train_history = []
    for epoch in range(epochs):
        # Train model:
        step_losses = 0
        model.train()
        for heatmap_train, init_train, fin_train in train_loader:
            # Reset optimizer:
            optimizer.zero_grad()
            # Send data to the device:
            heatmap_train = heatmap_train.to(device)
            init_train = init_train.to(device)
            fin_train = fin_train.to(device)
            # Forward:
            init_pred, fin_pred = model(heatmap_train)
            step_loss = loss_fn([init_pred, fin_pred], [init_train, fin_train])
            # Backprop:
            step_loss.backward()
            optimizer.step()
            step_losses += step_loss.item()

        # Compute epoch loss:
        loss = step_losses/len(train_loader)

        # Validation:
        step_val_losses = 0
        model.eval()
        with torch.no_grad():
            for heatmap_val, init_val, fin_val in val_loader:
                # Send data to the device:
                heatmap_val = heatmap_val.to(device)
                init_val = init_val.to(device)
                fin_val = fin_val.to(device)
                # Prediction:
                init_pred, fin_pred = model(heatmap_val)
                step_val_losses += loss_fn([init_pred, fin_pred], [init_val, fin_val]).item()

            # Compute epoch val_loss:
            val_loss = step_val_losses/len(val_loader)

        # Results to csv:
        global_train_history.append({'loss':loss,'val_loss':val_loss})
        pd.DataFrame(global_train_history).to_csv(file_train_history, index=False)

        # Model checkpoint callback
        print(f'📑 Epoch {epoch}/{epochs-1} ---> loss:{loss:.4f} | val_loss: {val_loss:.4f}') 
        if (val_loss < best_val_loss):
            print(f'🔥 Improvement found: {best_val_loss:.4f} --> {val_loss:.4f}. Saving model...')
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'global_train_history': global_train_history}, file_model)

    if file_model.exists():
        checkpoint = torch.load(file_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('✅ Best model loaded.')
    else:
        print('⚠️ No checkpoint was saved, returning last epoch model.')
    return model

def model_test2(test_loader:DataLoader, model:nn.Module, metrics2compute:list[str], shape:tuple[int,int], device:torch.device, results_dir:Path):
    """
    Evaluates the model on the test dataset, saving predictions and metrics.

    Args:
        test_loader (DataLoader): DataLoader containing the test dataset.
        model (nn.Module): Trained model to evaluate.
        metrics2compute (list[str]): List of torchmetrics metric names to compute
            (e.g., ['Accuracy', 'F1Score', 'JaccardIndex']).
        shape (tuple[int, int]): Spatial dimensions (height, width) of the boards.
        device (torch.device): Device to run the evaluation on (CPU or CUDA).
        results_dir (Path): Path to the directory where results will be saved.

    Returns:
        None

    Saves:
        predictions.csv: Each row contains the flattened predicted initial state
            (start_0, ..., start_N) and predicted final state (stop_0, ..., stop_N)
            for each test sample.
        test_results.csv: Single row containing the computed metrics for both
            initial and final state predictions, with keys in the format
            {metric_name}_init and {metric_name}_fin.
    """

    # Files to save test results
    file_predictions = results_dir/'predictions.csv'
    file_test_results = results_dir/'test_results.csv'

    # Headers of predictions.csv
    cells = shape[0] * shape[1]

    init_cols = [f'start_{i}' for i in range(cells)]
    stop_cols = [f'stop_{i}' for i in range(cells)]

    # Instantiate metrics for both init and fin:
    metrics = {
        'init': {metric: get_metric(metric, task='binary').to(device) for metric in metrics2compute},
        'fin':  {metric: get_metric(metric, task='binary').to(device) for metric in metrics2compute}
    }

    # Evaluation:
    predictions = []
    model.eval()
    with torch.no_grad():
        for heatmap, initial, final in test_loader:
            heatmap = heatmap.to(device)
            initial = initial.to(device)
            final = final.to(device)

            init_pred, fin_pred = model(heatmap)
            
            # Update metrics:
            for name, metric_fn in metrics['init'].items():
                metric_fn.update(init_pred, initial)
            for name, metric_fn in metrics['fin'].items():
                metric_fn.update(fin_pred, final)

            # Save predictions:
            B = initial.shape[0]
            init_pred_np = init_pred.cpu().numpy().reshape(B, -1)
            fin_pred_np = fin_pred.cpu().numpy().reshape(B, -1)
            for i in range(B):
                row = dict(zip(init_cols, init_pred_np[i]))
                row.update(dict(zip(stop_cols, fin_pred_np[i])))
                predictions.append(row)

    # Compute and save test results:
    test_results = {}
    for split, metric_dict in metrics.items():
        for name, metric_fn in metric_dict.items():
            test_results[f'{name}_{split}'] = metric_fn.compute().item()
            metric_fn.reset()

    pd.DataFrame([test_results]).to_csv(file_test_results, index=False)
    pd.DataFrame(predictions, columns=init_cols + stop_cols).to_csv(file_predictions, index=False)
