import sys
import torch
import utils.experiment_tools as tools
from utils.models import Regressor
from sacred import Experiment
from pathlib import Path

ex = Experiment('Regressor')

@ex.config
def config():

    seed:int = 42
    delta:int = 1
    shape:tuple = (15,15)
    path2save:str = './Data'
    n_hidden_filters:list[int] = [1,15,32,64]
    kernel_size:int = 3
    mlp_hidden:tuple[int] = (64,32)
    use_stats:bool = True
    padding_mode:str = 'circular'
    optimizer:str = 'Adam'
    learning_rate:float = 0.0001
    epochs:int = 1
    batch_size:int = 2**8
    metrics2compute:list= ['MeanAbsoluteError', 'MeanSquaredError', 'R2Score']
    # 14


@ex.main
def main(seed, delta, shape, path2save, n_hidden_filters, kernel_size, mlp_hidden, 
         use_stats, padding_mode, optimizer, learning_rate, epochs, batch_size, metrics2compute):

    H,W = shape
    # Device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print(f'❌ GPU is not detected, interrupting execution...')
        sys.exit(1)

    # Create dataset
    path2save = Path(path2save)/f'{H}x{W}_{delta}_{seed}' # ./Data/15x15_1_seed
    files2data = {'train': path2save/'trainGoL.csv', 'val': path2save/'valGoL.csv'}
    
    # Fix the seed
    tools.set_random_seed(seed)

    model = torch.compile(Regressor(n_hidden_filters=n_hidden_filters, kernel_size=kernel_size, mlp_hidden=mlp_hidden, use_stats=use_stats, padding_mode=padding_mode).to(device))
    optimizer = tools.get_optimizer(optimizer, model.parameters(), lr=learning_rate)
    loss_fn = tools.get_loss('HuberLoss', delta=0.05)   

    print(f' --- 🐲 Starting training Regressor Model ---')
    files2outputs = {'model': path2save/'model.pt', 'history':path2save/'train_history.csv'}
    model = tools.training_loop_regressor(model, optimizer, loss_fn, files2data, files2outputs, shape, epochs, batch_size, device)

    print(f'--- 🐲 Starting test Regressor Model ---')
    file2data_test = {'test': path2save/'testGoL.csv', 'results': path2save/'test_results.csv', 'predictions': path2save/'predictions.csv' }
    tools.test_loop_regressor(model, file2data_test, metrics2compute, shape, batch_size, device)
    

if __name__=='__main__':
    ex.run_commandline()
