import sys
sys.path.append('../..')
import torch
from sacred import Experiment
import utils.experiment_tools as tools
from utils.models import DiffGoLModel
from utils.data import data_generator, load_npz
from torch.utils.data import TensorDataset, DataLoader

ex = Experiment('ClassicDiffGoL Model')

@ex.config
def config():
    # Global configuration:
    seed:int = 1
    delta:int = 1
    shape:tuple = (15,15)

    parent_folder:str = './Data'
    model_names:str= 'ClassicDiffGoL'
    remove_model_data:bool = True
    remove_test:bool = False 
    remove_all:bool = False

    # Loss and optimizer configuration:
    optimizer:str = 'Adam'
    learning_rate:float = 0.0001
    lambda_init:float = 1.0
    lambda_phys:float = 1.0
    lambda_bin:float = 0.0
    lambda_dice:float = 0.0
    smooth:float = 1e-8
    gamma:float = 2.0
    alpha:float = 0.75
    apply_class_balancing:bool = False
    from_logits:bool = False

    # Training loop configuration
    epochs:int = 1
    batch_size:int = 2**8
    epochs2hold:int = 50
    num_outputs:int = 2
    tolerance_dts:int = 25
    tolerance_lr:int = 20
    factor:float = 0.5
    warmup_epochs:int = 0
    schedule_lambda_phys:dict[int,int] = {}
    schedule_lambda_bin:dict[int,int] = {}
    schedule_lambda_dice:dict[int, int] = {}
    
    # Test configurations:
    metrics2compute:list[str] = ['Accuracy', 'Recall', 'Specificity','Precision', 'F1Score']

    # Models configuration:
    kernel_size:int = 3
    n_hidden_convs:int = 6
    n_hidden_filters:int = 48
    epsilon:float=50.0
    order:float=4.0

    # Data generation configuration: (Usually default)
    val_split:float = 0.2
    states2generate:int = 60_000
    margin:int = 120_000
    min_density:float = 0.05
    max_density:float = 0.95
    generator_batch_size:int = 5_000
    generate_test:bool = False
    states2generate_test:int =15_000

@ex.main
def main(seed, delta, shape, parent_folder, model_names, remove_model_data, remove_test, remove_all,
         optimizer, learning_rate, lambda_init, lambda_phys, lambda_bin, lambda_dice, smooth, gamma, alpha, apply_class_balancing, from_logits,
         epochs, batch_size, epochs2hold, num_outputs, tolerance_dts, tolerance_lr, factor, warmup_epochs,  schedule_lambda_phys, schedule_lambda_bin,
         schedule_lambda_dice, metrics2compute, kernel_size, n_hidden_convs, n_hidden_filters, epsilon, order, 
         val_split, states2generate,  margin, min_density, max_density, generator_batch_size, generate_test, states2generate_test): 
        
    
    H,W = shape
    # Device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print(f'❌ GPU is not detected, interrupting execution...')
        sys.exit(1)

    # Seed:
    tools.set_random_seed(seed)
    
    print(f' --- 🐲 Starting training {model_names} Model ---')
    
    model_dirs = tools.build_dir_tree(parent_folder, seed, shape, delta, model_names, remove_model_data, remove_test, remove_all)

    model = torch.compile(DiffGoLModel(n_hidden_convs, n_hidden_filters, kernel_size, delta, epsilon, order).to(device))

    optimizer = tools.get_optimizer(optimizer, model.parameters(), lr=learning_rate)

    # loss_fn = tools.get_custom_loss('FocalBCEFuzz', lambda_phys=lambda_phys,lambda_bin=lambda_bin, gamma=gamma, alpha=alpha,
    #                                         apply_class_balancing=apply_class_balancing, from_logits=from_logits).to(device) 
    loss_fn = tools.get_custom_loss('FocalBCEDiceFuzz',num_outputs=num_outputs, lambda_init=lambda_init, lambda_phys=lambda_phys, lambda_bin=lambda_bin, 
                                            lambda_dice=lambda_dice, smooth=smooth, gamma=gamma, alpha=alpha, apply_class_balancing=apply_class_balancing, from_logits=from_logits).to(device)
     
    model = tools.training_loop(shape, delta, seed, model_dirs, device,
                  model, num_outputs, loss_fn, optimizer, epochs, batch_size, epochs2hold, tolerance_dts, tolerance_lr, factor, warmup_epochs, 
                  schedule_lambda_phys, schedule_lambda_bin, schedule_lambda_dice, val_split, 
                  states2generate, margin, min_density, max_density, generator_batch_size)
    
    # Test:
    dir_boards_test = model_dirs['test_data']
    if generate_test:
        data_generator(seed, delta, shape, dir_boards_test, f'test', False, val_split, states2generate_test,margin, min_density, max_density, generator_batch_size)

    file_test_name = dir_boards_test/'test.npz'
    _, x_test, y_test = load_npz(file_test_name, file_test_name, log=False)
    
    init_test = torch.from_numpy(x_test.reshape(-1,1,H,W)).float()
    fin_test = torch.from_numpy(y_test.reshape(-1,1,H,W)).float()
    
    test_dataset = TensorDataset(init_test, fin_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    tools.model_test(test_loader, model, num_outputs, metrics2compute, 
                     shape, delta, epsilon, order, device, model_dirs['results'])
    
if __name__=='__main__':
    ex.run_commandline()  