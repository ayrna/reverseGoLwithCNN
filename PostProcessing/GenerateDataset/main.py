import sys
sys.path.append('../')
import utils.data as data
from pathlib import Path
from sacred import Experiment


ex = Experiment('Generate Dataset')

@ex.config
def config():
    shape:tuple = (15,15)
    delta:int = 1
    seed:int = 100

    path2data:str = '../PyTorch/Data/'
    path2save:str = './Data'

    model_name:str = 'DiffGoL'
    samples2get:int = 60_000
    samples2get_test:int = 15_000

@ex.main
def main(shape, delta, seed, path2data, path2save, model_name, samples2get, samples2get_test):

    H,W = shape
    # Create Train dataset
    path2save = Path(path2save)/f'{H}x{W}_{delta}_{seed}' # ./Data/15x15_1_seed
    data.create_train_dataset(path2data=path2data, path2save=path2save, model_name=model_name, shape=shape, seed=seed, samples2get=samples2get)

    # Create Test dataset
    data.create_random_test(path2data=path2data, path2save=path2save, model=model_name, shape=shape,seed=seed, samples2get=samples2get_test)

if __name__=='__main__':
    ex.run_commandline()