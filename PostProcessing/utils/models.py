import torch 
import torch.nn as nn
from utils.gol import DiffGoL
import torch.nn.functional as F


### CNN models: Initial State predictors.
class ClassicModel(nn.Module):
    """
    Foundational CNN architecture formed by Conv2d-BatchNorm blocks:

    Conv2d --> BatchNorm --> ReLU --> Conv2d --> BatchNorm --> ReLU --> {...} --> Conv2d --> Sigmoid

    """
    def __init__(self, n_hidden_convs:int, n_hidden_filters:int, kernel_size:int):
        """
        Args:
            n_hidden_convs (int): number of hidden Conv2d-BatchNorm blocks.
            n_hidden_filters (int): number of filters in the different Conv2d layers
            kernel_size (int): size of the Conv2d kernel.
        """

        super().__init__()
        pad = kernel_size // 2 
        self.input_layer = nn.Sequential(nn.Conv2d(in_channels= 1, out_channels=n_hidden_filters, 
                                     kernel_size=kernel_size, padding = pad, padding_mode='circular'),
                                     nn.BatchNorm2d(num_features=n_hidden_filters),
                                     nn.ReLU())
        hidden_blocks = []
        for _ in range(n_hidden_convs):
            hidden_blocks.extend([nn.Conv2d(in_channels=n_hidden_filters, out_channels=n_hidden_filters, 
                                     kernel_size=kernel_size, padding = pad, padding_mode='circular'),
                                     nn.BatchNorm2d(num_features=n_hidden_filters),
                                     nn.ReLU()])
        
        self.hidden_layers = nn.Sequential(*hidden_blocks)

        self.output_layer = nn.Sequential(nn.Conv2d(in_channels=n_hidden_filters, out_channels=1, 
                                      kernel_size=kernel_size, padding = pad, padding_mode='circular'),
                                      nn.Sigmoid()) 

    def forward(self, X):
        assert (isinstance(X, torch.Tensor)), f'Expected a Tensor as input, but got {type(X)}'
        assert (X.ndim == 4), f'Expected a Rank 4 Tensor as input, but got Rank {X.ndim} Tensor --> (B,1, H,W)'

        X = self.input_layer(X)
        X = self.hidden_layers(X)

        return self.output_layer(X)
    
class DiffGoLModel(ClassicModel):
    """
    Extension of the Classic Model with the DiffGoL layer:

    ClassicModel --> DiffGoL

    """
    def __init__(self, n_hidden_convs:int, n_hidden_filters:int, kernel_size:int, 
                delta:int=1, epsilon:float=50.0, order:float=4.0):
        ClassicModel.__init__(self, n_hidden_convs, n_hidden_filters, kernel_size)
        
        self.diffGoL = DiffGoL(delta, epsilon, order)

    def forward(self, X):

        initial_state = ClassicModel.forward(self, X)

        final_state = self.diffGoL(initial_state)

        return initial_state, final_state

class Regressor(nn.Module):

    def __init__(self, n_hidden_filters: list[int] = [1, 15, 32,64], kernel_size: int = 3,
                 mlp_hidden: tuple[int] = (64, 32), use_stats: bool = True,
                 quantiles: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                 padding_mode: str = 'circular'):

        super().__init__()
        pad = kernel_size // 2

        cnn_block = []
        for i in range(len(n_hidden_filters) - 1):
            input_channels = n_hidden_filters[i]
            output_channels = n_hidden_filters[i + 1]
            cnn_block.extend([nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                        kernel_size=kernel_size, padding=pad, padding_mode=padding_mode),
                              nn.BatchNorm2d(num_features=output_channels),
                              nn.ReLU()])
        self.cnn_block = nn.Sequential(*cnn_block)

        # Parallel pooling: avg and max over the same feature map, concatenated in forward
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.use_stats = use_stats
        self.register_buffer('quantiles', torch.tensor(quantiles))
        n_stats = (3 + len(quantiles)) if use_stats else 0  # mean, std, fuzziness + quantiles

        in_features = 2 * n_hidden_filters[-1] + n_stats

        mlp_block = []
        prev = in_features
        for h in mlp_hidden:
            mlp_block.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        mlp_block.append(nn.Linear(prev, 1))
        self.mlp_block = nn.Sequential(*mlp_block)

    def _global_stats(self, heatmap):
        # heatmap: (B, 1, H, W) -> (B, n_stats)
        flat = heatmap.reshape(heatmap.shape[0], -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True)
        fuzz = (4.0 * flat * (1.0 - flat)).mean(dim=1, keepdim=True)
        q = torch.quantile(flat, self.quantiles.to(flat.device), dim=1).t()  # (B, n_quantiles)
        return torch.cat([mean, std, fuzz, q], dim=1)

    def forward(self, x):
        # x: (B, C_in, H, W). Assumes the heatmap is channel 0
        feats = self.cnn_block(x)
        avg = torch.flatten(self.avg_pool(feats), 1)
        mx = torch.flatten(self.max_pool(feats), 1)
        pooled = torch.cat([avg, mx], dim=1)

        if self.use_stats:
            pooled = torch.cat([pooled, self._global_stats(x[:, :1])], dim=1)

        return torch.sigmoid(self.mlp_block(pooled))
            
        
