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

### UNet: Heatmaps cleaner
class DoubleConv(nn.Module):
    """
    UNet block formed by two convolutions:

    Conv2d --> BatchNorm --> ReLU --> Conv2d --> BatchNorm --> ReLU
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:str|int = 1, 
                 padding_mode:str='circular', apply_batchnorm:bool=True):
        """
        Args:
            in_channels (int): number of channels of the input feature map.
            out_channels (int): number of channels of the output feature map.
            kernel_size (int): size of the kernel. Default: 3.
            stride (int): step between convolutions. Default: 1.
            padding (str or int): controls how the image is treated at the borders. It can be either an int, representing the number of rows/cols to be padded,
                or an str (e.g 'same', 'valid',...). .
            padding_mode (str): controls how the padding is applied. For example, 'circular' refers to periodic boundary. Default 'circular'.
        """
        super().__init__()
        
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                                  stride=stride, padding=padding, padding_mode=padding_mode),
                                                  nn.BatchNorm2d(num_features=out_channels) if apply_batchnorm else nn.Identity(),
                                                  nn.ReLU(),
                                                  
                                        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                                  stride=stride, padding=padding, padding_mode=padding_mode),
                                                  nn.BatchNorm2d(num_features=out_channels) if apply_batchnorm else nn.Identity(),
                                                  nn.ReLU())
    def forward(self, x):
        return self.double_conv(x)
    
class DecoderBlock(nn.Module):
    """
    UNet decoder block formed by:

    Up-Conv --> DoubleConv
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:str|int=1, padding_mode:str='circular', apply_batchnorm:bool=True, 
                 scale_factor:int=2, mode:str='bilinear'):
        """
        Args:
            in_channels (int): number of channels of the input feature map.
            out_channels (int): number of channels of the output feature map.
            kernel_size (int): size of the kernel. Default: 3.
            stride (int): step between convolutions. Default: 1.
            padding (str or int): controls how the image is treated at the borders. It can be either an int, representing the number of rows/cols to be padded,
                or an str (e.g 'same', 'valid',...). Default: 1.
            padding_mode (str): controls how the padding is applied. For example, 'circular' refers to periodic boundary. Default:'circular'.
            scale_factor (int): scale_factor used to increase/reduce the channels through the the UNet.
            mode (str): the upsampling algorithm, which can be one of 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'. Default: 'bilinear'.
        """
        super().__init__()
        self.up_sampling = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode=mode),
                                           nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1))
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                      stride=stride, padding=padding, padding_mode=padding_mode, apply_batchnorm=apply_batchnorm)

    def forward(self, x, skip):
        x_up = self.up_sampling(x)
        if x_up.shape != skip.shape:
            # F.pad ensures x_up matches skip dimensions after upsampling odd-sized inputs
            x_up = F.pad(x_up, (0, skip.shape[3] - x_up.shape[3], 0, skip.shape[2] - x_up.shape[2])) 

        return self.double_conv(torch.cat([x_up, skip],1))
    
class UNet(nn.Module):
    """
    UNet architecture adapted to the Conway's Reverse Game of Life Problem. The architecture implemented is:

    DoubleConv --> MaxPool --> Bottleneck --> DecoderBlock --> Conv2d --> Sigmoid
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:str|int=1, padding_mode:str='circular', apply_batchnorm:bool=True,
                 scale_factor:int=2, mode:str='bilinear', pool_kernel_size:int=2, pool_stride:int=2):
        """
        Args:
            in_channels (int): number of channels of the input feature map.
            out_channels (int): number of channels of the output feature map.
            kernel_size (int): size of the kernel. Default: 3.
            stride (int): step between convolutions. Default: 1.
            padding (str or int): controls how the image is treated at the borders. It can be either an int, representing the number of rows/cols to be padded,
                or an str (e.g 'same', 'valid',...). Default: 1.
            padding_mode (str): controls how the padding is applied. For example, 'circular' refers to periodic boundary. Default: 'circular'.
            scale_factor (int): scale_factor used to increase/reduce the channels through the the UNet.
            mode (str): the upsampling algorithm, which can be one of 'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'. Default: 'bilinear'.
            pool_kernel_size (int): size of the kernel used for the max pooling operation. Default: 2
            pool_stride (int): steps between pooling operations. Default: 2.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Encoder:  (C0 x H x W) --> (Cx H//2 x W//2)
        self.encoder1 = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   padding_mode=padding_mode, apply_batchnorm=apply_batchnorm) # (Old)

        self.encoder2 = DoubleConv(in_channels=out_channels, out_channels=out_channels*scale_factor, kernel_size=kernel_size, stride=stride, padding=1, 
                                   padding_mode='zeros', apply_batchnorm=apply_batchnorm)
        self.encoder3 = DoubleConv(in_channels=out_channels*scale_factor, out_channels=out_channels*(scale_factor)**2, kernel_size=kernel_size, stride=stride, padding=1, 
                                   padding_mode='zeros', apply_batchnorm=apply_batchnorm)
        self.encoder4 = DoubleConv(in_channels=out_channels*(scale_factor)**2, out_channels=out_channels*(scale_factor)**3, kernel_size=kernel_size, stride=stride, padding=1, 
                                   padding_mode='zeros', apply_batchnorm=apply_batchnorm)
        
        # # Bottleneck: (C x H//2 x W//2) --> (C' x H//2 x W//2)
        self.bottleneck = DoubleConv(in_channels = out_channels*(scale_factor)**3, out_channels=out_channels*(scale_factor)**4, kernel_size=kernel_size, stride=stride, padding=padding,
                                   padding_mode=padding_mode, apply_batchnorm=apply_batchnorm) 

        # # Bottleneck: (C x H//2 x W//2) --> (C' x H//2 x W//2) (Old)
        # self.bottleneck = DoubleConv(in_channels = out_channels, out_channels=out_channels*scale_factor, kernel_size=kernel_size, stride=stride, padding=padding,
        #                            padding_mode=padding_mode, apply_batchnorm=apply_batchnorm) 

        # Decoder: (C' x H//2 x W//2) --> (C x H x W)
        self.decoder1 = DecoderBlock(in_channels=out_channels*(scale_factor)**4, out_channels=out_channels*(scale_factor)**3, kernel_size=kernel_size, stride=stride, padding=1,
                                     padding_mode='zeros', apply_batchnorm=apply_batchnorm, scale_factor=scale_factor, mode=mode) 
        self.decoder2 = DecoderBlock(in_channels=out_channels*(scale_factor)**3, out_channels=out_channels*(scale_factor)**2, kernel_size=kernel_size, stride=stride, padding=1,
                                     padding_mode='zeros', apply_batchnorm=apply_batchnorm, scale_factor=scale_factor, mode=mode) 
        self.decoder3 = DecoderBlock(in_channels=out_channels*(scale_factor)**2, out_channels=out_channels*scale_factor, kernel_size=kernel_size, stride=stride, padding=1,
                                     padding_mode='zeros', apply_batchnorm=apply_batchnorm, scale_factor=scale_factor, mode=mode) 
        self.decoder4 = DecoderBlock(in_channels=out_channels*scale_factor, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                     padding_mode=padding_mode, apply_batchnorm=apply_batchnorm, scale_factor=scale_factor, mode=mode) # (Old) 
        
        # Output: # (C x H x W) --> (C0 x H x W)
        self.output_layer = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                                    padding_mode=padding_mode),
                                          nn.Sigmoid()) 
    def forward(self, x):
        # Encoder:
        x1 = self.encoder1(x) # (Old)
        x2 = self.encoder2(self.pool(x1))
        x3 = self.encoder3(self.pool(x2))
        x4 = self.encoder4(self.pool(x3))
        # Bottleneck:
        # x_bottle = self.bottleneck(self.pool(x1))# (Old)
        x_bottle = self.bottleneck(self.pool(x4))
        # Decoder:
        # x2 = self.decoder1(x_bottle, x1) # (Old)
        x5 = self.decoder1(x_bottle, x4)
        x6 = self.decoder2(x5, x3)
        x7 = self.decoder3(x6,x2)
        x8 = self.decoder4(x7,x1)
        # Output:
        # return self.output_layer(x2) # (Old)
        return self.output_layer(x8)  

class UNetDiffGoL(UNet):
    """
    UNet-based model for the reverse Conway's Game of Life problem.

    Extends UNet to predict the initial state of a Conway's Game of Life board
    from its final state, and subsequently evolves it forward using a differentiable
    Game of Life simulator.
    """
    def __init__(self, in_channels:int, out_channels:int, delta:int=1, epsilon:float=50.0, order:float=4.0, kernel_size:int=3, 
                 stride:int=1, padding:str|int=1, padding_mode:str='circular', apply_batchnorm:bool=True,
                 scale_factor:int=2, mode:str='bilinear', pool_kernel_size:int=2, pool_stride:int=2):
        """
        Args:
            in_channels (int): Number of channels of the input feature map.
            out_channels (int): Number of channels of the intermediate feature maps.
            delta (int): Number of Game of Life evolution steps. Default: 1.
            epsilon (float): Sharpness of the differentiable step function. Default: 50.0.
            order (float): Order of the differentiable approximation. Default: 4.0.
            kernel_size (int): Convolutional kernel size. Default: 3.
            stride (int): Convolution stride. Default: 1.
            padding (str or int): Padding applied to convolutions. Default: 1.
            padding_mode (str): Padding mode. Default: 'circular'.
            apply_batchnorm (bool): Whether to apply batch normalization. Default: True.
            scale_factor (int): Channel scaling factor across UNet levels. Default: 2.
            mode (str): Upsampling interpolation mode. Default: 'bilinear'.
            pool_kernel_size (int): MaxPool kernel size. Default: 2.
            pool_stride (int): MaxPool stride. Default: 2.
        """
        UNet.__init__(self, in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      padding_mode=padding_mode, apply_batchnorm=apply_batchnorm, scale_factor=scale_factor, 
                      mode=mode, pool_kernel_size=pool_kernel_size, pool_stride=pool_stride)
        self.diffGoL = DiffGoL(delta, epsilon, order)
    
    def forward(self, x):
        initial_state = UNet.forward(self, x)
        final_state = self.diffGoL(initial_state)
        return initial_state, final_state
    
### CNN: Heatmap cleaner.
class CNN(nn.Module):
    """
    CNN-based heatmap cleaner for the reverse Conway's Game of Life problem.

    Refines the heatmap produced by an upstream model into a clean binary board,
    using a bottleneck architecture that expands and then contracts the channel
    dimension while preserving spatial resolution throughout.

    The architecture is:
        Conv2d --> BN/Identity --> ReLU --> Conv2d --> BN/Identity --> ReLU -->
        Conv2d --> BN/Identity --> ReLU --> Conv2d --> Sigmoid

    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:str|int=1, 
                 padding_mode:str='circular', apply_batch_norm:bool=True, scale_factor:int = 2):
        """
        Args:
            in_channels (int): Number of channels of the input feature map.
            out_channels (int): Number of channels of the first hidden layer.
            kernel_size (int): Convolutional kernel size. Default: 3.
            stride (int): Convolution stride. Default: 1.
            padding (str or int): Padding applied to convolutions. Default: 1.
            padding_mode (str): Padding mode. Default: 'circular'.
            apply_batch_norm (bool): Whether to apply batch normalization. Default: True.
            scale_factor (int): Multiplier for the bottleneck channel dimension. Default: 2.
        """
        super().__init__()
        self.cnn_model = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                 padding=padding, padding_mode=padding_mode),
                                       nn.BatchNorm2d(num_features=out_channels) if apply_batch_norm else nn.Identity(),
                                       nn.ReLU(),
                                       
                                       nn.Conv2d(in_channels=out_channels, out_channels=out_channels*scale_factor, kernel_size=kernel_size, stride=stride,
                                                 padding=padding, padding_mode=padding_mode),
                                       nn.BatchNorm2d(num_features=out_channels*scale_factor) if apply_batch_norm else nn.Identity(),
                                       nn.ReLU(),
                                       
                                       nn.Conv2d(in_channels=out_channels*scale_factor, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                 padding=padding, padding_mode=padding_mode),
                                       nn.BatchNorm2d(num_features=out_channels) if apply_batch_norm else nn.Identity(),
                                       nn.ReLU(),
                                       
                                       nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                                                 padding=padding, padding_mode=padding_mode),
                                       nn.Sigmoid())
    def forward(self, x):
        return self.cnn_model(x)
    
class CNNDiffGoL(CNN):
    """CNN-based heatmap cleaner with integrated differentiable Game of Life evolution.

    Extends CNN to predict a clean initial state from an input heatmap and
    subsequently evolve it forward using a differentiable Game of Life simulator.
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:str|int=1, padding_mode:str='circular',
                  apply_batch_norm:bool=True, scale_factor:int=2, delta:int=1, epsilon:float=50.0, order:float=4.0):
        """
        Args:
            in_channels (int): Number of channels of the input feature map.
            out_channels (int): Number of channels of the first hidden layer.
            kernel_size (int): Convolutional kernel size. Default: 3.
            stride (int): Convolution stride. Default: 1.
            padding (str or int): Padding applied to convolutions. Default: 1.
            padding_mode (str): Padding mode. Default: 'circular'.
            apply_batch_norm (bool): Whether to apply batch normalization. Default: True.
            scale_factor (int): Multiplier for the bottleneck channel dimension. Default: 2.
            delta (int): Number of Game of Life evolution steps. Default: 1.
            epsilon (float): Sharpness of the differentiable step function. Default: 50.0.
            order (float): Order of the differentiable approximation. Default: 4.0.
        """
        CNN.__init__(self, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     padding_mode = padding_mode, apply_batch_norm=apply_batch_norm, scale_factor=scale_factor)
        self.diffGoL = DiffGoL(delta=delta, epsilon=epsilon, order=order)

    def forward(self, x):
        initial_state = CNN.forward(self, x)
        final_state = self.diffGoL(initial_state)
        return initial_state, final_state
    
### Previous Models: Tensorflow 
class ClassicModelTF(nn.Module):
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
        self.input_layer = nn.Sequential(nn.Conv2d(in_channels= 1, out_channels=n_hidden_filters, 
                                     kernel_size=kernel_size, padding = 'same', padding_mode='zeros'),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=n_hidden_filters),
                                     )
        hidden_blocks = []
        for _ in range(n_hidden_convs):
            hidden_blocks.extend([nn.Conv2d(in_channels=n_hidden_filters, out_channels=n_hidden_filters, 
                                     kernel_size=kernel_size, padding = 'same', padding_mode='zeros'),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=n_hidden_filters)
                                     ])
        
        self.hidden_layers = nn.Sequential(*hidden_blocks)

        self.output_layer = nn.Sequential(nn.Conv2d(in_channels=n_hidden_filters, out_channels=1, 
                                      kernel_size=kernel_size, padding = 'same', padding_mode='zeros'),
                                      nn.Sigmoid())

    def forward(self, X):
        assert (isinstance(X, torch.Tensor)), f'Expected a Tensor as input, but got {type(X)}'
        assert (X.ndim == 4), f'Expected a Rank 4 Tensor as input, but got Rank {X.ndim} Tensor --> (B,1, H,W)'

        X = self.input_layer(X)
        X = self.hidden_layers(X)

        return self.output_layer(X)
    
class DiffGoLModelTF(ClassicModelTF):
    """
    Extension of the Classic Model with the DiffGoL layer:

    ClassicModel --> DiffGoL

    """
    def __init__(self, n_hidden_convs:int, n_hidden_filters:int, kernel_size:int, 
                delta:int=1, epsilon:float=50.0, order:float=4.0):
        ClassicModelTF.__init__(self, n_hidden_convs, n_hidden_filters, kernel_size)
        
        self.diffGoL = DiffGoL(delta, epsilon, order)

    def forward(self, X):

        initial_state = ClassicModelTF.forward(self, X)

        final_state = self.diffGoL.forward(initial_state)

        return initial_state, final_state