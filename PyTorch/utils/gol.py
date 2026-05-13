import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Conway's Game of Life
# ------------------------------
class GameOfLife:
    """
    Conway's Game of Life cellular automaton with periodic boundary conditions.
    """
    kernel = torch.tensor([[1,1,1], 
                           [1,0,1], 
                           [1,1,1]], dtype=torch.float32).view(1, 1, 3, 3)
    # -------------------------------------------------------------------- 
    def __init__(self, delta:int=1):
    
        """
        Args:
            delta (int): Number of generations between the initial
                         and final states. Defaults to 1.
        """
        self.delta = delta
    # -------------------------------------------------------------------- 
    @staticmethod 
    def _state_to_tensor(state):
        """
        Converts any array-like input to a (1, 1, H, W) float tensor.

        Args:
            state: 2D array-like, or tensor of shape (H,W), (1,H,W) or (1,1,H,W).

        Returns:
            torch.Tensor of shape (1, 1, H, W).

        Raises:
            ValueError: if the input cannot be interpreted as a 2D grid.
        """

        # Check whether the input is already a tensor or not:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Remove all dims = 1
        state = state.float().squeeze()

        # Check whether the state is a Rank 2 tensor:
        if (state.ndim != 2):
            raise ValueError(f'Expected a 2D grid after squeezing, but got shape {tuple(state.shape)}')
        
        return state.unsqueeze(0).unsqueeze(0) # (1,1,H,W)
    # -------------------------------------------------------------------- 
    def step(self, state, return_as:str='torch'):
        """
        Advances the grid by one generation.

        Args:
            state: 2D array-like, or tensor of shape (H,W), (1,H,W) or (1,1,H,W).
            return_as (str): Output format — 'torch' (default) or 'numpy'.

        Returns:
            torch.Tensor of shape (1,1,H,W) or np.ndarray of shape (H,W).

        Raises:
            ValueError: if the input cannot be interpreted as a 2D grid or if return_as is not 'torch' or 'numpy'.

        """
        # Check the arguments:
        if return_as not in ('torch', 'numpy'):
            raise ValueError(f"return_as must be 'torch' or 'numpy', got '{return_as}'.")

        # Reshape of the state to Rank 4 tensor:
        if not isinstance(state, torch.Tensor) or state.ndim != 4:
            state = self._state_to_tensor(state)

        # Padding of the state:
        state_padded = F.pad(state, (1, 1, 1, 1), mode='circular') # (1,1, H+2, W+2)

        # Compute the neighbours
        neighbors = F.conv2d(state_padded, self.kernel) # (1,1, H,W)

        # Compute the next generation state:
        final_state =  ((neighbors == 3) | ((state == 1) & (neighbors == 2))).float() # (1,1,H,W)
        
        # return_as logic:
        if return_as=='numpy':
            final_state = final_state.squeeze().detach().numpy() # (H,W)
        
        return final_state
    # -------------------------------------------------------------------- 
    def evolution(self, state, return_as="torch"):
        """
        Advances the grid by delta generations.

        Args:
            state: 2D array-like, or tensor of shape (H,W), (1,H,W) or (1,1,H,W).
            return_as (str): Output format — 'torch' (default) or 'numpy'.

        Returns:
            torch.Tensor of shape (1,1,H,W) or np.ndarray of shape (H,W).

        Raises:
            ValueError: if the input cannot be interpreted as a 2D grid or if return_as is not 'torch' or 'numpy'.

        """

        # Check the arguments:
        if return_as not in ('torch', 'numpy'):
            raise ValueError(f"return_as must be 'torch' or 'numpy', got '{return_as}'.")
        
        state = self._state_to_tensor(state) # (1,1,H,W)

        # Compute delta generations:
        for _ in range(self.delta):
            state = self.step(state, "torch")

        # return_as logic:
        if return_as=='numpy':
            state = state.squeeze().detach().numpy() # (H,W)
        
        return state
    # -------------------------------------------------------------------- 
            
# ------------------------------
# Differentiable Game of Life
# ------------------------------
class DiffGoL(nn.Module, GameOfLife):
    """
    Differentiable Conway's Game of Life cellular automaton with periodic boundary conditions.
    """
    def __init__(self, delta:int=1, epsilon:float=50.0, order:float=4.0):
        """
        Args:
            delta (int): Number of generations between the initial
                         and final states. Defaults to 1.
            epsilon (float): steepness of the differentiable Conway's Game of Life rule function. Controls
        the strictness of aplication the target rule.
            order (float): exponent that controls the stricness of the binarisation.
        """
        nn.Module.__init__(self)
        GameOfLife.__init__(self,delta=delta)
        self.epsilon = epsilon
        self.order = order
    
    # -------------------------------------------------------------------- 
    def step(self, state, return_as:str='torch'):
        # Check the arguments:
        if return_as not in ('torch', 'numpy'):
            raise ValueError(f"return_as must be 'torch' or 'numpy', got '{return_as}'.")

        # Reshape of the state to Rank 4 tensor:
        if not isinstance(state, torch.Tensor) or state.ndim != 4:
            state = self._state_to_tensor(state)

        # Ensure (B, 1, H, W):
        if state.ndim == 3:
            state = state.unsqueeze(1)
        # Padding of the state:
        state_padded = F.pad(state, (1, 1, 1, 1), mode='circular') # (1,1, H+2, W+2)

        # Compute the neighbours
        neighbors = F.conv2d(state_padded, self.kernel.to(state.device)) # (1,1, H,W)

        # Lógica de actualización:
        survive = 2*(state**self.order)*(torch.sigmoid(-self.epsilon*((neighbors - 2)**2)) + torch.sigmoid(-self.epsilon*(neighbors - 3)**2) ) 
        born = 2*((1 - state)**self.order)* torch.sigmoid(-self.epsilon*(neighbors - 3) ** 2) 
        state = torch.clip(survive + born, 0, 1)
        
        # return_as logic:
        if return_as=='numpy':
            state = state.squeeze().detach().numpy() # (H,W)
        

        return state
    # -------------------------------------------------------------------- 
    def evolution(self, state):
        '''
        Overrides GameOfLife.evolution to support batched input.

        Args:
            state (torch.Tensor): shape (B, 1, H, W)

        Returns:
            torch.Tensor: shape (B, 1, H, W)
        '''
        if state.ndim == 3:
            state = state.unsqueeze(1)

        for _ in range(self.delta):
            state = self.step(state)

        return state
    # ----------------------------------------------------------------------
    def forward(self, state):
        
        return self.evolution(state)