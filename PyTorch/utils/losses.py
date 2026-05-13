import torch.nn as nn
import torch.nn.functional as F
import torch
 
class BinaryFocalCrossEntropy(nn.Module):
    """
    Computes focal cross-entropy loss between true labels and predictions. Binary cross-entropy loss 
    is often used for binary (0 or 1) classification tasks. According to 
    [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a "focal factor" to down-weight easy examples and focus more
    on hard examples. By default, the focal tensor is computed as follows:

    `focal_factor = (1 - output) ** gamma` for class 1

    `focal_factor = output ** gamma` for class 0
    
    where `gamma` is a focusing parameter. When `gamma=0`, this function is
    equivalent to the binary crossentropy loss.

    """
    def __init__(self, gamma:float=2.0, alpha:float=0.25, apply_class_balancing:bool=False, from_logits:bool=False):
        """
        Arguments:
            `gamma` (float): focusing parameter used to compute the focal factor, default is
            `2.0` as mentioned in the reference. 
            ([Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).)

            `alpha`(float): A weight balancing factor for class 1, default is `0.25` as
            mentioned in reference [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).  
            The weight for class 0 is `1.0 - alpha`

            `apply_class_balancing`(bool): A bool, whether to apply weight balancing on the
            binary classes 0 and 1. Default is `False`.

            `from_logits`: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` are probabilities (`False`).
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.apply_class_balancing = apply_class_balancing
        self.from_logits = from_logits

    def forward(self, pred, true):

        # from_logits?
        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(pred, true, reduction='none')
            probs = torch.sigmoid(pred)
        else:
            clamped_probs = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
            bce_loss = F.binary_cross_entropy(clamped_probs, true, reduction='none')
            probs = pred
        
        # Compute p_t --> p_t(1) = pred ; p_t(0) = 1 - pred
        p_t = true * probs + (1 - true) * (1 - probs)

        # Focal Factor:
        focal_factor = (1 - p_t)**self.gamma

        loss = focal_factor * bce_loss
        # apply_class_balancing?
        if self.apply_class_balancing:
            alpha_t = true*self.alpha + (1 - true)*(1 - self.alpha)
            loss = alpha_t * loss
        
        return loss.mean() 
    
class FuzzyLoss(nn.Module):

    """
    Computes the fuzziness index of a continuous probability prediction.

    The index is based on the scaled variance of the pixel probabilities, reaching 
    its maximum when probabilities are exactly 0.5 (maximum uncertainty).

    """

    def __init__(self, from_logits:bool=False):
        """
        Arguments:
            `from_logits`: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` are probabilities (`False`).
        """
        super().__init__()
        self.from_logits= from_logits
    
    def forward(self, pred, true=None):

        if self.from_logits:
            probs = torch.sigmoid(pred)
        else:
            probs = pred
        
        binary_loss = 4*torch.mean(probs*(1-probs))

        return binary_loss
    
class DiceLoss(nn.Module):
    """
    Soft Dice loss for the initial state predictions in DiffGoL-style models.

    Computes the complement of the Sørensen-Dice coefficient between predicted
    and ground-truth initial boards, averaged over the batch and channel
    dimensions. Acts as a region-overlap term complementary to per-pixel
    losses such as (focal) BCE.

        L = 1 - mean( (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth) )

    Predictions are assumed to be probabilities in [0, 1] (sigmoid already
    applied upstream). Only the initial boards are consumed; the final boards
    are ignored and handled by separate loss terms.
    """
    def __init__(self, smooth:float=1e-8):
        """
        Args:
            smooth (float): Stabilisation constant added to numerator and
                denominator. A small value (default 1e-8) only guards against
                the degenerate 0/0 case; larger values (e.g. 1.0) introduce
                additional regularisation that biases the ratio for samples
                with few positive pixels.
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, true):

        # Calculate intersection and union:
        I = (pred * true).sum(dim=(2,3))
        U = pred.sum(dim=(2,3)) + true.sum(dim=(2,3))

        # Compute loss:
        dice = (2.0 * I + self.smooth)/(U + self.smooth)

        return 1 - dice.mean()

 
class FocalBCEFuzz(BinaryFocalCrossEntropy, FuzzyLoss):
    """
    Combined loss for DiffGoL-style models that output paired predictions.

    Applies three components:
        - BinaryFocalCrossEntropy on the initial state predictions.
        - Standard BCE on the final state predictions (physics constraint).
        - FuzzyLoss on the initial state predictions (uncertainty penalty).

    Total loss = focal_loss + lambda_phys * bce_loss + lambda_bin * fuzzy_loss
    """

    def __init__(self, num_outputs:int = 2, lambda_init:float=1.0, lambda_phys:float=1.0, lambda_bin:float=1.0, gamma:float=2.0, alpha:float=0.25, 
                 apply_class_balancing:bool=False, from_logits:bool=False):
        """
        Args:
            num_outputs (int): Number of outputs of the model. If 2, the model returns
            (initial_state, final_state). If 1, returns only the initial state.
            lambda_init (float): Weight for the FBCE loss on initial states. Default is 1.0.
            lambda_phys (float): Weight for the BCE loss on final states. Default is 1.0.
            lambda_bin (float): Weight for the FuzzyLoss. Default is 1.0.
            gamma (float): Focusing parameter for the focal factor. Default is 2.0.
            alpha (float): Class balancing factor for class 1. Default is 0.25.
            apply_class_balancing (bool): Whether to apply alpha weighting. Default is False.
            from_logits (bool): Whether inputs are logits. Default is False.
        """
        
        BinaryFocalCrossEntropy.__init__(self, gamma, alpha, apply_class_balancing, from_logits)
        FuzzyLoss.__init__(self, from_logits)
        self.num_outputs = num_outputs
        self.lambda_init = lambda_init
        self.lambda_phys = lambda_phys
        self.lambda_bin = lambda_bin
    
    def forward(self, pred, true):

        # Extract the data:
        if self.num_outputs == 2:
            init_pred, fin_pred = pred
            init_true, fin_true = true
        else:
            init_pred = pred
            init_true = true

        # Focal Loss:
        if self.lambda_init:
            focal_loss = self.lambda_init * BinaryFocalCrossEntropy.forward(self, init_pred, init_true)
        else:
            focal_loss = 0

        # BCE loss:
        if self.lambda_phys:
            if self.from_logits:
                bce_loss = self.lambda_phys * F.binary_cross_entropy_with_logits(fin_pred, fin_true, reduction='mean')
            else:
                fin_pred_clamped = torch.clamp(fin_pred, min=1e-7, max=1.0 - 1e-7)
                bce_loss = self.lambda_phys * F.binary_cross_entropy(fin_pred_clamped, fin_true, reduction='mean')
        else:
            bce_loss = 0

        # Fuzzy Loss:
        if self.lambda_bin:
            fuzzy_loss = self.lambda_bin * FuzzyLoss.forward(self, init_pred)
        else:
            fuzzy_loss = 0

        return focal_loss + bce_loss + fuzzy_loss
        
class FocalFuzz(BinaryFocalCrossEntropy, FuzzyLoss):
    """
    Combined loss for single-output models.

    Applies two components:
        - BinaryFocalCrossEntropy on the predictions.
        - FuzzyLoss on the predictions (uncertainty penalty).

    Total loss = focal_loss + lambda_bin * fuzzy_loss
    """
    def __init__(self, lambda_bin:float=1.0, gamma:float=2.0, alpha:float=0.25, apply_class_balancing:bool=False, from_logits:bool=False):
        """
        Args:
            lambda_bin (float): Weight for the FuzzyLoss. Default is 1.0.
            gamma (float): Focusing parameter for the focal factor. Default is 2.0.
            alpha (float): Class balancing factor for class 1. Default is 0.25.
            apply_class_balancing (bool): Whether to apply alpha weighting. Default is False.
            from_logits (bool): Whether inputs are logits. Default is False.
        """
        BinaryFocalCrossEntropy.__init__(self, gamma, alpha, apply_class_balancing, from_logits)
        FuzzyLoss.__init__(self, from_logits)
        
        self.lambda_bin = lambda_bin
    
    def forward(self, pred, true):

        # Focal loss:
        focal_loss = BinaryFocalCrossEntropy.forward(self, pred, true)
        # Fuzzy Loss:
        fuzzy_loss = self.lambda_bin * FuzzyLoss.forward(self, pred, true)

        return focal_loss + fuzzy_loss

class FFFuzz(BinaryFocalCrossEntropy, FuzzyLoss):
    """
    Combined loss for DiffGoL-style models that output paired predictions.

    Extends FocalBCEFuzz by replacing the standard BCE on final states with
    BinaryFocalCrossEntropy, applying the focal factor to both states.

    Applies three components:
        - BinaryFocalCrossEntropy on the initial state predictions.
        - BinaryFocalCrossEntropy on the final state predictions.
        - FuzzyLoss on the initial state predictions (uncertainty penalty).

    Total loss = init_focal_loss + lambda_phys * fin_focal_loss + lambda_bin * fuzzy_loss
    """

    def __init__(self, lambda_phys:float=1.0, lambda_bin:float=1.0, gamma:float=2.0, alpha:float=0.25, 
                 apply_class_balancing:bool=False, from_logits:bool=False):
        """
        Args:
            lambda_phys (float): Weight for the focal loss on final states. Default is 1.0.
            lambda_bin (float): Weight for the FuzzyLoss. Default is 1.0.
            gamma (float): Focusing parameter for the focal factor. Default is 2.0.
            alpha (float): Class balancing factor for class 1. Default is 0.25.
            apply_class_balancing (bool): Whether to apply alpha weighting. Default is False.
            from_logits (bool): Whether inputs are logits. Default is False.
        """
        
        BinaryFocalCrossEntropy.__init__(self, gamma, alpha, apply_class_balancing, from_logits)
        FuzzyLoss.__init__(self, from_logits)
        
        self.lambda_phys = lambda_phys
        self.lambda_bin = lambda_bin
    
    def forward(self, pred, true):

        # Extract the data:
        init_pred, fin_pred = pred
        init_true, fin_true = true

        # Focal Loss init:
        init_focal_loss = BinaryFocalCrossEntropy.forward(self, init_pred, init_true)
        
        # Focal Loss fin:
        fin_focal_loss = BinaryFocalCrossEntropy.forward(self, fin_pred, fin_true)
    
        # Fuzzy Loss:
        fuzzy_loss = self.lambda_bin * FuzzyLoss.forward(self, init_pred)

        return init_focal_loss + fin_focal_loss + fuzzy_loss
    

class FocalBCE(BinaryFocalCrossEntropy):
    """
    Combined loss for DiffGoL-style models that output paired predictions.

    Applies three components:
        - BinaryFocalCrossEntropy on the initial state predictions.
        - Standard BCE on the final state predictions (physics constraint).
    
    Total loss = focal_loss + lambda_phys * bce_loss
    """

    def __init__(self, lambda_phys:float=1.0, gamma:float=2.0, alpha:float=0.25, 
                 apply_class_balancing:bool=False, from_logits:bool=False):
        """
        Args:
            lambda_phys (float): Weight for the BCE loss on final states. Default is 1.0.
            gamma (float): Focusing parameter for the focal factor. Default is 2.0.
            alpha (float): Class balancing factor for class 1. Default is 0.25.
            apply_class_balancing (bool): Whether to apply alpha weighting. Default is False.
            from_logits (bool): Whether inputs are logits. Default is False.
        """
        
        BinaryFocalCrossEntropy.__init__(self, gamma, alpha, apply_class_balancing, from_logits)
        
        self.lambda_phys = lambda_phys
        
    
    def forward(self, pred, true):

        # Extract the data:
        init_pred, fin_pred = pred
        init_true, fin_true = true

        # Focal Loss:
        focal_loss = BinaryFocalCrossEntropy.forward(self, init_pred, init_true)

        # BCE loss:
        if self.from_logits:
            bce_loss = self.lambda_phys * F.binary_cross_entropy_with_logits(fin_pred, fin_true, reduction='mean')
        else:
            fin_pred_clamped = torch.clamp(fin_pred, min=1e-7, max=1.0 - 1e-7)
            bce_loss = self.lambda_phys * F.binary_cross_entropy(fin_pred_clamped, fin_true, reduction='mean')

        return focal_loss + bce_loss 
    

class FocalBCEDiceFuzz(BinaryFocalCrossEntropy, FuzzyLoss, DiceLoss):
    """
    Combined loss for DiffGoL-style models that output paired predictions.

    Applies three components:
        - BinaryFocalCrossEntropy on the initial state predictions.
        - Standard BCE on the final state predictions (physics constraint).
        - FuzzyLoss on the initial state predictions (uncertainty penalty).

    Total loss = focal_loss + lambda_phys * bce_loss + lambda_bin * fuzzy_loss
    """

    def __init__(self, num_outputs:int = 2, lambda_init:float=1.0, lambda_phys:float=1.0, lambda_bin:float=1.0, 
                 lambda_dice:float=0.0, smooth:float=1e-8,
                 gamma:float=2.0, alpha:float=0.25, apply_class_balancing:bool=False, from_logits:bool=False):
        """
        Args:
            num_outputs (int): Number of outputs of the model. If 2, the model returns
            (initial_state, final_state). If 1, returns only the initial state.
            lambda_init (float): Weight for the FBCE loss on initial states. Default is 1.0.
            lambda_phys (float): Weight for the BCE loss on final states. Default is 1.0.
            lambda_bin (float): Weight for the FuzzyLoss. Default is 1.0.
            gamma (float): Focusing parameter for the focal factor. Default is 2.0.
            alpha (float): Class balancing factor for class 1. Default is 0.25.
            apply_class_balancing (bool): Whether to apply alpha weighting. Default is False.
            from_logits (bool): Whether inputs are logits. Default is False.
        """
        
        BinaryFocalCrossEntropy.__init__(self, gamma, alpha, apply_class_balancing, from_logits)
        FuzzyLoss.__init__(self, from_logits)
        DiceLoss.__init__(self, smooth)
    
        self.num_outputs = num_outputs
        self.lambda_init = lambda_init
        self.lambda_phys = lambda_phys
        self.lambda_bin = lambda_bin
        self.lambda_dice = lambda_dice
    
    def forward(self, pred, true):

        # Extract the data:
        if self.num_outputs == 2:
            init_pred, fin_pred = pred
            init_true, fin_true = true
        else:
            init_pred = pred
            init_true = true

        # Focal Loss:
        if self.lambda_init:
            focal_loss = self.lambda_init * BinaryFocalCrossEntropy.forward(self, init_pred, init_true)
        else:
            focal_loss = 0

        # BCE loss:
        if self.lambda_phys:
            if self.from_logits:
                bce_loss = self.lambda_phys * F.binary_cross_entropy_with_logits(fin_pred, fin_true, reduction='mean')
            else:
                fin_pred_clamped = torch.clamp(fin_pred, min=1e-7, max=1.0 - 1e-7)
                bce_loss = self.lambda_phys * F.binary_cross_entropy(fin_pred_clamped, fin_true, reduction='mean')
        else:
            bce_loss = 0

        # Fuzzy Loss:
        if self.lambda_bin:
            fuzzy_loss = self.lambda_bin * FuzzyLoss.forward(self, init_pred)
        else:
            fuzzy_loss = 0

        # Dice Loss:
        if self.lambda_bin:
            dice_loss = self.lambda_dice * DiceLoss.forward(self, init_pred, init_true)
        else:
            dice_loss = 0

        return focal_loss + bce_loss + fuzzy_loss + dice_loss