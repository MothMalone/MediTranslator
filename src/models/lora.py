"""
LoRA (Low-Rank Adaptation) Implementation for Transformer
Efficient fine-tuning by adapting only low-rank decomposition matrices.

Reference: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class LoRALayer(nn.Module):
    """
    LoRA Layer: Adds low-rank adaptation to a linear layer.
    
    W = W0 + BA, where:
        - W0 is frozen pretrained weight
        - B is (d_out, r) trainable matrix
        - A is (r, d_in) trainable matrix
        - r << min(d_in, d_out) is the rank
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of low-rank decomposition
        alpha: Scaling factor (usually equal to rank)
        dropout: Dropout probability for LoRA weights
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA weights (trainable)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with Kaiming uniform (like nn.Linear)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros (so initially LoRA has no effect)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA adaptation: BA * x
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            LoRA output (..., out_features)
        """
        # x @ A^T -> (..., rank)
        # result @ B^T -> (..., out_features)
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with optional LoRA adaptation.
    
    Wraps nn.Linear and adds LoRA on top of it.
    The original linear layer is frozen during fine-tuning.
    
    Args:
        linear_layer: Existing nn.Linear layer to adapt
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
    """
    
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Freeze original layer
        self.linear = linear_layer
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Add LoRA
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Original linear + LoRA adaptation
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.linear(x) + self.lora(x)
    
    def merge_weights(self):
        """Merge LoRA weights into the base linear layer."""
        if hasattr(self, '_merged') and self._merged:
            return  # Already merged
        
        # Compute the LoRA weight update: scaling * B @ A
        lora_weight = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
        
        # Add to base weight
        self.linear.weight.data += lora_weight
        self._merged = True
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from the base linear layer."""
        if not hasattr(self, '_merged') or not self._merged:
            return  # Not merged
        
        # Compute the LoRA weight update: scaling * B @ A
        lora_weight = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
        
        # Subtract from base weight
        self.linear.weight.data -= lora_weight
        self._merged = False


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
    target_modules: Optional[list] = None
):
    """
    Apply LoRA to specified modules in the model.
    
    Args:
        model: Transformer model
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        target_modules: List of module name patterns to apply LoRA to.
                       Default: ['query', 'key', 'value', 'output'] (attention layers)
    
    Returns:
        Modified model with LoRA layers
    """
    if target_modules is None:
        # Default: Apply to all attention projection layers
        target_modules = ['query', 'key', 'value', 'output']
    
    lora_count = 0
    
    # Recursively find and replace linear layers
    for name, module in model.named_modules():
        # Skip root module
        if name == '':
            continue
        
        # Check if this module should have LoRA
        should_apply_lora = any(target in name for target in target_modules)
        
        if should_apply_lora and isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            
            # Replace with LoRA version
            lora_layer = LinearWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_layer)
            lora_count += 1
    
    return model, lora_count


def mark_only_lora_as_trainable(model: nn.Module):
    """
    Freeze all parameters except LoRA parameters.
    
    Args:
        model: Model with LoRA layers
    """
    for name, param in model.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True


def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Get only LoRA parameters from model state dict.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Dictionary containing only LoRA parameters
    """
    lora_state_dict = {
        name: param for name, param in model.named_parameters()
        if 'lora' in name.lower()
    }
    return lora_state_dict


def load_lora_weights(model: nn.Module, lora_state_dict: dict):
    """
    Load LoRA weights into model.
    
    Args:
        model: Model with LoRA layers
        lora_state_dict: Dictionary containing LoRA parameters
    """
    model.load_state_dict(lora_state_dict, strict=False)


def merge_lora_weights(model: nn.Module):
    """
    Merge LoRA weights into the base model for inference.
    This eliminates LoRA overhead during inference.
    
    Args:
        model: Model with LoRA layers
    
    Returns:
        Model with merged weights (LoRA layers removed)
    """
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            # Compute merged weight: W0 + scaling * B @ A
            merged_weight = module.linear.weight.data + (
                module.lora.lora_B @ module.lora.lora_A * module.lora.scaling
            )
            
            # Create new linear layer with merged weights
            merged_linear = nn.Linear(
                module.linear.in_features,
                module.linear.out_features,
                bias=module.linear.bias is not None
            )
            merged_linear.weight.data = merged_weight
            if module.linear.bias is not None:
                merged_linear.bias.data = module.linear.bias.data
            
            # Replace in model
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            
            setattr(parent, attr_name, merged_linear)
    
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count parameters in model.
    
    Args:
        model: Model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_lora_parameters(model: nn.Module):
    """
    Get only LoRA parameters for optimization.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            lora_params.extend(module.lora.parameters())
    return lora_params


def unmerge_lora_weights(model: nn.Module):
    """
    Unmerge all LoRA weights from base layers.
    
    Args:
        model: Model with LoRA layers
    """
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            module.unmerge_weights()


def print_lora_info(model: nn.Module):
    """
    Print LoRA configuration info.
    
    Args:
        model: Model with LoRA layers
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print("=" * 60)
    print("LoRA Configuration")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Count LoRA layers
    lora_layers = sum(1 for name, _ in model.named_modules() 
                     if isinstance(_, LinearWithLoRA))
    print(f"Number of LoRA layers: {lora_layers}")
    print("=" * 60)
