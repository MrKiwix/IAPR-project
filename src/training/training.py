# Contains functions and optimization tools for the training of the model
import torch
from torch import nn

def get_optimization_groups(model, lr_head=1e-3, lr_backbone=1e-4):
    """ 
    This function returns the optimization groups for the model.
    It separates the parameters of the model into two groups:
    1. The parameters of the head (the last layer) with a different learning rate.
    2. The parameters of the rest of the model with a different learning rate.

    Args:
        model (torch.nn.Module): The model to be optimized.
        lr_head (float): The learning rate for the head parameters. Defaults to 1e-3.
        lr_backbone (float): The learning rate for the backbone parameters. Defaults to 1e-4.
    Returns:
        list: A list of dictionaries containing the parameters and their corresponding learning rates.
    """    
    
    return [
        {"params": [p for n, p in model.named_parameters() if n.startswith("head.")],
        "lr": lr_head},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("head.")],
        "lr": lr_backbone},
    ]

def get_optimizer(model, weight_decay=1e-4):
    """
    This function returns the optimizer for the model.
    It uses the AdamW optimizer with weight decay.

    Args:
        weight_decay (float): The weight decay for the optimizer. Defaults to 1e-4.
        momentum (float): The momentum for the optimizer. Defaults to 0.9.
    Returns:
        torch.optim.Optimizer: The optimizer for the model.
    """    
    
    return torch.optim.AdamW(
        get_optimization_groups(model),
        lr=1e-3,
        weight_decay=weight_decay,
    )
    

    
###########################
# Training and Eval Epochs

def train_epoch(loader, model, loss_fn, optim, device):
    
    model.train()
    running_loss = 0.0
    
    for imgs, targets in loader:
        
        imgs     = imgs.to(device, non_blocking=True)
        targets  = targets.float().to(device, non_blocking=True)

        preds = model(imgs)
        loss  = loss_fn(preds, targets)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(loader, model, loss_fn, num_classes, device):
    
    # Set the model to evaluation mode
    model.eval()
    
    total_loss = 0.0
    mae_sum = torch.zeros(num_classes, device=device)
    f1_sum = 0.0

    for imgs, targets in loader:
        
        # Soft validation -> no clamping and no rounding
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.float().to(device, non_blocking=True)
        preds = model(imgs)
        total_loss += loss_fn(preds, targets).item() * imgs.size(0)

        # Hard validation -> clamping and rounding
        hard = torch.round(torch.nn.functional.relu(preds))
        # Compute per-image TP and FPN
        tp_per_image  = torch.min(hard, targets).sum(dim=1)         # shape: [batch]
        fpn_per_image = torch.abs(hard - targets).sum(dim=1)        # shape: [batch]

        smooth=1e-10
        # Compute per-image F1, then sum them
        f1_per_image = (2*tp_per_image + smooth) \
                    / (2*tp_per_image + fpn_per_image + smooth)
        f1_sum += f1_per_image.sum().item()
        
        mae_sum += (preds - targets).abs().sum(dim=0)
        
        
    avg_loss = total_loss / len(loader.dataset)
    avg_f1 = f1_sum / len(loader.dataset)  # per-class F1 score
    mae = (mae_sum / len(loader.dataset)).cpu()   # per-class MAE

    return avg_loss, avg_f1, mae