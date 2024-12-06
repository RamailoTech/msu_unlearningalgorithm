import torch
import gc

def get_param(net):
    """
    Clone the parameters of the network.

    Args:
        net (torch.nn.Module): The network to clone parameters from.

    Returns:
        List[torch.Tensor]: Cloned parameters.
    """
    new_param = []
    with torch.no_grad():
        for name, param in net.named_parameters():
            new_param.append(param.clone())
    torch.cuda.empty_cache()
    gc.collect()
    return new_param

def set_param(net, old_param):
    """
    Set the network's parameters from a list of tensors.

    Args:
        net (torch.nn.Module): The network to set parameters for.
        old_param (List[torch.Tensor]): The list of parameter tensors.

    Returns:
        torch.nn.Module: The network with updated parameters.
    """
    with torch.no_grad():
        for param, old_p in zip(net.parameters(), old_param):
            param.copy_(old_p)
    torch.cuda.empty_cache()
    gc.collect()
    return net
