import torch
import logging


def prepare_gpu(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()  # number of available gpu
    if n_gpu_use > 0 and n_gpu == 0:
        logging.warning(
            "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logging.warning(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
