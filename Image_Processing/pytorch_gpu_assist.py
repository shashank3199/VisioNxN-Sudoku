"""
  pytorch_gpu_assist.py :   This file contains helper functions to implement training on GPUs.
  File created by       :   Shashank Goyal
  Last commit done by   :   Shashank Goyal
  Last commit date      :   3rd September
"""

# import the pytorch module
import torch


def get_default_device():
    """Pick GPU if available, else CPU"""

    # if NVIDIA GPU is available for computation
    if torch.cuda.is_available():
        # return cuda as computation device
        return torch.device('cuda')
    else:
        # return cpu as computation device
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""

    # if the data is of list or tuple type
    if isinstance(data, (list, tuple)):
        # move the list to computation device
        return [to_device(x, device) for x in data]
    # move the data to the computation device
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        """default initialization"""

        # data loader object
        self.dl = dl
        # device
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""

        # for each iterable in the data loader object
        for b in self.dl:
            # yield the moved data
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
