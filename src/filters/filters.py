import torch
from collections.abc import Callable

# Map (B,1,H,W) --> Angle scores (B,N_angles)
Filter = Callable[[torch.Tensor], torch.Tensor]
