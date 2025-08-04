import torch

NUM_CLASSES = 4


def one_hot(num: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(num), num_classes=NUM_CLASSES)
