import torch

def softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=1)

def naive_softmax(x:torch.Tensor) -> torch.Tensor:
    # compute softmax in naive way, without using torch.softmax, for benchmarking purposes
    x_max = x.max(dim=1, keepdim=True)[0]
    z = x - x_max
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1, keepdim=True)
    return numerator / denominator

    