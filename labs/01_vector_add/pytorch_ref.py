import torch

def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.rand()