import torch

def relu_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.zeros_like(x))

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(100_000, device="cuda")
    print("relu(x):", relu_ref(x)[:5])