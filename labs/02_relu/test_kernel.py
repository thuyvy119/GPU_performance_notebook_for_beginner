import pytest
import torch

from pytorch_ref import relu_ref
from triton_kernel import relu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

@pytest.mark.parametrize("n", [1, 17, 1024, 98_432, 100_000, 1_000_000])
def test_relu_matches_torch(n):
    torch.manual_seed(0)
    x = torch.rand(n, device="cuda")
    torch.testing.assert_close(relu(x), relu_ref(x))
    
def test_relu_zeros_negatives():
    x = torch.tensor([-3.0, -0.5, -1e-6, 0.0, 1e-6, 0.5, 3.0], device="cuda")
    output = relu(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 1e-6, 0.5, 3.0], device="cuda")
    torch.testing.assert_close(output, expected)
    
def test_relu_all_negative():
    x = -torch.rand(5000, device="cuda") - 0.01
    output = relu(x)
    assert bool((output == 0).all())
    
def test_relu_all_positive():
    x = torch.rand(5000, device="cuda") + 0.01
    output = relu(x)
    torch.testing.assert_close(output, x)