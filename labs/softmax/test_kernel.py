import pytest, torch
from triton_kernel import softmax
from pytorch_ref import softmax as softmax_ref

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@pytest.mark.parametrize("M,N", [
    (1, 1024),
    (4096, 1024),    # running example, BLOCK_SIZE = 1024, no padding
    (4096, 1000),    # BLOCK_SIZE = 1024, padding required
    (17, 512),
    (100, 2000),     # BLOCK_SIZE = 2048
])
def test_softmax(M, N):
    x = torch.randn(M, N, device=DEVICE)
    assert torch.allclose(softmax(x), softmax_ref(x), rtol=1e-5, atol=1e-5)