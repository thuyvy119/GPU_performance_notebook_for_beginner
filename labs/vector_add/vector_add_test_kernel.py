import pytest, torch
from triton_kernel import vector_add
from pytorch_ref import vector_add_ref

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@pytest.mark.parametrize("n", [1, 17, 1024, 98_432, 100_000, 1_000_000])
def test_vector_add_matches_torch(n):
    x = torch.rand(n, device=DEVICE)
    y = torch.rand(n, device=DEVICE)
    torch.testing.assert_close(vector_add(x, y), vector_add_ref(x, y))
