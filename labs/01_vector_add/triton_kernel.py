'''
Triton vector addition: z = x + y.

Running example used in note.md and README.md:
x, y are length-100000 float32 vectors
BLOCK_SIZE = 1024, grid = ceil(100_000 / 1024) = 98 programs, pid in [0, 97]
pid 0..96 each handle 1024 elements, only pid 97 handles only 672 (the rest is masked)
'''
import torch
import triton
import triton.language as tl

# Triton launches many copies of this function ("programs") at the same time
# Each copy works on a different chunk (block) of the vectors
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # "Which copy am I?"   pid in {0, 1, ..., 97}   
    pid = tl.program_id(axis=0)
    
    # Where does my chunk begin? 
    # pid=0 -> block_start = 0
    # pid=97 -> block_start = 97*1024 = 99,328
    block_start = pid * BLOCK_SIZE

    # The 1024 positions I am responsible for
    # pid=0 -> 0 + [0, 1, ..., 1023] = [0, 1, ..., 1023]
    # pid=97 -> 99328 + [0, 1, ..., 1023] = [99328, 99329, ..., 100351]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Which of my positions really exist? (the vectors end at index 99999)
    # -> 99328..99999 are < 100000  -> True  (672 of them)
    # -> 100000..100351 are not     -> False (352 of them)
    # For pid 0..96 the mask is all True
    mask = offsets < n_elements

    # Load
    # Masked slots are not read from memory (safe default = 0)
    # pid=97: read x[99328] .. x[99999] from GPU memory (672 values)
    # 352 masked slots are not read from memory, they get 0.0 because we passed other=0.0
    # (without `other`, masked values are undefined, so do not rely on them being zero)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute. 
    # 1024 additions happen at once. The last 352 are 0.0 + 0.0 and are thrown away below
    output = x + y
    
    # pid=97: write only the 672 valid results: output[99328] .. output[99999]
    # Nothing is written past index 99999, so we never touch memory we do not own
    tl.store(output_ptr + offsets, output, mask=mask)
    

def vector_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "tensors must be on the GPU"
    assert x.shape == y.shape, "x and y must have the same shape"
    
    # Space for the result (allocated on the GPU, same shape and dtype as x)
    output = torch.empty_like(x)
    
    # 100_000 in our running example
    n_elements = output.numel()

    # How many programs to launch? ceil(n_elements / BLOCK_SIZE) = ceil(100000 / 1024) = 98
    # The grid is a tuple because it can be 1D, 2D or 3D, here it is 1D
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch 98 programs. Launches are asynchronous: this line returns before the GPU finishes
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    
    return output

