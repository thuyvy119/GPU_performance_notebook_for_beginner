'''
ReLU(x) = max(x, 0)  applied element-wise

Similar to lab 01.vector_add, but instead of adding two vectors, we are applying the ReLU function to a single vector
x is a length-100000 float32 vector
BLOCK_SIZE = 1024, grid = ceil(100_000 / 1024) = 98 programs, pid in [0, 97]
pid 0..96 each handle 1024 elements, only pid 97 handles only 672 (the rest is masked).
'''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
    x = tl.load(x_ptr +  offsets, mask=mask, other=0.0)
    
    # tl.maximum is the Triton equivalent of torch.clamp_min(x, 0) or torch.nn.functional.relu
    # It is element-wise on the 1024-lane, register tile, so it processes all BLOCK_SIZE values in parallel
    # For pid=97 the last 352 lanes contain 0.0 (from `other`), 
    # so they compute max(0, 0) = 0, harmless and never stored
    output = tl.maximum(x, 0.0)
    
    # store result back to memory
    # similar to lab 01.vector_add: pid=97 write only the 672 valid output[99328 .. 99999]
    # and skips the 352 phantom slots
    tl.store(output_ptr + offsets, output, mask=mask)
    
def relu(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.is_cuda, "tensor must be on the GPU"
    
    # Space for the result (allocated on the GPU, same shape and dtype as x)
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # similar to lab 01.vector_add: ceil(100000 / 1024) = 98 programs to launch
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=block_size)
    return output

